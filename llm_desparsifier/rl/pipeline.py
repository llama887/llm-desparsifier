
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, Callable, Mapping, Optional, Protocol, TypedDict

import flax
import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
import distrax
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.training.train_state import TrainState

import xminigrid
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import EnvParams, Environment
from xminigrid.wrappers import GymAutoResetWrapper

from llm_desparsifier.rl.structures import RolloutStats, Transition
from llm_desparsifier.rl.wrappers import DesparsifyRewardWrapper


class RewardGeneratorProtocol(Protocol):
    """Interface for reward generators consumed by the RL pipeline."""

    def generate(self, env: Environment, env_params: EnvParams) -> tuple[Callable[..., Any], str]:
        """Return a dense reward function and the emitted source code."""


@dataclass
class TrainingResult:
    """Structured output from `run_training_with_reward`."""

    config: "TrainConfig"
    train_info: Mapping[str, Any]
    artifacts: Mapping[str, str]
    final_metrics: Mapping[str, float]
    emitted_reward_code: str


# ======================
# Networks
# ======================

# Model adapted from minigrid baselines:
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py

# custom RNN cell, which is more convenient that default in flax
class GRU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param("Wi", glorot_normal(in_axis=1, out_axis=0), (self.hidden_dim * 3, input_dim))
        Wh = self.param("Wh", orthogonal(column_axis=0), (self.hidden_dim * 3, self.hidden_dim))
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,))
        bn = self.param("bn", zeros_init(), (self.hidden_dim,))

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state

class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(hidden_dim=self.hidden_dim)(xs, init_state[layer])
            outs.append(xs)
            states.append(state)

        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)

BatchedRNNModel = flax.linen.vmap(
    RNNModel, variable_axes={"params": None}, split_rngs={"params": False}, axis_name="batch"
)


class ActorCriticInput(TypedDict):
    observation: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class ActorCriticRNN(nn.Module):
    num_actions: int
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    img_obs: bool = False

    @nn.compact
    def __call__(self, inputs: ActorCriticInput, hidden: jax.Array):
        B, S = inputs["observation"].shape[:2]
        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        if self.img_obs:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(16, (3, 3), strides=2, padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    nn.Conv(32, (3, 3), strides=2, padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    nn.Conv(32, (3, 3), strides=2, padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    nn.Conv(32, (3, 3), strides=2, padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                ]
            )
        else:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(16, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                ]
            )
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        rnn_core = BatchedRNNModel(self.rnn_hidden_dim, self.rnn_num_layers)
        actor = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2)),
                nn.tanh,
                nn.Dense(self.num_actions, kernel_init=orthogonal(0.01)),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2)),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        # [batch_size, seq_len, ...]
        obs_emb = img_encoder(inputs["observation"]).reshape(B, S, -1)
        act_emb = action_encoder(inputs["prev_action"])
        # [batch_size, seq_len, hidden_dim + act_emb_dim + 1]
        out = jnp.concatenate([obs_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)
        # core networks
        out, new_hidden = rnn_core(out, hidden)
        dist = distrax.Categorical(logits=actor(out))
        values = critic(out)

        return dist, jnp.squeeze(values, axis=-1), new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim))


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
):
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "observation": transitions.obs,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean((loss, vloss, aloss, entropy, grads), axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "observation": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        extras = getattr(timestep, "extras", None)
        if extras is not None:
            get_fn = getattr(extras, "get", None)
            if callable(get_fn):
                gt_reward = get_fn("ground_truth_reward", timestep.reward)
            elif "ground_truth_reward" in extras:
                gt_reward = extras["ground_truth_reward"]
            else:
                gt_reward = timestep.reward
        else:
            gt_reward = timestep.reward

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            ground_truth_reward=stats.ground_truth_reward + gt_reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]

# ======================
# Training
# ======================

@dataclass
class TrainConfig:
    env_id: str = "XLand-MiniGrid-R1-8x8"
    benchmark_id: str = "trivial-1m"
    img_obs: bool = False
    # agent
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    # training
    num_envs: int = 2048
    num_steps_per_env: int = 4096
    num_steps_per_update: int = 32
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 1_000_000_000
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_num_envs: int = 256
    eval_num_episodes: int = 50
    eval_seed: int = 42
    train_seed: int = 42

    def __post_init__(self):
        num_devices = jax.local_device_count()
        
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_num_envs_per_device = self.eval_num_envs // num_devices
        assert self.num_envs % num_devices == 0
        
        self.num_meta_updates = round(self.total_timesteps_per_device / (self.num_envs_per_device * self.num_steps_per_env))
        self.num_inner_updates = self.num_steps_per_env // self.num_steps_per_update
        assert self.num_steps_per_env % self.num_steps_per_update == 0
        print(f"Num devices: {num_devices}, Num meta updates: {self.num_meta_updates}")

def make_states(
    config: "TrainConfig",
    reward_generator: RewardGeneratorProtocol,
    output_dir: str,
    ctx_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
):
    # for learning rage scheduling
    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_inner_updates
        frac = 1.0 - (count // total_inner_updates) / config.num_meta_updates
        return config.lr * frac

    # setup environment
    if "XLand" not in config.env_id:
        raise ValueError("Only meta-task environments are supported.")

    env, env_params = xminigrid.make(config.env_id)
    env = GymAutoResetWrapper(env)

    data_root = os.environ.setdefault(
        "XLAND_MINIGRID_DATA", os.path.join(os.getcwd(), "data", "xland_minigrid")
    )
    os.makedirs(data_root, exist_ok=True)

    benchmark = xminigrid.load_benchmark(config.benchmark_id)
    # Use a deterministic ruleset example so the LLM sees a concrete task description.
    example_ruleset = benchmark.get_ruleset(0)
    env_params = env_params.replace(ruleset=example_ruleset)

    dense_reward, emitted_code = reward_generator.generate(env, env_params)
    os.makedirs(output_dir, exist_ok=True)
    dense_path = os.path.join(output_dir, "dense_reward_synthesized.py")
    with open(dense_path, "w", encoding="utf-8") as f:
        f.write(emitted_code)
    env = DesparsifyRewardWrapper(env, dense_fn=dense_reward, ctx_fn=ctx_fn)

    # enabling image observations if needed
    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)
    
    # set up training state
    rng = jax.random.key(config.train_seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )
    # [batch_size, seq_len, ...]
    init_obs = {
        "observation": jnp.zeros((config.num_envs_per_device, 1, *env.observation_shape(env_params))),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    return rng, env, env_params, benchmark, init_hstate, train_state, emitted_code, dense_path


def make_train(
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
        # META TRAIN LOOP
        def _meta_step(meta_state, _):
            rng, train_state = meta_state

            # INIT ENV
            rng, _rng1, _rng2 = jax.random.split(rng, num=3)
            ruleset_rng = jax.random.split(rng, num=config.num_envs_per_device)
            reset_rng = jax.random.split(rng, num=config.num_envs_per_device)

            # sample rulesets for this meta update
            rulesets = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)
            meta_env_params = env_params.replace(ruleset=rulesets)

            timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
            prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
            prev_reward = jnp.zeros(config.num_envs_per_device)

            # INNER TRAIN LOOP
            def _update_step(runner_state, _):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, _):
                    rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    dist, value, hstate = train_state.apply_fn(
                        train_state.params,
                        {
                            # [batch_size, seq_len=1, ...]
                            "observation": prev_timestep.observation[:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                        },
                        prev_hstate,
                    )
                    action, log_prob = dist.sample_and_log_prob(seed=_rng)
                    # squeeze seq_len where possible
                    action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                    # STEP ENV
                    timestep = jax.vmap(env.step, in_axes=0)(meta_env_params, prev_timestep, action)
                    transition = Transition(
                        # ATTENTION: done is always false, as we optimize for entire meta-rollout
                        done=jnp.zeros_like(timestep.last()),
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        log_prob=log_prob,
                        obs=prev_timestep.observation,
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                    )
                    runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                    return runner_state, transition

                initial_hstate = runner_state[-1]
                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)

                # CALCULATE ADVANTAGE
                rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
                # calculate value of the last step for bootstrapping
                _, last_val, _ = train_state.apply_fn(
                    train_state.params,
                    {
                        "observation": timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    hstate,
                )
                advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            init_hstate=init_hstate.squeeze(1),
                            advantages=advantages,
                            targets=targets,
                            clip_eps=config.clip_eps,
                            vf_coef=config.vf_coef,
                            ent_coef=config.ent_coef,
                        )
                        return new_train_state, update_info

                    rng, train_state, init_hstate, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                    # [seq_len, batch_size, ...]
                    batch = (init_hstate, transitions, advantages, targets)
                    # [batch_size, seq_len, ...], as our model assumes
                    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                    # [num_minibatches, minibatch_size, ...]
                    minibatches = jtu.tree_map(
                        lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                    )
                    train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                    update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                    return update_state, update_info

                # hstate shape: [seq_len=None, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
                # WARN: do not forget to get updated params
                rng, train_state = update_state[:2]

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                runner_state = (rng, train_state, timestep, prev_action, prev_reward, hstate)
                return runner_state, loss_info

            # on each meta-update we reset hidden to init_hstate
            runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)
            runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_inner_updates)
            # WARN: do not forget to get updated params
            rng, train_state = runner_state[:2]

            # EVALUATE AGENT
            eval_ruleset_rng, eval_reset_rng = jax.random.split(jax.random.key(config.eval_seed))
            eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs_per_device)
            eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs_per_device)

            eval_ruleset = jax.vmap(benchmark.sample_ruleset)(eval_ruleset_rng)
            eval_env_params = env_params.replace(ruleset=eval_ruleset)

            eval_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
                eval_reset_rng,
                env,
                eval_env_params,
                train_state,
                # TODO: make this a static method?
                jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                config.eval_num_episodes,
            )
            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")

            # averaging over inner updates, adding evaluation metrics
            loss_info = jtu.tree_map(lambda x: x.mean(-1), loss_info)
            loss_info.update(
                {
                    "eval/returns_mean": eval_stats.reward.mean(0),
                    "eval/returns_median": jnp.median(eval_stats.reward),
                    "eval/ground_truth_returns_mean": eval_stats.ground_truth_reward.mean(0),
                    "eval/ground_truth_returns_median": jnp.median(eval_stats.ground_truth_reward),
                    "eval/lengths": eval_stats.length.mean(0),
                    "eval/lengths_20percentile": jnp.percentile(eval_stats.length, q=20),
                    "eval/returns_20percentile": jnp.percentile(eval_stats.reward, q=20),
                    "eval/ground_truth_returns_20percentile": jnp.percentile(
                        eval_stats.ground_truth_reward, q=20
                    ),
                    "eval/returns_abs_gap_mean": jnp.mean(
                        jnp.abs(eval_stats.reward - eval_stats.ground_truth_reward)
                    ),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )
            meta_state = (rng, train_state)
            return meta_state, loss_info

        meta_state = (rng, train_state)
        meta_state, loss_info = jax.lax.scan(_meta_step, meta_state, None, config.num_meta_updates)
        return {"state": meta_state[-1], "loss_info": loss_info}

    return train


def run_training_with_reward(
    reward_generator: RewardGeneratorProtocol,
    output_dir: str,
    *,
    ctx_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    config_override: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[int, Mapping[str, float]], None]] = None,
) -> TrainingResult:
    """Execute the full RL pipeline with a supplied dense reward generator."""

    config_kwargs = dict(config_override or {})
    config = TrainConfig(**config_kwargs)

    (
        rng,
        env,
        env_params,
        benchmark,
        init_hstate,
        train_state,
        emitted_code,
        dense_path,
    ) = make_states(config, reward_generator, output_dir, ctx_fn=ctx_fn)

    # Replicate args across devices.
    rng_devices = jax.random.split(rng, num=jax.local_device_count())
    train_state_devices = replicate(train_state, jax.local_devices())
    init_hstate_devices = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    compile_start = time.time()
    train_fn = make_train(env, env_params, benchmark, config)
    train_fn = train_fn.lower(rng_devices, train_state_devices, init_hstate_devices).compile()
    compile_elapsed = time.time() - compile_start
    print(f"Done in {compile_elapsed:.2f}s.")

    print("Training...")
    train_start = time.time()
    train_info = jax.block_until_ready(train_fn(rng_devices, train_state_devices, init_hstate_devices))
    train_elapsed = time.time() - train_start
    print(f"Done in {train_elapsed / 60:.2f}min")

    # Bring results back to host.
    train_info = unreplicate(train_info)
    loss_info = jtu.tree_map(lambda x: jnp.asarray(x), train_info["loss_info"])

    final_dense_return = float(loss_info["eval/returns_mean"][-1])
    final_gt_return = float(loss_info["eval/ground_truth_returns_mean"][-1])
    final_abs_gap = float(loss_info["eval/returns_abs_gap_mean"][-1])
    print("Final dense return:", final_dense_return)
    print("Final ground-truth return:", final_gt_return)
    print("Final |dense-ground| mean gap:", final_abs_gap)

    if progress_callback is not None:
        steps = loss_info["eval/returns_mean"].shape[0]
        for idx in range(steps):
            step_metrics: dict[str, float] = {}
            for name, value in loss_info.items():
                value_arr = jnp.asarray(value)
                if value_arr.ndim == 0:
                    step_metrics[name] = float(value_arr)
                else:
                    step_metrics[name] = float(value_arr[idx])
            progress_callback(idx, step_metrics)

    meta_updates = jnp.arange(config.num_meta_updates)
    dense_series = loss_info["eval/returns_mean"]
    gt_series = loss_info["eval/ground_truth_returns_mean"]

    gt_curve_path = os.path.join(output_dir, "training_curve_ground_truth.png")
    dense_curve_path = os.path.join(output_dir, "training_curve_dense.png")
    combined_curve_path = os.path.join(output_dir, "training_curve.png")

    plt.figure()
    plt.plot(meta_updates, gt_series, label="Ground-truth reward")
    plt.title("Ground-truth Eval Returns over Meta Updates")
    plt.xlabel("Meta Update")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(gt_curve_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(meta_updates, dense_series, label="Dense reward", color="tab:orange")
    plt.title("Dense Eval Returns over Meta Updates")
    plt.xlabel("Meta Update")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(dense_curve_path, dpi=150)
    plt.close()

    def _normalize_to_unit_interval(series):
        min_val = jnp.min(series)
        max_val = jnp.max(series)
        range_val = max_val - min_val
        inv_range = jnp.where(range_val > 0, 1.0 / range_val, 0.0)
        return jnp.where(range_val > 0, (series - min_val) * inv_range, jnp.zeros_like(series))

    dense_series_norm = _normalize_to_unit_interval(dense_series)
    gt_series_norm = _normalize_to_unit_interval(gt_series)

    plt.figure()
    plt.plot(meta_updates, gt_series_norm, label="Ground-truth reward (normalized)")
    plt.plot(
        meta_updates,
        dense_series_norm,
        label="Dense reward (normalized)",
        linestyle="--",
    )
    plt.title("Normalized Eval Returns over Meta Updates")
    plt.xlabel("Meta Update")
    plt.ylabel("Normalized Return [0, 1]")
    plt.legend()
    plt.savefig(combined_curve_path, dpi=150)
    plt.close()

    # ========== Evaluation ==========
    from xminigrid.rendering.text_render import print_ruleset

    META_EPISODES = 10

    env_local, env_params_local = xminigrid.make(config.env_id)
    env_local = GymAutoResetWrapper(env_local)

    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env_local = RGBImgObservationWrapper(env_local)

    ruleset = xminigrid.load_benchmark(config.benchmark_id).get_ruleset(ruleset_id=0)
    env_params_local = env_params_local.replace(ruleset=ruleset)

    params = train_info["state"].params
    model = ActorCriticRNN(
        num_actions=env_local.num_actions(env_params_local),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )

    apply_fn = jax.jit(model.apply)
    reset_fn = jax.jit(env_local.reset)
    step_fn = jax.jit(env_local.step)

    total_reward, num_episodes = 0.0, 0
    rendered_imgs = []

    eval_rng = jax.random.key(1)
    eval_rng, _rng = jax.random.split(eval_rng)

    hidden = model.initialize_carry(1)
    prev_reward = jnp.asarray(0)
    prev_action = jnp.asarray(0)

    timestep = reset_fn(env_params_local, _rng)
    rendered_imgs.append(env_local.render(env_params_local, timestep))

    while num_episodes < META_EPISODES:
        eval_rng, _rng = jax.random.split(eval_rng)
        dist, _, hidden = apply_fn(
            params,
            {
                "observation": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hidden,
        )
        action = dist.sample(seed=_rng).squeeze()

        timestep = step_fn(env_params_local, timestep, action)
        prev_action = action
        prev_reward = timestep.reward

        total_reward += float(timestep.reward)
        num_episodes += int(timestep.last())
        rendered_imgs.append(env_local.render(env_params_local, timestep))

    rollout_path = os.path.join(output_dir, "eval_rollout.mp4")
    print("Reward:", total_reward)
    print("Ruleset:")
    print_ruleset(ruleset)
    imageio.mimsave(rollout_path, rendered_imgs, fps=16, format="mp4")
    print(f"Saved artifacts to {output_dir}")

    final_metrics = {
        "dense_return": final_dense_return,
        "ground_truth_return": final_gt_return,
        "dense_ground_abs_gap": final_abs_gap,
        "total_eval_reward": total_reward,
        "num_eval_episodes": num_episodes,
    }

    artifacts = {
        "dense_reward_path": dense_path,
        "training_curve_ground_truth": gt_curve_path,
        "training_curve_dense": dense_curve_path,
        "training_curve_combined": combined_curve_path,
        "eval_rollout": rollout_path,
    }

    return TrainingResult(
        config=config,
        train_info=train_info,
        artifacts=artifacts,
        final_metrics=final_metrics,
        emitted_reward_code=emitted_code,
    )
