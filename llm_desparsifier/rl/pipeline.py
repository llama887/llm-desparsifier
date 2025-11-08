
from __future__ import annotations

import math
import os
import time
import csv
import json
from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, TypedDict
from typing import Literal

import flax
import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
import distrax
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.training.train_state import TrainState

import xminigrid
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import EnvParams, Environment
from xminigrid.wrappers import GymAutoResetWrapper

from llm_desparsifier.rl.eval import GroundTruthEvalConfig, run_ground_truth_eval
from llm_desparsifier.rl.metrics import (
    GroundTruthLogRow,
    compute_ground_truth_summary,
    dump_ground_truth_logs,
    dump_ground_truth_summary,
)
from llm_desparsifier.rl.structures import RolloutStats, Transition
from llm_desparsifier.rl.wrappers import DesparsifyRewardWrapper
from llm_desparsifier.utils import extract_xland_ctx


class RewardGeneratorProtocol(Protocol):
    """Interface for reward generators consumed by the RL pipeline."""

    def generate(self, env: Environment, env_params: EnvParams) -> tuple[Callable[..., Any], str]:
        """Return a dense reward function and the emitted source code."""

RewardMode = Literal["dense", "sparse"]


@dataclass
class TrainingResult:
    """Structured output from `run_training_with_reward`."""

    config: "TrainConfig"
    train_info: Mapping[str, Any]
    artifacts: Mapping[str, str]
    final_metrics: Mapping[str, float]
    emitted_reward_code: str
    reward_mode: RewardMode = "dense"
    ground_truth_eval: Optional[Mapping[str, Any]] = None


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


def _load_ground_truth_series(csv_path: str):
    steps, returns, stds = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["global_step"]))
            returns.append(float(row["gt_return"]))
            stds.append(float(row.get("gt_return_std", 0.0)))
    if not steps:
        return None
    return {
        "steps": np.asarray(steps),
        "returns": np.asarray(returns),
        "std": np.asarray(stds),
    }


def _load_summary(summary_path: str):
    if not summary_path:
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_gt_comparison(csv_paths: Mapping[str, str], summary_paths: Mapping[str, str], out_path: str) -> Optional[str]:
    run_series: dict[str, dict[str, np.ndarray]] = {}
    run_summaries: dict[str, Mapping[str, Any]] = {}
    for mode, csv_path in csv_paths.items():
        if not csv_path or not os.path.exists(csv_path):
            continue
        series = _load_ground_truth_series(csv_path)
        if series is None:
            continue
        run_series[mode] = series
        summary_path = summary_paths.get(mode)
        if summary_path and os.path.exists(summary_path):
            run_summaries[mode] = _load_summary(summary_path)

    if not run_series:
        return None

    modes = sorted(run_series.keys())
    colors = {
        "dense": "tab:blue",
        "sparse": "tab:orange",
    }

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_main = fig.add_subplot(grid[0])
    ax_diff = fig.add_subplot(grid[1], sharex=ax_main)

    for mode in modes:
        series = run_series[mode]
        steps = series["steps"]
        returns = series["returns"]
        std = series["std"]
        color = colors.get(mode, None)
        label = f"{mode.capitalize()} GT"
        ax_main.plot(steps, returns, label=label, color=color)
        if np.any(std):
            ax_main.fill_between(steps, returns - std, returns + std, color=color, alpha=0.2)

        summary = run_summaries.get(mode)
        if summary and summary.get("max_return") is not None and summary.get("argmax_step") is not None:
            ax_main.scatter(summary["argmax_step"], summary["max_return"], color=color, marker="o")
            ax_main.annotate(
                f"max {summary['max_return']:.1f}",
                xy=(summary["argmax_step"], summary["max_return"]),
                xytext=(5, 5),
                textcoords="offset points",
                color=color,
                fontsize=8,
            )

    ax_main.set_title("Ground-truth Returns vs Environment Steps")
    ax_main.set_ylabel("Return")
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)

    if "dense" in run_series and "sparse" in run_series:
        dense_steps = run_series["dense"]["steps"]
        dense_returns = run_series["dense"]["returns"]
        sparse_steps = run_series["sparse"]["steps"]
        sparse_returns = run_series["sparse"]["returns"]
        shared_steps = np.intersect1d(dense_steps, sparse_steps)
        if shared_steps.size:
            dense_interp = np.interp(shared_steps, dense_steps, dense_returns)
            sparse_interp = np.interp(shared_steps, sparse_steps, sparse_returns)
            diff = dense_interp - sparse_interp
            ax_diff.plot(shared_steps, diff, color="tab:purple", label="Dense - Sparse")
            ax_diff.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax_diff.set_ylabel("Δ Return")
            ax_diff.set_xlabel("Environment Steps")
            ax_diff.grid(True, alpha=0.3)
            ax_diff.legend()
        else:
            ax_diff.set_visible(False)
    else:
        ax_diff.set_visible(False)

    if run_summaries:
        table_lines = ["Run | Final | Max | Argmax | AUC | T_thresh"]
        for mode in modes:
            summary = run_summaries.get(mode)
            if summary:
                table_lines.append(
                    (
                        f"{mode:<5}| "
                        f"{summary.get('final_return', '-')!s:<6} | "
                        f"{summary.get('max_return', '-')!s:<6} | "
                        f"{summary.get('argmax_step', '-')!s:<7} | "
                        f"{summary.get('auc_ground_truth', '-')!s:<6} | "
                        f"{summary.get('time_to_threshold', '-')!s:<8}"
                    )
                )
        fig.text(0.02, 0.01, "\n".join(table_lines), fontsize=8, family="monospace")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

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
    gt_success_threshold: Optional[float] = None

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
    *,
    reward_mode: RewardMode = "dense",
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

    emitted_code = ""
    dense_path = ""
    ctx_fn_used = ctx_fn if reward_mode == "dense" else None

    if reward_mode == "dense":
        if reward_generator is None:
            raise ValueError("reward_generator must be provided for dense training runs")
        dense_reward, emitted_code = reward_generator.generate(env, env_params)
        os.makedirs(output_dir, exist_ok=True)
        dense_path = os.path.join(output_dir, "dense_reward_synthesized.py")
        with open(dense_path, "w", encoding="utf-8") as f:
            f.write(emitted_code)
        env = DesparsifyRewardWrapper(env, dense_fn=dense_reward, ctx_fn=ctx_fn)
    else:
        os.makedirs(output_dir, exist_ok=True)

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

    return rng, env, env_params, benchmark, init_hstate, train_state, emitted_code, dense_path, ctx_fn_used


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
                    "eval/returns_std": jnp.std(eval_stats.reward),
                    "eval/ground_truth_returns_mean": eval_stats.ground_truth_reward.mean(0),
                    "eval/ground_truth_returns_median": jnp.median(eval_stats.ground_truth_reward),
                    "eval/ground_truth_returns_std": jnp.std(eval_stats.ground_truth_reward),
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
    reward_mode: RewardMode = "dense",
) -> TrainingResult:
    """Execute the full RL pipeline with a supplied dense reward generator."""

    if reward_mode not in ("dense", "sparse"):
        raise ValueError(f"Unsupported reward_mode '{reward_mode}'. Expected 'dense' or 'sparse'.")

    config_kwargs = dict(config_override or {})
    config = TrainConfig(**config_kwargs)

    ctx_fn_to_use = ctx_fn
    if reward_mode == "dense":
        if ctx_fn_to_use is None and "XLand" in config.env_id:
            ctx_fn_to_use = extract_xland_ctx
    else:
        ctx_fn_to_use = None

    (
        rng,
        env,
        env_params,
        benchmark,
        init_hstate,
        train_state,
        emitted_code,
        dense_path,
        ctx_fn_used,
    ) = make_states(
        config,
        reward_generator,
        output_dir,
        ctx_fn=ctx_fn_to_use,
        reward_mode=reward_mode,
    )

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

    final_train_reward = float(loss_info["eval/returns_mean"][-1])
    final_gt_return = float(loss_info["eval/ground_truth_returns_mean"][-1])
    final_abs_gap = float(loss_info["eval/returns_abs_gap_mean"][-1])
    if reward_mode == "dense":
        print("Final dense return:", final_train_reward)
        print("Final ground-truth return:", final_gt_return)
        print("Final |dense-ground| mean gap:", final_abs_gap)
    else:
        print("Final sparse (ground-truth) return:", final_gt_return)

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
    gt_std_series = loss_info.get("eval/ground_truth_returns_std")

    steps_per_meta = config.num_envs * config.num_steps_per_env
    total_meta_updates = int(meta_updates.shape[0]) if meta_updates.size else 0
    run_id = os.path.basename(os.path.normpath(output_dir)) or "run"
    ground_truth_rows: list[GroundTruthLogRow] = []
    if total_meta_updates > 0:
        for idx in range(total_meta_updates):
            global_step = int((idx + 1) * steps_per_meta)
            wall_time = float(train_elapsed * ((idx + 1) / total_meta_updates)) if total_meta_updates else 0.0
            gt_return = float(gt_series[idx])
            gt_std = float(gt_std_series[idx]) if gt_std_series is not None else 0.0
            ground_truth_rows.append(
                GroundTruthLogRow(
                    run_id=run_id,
                    reward_mode=reward_mode,
                    global_step=global_step,
                    episode=idx,
                    gt_return=gt_return,
                    gt_return_std=gt_std,
                    wall_time=wall_time,
                    checkpoint_path="",
                )
            )

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
    dense_label = "Dense reward" if reward_mode == "dense" else "Training reward"
    plt.plot(meta_updates, dense_series, label=dense_label, color="tab:orange")
    plt.title(f"{dense_label} Eval Returns over Meta Updates")
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
        label=("Dense reward (normalized)" if reward_mode == "dense" else "Training reward (normalized)"),
        linestyle="--",
    )
    plt.title("Normalized Eval Returns over Meta Updates")
    plt.xlabel("Meta Update")
    plt.ylabel("Normalized Return [0, 1]")
    plt.legend()
    plt.savefig(combined_curve_path, dpi=150)
    plt.close()

    metrics_csv_path = None
    metrics_summary_path = None
    if ground_truth_rows:
        metrics_csv_path = dump_ground_truth_logs(output_dir, ground_truth_rows)
        summary_payload = compute_ground_truth_summary(
            ground_truth_rows, threshold=config.gt_success_threshold
        )
        summary_payload.update({
            "run_id": run_id,
            "reward_mode": reward_mode,
            "threshold": config.gt_success_threshold,
        })
        metrics_summary_path = dump_ground_truth_summary(output_dir, summary_payload)

    # ========== Evaluation via ground-truth harness ==========
    eval_model = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )
    eval_cfg = GroundTruthEvalConfig(
        env_id=config.env_id,
        benchmark_id=config.benchmark_id,
        num_episodes=config.eval_num_episodes,
        seed=config.eval_seed,
        img_obs=config.img_obs,
        capture_video=True,
    )
    gt_eval_result = run_ground_truth_eval(
        train_state=train_info["state"],
        model=eval_model,
        cfg=eval_cfg,
    )

    rollout_path = os.path.join(output_dir, "eval_rollout.mp4")
    if gt_eval_result.frames:
        imageio.mimsave(rollout_path, gt_eval_result.frames, fps=16, format="mp4")
    else:
        rollout_path = ""
    print(
        "Ground-truth eval: mean=%.4f ± %.4f over %d episodes"
        % (gt_eval_result.mean_return, gt_eval_result.std_return, len(gt_eval_result.returns))
    )

    ground_truth_eval = {
        "returns": gt_eval_result.returns,
        "lengths": gt_eval_result.lengths,
        "mean": gt_eval_result.mean_return,
        "std": gt_eval_result.std_return,
        "total_steps": gt_eval_result.total_steps,
    }

    final_metrics = {
        "dense_return": final_train_reward,
        "ground_truth_return": final_gt_return,
        "dense_ground_abs_gap": final_abs_gap,
        "total_eval_reward": gt_eval_result.mean_return,
        "num_eval_episodes": len(gt_eval_result.returns),
    }

    artifacts = {
        "dense_reward_path": dense_path,
        "training_curve_ground_truth": gt_curve_path,
        "training_curve_dense": dense_curve_path,
        "training_curve_combined": combined_curve_path,
        "eval_rollout": rollout_path,
        "ctx_fn": "None" if ctx_fn_used is None else f"{ctx_fn_used.__module__}.{ctx_fn_used.__name__}",
        "reward_mode": reward_mode,
        "ground_truth_metrics_csv": "" if metrics_csv_path is None else str(metrics_csv_path),
        "ground_truth_summary": "" if metrics_summary_path is None else str(metrics_summary_path),
    }

    return TrainingResult(
        config=config,
        train_info=train_info,
        artifacts=artifacts,
        final_metrics=final_metrics,
        emitted_reward_code=emitted_code,
        reward_mode=reward_mode,
        ground_truth_eval=ground_truth_eval,
    )


def run_dense_and_sparse(
    reward_generator: RewardGeneratorProtocol,
    output_dir: str,
    *,
    ctx_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    config_override: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[int, Mapping[str, float]], None]] = None,
    compare_dense_vs_sparse: bool = False,
    default_reward_mode: RewardMode = "dense",
) -> list[TrainingResult]:
    """Run one or two training jobs sequentially depending on the compare flag.

    When ``compare_dense_vs_sparse`` is True, runs the dense configuration first and
    the sparse baseline second, writing each run into ``output_dir/<mode>``.
    Otherwise runs a single job using ``default_reward_mode`` and writes to
    ``output_dir`` directly (preserving legacy behavior).
    """

    if default_reward_mode not in ("dense", "sparse"):
        raise ValueError(
            f"Unsupported default_reward_mode '{default_reward_mode}'. Expected 'dense' or 'sparse'."
        )

    reward_modes: Sequence[RewardMode]
    if compare_dense_vs_sparse:
        reward_modes = ("dense", "sparse")
    else:
        reward_modes = (default_reward_mode,)

    multi_run = len(reward_modes) > 1
    results: list[TrainingResult] = []
    for mode in reward_modes:
        mode_output_dir = os.path.join(output_dir, mode) if multi_run else output_dir
        result = run_training_with_reward(
            reward_generator,
            output_dir=mode_output_dir,
            ctx_fn=ctx_fn,
            config_override=config_override,
            progress_callback=progress_callback,
            reward_mode=mode,
        )
        results.append(result)

    if multi_run:
        csv_paths = {}
        summary_paths = {}
        for result in results:
            csv_path = result.artifacts.get("ground_truth_metrics_csv")
            summary_path = result.artifacts.get("ground_truth_summary")
            csv_paths[result.reward_mode] = csv_path
            summary_paths[result.reward_mode] = summary_path
        comparison_plot_path = os.path.join(output_dir, "plots", "dense_vs_sparse_gt.png")
        plotted = plot_gt_comparison(csv_paths, summary_paths, comparison_plot_path)
        if plotted:
            for result in results:
                result.artifacts["dense_vs_sparse_gt_plot"] = plotted

    return results
