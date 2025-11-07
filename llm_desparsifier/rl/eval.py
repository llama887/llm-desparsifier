"""Ground-truth evaluation helpers for trained policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
from flax.training import checkpoints

import xminigrid
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import EnvParams, Environment
from xminigrid.wrappers import GymAutoResetWrapper


@dataclass
class GroundTruthEvalConfig:
    """Configuration for sparse-reward evaluation rollouts."""

    env_id: str
    benchmark_id: str
    num_episodes: int = 10
    seed: int = 0
    img_obs: bool = False
    capture_video: bool = False
    video_episode_index: int = 0


@dataclass
class GroundTruthEvalResult:
    returns: List[float]
    lengths: List[int]
    mean_return: float
    std_return: float
    total_steps: int
    frames: Optional[List] = None


def _build_eval_env(cfg: GroundTruthEvalConfig) -> tuple[Environment, EnvParams, Benchmark]:
    env, env_params = xminigrid.make(cfg.env_id)
    env = GymAutoResetWrapper(env)

    if cfg.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    benchmark = xminigrid.load_benchmark(cfg.benchmark_id)
    return env, env_params, benchmark


def _maybe_restore_train_state(train_state, checkpoint_path: Optional[str]):
    if checkpoint_path is None:
        return train_state
    if train_state is None:
        raise ValueError("train_state template required to restore from checkpoint")
    return checkpoints.restore_checkpoint(checkpoint_path, target=train_state)


def run_ground_truth_eval(
    train_state,
    model,
    cfg: GroundTruthEvalConfig,
    *,
    checkpoint_path: Optional[str] = None,
) -> GroundTruthEvalResult:
    """Roll out a policy on the sparse environment and report per-episode stats."""

    resolved_state = _maybe_restore_train_state(train_state, checkpoint_path)
    env, env_params, benchmark = _build_eval_env(cfg)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    apply_fn = jax.jit(model.apply)

    rng = jax.random.key(cfg.seed)
    returns: List[float] = []
    lengths: List[int] = []
    frames: Optional[List] = [] if cfg.capture_video else None

    for episode_idx in range(cfg.num_episodes):
        rng, ruleset_key, reset_key = jax.random.split(rng, 3)
        ruleset = benchmark.sample_ruleset(ruleset_key)
        episode_params = env_params.replace(ruleset=ruleset)

        timestep = reset_fn(episode_params, reset_key)
        hidden = model.initialize_carry(1)
        prev_action = jnp.asarray(0)
        prev_reward = jnp.asarray(0)

        ep_return = 0.0
        ep_length = 0

        if frames is not None and episode_idx == cfg.video_episode_index:
            frames.append(env.render(episode_params, timestep))

        while True:
            rng, action_key = jax.random.split(rng)
            dist, _, hidden = apply_fn(
                resolved_state.params,
                {
                    "observation": timestep.observation[None, None, ...],
                    "prev_action": prev_action[None, None, ...],
                    "prev_reward": prev_reward[None, None, ...],
                },
                hidden,
            )
            action = dist.sample(seed=action_key).squeeze()
            timestep = step_fn(episode_params, timestep, action)

            reward = float(timestep.reward)
            ep_return += reward
            ep_length += 1

            prev_action = action
            prev_reward = timestep.reward

            if frames is not None and episode_idx == cfg.video_episode_index:
                frames.append(env.render(episode_params, timestep))

            if bool(timestep.last()):
                break

        returns.append(ep_return)
        lengths.append(ep_length)

    returns_arr = jnp.asarray(returns)
    mean_return = float(jnp.mean(returns_arr))
    std_return = float(jnp.std(returns_arr))
    total_steps = int(sum(lengths))

    return GroundTruthEvalResult(
        returns=returns,
        lengths=lengths,
        mean_return=mean_return,
        std_return=std_return,
        total_steps=total_steps,
        frames=frames,
    )
