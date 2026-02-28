"""Ground-truth evaluation helpers for trained policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import xminigrid
from flax.training import checkpoints
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

from llm_desparsifier.rewards.parser import describe_ruleset

DEFAULT_RULESET_INDEX = 42


@dataclass
class GroundTruthEvalConfig:
    """Configuration for sparse-reward evaluation rollouts.

    This config collects every knob needed by `run_ground_truth_eval` so the
    evaluation loop can be reproduced independently of training. It is needed
    because training and evaluation run in different contexts (JAX-compiled
    training loop vs host-side evaluation), and it differs from training config
    structures by including trajectory-capture settings that exist solely to
    support post-run visualization.
    """

    env_id: str
    benchmark_id: str
    num_episodes: int = 10
    seed: int = 0
    img_obs: bool = False
    capture_video: bool = False
    video_episode_index: int = 0
    deterministic_rulesets: bool = False
    fixed_ruleset_seed: Optional[int] = None
    capture_trajectory: bool = True
    trajectory_episode_index: int = 0


@dataclass
class GroundTruthEvalResult:
    returns: List[float]
    lengths: List[int]
    mean_return: float
    std_return: float
    total_steps: int
    frames: Optional[List] = None
    trajectory: Optional[dict] = None


def _key_to_list(key: jax.Array) -> list[int]:
    """Convert a JAX PRNG key into a JSON-friendly list of uint32 integers.

    This helper serializes PRNG keys emitted during evaluation so trajectories
    can be replayed exactly in downstream tooling (for example, video
    generation). It is needed because modern JAX uses typed keys (for example
    `key<fry>`) that cannot be directly cast with `jnp.asarray(...,
    dtype=jnp.uint32)`, which can crash long GEPA runs right at eval-time. It
    differs from naïve `tolist()` conversion by routing through
    `jax.random.key_data`, which works for both typed keys (`jax.random.key`)
    and legacy uint32 keys (`jax.random.PRNGKey`) while preserving the exact
    underlying key bits required for deterministic replay.

    Args:
        key: A JAX PRNG key array in either typed-key or legacy uint32 format.

    Returns:
        A flat list of Python integers containing the raw uint32 key data.
    """
    arr = jnp.asarray(jax.random.key_data(key), dtype=jnp.uint32).reshape(-1)
    return [int(value) for value in arr.tolist()]


def _build_eval_env(
    cfg: GroundTruthEvalConfig,
) -> tuple[Environment, EnvParams, Benchmark]:
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
    """Roll out a trained policy on sparse rewards and return episode stats.

    This evaluation isolates the ground-truth success signal so GEPA can score
    candidate dense rewards without being biased by shaping scale. It is needed
    because the training loop evaluates on dense rewards, while GEPA requires a
    consistent sparse metric; it differs from the training-time eval inside
    `make_train` by running on the host with optional deterministic rulesets and
    by returning trajectory metadata tailored for deterministic replay.

    In addition to sparse returns and episode lengths, this routine can capture
    one replayable trajectory containing: the benchmark/ruleset selection,
    reset seed material, action sequence, and a human-readable environment text
    description. That payload is consumed by video tooling to regenerate the
    exact rollout and overlay dense reward diagnostics without repeating policy
    inference.

    Args:
        train_state: Trained model state (or template when restoring checkpoint).
        model: Policy network used to produce actions during evaluation.
        cfg: Evaluation configuration including environment and capture options.
        checkpoint_path: Optional path to restore parameters before rollout.

    Returns:
        A `GroundTruthEvalResult` with per-episode sparse metrics and optional
        captured trajectory payload for replay.
    """

    resolved_state = _maybe_restore_train_state(train_state, checkpoint_path)
    env, env_params, benchmark = _build_eval_env(cfg)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    apply_fn = jax.jit(model.apply)

    rng = jax.random.key(cfg.seed)
    returns: List[float] = []
    lengths: List[int] = []
    frames: Optional[List] = [] if cfg.capture_video else None
    trajectory_payload: Optional[dict] = None
    trajectory_actions: Optional[list[int]] = None
    trajectory_ruleset_key: Optional[list[int]] = None
    trajectory_reset_key: Optional[list[int]] = None
    trajectory_ruleset_index: Optional[int] = None
    trajectory_episode_return: Optional[float] = None
    trajectory_episode_length: Optional[int] = None
    trajectory_env_text: Optional[str] = None

    fixed_ruleset = None
    if cfg.deterministic_rulesets:
        if cfg.fixed_ruleset_seed is None:
            fixed_ruleset = benchmark.get_ruleset(DEFAULT_RULESET_INDEX)
        else:
            fixed_ruleset = benchmark.sample_ruleset(jax.random.key(cfg.fixed_ruleset_seed))

    for episode_idx in range(cfg.num_episodes):
        rng, ruleset_key, reset_key = jax.random.split(rng, 3)
        if cfg.deterministic_rulesets:
            ruleset = fixed_ruleset
        else:
            ruleset = benchmark.sample_ruleset(ruleset_key)
        episode_params = env_params.replace(ruleset=ruleset)

        capture_this = cfg.capture_trajectory and (
            episode_idx == cfg.trajectory_episode_index
        )
        if capture_this:
            trajectory_actions = []
            trajectory_reset_key = _key_to_list(reset_key)
            if cfg.deterministic_rulesets:
                if cfg.fixed_ruleset_seed is None:
                    trajectory_ruleset_index = int(DEFAULT_RULESET_INDEX)
                    trajectory_ruleset_key = None
                else:
                    trajectory_ruleset_key = None
                    trajectory_ruleset_index = None
            else:
                trajectory_ruleset_key = _key_to_list(ruleset_key)
                trajectory_ruleset_index = None
            trajectory_env_text = f"{cfg.env_id} | benchmark={cfg.benchmark_id}"
            try:
                described = describe_ruleset(env, episode_params)
                if isinstance(described, str) and described.strip():
                    trajectory_env_text = described
            except Exception:
                # Keep evaluation robust even if environment text extraction fails.
                pass

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
            if capture_this and trajectory_actions is not None:
                trajectory_actions.append(int(jnp.asarray(action)))
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
        if capture_this:
            trajectory_episode_return = ep_return
            trajectory_episode_length = ep_length

    returns_arr = jnp.asarray(returns)
    mean_return = float(jnp.mean(returns_arr))
    std_return = float(jnp.std(returns_arr))
    total_steps = int(sum(lengths))

    if cfg.capture_trajectory and trajectory_actions is not None:
        trajectory_payload = {
            "version": 1,
            "env_id": cfg.env_id,
            "benchmark_id": cfg.benchmark_id,
            "deterministic_rulesets": bool(cfg.deterministic_rulesets),
            "fixed_ruleset_seed": cfg.fixed_ruleset_seed,
            "ruleset_index": trajectory_ruleset_index,
            "ruleset_key": trajectory_ruleset_key,
            "reset_key": trajectory_reset_key,
            "episode_index": cfg.trajectory_episode_index,
            "episode_length": trajectory_episode_length,
            "episode_return": trajectory_episode_return,
            "actions": trajectory_actions,
            "num_eval_episodes": cfg.num_episodes,
            "eval_seed": cfg.seed,
            "env_seed": cfg.seed,
            "env_text": trajectory_env_text,
            "img_obs": cfg.img_obs,
        }

    return GroundTruthEvalResult(
        returns=returns,
        lengths=lengths,
        mean_return=mean_return,
        std_return=std_return,
        total_steps=total_steps,
        frames=frames,
        trajectory=trajectory_payload,
    )
