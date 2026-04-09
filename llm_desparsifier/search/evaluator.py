"""A* search evaluator used as an alternative GEPA inner loop.

This module runs deterministic A* search over one XLand task instance and
produces a compact evaluation result with planner-oriented metrics. It is
needed because the GEPA loop now supports a non-RL backend that still needs to
generate replay artifacts, reward diagnostics, and comparable scalar scores,
and it differs from the PPO pipeline by evaluating one search episode instead
of optimizing a recurrent policy over many updates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import jax
import jax.numpy as jnp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper

from llm_desparsifier.rewards.generator import (
    GeneratedReward,
    GeneratedRewardValidation,
    persist_generated_reward_artifacts,
)
from llm_desparsifier.rewards.parser import describe_ruleset
from llm_desparsifier.rl.eval import DEFAULT_RULESET_INDEX
from llm_desparsifier.rl.wrappers import DesparsifyRewardWrapper
from llm_desparsifier.search.astar import AStarPlanResult, plan_with_astar
from llm_desparsifier.utils import extract_xland_ctx


class RewardGeneratorProtocol(Protocol):
    """Interface for reward synthesis backends used by the A* evaluator.

    This protocol mirrors the subset of `RewardGenerator` consumed by the
    search evaluator. It is needed because the GEPA runner passes in concrete
    reward generators while sparse baselines intentionally skip reward
    generation, and it differs from the RL pipeline's protocol by not requiring
    any training-specific behavior.
    """

    def generate(self, env: Any, env_params: Any) -> GeneratedReward:
        """Return the canonical generated dense reward payload for one task."""


def _coerce_generated_reward_payload(
    generated_reward: GeneratedReward | tuple[Any, str],
) -> GeneratedReward:
    """Normalize evaluator reward output into the canonical generated-reward type.

    This helper preserves backwards compatibility for lightweight tests and any
    older local call sites that still return the historical
    `(dense_fn, raw_code)` tuple. It is needed because the evaluator now
    persists richer validation artifacts from `GeneratedReward`, and it differs
    from changing every caller at once by synthesizing a minimal canonical
    payload only when legacy output is encountered.

    Args:
        generated_reward: Either the new `GeneratedReward` payload or the legacy
            `(dense_fn, raw_code)` tuple produced by older test doubles.

    Returns:
        Canonical `GeneratedReward` instance suitable for artifact persistence
        and downstream validation checks.
    """
    if isinstance(generated_reward, GeneratedReward):
        return generated_reward
    dense_fn, raw_code = generated_reward
    component_keys = tuple(
        str(name)
        for name in getattr(dense_fn, "__reward_component_keys__", ())
    )
    return GeneratedReward(
        dense_fn=dense_fn,
        raw_code=str(raw_code),
        sanitized_code=str(raw_code),
        component_keys=component_keys,
        validation=GeneratedRewardValidation(
            status="ok",
            failure_reason=None,
            raw_code_sha16="",
            sanitized_code_sha16="",
            component_keys=component_keys,
            diagnostics={
                "referenced_object_keys": [],
                "task_object_keys": [],
                "missing_from_task": [],
            },
        ),
    )


@dataclass(frozen=True)
class AStarEvalConfig:
    """Configuration describing one deterministic A* evaluation job.

    This config captures the subset of environment and replay knobs needed to
    reproduce one search-based evaluation. It is needed because the GEPA runner
    passes around larger RL-shaped config dictionaries, and it differs from the
    PPO `TrainConfig` by keeping only the values that matter to A* planning and
    trajectory capture.
    """

    env_id: str
    benchmark_id: str
    eval_seed: int
    img_obs: bool = False
    deterministic_rulesets: bool = False
    fixed_ruleset_seed: Optional[int] = None


@dataclass(frozen=True)
class AStarEvaluationResult:
    """Structured output from `run_astar_with_reward`.

    This result gives the GEPA runner a backend-specific but stable payload
    containing the emitted reward, scalar score, planner stats, and replay
    artifacts for one environment. It is needed because A* evaluation should
    not pretend to be PPO training, and it differs from `TrainingResult` by
    reporting planner diagnostics instead of policy-training curves.
    """

    config: AStarEvalConfig
    score: float
    solved: bool
    search_stats: Mapping[str, Any]
    final_metrics: Mapping[str, float]
    artifacts: Mapping[str, str]
    emitted_reward_code: str
    reward_mode: str
    raw_reward_code: str = ""
    env_description: Optional[str] = None
    reward_validation: Optional[Mapping[str, Any]] = None


def astar_score(*, solved: bool, expanded_states: int, max_expansions: int) -> float:
    """Map one planner outcome onto the GEPA score range `[0, 1]`.

    This helper implements the lexicographic scoring rule requested for the A*
    backend: all solved runs must score above all unsolved runs, and among runs
    with the same solve status, fewer expanded states are better. It is needed
    because GEPA expects a scalar objective rather than a tuple, and it differs
    from plain solve-rate scoring by preserving a strong search-efficiency
    signal once the planner can solve the task.

    Args:
        solved: Whether the planner found a sparse-success state.
        expanded_states: Number of states expanded from the open set.
        max_expansions: Configured planner expansion budget.

    Returns:
        Scalar score in `[0, 1]` following the requested lexicographic rule.
    """
    if max_expansions <= 0:
        raise ValueError("max_expansions must be > 0")
    bounded_expansions = min(max(0, int(expanded_states)), int(max_expansions))
    efficiency = 1.0 - (bounded_expansions / float(max_expansions))
    if solved:
        return 0.5 + 0.5 * efficiency
    return 0.5 * efficiency


def _key_to_list(key: jax.Array) -> list[int]:
    """Convert a JAX PRNG key into raw uint32 words for JSON serialization.

    This helper preserves the exact key material used to initialize the search
    rollout so downstream video tooling can replay the same task instance. It is
    needed because modern JAX typed keys are not directly JSON serializable, and
    it differs from naive `tolist()` conversion by routing through
    `jax.random.key_data` to support both typed and legacy key formats.
    """
    arr = jnp.asarray(jax.random.key_data(key), dtype=jnp.uint32).reshape(-1)
    return [int(value) for value in arr.tolist()]


def _build_reset_and_step_fns(env: Any) -> tuple[Any, Any]:
    """Build reset and step callables for deterministic A* evaluation.

    This helper intentionally returns the raw environment methods instead of
    JIT-wrapping them. It is needed because the search evaluator must remain
    robust under smoke-test and workstation memory limits, and recent runs have
    shown LLVM/JAX compilation failures during A* planning that block
    experimentation entirely. It differs from the earlier implementation by
    preferring predictable execution over speculative compilation speedups.
    """
    return env.reset, env.step


def _resolve_ruleset(config: AStarEvalConfig, benchmark: Any, ruleset_key: Any) -> tuple[Any, int | None]:
    """Resolve the concrete benchmark ruleset for one A* evaluation.

    This helper ensures reward generation, planner execution, and replay all see
    the exact same task instance. It is needed because XLand benchmarks may
    sample rulesets stochastically, and it differs from the RL training path by
    producing exactly one ruleset rather than batched train/eval ruleset sets.
    """
    if config.deterministic_rulesets:
        if config.fixed_ruleset_seed is None:
            return benchmark.get_ruleset(DEFAULT_RULESET_INDEX), None
        return (
            benchmark.sample_ruleset(jax.random.key(config.fixed_ruleset_seed)),
            int(config.fixed_ruleset_seed),
        )
    return benchmark.sample_ruleset(ruleset_key), None


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write a JSON artifact and return its string path.

    This helper centralizes artifact serialization so the evaluator can emit
    replay and planner summaries with one consistent format. It is needed
    because A* mode writes several small JSON files consumed by downstream tools,
    and it differs from ad-hoc `write_text` calls by always sorting keys and
    creating parent directories first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _build_config(config_override: Mapping[str, Any] | None) -> AStarEvalConfig:
    """Construct an `AStarEvalConfig` from a larger job config mapping.

    This helper filters the RL-shaped job dictionaries used by the GEPA runner
    down to the subset needed for A* evaluation. It is needed because job
    configs also contain PPO budget knobs that are irrelevant to search, and it
    differs from directly unpacking the mapping into a dataclass by explicitly
    ignoring unrelated keys rather than failing on them.
    """
    cfg = dict(config_override or {})
    return AStarEvalConfig(
        env_id=str(cfg["env_id"]),
        benchmark_id=str(cfg.get("benchmark_id", "trivial-1m")),
        eval_seed=int(cfg.get("eval_seed", 0)),
        img_obs=bool(cfg.get("img_obs", False)),
        deterministic_rulesets=bool(cfg.get("deterministic_rulesets", False)),
        fixed_ruleset_seed=(
            None
            if cfg.get("fixed_ruleset_seed") is None
            else int(cfg["fixed_ruleset_seed"])
        ),
    )


def run_astar_with_reward(
    reward_generator: RewardGeneratorProtocol | None,
    output_dir: str,
    *,
    config_override: Optional[Mapping[str, Any]] = None,
    max_nodes: int,
    max_expansions: int,
    reward_mode: str = "dense",
    use_dense_heuristic: bool = True,
) -> AStarEvaluationResult:
    """Run deterministic A* search with optional dense-reward synthesis.

    This entrypoint mirrors the role of `run_training_with_reward` for the A*
    backend: it prepares the environment, optionally synthesizes the dense
    reward, plans a path, writes replay artifacts, and returns a compact result.
    It is needed because the GEPA runner should be able to switch between RL and
    A* without rewriting all downstream orchestration, and it differs from the
    PPO pipeline by producing planner statistics instead of training curves.

    Args:
        reward_generator: Dense reward synthesizer used in `reward_mode="dense"`.
        output_dir: Directory where replay and planner artifacts should be saved.
        config_override: Job-level environment config from the GEPA runner.
        max_nodes: Planner node-generation budget.
        max_expansions: Planner expansion budget.
        reward_mode: Either `dense` or `sparse`.
        use_dense_heuristic: Whether the planner should use the synthesized dense
            reward as its heuristic signal.

    Returns:
        `AStarEvaluationResult` describing the planner outcome and artifacts.
    """
    if reward_mode not in ("dense", "sparse"):
        raise ValueError(
            f"Unsupported reward_mode '{reward_mode}'. Expected 'dense' or 'sparse'."
        )
    if reward_mode == "sparse" and use_dense_heuristic:
        raise ValueError("Sparse A* runs cannot use the dense heuristic")

    config = _build_config(config_override)
    if "XLand" not in config.env_id:
        raise ValueError("Only meta-task environments are supported.")

    env, env_params = xminigrid.make(config.env_id)
    env = GymAutoResetWrapper(env)
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    rng = jax.random.key(config.eval_seed)
    rng, ruleset_key, reset_key = jax.random.split(rng, 3)
    ruleset, _ruleset_seed = _resolve_ruleset(config, benchmark, ruleset_key)
    env_params = env_params.replace(ruleset=ruleset)

    emitted_code = ""
    raw_reward_code = ""
    dense_path = ""
    env_description = None
    reward_validation: Optional[Mapping[str, Any]] = None
    generated_artifact_paths: dict[str, str] = {}
    if reward_mode == "dense":
        if reward_generator is None:
            raise ValueError("Dense A* runs require a reward generator")
        generated_reward = _coerce_generated_reward_payload(
            reward_generator.generate(env, env_params)
        )
        emitted_code = generated_reward.sanitized_code
        raw_reward_code = generated_reward.raw_code
        reward_validation = generated_reward.validation.to_dict()
        env_description = getattr(reward_generator, "last_env_description", None)
        generated_artifact_paths = persist_generated_reward_artifacts(
            Path(output_dir),
            generated_reward,
        )
        dense_path = generated_artifact_paths["dense_reward_path"]
        failure_reason = reward_validation.get("failure_reason")
        if failure_reason:
            raise ValueError(str(failure_reason))
        ctx_fn = extract_xland_ctx if "XLand" in config.env_id else None
        env = DesparsifyRewardWrapper(
            env,
            dense_fn=generated_reward.dense_fn,
            ctx_fn=ctx_fn,
        )
    else:
        os.makedirs(output_dir, exist_ok=True)

    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    reset_fn, step_fn = _build_reset_and_step_fns(env)
    root_timestep = reset_fn(env_params, reset_key)
    plan: AStarPlanResult = plan_with_astar(
        env=env,
        env_params=env_params,
        step_fn=step_fn,
        root_timestep=root_timestep,
        use_dense_heuristic=use_dense_heuristic,
        max_nodes=max_nodes,
        max_expansions=max_expansions,
    )

    search_stats = dict(plan.search_stats)
    solved = bool(search_stats.get("solved", False))
    expanded_states = int(search_stats.get("expanded_states", 0))
    generated_states = int(search_stats.get("generated_states", 0))
    solution_length = int(search_stats.get("solution_length", 0))
    score = astar_score(
        solved=solved,
        expanded_states=expanded_states,
        max_expansions=max_expansions,
    )
    if env_description is None:
        try:
            described = describe_ruleset(env, env_params)
        except Exception:
            described = f"{config.env_id} | benchmark={config.benchmark_id}"
        env_description = described

    trajectory_payload = {
        "version": 1,
        "env_id": config.env_id,
        "benchmark_id": config.benchmark_id,
        "deterministic_rulesets": bool(config.deterministic_rulesets),
        "fixed_ruleset_seed": config.fixed_ruleset_seed,
        "ruleset_index": int(DEFAULT_RULESET_INDEX)
        if config.deterministic_rulesets and config.fixed_ruleset_seed is None
        else None,
        "ruleset_key": None if config.deterministic_rulesets else _key_to_list(ruleset_key),
        "reset_key": _key_to_list(reset_key),
        "episode_index": 0,
        "episode_length": solution_length,
        "episode_return": float(search_stats.get("final_sparse_reward", 0.0)),
        "actions": list(plan.actions),
        "num_eval_episodes": 1,
        "eval_seed": config.eval_seed,
        "env_seed": config.eval_seed,
        "env_text": env_description,
        "img_obs": config.img_obs,
        "search_stats": search_stats,
        "score": score,
    }
    output_path = Path(output_dir)
    trajectory_path = _write_json(output_path / "eval_trajectory.json", trajectory_payload)
    search_stats_path = _write_json(
        output_path / "astar_search_stats.json",
        {
            "score": score,
            "solved": solved,
            "env_id": config.env_id,
            "benchmark_id": config.benchmark_id,
            "reward_mode": reward_mode,
            "search_stats": search_stats,
        },
    )
    final_metrics = {
        "score": float(score),
        "solve_rate": 1.0 if solved else 0.0,
        "expanded_states": float(expanded_states),
        "generated_states": float(generated_states),
        "solution_length": float(solution_length),
        "eval_successes": 1.0 if solved else 0.0,
    }
    artifacts = {
        "dense_reward_path": dense_path,
        "dense_reward_raw_response": generated_artifact_paths.get(
            "dense_reward_raw_response",
            "",
        ),
        "eval_trajectory": trajectory_path,
        "astar_search_stats": search_stats_path,
        "reward_mode": reward_mode,
        "reward_validation": generated_artifact_paths.get("reward_validation", ""),
    }
    return AStarEvaluationResult(
        config=config,
        score=score,
        solved=solved,
        search_stats=search_stats,
        final_metrics=final_metrics,
        artifacts=artifacts,
        emitted_reward_code=emitted_code,
        raw_reward_code=raw_reward_code,
        reward_mode=reward_mode,
        env_description=env_description,
        reward_validation=reward_validation,
    )
