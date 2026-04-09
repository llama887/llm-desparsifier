"""Search backend for heuristic evaluation.

This module exposes the `SearchBackend` interface requested by the refactor
plan. It is needed because the batch runner should depend on a search backend
abstraction rather than planner-specific details, and it differs from the old
search evaluator by returning multi-seed batch results and heuristic validation
diagnostics directly.

The current implementation is a repo-local compatibility backend that preserves
the JAxtar-oriented boundary and metadata contract while XLand-specific
integration remains internal to this repository. The upstream JAxtar repository
currently targets its own `Puzzle` abstraction and is pinned in project
metadata for future native backend replacement.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .metrics import SearchBatchResult, SearchSeedResult, compute_seed_score, summarize_batch
from .xland_adapter import XLandTaskInstance, build_heuristic_ctx

JAXTAR_GIT_URL = "https://github.com/tinker495/JAxtar.git"
JAXTAR_COMMIT = "0e9190be2ff74f65814ad40bf4595935b9cda89a"


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for one multi-seed search solve.

    This config captures the planner budget and admissibility tolerances used by
    the compatibility backend. It is needed because the batch runner and tests
    should pass one stable config object into the backend, and it differs from
    the GEPA job config by containing only solver-facing parameters.
    """

    max_nodes: int
    max_expansions: int
    wall_clock_timeout_seconds: float | None = None
    goal_zero_tolerance: float = 1e-6
    consistency_tolerance: float = 1e-6
    path_tolerance: float = 1e-6
    max_trace_states: int = 128


@dataclass(frozen=True)
class SearchTask:
    """Concrete per-seed task bundle consumed by the search backend.

    This dataclass carries the environment, initial state, transition function,
    and replay metadata needed to solve one seed. It is needed because the
    backend should not know how tasks are sampled from benchmarks, and it
    differs from `XLandTaskInstance` by containing live runtime objects in
    addition to serializable metadata.
    """

    env: Any
    env_params: Any
    step_fn: Any
    root_timestep: Any
    task_instance: XLandTaskInstance


@dataclass(frozen=True)
class _SearchNode:
    """Immutable node record used by the compatibility planner.

    This structure stores ancestry and score data so solved or fallback paths
    can be reconstructed after search terminates. It is needed because replay
    artifacts require the chosen action sequence and intermediate g/h/f values,
    and it differs from heap entries by preserving full parent linkage.
    """

    key: bytes
    timestep: Any
    parent_key: bytes | None
    parent_action: int | None
    g_cost: int
    h_cost: float
    f_cost: float


class SearchBackend(Protocol):
    """Protocol implemented by search backends used by the heuristic runner.

    This protocol gives the runner one stable way to evaluate many seeds with a
    synthesized heuristic. It is needed because the new architecture centers on
    search rather than PPO, and it differs from the old evaluator by returning a
    `SearchBatchResult` rather than a single-score payload.
    """

    def solve_many(
        self,
        task_batch: list[SearchTask],
        heuristic_fn: Callable[[Any, Any, dict[str, Any]], float],
        search_config: SearchConfig,
    ) -> SearchBatchResult:
        """Solve a batch of tasks with one synthesized heuristic."""


def _state_cache_key_from_timestep(timestep: Any) -> bytes:
    """Build a stable bytes key for one timestep state.

    This helper lets the planner deduplicate revisited XLand states in Python
    dictionaries. It is needed because graph search correctness depends on
    revisitation checks, and it differs from object identity by hashing the
    actual tree leaf values, dtypes, and shapes.
    """

    leaves: list[bytes] = []
    for leaf in jax.tree_util.tree_leaves(timestep.state):
        try:
            arr = np.asarray(leaf)
        except TypeError:
            arr = np.asarray(jax.random.key_data(leaf))
        leaves.append(
            arr.dtype.str.encode("ascii")
            + b"|"
            + str(arr.shape).encode("ascii")
            + b"|"
            + arr.tobytes()
            + b";"
        )
    return b"".join(leaves)


def _is_sparse_success(timestep: Any) -> bool:
    """Return whether the current timestep satisfies the sparse success rule.

    This helper preserves the project-wide solve criterion: a state is solved
    when the sparse reward is positive. It is needed because the compatibility
    backend uses the unwrapped environment instead of the legacy dense wrapper,
    and it differs from generic termination checks by targeting actual goal
    success rather than any terminal state.
    """

    reward = float(jnp.asarray(getattr(timestep, "reward")))
    return reward > 0.0


def _num_actions(env: Any, env_params: Any) -> int:
    """Return the discrete action count for the search environment.

    This helper normalizes the `xminigrid` action-count API into a Python int.
    It is needed because the planner expands actions in Python loops, and it
    differs from direct calls by handling array-valued counts robustly.
    """

    return int(jnp.asarray(env.num_actions(env_params)))


def _call_heuristic(
    heuristic_fn: Callable[[Any, Any, dict[str, Any]], float],
    *,
    task: SearchTask,
    timestep: Any,
) -> tuple[float, dict[str, Any]]:
    """Evaluate the heuristic and capture contract diagnostics.

    This helper builds the runtime `ctx` mapping, executes the synthesized
    heuristic, and normalizes the result into a finite non-negative float when
    possible. It is needed because the planner should surface heuristic failures
    as validation diagnostics instead of crashing mid-search, and it differs
    from direct calls by returning structured error metadata.
    """

    ctx = build_heuristic_ctx(
        ts=timestep,
        env_params=task.env_params,
        env_id=task.task_instance.env_id,
        benchmark_id=task.task_instance.benchmark_id,
        ruleset_text=task.task_instance.ruleset_text,
        goal_description=task.task_instance.goal_description,
    )
    try:
        raw_value = heuristic_fn(timestep, task.env_params, ctx)
        value = float(raw_value)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return 0.0, {"error": f"{exc.__class__.__name__}: {exc}", "ctx": ctx}
    if not math.isfinite(value):
        return 0.0, {"error": "heuristic returned a non-finite value", "ctx": ctx}
    if value < 0.0:
        return 0.0, {"warning": "heuristic returned a negative value", "ctx": ctx}
    return value, {"ctx": ctx}


def _reconstruct_action_path(nodes: dict[bytes, _SearchNode], leaf_key: bytes) -> list[int]:
    """Reconstruct the forward action sequence from root to a selected node.

    This helper materializes the planned action list after search terminates. It
    is needed because storing full prefixes at every node would waste memory,
    and it differs from heap-based path copies by reconstructing the sequence
    only once from parent links.
    """

    actions: list[int] = []
    current_key: bytes | None = leaf_key
    while current_key is not None:
        node = nodes[current_key]
        if node.parent_action is not None:
            actions.append(int(node.parent_action))
        current_key = node.parent_key
    actions.reverse()
    return actions


class JAxtarSearchBackend:
    """Compatibility search backend with a JAxtar-shaped boundary.

    This class provides the backend interface used by the new heuristic-only
    runner. It is needed because the repo should expose a JAxtar-oriented search
    entrypoint immediately, and it differs from the previous evaluator by
    returning one `SearchBatchResult` spanning many seeds.
    """

    def solve_many(
        self,
        task_batch: list[SearchTask],
        heuristic_fn: Callable[[Any, Any, dict[str, Any]], float],
        search_config: SearchConfig,
    ) -> SearchBatchResult:
        """Solve a batch of tasks with one synthesized heuristic.

        This method runs one independent search per seed and aggregates the
        results into the job-level score. It is needed because GEPA evaluates
        prompt candidates over many seeds for each job, and it differs from the
        legacy search entrypoint by returning the per-seed list together with
        aggregate statistics.
        """

        seed_results = [
            self._solve_one(task=task, heuristic_fn=heuristic_fn, search_config=search_config)
            for task in task_batch
        ]
        return summarize_batch(seed_results)

    def _solve_one(
        self,
        *,
        task: SearchTask,
        heuristic_fn: Callable[[Any, Any, dict[str, Any]], float],
        search_config: SearchConfig,
    ) -> SearchSeedResult:
        """Solve one deterministic XLand task with A* and collect diagnostics.

        This method performs host-side graph search while tracking the
        admissibility-oriented counters required by the new artifact contract. It
        is needed because the runner must produce replayable plans and stable
        validation data for each seed, and it differs from the legacy dense
        heuristic planner by calling the synthesized heuristic directly on
        full-information `ctx` values.
        """

        timeout_seconds = search_config.wall_clock_timeout_seconds
        if timeout_seconds is not None and timeout_seconds <= 0.0:
            raise ValueError("wall_clock_timeout_seconds must be > 0 when provided")
        num_actions = _num_actions(task.env, task.env_params)
        if num_actions <= 0:
            raise ValueError("environment returned zero actions")
        start_time = time.monotonic()
        root_key = _state_cache_key_from_timestep(task.root_timestep)
        root_h, root_diag = _call_heuristic(heuristic_fn, task=task, timestep=task.root_timestep)
        root_h = max(0.0, root_h)
        nodes: dict[bytes, _SearchNode] = {
            root_key: _SearchNode(
                key=root_key,
                timestep=task.root_timestep,
                parent_key=None,
                parent_action=None,
                g_cost=0,
                h_cost=root_h,
                f_cost=root_h,
            )
        }
        best_g: dict[bytes, int] = {root_key: 0}
        heuristic_cache: dict[bytes, float] = {root_key: root_h}
        open_heap: list[tuple[float, int, bytes]] = [(root_h, 0, root_key)]
        tie_counter = 1
        expanded_states = 0
        generated_states = 1
        solved_key: bytes | None = None
        best_fallback_key = root_key
        termination_reason = "open_set_exhausted"
        heuristic_eval_count = 1
        frontier_sizes: list[int] = []
        expanded_trace: list[dict[str, Any]] = []
        contract_violations: list[str] = []
        nonnegative_pass = True
        goal_zero_pass = True
        consistency_pass = True
        consistency_violation_count = 0
        goal_zero_violation_count = 0
        root_error = root_diag.get("error")
        if isinstance(root_error, str):
            contract_violations.append(root_error)
        if root_diag.get("warning"):
            nonnegative_pass = False

        def _timed_out() -> bool:
            """Return whether the optional wall-clock timeout has elapsed.

            The calibration workflow needs blind A* to stop after a real-time
            budget rather than only after synthetic node counters. This helper
            is needed because the backend otherwise runs until search limits or
            completion, and it differs from those existing limits by checking
            elapsed host time while preserving the default timeout-free path.
            """

            if timeout_seconds is None:
                return False
            return (time.monotonic() - start_time) >= timeout_seconds

        while open_heap:
            if _timed_out():
                termination_reason = "wall_clock_timeout"
                break
            current_f, _, current_key = heapq.heappop(open_heap)
            current_node = nodes[current_key]
            current_best_g = best_g.get(current_key)
            if current_best_g is None or current_best_g != current_node.g_cost:
                continue
            if current_f > current_node.f_cost + 1e-8:
                continue

            expanded_states += 1
            best_fallback_key = current_key
            if expanded_states > search_config.max_expansions:
                termination_reason = "max_expansions_reached"
                break
            if _is_sparse_success(current_node.timestep):
                solved_key = current_key
                termination_reason = "solved"
                if abs(current_node.h_cost) > search_config.goal_zero_tolerance:
                    goal_zero_pass = False
                    goal_zero_violation_count += 1
                break
            if bool(current_node.timestep.last()):
                continue
            if len(expanded_trace) < search_config.max_trace_states:
                expanded_trace.append(
                    {
                        "g": current_node.g_cost,
                        "h": current_node.h_cost,
                        "f": current_node.f_cost,
                        "frontier_size": len(open_heap),
                    }
                )
            frontier_sizes.append(len(open_heap))

            for action_value in range(num_actions):
                next_ts = task.step_fn(task.env_params, current_node.timestep, jnp.asarray(action_value))
                next_key = _state_cache_key_from_timestep(next_ts)
                next_g = current_node.g_cost + 1
                prev_best = best_g.get(next_key)
                if prev_best is not None and next_g >= prev_best:
                    continue
                if prev_best is None and len(nodes) >= search_config.max_nodes:
                    termination_reason = "max_nodes_reached"
                    break
                if _timed_out():
                    termination_reason = "wall_clock_timeout"
                    break

                next_h = heuristic_cache.get(next_key)
                next_diag: dict[str, Any] = {}
                if next_h is None:
                    next_h, next_diag = _call_heuristic(heuristic_fn, task=task, timestep=next_ts)
                    next_h = max(0.0, next_h)
                    heuristic_cache[next_key] = next_h
                    heuristic_eval_count += 1
                if current_node.h_cost > 1.0 + next_h + search_config.consistency_tolerance:
                    consistency_pass = False
                    consistency_violation_count += 1
                next_error = next_diag.get("error")
                if isinstance(next_error, str):
                    contract_violations.append(next_error)
                if next_diag.get("warning"):
                    nonnegative_pass = False
                next_f = float(next_g) + float(next_h)
                best_g[next_key] = next_g
                nodes[next_key] = _SearchNode(
                    key=next_key,
                    timestep=next_ts,
                    parent_key=current_key,
                    parent_action=action_value,
                    g_cost=next_g,
                    h_cost=next_h,
                    f_cost=next_f,
                )
                generated_states = max(generated_states, len(nodes))
                heapq.heappush(open_heap, (next_f, tie_counter, next_key))
                tie_counter += 1
            if termination_reason in {
                "max_nodes_reached",
                "max_expansions_reached",
                "wall_clock_timeout",
            }:
                break

        target_key = solved_key or best_fallback_key
        actions = _reconstruct_action_path(nodes, target_key)
        solution_length = len(actions) if solved_key is not None else None
        path_nodes: list[_SearchNode] = []
        cursor: bytes | None = target_key if solved_key is not None else None
        while cursor is not None:
            path_nodes.append(nodes[cursor])
            cursor = nodes[cursor].parent_key
        path_nodes.reverse()
        path_overestimate_count = 0
        max_path_overestimate = 0.0
        if solved_key is not None:
            for idx, node in enumerate(path_nodes):
                remaining_cost = max(0, len(actions) - idx)
                overestimate = node.h_cost - float(remaining_cost)
                if overestimate > search_config.path_tolerance:
                    path_overestimate_count += 1
                    max_path_overestimate = max(max_path_overestimate, overestimate)
        admissibility_pass = (
            goal_zero_pass
            and nonnegative_pass
            and consistency_pass
            and path_overestimate_count == 0
            and not contract_violations
        )
        seed_score, candidate_cost = compute_seed_score(
            solved=solved_key is not None,
            expanded_states=expanded_states,
            solution_length=solution_length,
            astar_max_expansions=search_config.max_expansions,
        )
        return SearchSeedResult(
            seed=task.task_instance.seed,
            solved=solved_key is not None,
            expanded_states=expanded_states,
            generated_states=generated_states,
            solution_length=solution_length,
            termination_reason=termination_reason,
            actions=actions,
            search_trace={
                "seed": task.task_instance.seed,
                "expanded_trace": expanded_trace,
                "frontier_sizes": frontier_sizes[: search_config.max_trace_states],
                "admissibility_events": [],
                "terminated_reason": termination_reason,
                "heuristic_eval_count": heuristic_eval_count,
            },
            validation={
                "contract_violations": contract_violations,
                "goal_zero_pass": goal_zero_pass,
                "nonnegative_pass": nonnegative_pass,
                "consistency_pass": consistency_pass,
                "admissibility_goal_violation_count": goal_zero_violation_count,
                "consistency_violation_count": consistency_violation_count,
                "consistency_violation_rate": (
                    float(consistency_violation_count) / float(max(1, generated_states - 1))
                ),
                "path_overestimate_count": path_overestimate_count,
                "max_path_overestimate": max_path_overestimate,
                "admissibility_pass": admissibility_pass,
            },
            seed_score=seed_score,
            candidate_cost=candidate_cost,
        )


__all__ = [
    "JAXTAR_COMMIT",
    "JAXTAR_GIT_URL",
    "JAxtarSearchBackend",
    "SearchBackend",
    "SearchConfig",
    "SearchTask",
]
