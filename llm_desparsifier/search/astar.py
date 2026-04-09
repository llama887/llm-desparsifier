"""Reusable deterministic A* planning helpers.

This module houses the pure planner logic used by both the GEPA A* evaluator
and the video replay tooling. It is needed because the project now depends on
one consistent definition of A* search statistics and heuristic behavior across
training-time evaluation and post-hoc visualization, and it differs from the
video script's orchestration code by containing only environment-agnostic graph
search logic.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class _AStarNode:
    """Immutable record for one explored search node.

    This structure stores the state identity, parent linkage, and accumulated
    search costs needed to reconstruct a plan after A* terminates. It is needed
    because both the evaluator and the video tooling must recover the selected
    action sequence together with the final planner metrics, and it differs from
    transient priority-queue entries by preserving the full ancestry needed for
    path reconstruction and diagnostics.
    """

    key: bytes
    timestep: Any
    parent_key: bytes | None
    parent_action: int | None
    g_cost: int
    h_cost: float
    f_cost: float


@dataclass(frozen=True)
class AStarPlanResult:
    """Deterministic A* output consumed by evaluators and replay tooling.

    This result bundles the planned action sequence and aggregated search
    statistics under one stable return type. It is needed because search is now
    reused across multiple callers that each need the plan and the diagnostic
    counters, and it differs from the earlier video-local planner return type by
    deliberately avoiding rollout-script-specific metadata so other backends can
    reuse it directly.
    """

    actions: list[int]
    search_stats: dict[str, Any]


def extract_step_reward_details(timestep: Any) -> tuple[float, float, dict[str, Any]]:
    """Extract dense and sparse rewards from an environment timestep.

    This helper normalizes wrapped environment outputs into the two reward views
    relevant for search: the dense shaping reward used to build the heuristic
    and the sparse reward used to determine true task success. It is needed
    because wrapped timesteps may store sparse reward inside `extras` while the
    dense wrapper replaces `timestep.reward`, and it differs from direct field
    access by providing one consistent normalization path for all A* callers.

    Args:
        timestep: Environment timestep produced by the wrapped search env.

    Returns:
        Tuple of dense reward, sparse reward, and raw reward-components mapping.
    """
    import jax.numpy as jnp

    extras = getattr(timestep, "extras", None)
    dense_reward_value = float(jnp.asarray(timestep.reward))
    sparse_reward_value = dense_reward_value
    reward_components: dict[str, Any] = {}
    if extras is not None:
        sparse_reward_value = float(
            jnp.asarray(extras.get("ground_truth_reward", dense_reward_value))
        )
        reward_components = extras.get("reward_components") or {}
    return dense_reward_value, sparse_reward_value, reward_components


def state_cache_key_from_timestep(timestep: Any) -> bytes:
    """Build a stable hashable key from one timestep's environment state.

    This helper converts the JAX pytree stored inside `timestep.state` into a
    deterministic byte string so Python dictionaries can deduplicate states
    during graph search. It is needed because A* correctness depends on
    revisitation checks for logically identical states, and it differs from
    object-identity hashing by using the actual leaf values, dtypes, and shapes.

    Args:
        timestep: Environment timestep whose state pytree should be keyed.

    Returns:
        Byte string that uniquely identifies the timestep state contents.
    """
    import jax

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


def is_sparse_success(timestep: Any) -> bool:
    """Return whether a timestep satisfies the sparse success condition.

    This helper enforces the same ground-truth success rule used throughout the
    project: a state is solved only when the sparse reward is strictly positive.
    It is needed because dense reward can be positive on non-goal states, and it
    differs from generic episode termination checks by targeting actual task
    completion instead of any terminal outcome.

    Args:
        timestep: Environment timestep to classify.

    Returns:
        True when sparse reward indicates success; otherwise False.
    """
    _, sparse_reward_value, _ = extract_step_reward_details(timestep)
    return sparse_reward_value > 0.0


def estimate_dense_qmax(
    *,
    env: Any,
    timestep: Any,
    step_fn: Any,
    env_params: Any,
) -> float:
    """Estimate the best one-step dense reward available from a state.

    This helper scores all discrete actions from the current state and returns
    the maximum immediate dense reward. It is needed because the dense-heuristic
    A* mode uses reward advantage relative to the root state as a heuristic
    proxy, and it differs from rollout action selection by evaluating every
    action only to rank future states rather than to execute a policy.

    Args:
        env: Wrapped environment exposing `num_actions`.
        timestep: Current search state.
        step_fn: Callable implementing one environment transition.
        env_params: Environment parameters for the current task instance.

    Returns:
        Maximum immediate dense reward across all available actions.
    """
    import jax.numpy as jnp

    num_actions = int(env.num_actions(env_params))
    if num_actions <= 0:
        return 0.0
    best = float("-inf")
    for action_value in range(num_actions):
        next_ts = step_fn(env_params, timestep, jnp.asarray(action_value))
        candidate_dense = float(jnp.asarray(next_ts.reward))
        best = max(best, candidate_dense)
    if best == float("-inf"):
        return 0.0
    return best


def _reconstruct_action_path(nodes: Mapping[bytes, _AStarNode], leaf_key: bytes) -> list[int]:
    """Reconstruct the forward action sequence from root to a selected node.

    This helper walks parent links emitted during graph search and materializes
    the final action plan. It is needed because storing the full action prefix at
    every node would waste memory on large searches, and it differs from a
    one-shot path copy strategy by reconstructing the path only once after the
    planner has already chosen a terminal or fallback node.

    Args:
        nodes: Mapping of state keys to stored node records.
        leaf_key: State key whose ancestry should be reconstructed.

    Returns:
        Ordered list of action ids from root to `leaf_key`.
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


def plan_with_astar(
    *,
    env: Any,
    env_params: Any,
    step_fn: Any,
    root_timestep: Any,
    use_dense_heuristic: bool,
    max_nodes: int,
    max_expansions: int,
) -> AStarPlanResult:
    """Plan a deterministic action sequence with A* and collect diagnostics.

    This function performs graph search from one fixed initial state and returns
    the best solved path found within the configured search budgets, or the best
    fallback prefix if the planner terminates early. It is needed because both
    GEPA evaluation and debugging videos now rely on the same A* semantics, and
    it differs from JAxtar-style batched search by using a Python priority queue
    for straightforward instrumentation and predictable trace generation.

    Args:
        env: Wrapped environment instance exposing `num_actions`.
        env_params: Parameters for the current task instance.
        step_fn: Callable that steps the environment one action at a time.
        root_timestep: Initial state from which search should begin.
        use_dense_heuristic: Whether to use dense reward as the heuristic.
        max_nodes: Maximum number of unique states the planner may generate.
        max_expansions: Maximum number of states popped from the open set.

    Returns:
        `AStarPlanResult` containing the chosen action sequence and search stats.
    """
    import jax.numpy as jnp

    if max_nodes <= 0:
        raise ValueError("max_nodes must be > 0")
    if max_expansions <= 0:
        raise ValueError("max_expansions must be > 0")

    num_actions = int(env.num_actions(env_params))
    if num_actions <= 0:
        raise ValueError("Environment returned zero actions for A* search")

    root_key = state_cache_key_from_timestep(root_timestep)
    root_qmax = estimate_dense_qmax(
        env=env,
        timestep=root_timestep,
        step_fn=step_fn,
        env_params=env_params,
    )
    nodes: dict[bytes, _AStarNode] = {
        root_key: _AStarNode(
            key=root_key,
            timestep=root_timestep,
            parent_key=None,
            parent_action=None,
            g_cost=0,
            h_cost=0.0,
            f_cost=0.0,
        )
    }
    best_g: dict[bytes, int] = {root_key: 0}
    heuristic_cache: dict[bytes, float] = {root_key: root_qmax}
    open_heap: list[tuple[float, int, bytes]] = [(0.0, 0, root_key)]
    tie_counter = 1
    expanded_states = 0
    termination_reason = "open_set_exhausted"
    solved_key: bytes | None = None
    best_fallback_key = root_key

    while open_heap:
        current_f, _, current_key = heapq.heappop(open_heap)
        current_node = nodes[current_key]
        current_best_g = best_g.get(current_key)
        if current_best_g is None or current_best_g != current_node.g_cost:
            continue
        if current_f > current_node.f_cost + 1e-8:
            continue

        expanded_states += 1
        best_fallback_key = current_key
        if expanded_states > max_expansions:
            termination_reason = "max_expansions_reached"
            break
        if is_sparse_success(current_node.timestep):
            solved_key = current_key
            termination_reason = "solved"
            break
        if bool(current_node.timestep.last()):
            continue

        for action_value in range(num_actions):
            next_ts = step_fn(env_params, current_node.timestep, jnp.asarray(action_value))
            next_key = state_cache_key_from_timestep(next_ts)
            next_g = current_node.g_cost + 1
            prev_best = best_g.get(next_key)
            if prev_best is not None and next_g >= prev_best:
                continue
            if prev_best is None and len(nodes) >= max_nodes:
                termination_reason = "max_nodes_reached"
                break

            qmax = heuristic_cache.get(next_key)
            if qmax is None:
                qmax = estimate_dense_qmax(
                    env=env,
                    timestep=next_ts,
                    step_fn=step_fn,
                    env_params=env_params,
                )
                heuristic_cache[next_key] = qmax
            h_value = max(0.0, root_qmax - qmax) if use_dense_heuristic else 0.0
            f_value = float(next_g) + float(h_value)

            best_g[next_key] = next_g
            nodes[next_key] = _AStarNode(
                key=next_key,
                timestep=next_ts,
                parent_key=current_key,
                parent_action=action_value,
                g_cost=next_g,
                h_cost=h_value,
                f_cost=f_value,
            )
            heapq.heappush(open_heap, (f_value, tie_counter, next_key))
            tie_counter += 1
            if is_sparse_success(next_ts):
                solved_key = next_key
                termination_reason = "solved"
                break

        if solved_key is not None or termination_reason == "max_nodes_reached":
            break

    final_key = solved_key if solved_key is not None else best_fallback_key
    planned_actions = _reconstruct_action_path(nodes, final_key)
    final_node = nodes[final_key]
    final_dense, final_sparse, _ = extract_step_reward_details(final_node.timestep)
    search_stats = {
        "planner": "python_astar_dense_proxy",
        "solved": solved_key is not None,
        "terminated_reason": termination_reason,
        "generated_states": int(len(nodes)),
        "expanded_states": int(expanded_states),
        "max_nodes": int(max_nodes),
        "max_expansions": int(max_expansions),
        "solution_length": int(len(planned_actions)),
        "solution_cost": int(len(planned_actions)),
        "final_dense_reward": float(final_dense),
        "final_sparse_reward": float(final_sparse),
        "use_dense_heuristic": bool(use_dense_heuristic),
        "heuristic_reference_qmax": float(root_qmax),
    }
    return AStarPlanResult(actions=planned_actions, search_stats=search_stats)
