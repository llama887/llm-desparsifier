"""Python A* planner for PuzzleScript using C++ engine for stepping.

Uses engine.backup_level() / restore_level() / process_input() for state
transitions, and an external Python heuristic function for guidance.
This lets GEPA plug in LLM-generated heuristics while keeping the fast
C++ engine for game simulation.
"""

from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from puzzlescript_adapter import (
    build_puzzlescript_ctx,
)

MAX_AGAIN = 50


@dataclass
class PuzzleScriptSearchResult:
    """Result from one A* search episode."""
    solved: bool
    actions: list[int]
    expanded_states: int
    generated_states: int
    solution_length: int
    time_s: float
    score: float  # GEPA-compatible scalar score
    trace_summary: dict[str, Any] = field(default_factory=dict)


def _process_input(engine, action: int) -> bool:
    """Process input and handle 'again' loops."""
    changed = engine.process_input(action)
    again = 0
    while engine.is_againing() and again < MAX_AGAIN:
        changed = engine.process_input(-1) or changed
        again += 1
    return changed


def _state_key(engine) -> tuple:
    """Get a hashable state representation from the engine."""
    return tuple(engine.get_objects())


def _compact_ctx_snapshot(ctx: dict[str, Any]) -> dict[str, Any]:
    """Return a small state summary suitable for reflection prompts."""

    object_counts = {
        name: len(positions)
        for name, positions in sorted(ctx.get("object_positions", {}).items())
        if positions
    }
    return {
        "score_normalized": float(ctx.get("score_normalized", 0.0)),
        "is_winning": bool(ctx.get("is_winning", False)),
        "object_counts": object_counts,
        "ascii_state": str(ctx.get("ascii_state", "")),
    }


def puzzlescript_astar(
    engine,
    compiled_json: dict[str, Any],
    heuristic_fn: Callable[[dict[str, Any]], float],
    max_expansions: int = 50_000,
    timeout_s: float = 30.0,
) -> PuzzleScriptSearchResult:
    """Run A* search on a PuzzleScript level with a custom Python heuristic.

    Args:
        engine: C++ PuzzleScript Engine with a level already loaded.
        compiled_json: Parsed compiled game JSON (for context building).
        heuristic_fn: Callable that takes a ctx dict and returns a
            non-negative cost estimate. Lower = closer to goal.
        max_expansions: Maximum states to expand.
        timeout_s: Wall-clock timeout in seconds.

    Returns:
        PuzzleScriptSearchResult with solve status and GEPA-compatible score.
    """
    t0 = time.perf_counter()
    initial_backup = engine.backup_level()
    initial_key = _state_key(engine)

    # Compute initial heuristic
    ctx = build_puzzlescript_ctx(engine, compiled_json)
    try:
        h0 = float(heuristic_fn(ctx))
    except Exception:
        h0 = 0.0

    # Priority queue: (f_cost, tie_break, g_cost, state_backup, state_key)
    counter = 0
    open_set: list[tuple[float, int, int, Any, tuple]] = []
    heapq.heappush(open_set, (h0, counter, 0, initial_backup, initial_key))

    # Closed set + parent tracking
    came_from: dict[tuple, tuple[tuple, int]] = {}  # state_key -> (parent_key, action)
    g_cost: dict[tuple, int] = {initial_key: 0}
    expanded = 0
    generated = 1
    root_snapshot = _compact_ctx_snapshot(ctx)
    expansion_samples: list[dict[str, Any]] = []
    best_seen_f = h0
    best_seen_h = h0
    terminated_reason = "open_set_exhausted"

    has_action = not engine.has_metadata("noaction") if hasattr(engine, "has_metadata") else True
    n_actions = 5 if has_action else 4

    while open_set and expanded < max_expansions:
        if time.perf_counter() - t0 > timeout_s:
            terminated_reason = "timeout"
            break

        current_f, _, g, current_backup, current_key = heapq.heappop(open_set)

        # Skip if we already found a better path
        if g > g_cost.get(current_key, float("inf")):
            continue

        expanded += 1
        engine.restore_level(current_backup)
        current_ctx = build_puzzlescript_ctx(engine, compiled_json)
        try:
            current_h = max(0.0, float(heuristic_fn(current_ctx)))
        except Exception:
            current_h = 0.0
        if len(expansion_samples) < 6:
            expansion_samples.append(
                {
                    "expanded_index": expanded,
                    "g_cost": g,
                    "h_cost": current_h,
                    "f_cost": float(current_f),
                    "snapshot": _compact_ctx_snapshot(current_ctx),
                }
            )
        best_seen_f = min(best_seen_f, float(current_f))
        best_seen_h = min(best_seen_h, current_h)

        for action in range(n_actions):
            engine.restore_level(current_backup)
            changed = _process_input(engine, action)
            if not changed:
                continue

            next_key = _state_key(engine)
            next_g = g + 1

            if next_g >= g_cost.get(next_key, float("inf")):
                continue

            generated += 1
            g_cost[next_key] = next_g
            came_from[next_key] = (current_key, action)

            if engine.is_winning():
                # Reconstruct solution
                actions = _reconstruct(came_from, next_key)
                elapsed = time.perf_counter() - t0
                score = _gepa_score(True, expanded, max_expansions)
                trace_summary = {
                    "terminated_reason": "solved",
                    "root_snapshot": root_snapshot,
                    "sampled_states": expansion_samples,
                    "best_seen_f": float(min(best_seen_f, float(next_g))),
                    "best_seen_h": float(min(best_seen_h, 0.0)),
                    "open_set_size_at_end": len(open_set),
                }
                return PuzzleScriptSearchResult(
                    solved=True, actions=actions,
                    expanded_states=expanded,
                    generated_states=generated,
                    solution_length=len(actions),
                    time_s=elapsed, score=score,
                    trace_summary=trace_summary,
                )

            next_backup = engine.backup_level()
            ctx = build_puzzlescript_ctx(engine, compiled_json)
            try:
                h = float(heuristic_fn(ctx))
            except Exception:
                h = 0.0
            h = max(0.0, h)
            best_seen_h = min(best_seen_h, h)

            counter += 1
            heapq.heappush(
                open_set,
                (next_g + h, counter, next_g, next_backup, next_key),
            )

            best_seen_f = min(best_seen_f, float(next_g + h))

            if len(expansion_samples) < 6 and action == 0:
                expansion_samples.append(
                    {
                        "expanded_index": expanded,
                        "successor_action": action,
                        "g_cost": next_g,
                        "h_cost": h,
                        "f_cost": float(next_g + h),
                        "snapshot": _compact_ctx_snapshot(ctx),
                    }
                )

    if expanded >= max_expansions:
        terminated_reason = "expansion_budget"

    trace_summary = {
        "terminated_reason": terminated_reason,
        "root_snapshot": root_snapshot,
        "sampled_states": expansion_samples,
        "best_seen_f": float(best_seen_f),
        "best_seen_h": float(best_seen_h),
        "open_set_size_at_end": len(open_set),
    }

    elapsed = time.perf_counter() - t0
    score = _gepa_score(False, expanded, max_expansions)
    return PuzzleScriptSearchResult(
        solved=False, actions=[],
        expanded_states=expanded,
        generated_states=generated,
        solution_length=0,
        time_s=elapsed, score=score,
        trace_summary=trace_summary,
    )


def _reconstruct(
    came_from: dict[tuple, tuple[tuple, int]],
    goal_key: tuple,
) -> list[int]:
    """Trace back the action sequence from goal to start."""
    actions = []
    current = goal_key
    while current in came_from:
        parent_key, action = came_from[current]
        actions.append(action)
        current = parent_key
    actions.reverse()
    return actions


def _gepa_score(solved: bool, expanded: int, max_expansions: int) -> float:
    """Compute GEPA-compatible score matching the XLand metric.

    Score = ((N+1) - S) / (N+1) where:
      N = max_expansions budget
      S = expanded_states if solved, else N+1
    Solved runs always score > 0.5, unsolved < 0.5.
    """
    n = max_expansions
    s = expanded if solved else n + 1
    return ((n + 1) - s) / (n + 1)


def blind_heuristic(ctx: dict[str, Any]) -> float:
    """Blind heuristic (h=0). Used as baseline."""
    return 0.0


def builtin_heuristic(ctx: dict[str, Any]) -> float:
    """Use the engine's built-in score as heuristic."""
    return ctx.get("score", 0.0)
