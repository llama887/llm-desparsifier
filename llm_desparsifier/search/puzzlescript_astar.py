"""Python A* planner for PuzzleScript using C++ engine for stepping.

Uses engine.backup_level() / restore_level() / process_input() for state
transitions, and an external Python heuristic function for guidance.
This lets GEPA plug in LLM-generated heuristics while keeping the fast
C++ engine for game simulation.
"""

from __future__ import annotations

import heapq
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from puzzlescript_adapter import build_puzzlescript_ctx

MAX_AGAIN = 50

# Number of decoded (key, ctx) pairs kept resident per search. Everything
# beyond this is rebuilt from the engine backup on demand; the decoded ctx of
# one state costs tens of kilobytes, so retaining one per generated state is
# what drove multi-gigabyte search workers.
STATE_CTX_CACHE_SIZE = 256


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
    """Return a compact, candidate-independent state summary for reflection."""

    object_counts = {
        name: len(positions)
        for name, positions in sorted(ctx.get("object_positions", {}).items())
        if positions
    }
    engine_progress = max(0.0, min(1.0, float(ctx.get("score_normalized", 0.0))))
    win_progress_value = ctx.get("win_condition_progress")
    win_progress = (
        None
        if win_progress_value is None
        else max(0.0, min(1.0, float(win_progress_value)))
    )
    progress_score = (
        engine_progress
        if win_progress is None
        else 0.5 * engine_progress + 0.5 * win_progress
    )
    return {
        "score_normalized": engine_progress,
        "win_condition_progress": win_progress,
        "progress_score": progress_score,
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
    progress_samples: deque[dict[str, Any]] = deque(maxlen=12)
    late_states: deque[dict[str, Any]] = deque(maxlen=6)
    best_progress = float(root_snapshot["progress_score"])
    best_progress_snapshot = root_snapshot
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
        current_snapshot = _compact_ctx_snapshot(current_ctx)
        trace_state = {
            "expanded_index": expanded,
            "g_cost": g,
            "h_cost": current_h,
            "f_cost": float(current_f),
            "snapshot": current_snapshot,
        }
        late_states.append(trace_state)
        current_progress = float(current_snapshot["progress_score"])
        if current_progress > best_progress + 1e-12:
            best_progress = current_progress
            best_progress_snapshot = current_snapshot
            progress_samples.append(trace_state)
        if len(expansion_samples) < 6:
            expansion_samples.append(trace_state)
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
                    "search_strategy": "legacy_astar",
                    "search_algorithm": "astar",
                    "terminated_reason": "solved",
                    "root_snapshot": root_snapshot,
                    "sampled_states": expansion_samples,
                    "progress_samples": list(progress_samples),
                    "late_states": list(late_states),
                    "best_progress": best_progress,
                    "best_progress_snapshot": best_progress_snapshot,
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
        "search_strategy": "legacy_astar",
        "search_algorithm": "astar",
        "terminated_reason": terminated_reason,
        "root_snapshot": root_snapshot,
        "sampled_states": expansion_samples,
        "progress_samples": list(progress_samples),
        "late_states": list(late_states),
        "best_progress": best_progress,
        "best_progress_snapshot": best_progress_snapshot,
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


class _SearchStopped(RuntimeError):
    pass


class PuzzleScriptSearchAPI:
    """Budgeted engine boundary exposed to generated search code."""

    def __init__(
        self,
        engine,
        compiled_json: dict[str, Any],
        max_expansions: int,
        timeout_s: float,
        ctx_cache_size: int = STATE_CTX_CACHE_SIZE,
    ):
        self._engine = engine
        self._compiled = compiled_json
        self._max_expansions = max(1, max_expansions)
        self._deadline = time.perf_counter() + max(0.001, timeout_s)
        # Only the engine backup and the terminal flag are retained per state.
        # The decoded key and ctx are ~99% of the per-state footprint, so they
        # live in a bounded LRU and are rebuilt from the backup after eviction.
        self._states: list[tuple[Any, bool]] = []
        self._decoded: OrderedDict[int, tuple[tuple, dict[str, Any]]] = OrderedDict()
        self._decoded_limit = max(1, int(ctx_cache_size))
        self.expanded = 0
        self.generated = 0
        self.samples: list[dict[str, Any]] = []
        self.best_snapshot: dict[str, Any] = {}
        self.best_progress = 0.0
        self._add_state()

    def _check(self) -> None:
        if time.perf_counter() > self._deadline:
            raise _SearchStopped("timeout")
        if self.expanded >= self._max_expansions:
            raise _SearchStopped("expansion_budget")

    def _add_state(self) -> int:
        ctx = build_puzzlescript_ctx(self._engine, self._compiled)
        snapshot = _compact_ctx_snapshot(ctx)
        progress = float(snapshot["progress_score"])
        if not self._states or progress > self.best_progress:
            self.best_progress = progress
            self.best_snapshot = snapshot
        index = len(self._states)
        self._states.append((self._engine.backup_level(), self._engine.is_winning()))
        self._remember(index, _state_key(self._engine), ctx)
        self.generated += 1
        return index

    def _remember(self, index: int, key: tuple, ctx: dict[str, Any]) -> None:
        """Cache one decoded state, evicting the least recently used entry."""
        decoded = self._decoded
        decoded[index] = (key, ctx)
        decoded.move_to_end(index)
        while len(decoded) > self._decoded_limit:
            decoded.popitem(last=False)

    def _decode(self, state: int) -> tuple[tuple, dict[str, Any]]:
        """Return the (key, ctx) pair for a state, rebuilding it after eviction.

        Rebuilding repositions the engine on the requested state. Callers that
        also need the engine positioned elsewhere must restore it afterwards.
        """
        decoded = self._decoded
        entry = decoded.get(state)
        if entry is not None:
            decoded.move_to_end(state)
            return entry
        self._engine.restore_level(self._states[state][0])
        entry = (_state_key(self._engine), build_puzzlescript_ctx(self._engine, self._compiled))
        self._remember(state, entry[0], entry[1])
        return entry

    def initial(self) -> int:
        return 0

    def key(self, state: int) -> tuple:
        return self._decode(state)[0]

    def ctx(self, state: int) -> dict[str, Any]:
        return self._decode(state)[1]

    def is_winning(self, state: int) -> bool:
        return self._states[state][1]

    def expansion_budget(self) -> int:
        """Return the same hard expansion limit enforced by successors()."""
        return self._max_expansions

    def successors(self, state: int) -> list[tuple[int, int]]:
        self._check()
        self.expanded += 1
        if len(self.samples) < 12:
            self.samples.append(
                {
                    "expanded_index": self.expanded,
                    "snapshot": _compact_ctx_snapshot(self.ctx(state)),
                }
            )
        backup = self._states[state][0]
        self._engine.restore_level(backup)
        has_action = (
            not self._engine.has_metadata("noaction")
            if hasattr(self._engine, "has_metadata")
            else True
        )
        children = []
        for action in range(5 if has_action else 4):
            self._engine.restore_level(backup)
            if _process_input(self._engine, action):
                children.append((action, self._add_state()))
        return children


def puzzlescript_custom_search(
    engine,
    compiled_json: dict[str, Any],
    search_fn: Callable[[PuzzleScriptSearchAPI, int], list[int]],
    max_expansions: int = 50_000,
    timeout_s: float = 30.0,
    seed: int = 0,
) -> PuzzleScriptSearchResult:
    """Run generated search code through the same result and budget boundary."""
    t0 = time.perf_counter()
    initial_backup = engine.backup_level()
    api = PuzzleScriptSearchAPI(engine, compiled_json, max_expansions, timeout_s)
    reason = "search_exhausted"
    actions: list[int] = []
    try:
        proposed = search_fn(api, int(seed))
        if not isinstance(proposed, list) or any(
            not isinstance(action, int) or isinstance(action, bool) or action not in range(5)
            for action in proposed
        ):
            reason = "invalid_solution"
        else:
            engine.restore_level(initial_backup)
            valid = all(_process_input(engine, action) for action in proposed)
            if valid and engine.is_winning():
                actions = proposed
                reason = "solved"
            elif proposed:
                reason = "invalid_solution"
    except _SearchStopped as exc:
        reason = str(exc)
    except Exception as exc:
        reason = "custom_search_error"
        error = f"{type(exc).__name__}: {exc}"
    solved = reason == "solved"
    root_snapshot = _compact_ctx_snapshot(api.ctx(api.initial()))
    trace = {
        "search_strategy": "custom_search",
        "search_algorithm": getattr(search_fn, "_search_algorithm", "custom_unspecified"),
        "seed": int(seed),
        "terminated_reason": reason,
        "root_snapshot": root_snapshot,
        "sampled_states": api.samples[:6],
        "progress_samples": api.samples,
        "late_states": api.samples[-6:],
        "best_progress": api.best_progress,
        "best_progress_snapshot": api.best_snapshot,
        "open_set_size_at_end": 0,
    }
    if reason == "custom_search_error":
        trace["error"] = error
    elapsed = time.perf_counter() - t0
    return PuzzleScriptSearchResult(
        solved=solved,
        actions=actions,
        expanded_states=api.expanded,
        generated_states=api.generated,
        solution_length=len(actions),
        time_s=elapsed,
        score=_gepa_score(solved, api.expanded, max_expansions),
        trace_summary=trace,
    )


def puzzlescript_search(
    engine,
    compiled_json: dict[str, Any],
    strategy: str,
    artifact_fn: Callable[..., Any],
    max_expansions: int = 50_000,
    timeout_s: float = 30.0,
    seed: int = 0,
) -> PuzzleScriptSearchResult:
    """Dispatch a legacy heuristic or generated search through one evaluator."""
    if strategy == "heuristic":
        return puzzlescript_astar(
            engine, compiled_json, artifact_fn, max_expansions=max_expansions, timeout_s=timeout_s
        )
    if strategy == "custom_search":
        return puzzlescript_custom_search(
            engine,
            compiled_json,
            artifact_fn,
            max_expansions=max_expansions,
            timeout_s=timeout_s,
            seed=seed,
        )
    raise ValueError(f"unknown search strategy: {strategy}")


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
