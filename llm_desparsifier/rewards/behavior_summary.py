"""Compact trajectory summarization utilities for GEPA reflection feedback.

This module converts replay trajectories into concise, behavior-oriented text that
is cheap to include in LLM context windows. The summaries are designed to be an
additive signal alongside existing sparse-curve, component, and aggregate metric
feedback used by the reflection model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ACTION_NAMES = {
    0: "move_forward",
    1: "turn_right",
    2: "turn_left",
    3: "pick_up",
    4: "put_down",
    5: "toggle",
}

MANIPULATION_ACTIONS = {"pick_up", "put_down", "toggle"}
TURN_ACTIONS = {"turn_left", "turn_right"}


def summarize_trajectory_behavior(
    trajectory: Mapping[str, Any],
    *,
    long_repeat_threshold: int = 4,
    max_event_windows: int = 3,
    max_window_actions: int = 12,
) -> str:
    """Summarize trajectory behavior into a compact reflection-friendly report.

    This function converts a single replayable trajectory payload into a concise
    behavior report that highlights action composition, manipulation churn,
    oscillatory turning, repetitive loops, and suspicious local windows. It is
    needed because full action traces are often too long for reflection-model
    context, and it differs from raw trace logging by prioritizing diagnostics
    that correlate with reward misspecification (for example unnecessary pickup
    and put-down behavior in primarily navigation tasks).

    Args:
        trajectory: Parsed trajectory payload, typically loaded from
            ``eval_trajectory.json``.
        long_repeat_threshold: Minimum run length that contributes to the
            long-repeat metric.
        max_event_windows: Maximum number of suspicious local windows reported
            in the event-sketch section.
        max_window_actions: Maximum number of actions rendered per event window.

    Returns:
        A multi-line summary string suitable for the reflection input channel.
        If actions are missing or malformed, a stable fallback summary is
        returned instead of raising.
    """
    actions_raw = trajectory.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        return (
            "Behavior summary: trajectory actions unavailable. "
            "Cannot compute action-level diagnostics."
        )

    actions: list[int] = []
    for value in actions_raw:
        try:
            actions.append(int(value))
        except (TypeError, ValueError):
            continue
    if not actions:
        return (
            "Behavior summary: trajectory actions unavailable. "
            "Cannot compute action-level diagnostics."
        )

    action_names = [_action_name(action_id) for action_id in actions]
    total_steps = len(action_names)

    counts: dict[str, int] = {}
    for name in action_names:
        counts[name] = counts.get(name, 0) + 1

    pick_up_count = counts.get("pick_up", 0)
    put_down_count = counts.get("put_down", 0)
    toggle_count = counts.get("toggle", 0)
    manip_count = pick_up_count + put_down_count + toggle_count
    manipulation_rate = manip_count / float(max(1, total_steps))
    pickup_putdown_churn = min(pick_up_count, put_down_count) / float(
        max(1, pick_up_count + put_down_count)
    )

    turn_count = counts.get("turn_left", 0) + counts.get("turn_right", 0)
    oscillation_pairs = _count_turn_oscillations(action_names)
    oscillation_rate = oscillation_pairs / float(max(1, total_steps - 1))

    long_repeat_steps = _count_long_repeat_steps(
        action_names, threshold=max(2, int(long_repeat_threshold))
    )
    long_repeat_rate = long_repeat_steps / float(max(1, total_steps))

    forward_count = counts.get("move_forward", 0)
    forward_turn_balance = forward_count / float(max(1, turn_count))

    first_sparse_reward_step = _extract_first_sparse_reward_step(trajectory)

    lines: list[str] = [
        "Behavior summary (additive signal; existing sparse/component/metric feedback still applies):",
        f"- total_steps={total_steps}",
        "- action_histogram="
        + ", ".join(
            f"{name}:{counts.get(name, 0)} ({counts.get(name, 0) / float(total_steps):.1%})"
            for name in _ordered_action_names(counts)
        ),
        (
            "- manipulation_rate="
            f"{manipulation_rate:.3f} "
            f"(pick_up={pick_up_count}, put_down={put_down_count}, toggle={toggle_count})"
        ),
        f"- pickup_putdown_churn={pickup_putdown_churn:.3f}",
        f"- turn_oscillation_rate={oscillation_rate:.3f} (oscillation_pairs={oscillation_pairs})",
        f"- long_repeat_rate={long_repeat_rate:.3f} (threshold={max(2, int(long_repeat_threshold))})",
        f"- forward_turn_balance={forward_turn_balance:.3f} (forward/turn)",
        "- first_sparse_reward_step="
        + (
            str(first_sparse_reward_step) if first_sparse_reward_step is not None else "unavailable"
        ),
    ]

    event_lines = _build_event_sketches(
        action_names,
        max_windows=max(1, int(max_event_windows)),
        max_window_actions=max(3, int(max_window_actions)),
    )
    lines.append("Event sketch (top suspicious windows):")
    if event_lines:
        lines.extend(f"- {line}" for line in event_lines)
    else:
        lines.append("- none detected")

    return "\n".join(lines)


def summarize_trajectory_behavior_from_path(
    trajectory_path: Optional[str | Path],
    *,
    long_repeat_threshold: int = 4,
    max_event_windows: int = 3,
    max_window_actions: int = 12,
) -> str:
    """Load a trajectory artifact and summarize behavior with robust fallbacks.

    This helper encapsulates file loading and validation for trajectory-backed
    behavior summaries. It is needed because metric code should not fail when a
    trajectory artifact is missing or malformed, and it differs from direct JSON
    parsing by always returning a human-readable fallback summary rather than
    propagating exceptions into GEPA scoring.

    Args:
        trajectory_path: Path to ``eval_trajectory.json``. ``None`` or empty
            paths are treated as unavailable artifacts.
        long_repeat_threshold: Forwarded to ``summarize_trajectory_behavior``.
        max_event_windows: Forwarded to ``summarize_trajectory_behavior``.
        max_window_actions: Forwarded to ``summarize_trajectory_behavior``.

    Returns:
        A behavior summary string suitable for reflection inputs.
    """
    if trajectory_path is None:
        return (
            "Behavior summary: eval trajectory artifact unavailable (no trajectory path provided)."
        )

    path = Path(str(trajectory_path))
    if not path.exists():
        return f"Behavior summary: eval trajectory artifact missing at {path}."

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Behavior summary: failed to parse eval trajectory artifact at {path}: {exc}"

    if not isinstance(payload, Mapping):
        return (
            f"Behavior summary: invalid eval trajectory payload type at {path}; expected mapping."
        )

    return summarize_trajectory_behavior(
        payload,
        long_repeat_threshold=long_repeat_threshold,
        max_event_windows=max_event_windows,
        max_window_actions=max_window_actions,
    )


def _action_name(action_id: int) -> str:
    return ACTION_NAMES.get(action_id, f"action_{action_id}")


def _ordered_action_names(counts: Mapping[str, int]) -> list[str]:
    ordered: list[str] = [ACTION_NAMES[idx] for idx in sorted(ACTION_NAMES.keys())]
    extras = sorted(name for name in counts.keys() if name not in ordered)
    return ordered + extras


def _count_turn_oscillations(action_names: list[str]) -> int:
    count = 0
    for prev, curr in zip(action_names[:-1], action_names[1:]):
        if (prev, curr) in {
            ("turn_left", "turn_right"),
            ("turn_right", "turn_left"),
        }:
            count += 1
    return count


def _count_long_repeat_steps(action_names: list[str], *, threshold: int) -> int:
    if not action_names:
        return 0
    total = 0
    run_len = 1
    for idx in range(1, len(action_names)):
        if action_names[idx] == action_names[idx - 1]:
            run_len += 1
        else:
            if run_len >= threshold:
                total += run_len
            run_len = 1
    if run_len >= threshold:
        total += run_len
    return total


def _extract_first_sparse_reward_step(trajectory: Mapping[str, Any]) -> Optional[int]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    for idx, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        reward = step.get("sparse_reward")
        try:
            if reward is not None and float(reward) > 0.0:
                return int(step.get("step", idx))
        except (TypeError, ValueError):
            continue
    return None


def _build_event_sketches(
    action_names: list[str], *, max_windows: int, max_window_actions: int
) -> list[str]:
    if not action_names:
        return []

    window = max_window_actions
    scored: list[tuple[int, int, int]] = []
    for start in range(0, len(action_names)):
        end = min(len(action_names), start + window)
        segment = action_names[start:end]
        if not segment:
            continue
        manip = sum(1 for name in segment if name in MANIPULATION_ACTIONS)
        oscillations = _count_turn_oscillations(segment)
        repeat_steps = _count_long_repeat_steps(segment, threshold=3)
        score = (2 * manip) + oscillations + repeat_steps
        if score <= 0:
            continue
        scored.append((score, start, end))

    scored.sort(key=lambda item: (item[0], item[2] - item[1]), reverse=True)

    chosen: list[tuple[int, int, int]] = []
    for score, start, end in scored:
        overlaps = any(not (end <= c_start or start >= c_end) for _, c_start, c_end in chosen)
        if overlaps:
            continue
        chosen.append((score, start, end))
        if len(chosen) >= max_windows:
            break

    lines: list[str] = []
    for score, start, end in chosen:
        sketch = _run_length_encode(action_names[start:end])
        lines.append(f"steps[{start}:{end}] score={score} actions={sketch}")
    return lines


def _run_length_encode(action_names: Iterable[str]) -> str:
    sequence = list(action_names)
    if not sequence:
        return ""
    parts: list[str] = []
    current = sequence[0]
    run_len = 1
    for item in sequence[1:]:
        if item == current:
            run_len += 1
        else:
            parts.append(f"{current}x{run_len}")
            current = item
            run_len = 1
    parts.append(f"{current}x{run_len}")
    return " -> ".join(parts)


__all__ = [
    "summarize_trajectory_behavior",
    "summarize_trajectory_behavior_from_path",
    "ACTION_NAMES",
]
