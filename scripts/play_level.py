#!/usr/bin/env python3
"""Play a saved XLand level interactively with dense-reward diagnostics.

This script mirrors the deterministic replay setup used by
`scripts/generate_training_video.py` but swaps prerecorded policy actions for
human keyboard input. It is needed for reward-debugging workflows where humans
want to intentionally execute a high-quality action sequence and immediately see
which dense-reward components fire at each step. It differs from video
rendering by running an interactive event loop, and it differs from standard
`xminigrid.manual_control` by preserving the project's custom right-panel
sidebar with dense/sparse totals and per-component trend plots.
"""

from __future__ import annotations

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,no-member
import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from generate_training_video import (
    DEFAULT_STATE_ROOT,
    DENSE_REWARD_FILENAME,
    TRAJECTORY_FILENAME,
    _build_env,
    _build_replay_step_fns,
    _coerce_key,
    _configure_replay_jax_runtime,
    _draw_overlay,
    _format_overlay_lines,
    _load_dense_reward,
    _load_json,
    _normalize_component_map,
    _resolve_ruleset,
    _resolve_run_dir,
    _summarize_env_text,
    _wrap_env_with_dense_reward,
    _write_trace_payload,
)

from llm_desparsifier.rewards.behavior_summary import ACTION_NAMES as BEHAVIOR_ACTION_NAMES
from llm_desparsifier.rewards.parser import describe_ruleset

DEFAULT_TRACE_NAME = "play_level_trace.json"
DEFAULT_WINDOW_SIZE = 960
DEFAULT_FPS = 30
PLAY_OVERLAY_WRAP_CHARS = 58

pygame: Any
try:
    import pygame as _pygame

    pygame = _pygame
except ImportError:  # pragma: no cover - runtime dependency for interactive mode only
    pygame = None

ACTION_NAMES: dict[int, str] = dict(BEHAVIOR_ACTION_NAMES)

KEY_TO_ACTION: dict[str, int] = {
    "up": 0,
    "w": 0,
    "right": 1,
    "d": 1,
    "left": 2,
    "a": 2,
    "tab": 3,
    "e": 3,
    "left shift": 4,
    "q": 4,
    "space": 5,
}


def _extract_human_objective_lines(
    env_text: Any,
    env_summary: str | None,
) -> tuple[str | None, str | None]:
    """Extract concise objective text for human-facing play overlays.

    This helper parses the verbose environment-description payload and isolates
    the two clauses that matter most to a human actively controlling the agent:
    the direct task statement (`Your task is ...`) and the exact success
    criterion (`Success when ...`). It is needed because `env_text` is optimized
    for reward-synthesis code generation and often contains implementation
    details that distract manual players. It differs from
    `_summarize_env_text` by intentionally selecting only objective-critical
    clauses instead of preserving the full normalized description.

    Args:
        env_text: Raw environment description value from trajectory metadata.
        env_summary: Fallback normalized summary string when `env_text` is
            missing or unusable.

    Returns:
        A tuple `(objective_line, win_condition_line)` where each element is a
        punctuation-normalized sentence or `None` when unavailable.
    """

    def _normalize_sentence(sentence: str | None) -> str | None:
        if sentence is None:
            return None
        normalized = " ".join(sentence.split()).strip()
        if not normalized:
            return None
        if normalized[-1] not in ".!?":
            normalized = f"{normalized}."
        return normalized

    source_text: str | None = env_text if isinstance(env_text, str) else env_summary
    if not isinstance(source_text, str):
        return None, None

    normalized_source = " ".join(source_text.split())
    if not normalized_source:
        return None, None

    objective_match = re.search(
        r"(Your task is .*?)(?:[.!?](?:\s|$)|$)", normalized_source, flags=re.IGNORECASE
    )
    win_condition_match = re.search(
        r"(Success when .*?)(?:[.!?](?:\s|$)|$)", normalized_source, flags=re.IGNORECASE
    )

    objective_line = _normalize_sentence(objective_match.group(1) if objective_match else None)
    win_condition_line = _normalize_sentence(
        win_condition_match.group(1) if win_condition_match else None
    )

    if objective_line is None and env_summary is not None:
        objective_line = _normalize_sentence(env_summary)

    return objective_line, win_condition_line


def _trim_leading_phrase(text: str, phrase: str) -> str:
    """Remove a case-insensitive sentence prefix from display text.

    This helper strips boilerplate lead-ins (for example `Your task is`) so
    overlay labels can read cleanly (`OBJECTIVE: ...`) without repeating the
    same framing twice. It is needed because extracted objective clauses already
    include natural-language prefixes intended for prose, and it differs from a
    generic replace operation by removing only one anchored leading phrase.

    Args:
        text: Full sentence to post-process.
        phrase: Prefix phrase to remove when present at sentence start.

    Returns:
        Sentence text with at most one leading prefix removed.
    """
    lowered_text = text.lower()
    lowered_phrase = phrase.lower()
    if lowered_text.startswith(lowered_phrase):
        return text[len(phrase) :].strip()
    return text


def _wrap_labeled_line(label: str, text: str, *, max_chars: int) -> list[str]:
    """Wrap one labeled overlay field into fixed-width panel-friendly lines.

    This helper formats one diagnostics field with a stable uppercase label and
    continuation indentation so long objective/control text remains readable in
    the sidebar. It is needed because manual-play overlays mix short metrics with
    long prose snippets, and it differs from plain `textwrap.wrap` by keeping a
    persistent `LABEL: ` prefix on the first line and aligned continuations on
    later lines.

    Args:
        label: Left-hand field label (for example `OBJECTIVE`).
        text: Unwrapped body text for the field.
        max_chars: Maximum wrapped chunk width before continuation.

    Returns:
        One or more fully formatted lines ready for overlay rendering.
    """
    prefix = f"{label}: "
    continuation_prefix = " " * len(prefix)
    chunks = textwrap.wrap(
        text,
        width=max(12, max_chars),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not chunks:
        return [prefix.rstrip()]
    wrapped_lines: list[str] = []
    for idx, chunk in enumerate(chunks):
        line_prefix = prefix if idx == 0 else continuation_prefix
        wrapped_lines.append(f"{line_prefix}{chunk}")
    return wrapped_lines


def _build_human_overlay_context_lines(
    objective_line: str | None,
    win_condition_line: str | None,
) -> list[str]:
    """Build the human-only objective and controls block for the play overlay.

    This helper composes a concise context block that appears only during manual
    keyboard play. It is needed because human users require immediate objective
    clarity and explicit control hints while acting in real time, and it differs
    from policy-rollout overlays by prioritizing task and input guidance over the
    full synthesis-oriented environment description.

    Args:
        objective_line: Optional extracted objective sentence.
        win_condition_line: Optional extracted success-condition sentence.

    Returns:
        Ordered overlay lines containing objective/win-condition text and a
        compact control legend.
    """
    lines: list[str] = []
    if objective_line:
        objective_text = _trim_leading_phrase(objective_line, "Your task is")
        lines.extend(
            _wrap_labeled_line(
                "OBJECTIVE",
                objective_text,
                max_chars=PLAY_OVERLAY_WRAP_CHARS,
            )
        )
    if win_condition_line:
        win_condition_text = _trim_leading_phrase(win_condition_line, "Success when")
        lines.extend(
            _wrap_labeled_line(
                "WIN CONDITION",
                win_condition_text,
                max_chars=PLAY_OVERLAY_WRAP_CHARS,
            )
        )

    control_text = (
        "Up/W=move_forward; Left/A=turn_left; Right/D=turn_right; "
        "Tab/E=pick_up; Shift/Q=put_down; Space=toggle; R=reset; Esc=quit."
    )
    lines.extend(
        _wrap_labeled_line(
            "CONTROLS",
            control_text,
            max_chars=PLAY_OVERLAY_WRAP_CHARS,
        )
    )
    return lines


def _resolve_human_overlay_objective_lines(
    *,
    env: Any,
    env_params: Any,
    trajectory_env_text: Any,
    trajectory_env_summary: str | None,
) -> tuple[str | None, str | None]:
    """Resolve play-overlay objective text, preferring live ruleset semantics.

    This helper computes human-facing objective lines for manual play by first
    regenerating environment text from the currently reconstructed ruleset and
    then extracting concise objective/success clauses from that fresh text. It is
    needed because historical `eval_trajectory.json` artifacts may contain stale
    or previously misinterpreted direction wording, and it differs from directly
    consuming `trajectory["env_text"]` by prioritizing the live, authoritative
    ruleset semantics used by the current replay session.

    Args:
        env: Unwrapped XMiniGrid environment instance used for deterministic play.
        env_params: Environment parameters with the selected ruleset installed.
        trajectory_env_text: `env_text` string loaded from trajectory metadata.
        trajectory_env_summary: Pre-normalized summary from trajectory metadata.

    Returns:
        Tuple `(objective_line, win_condition_line)` suitable for overlay display.
    """
    try:
        live_env_text = describe_ruleset(env, env_params)
    except Exception:
        live_env_text = None

    if isinstance(live_env_text, str) and live_env_text.strip():
        live_summary = _summarize_env_text(live_env_text)
        objective_line, win_condition_line = _extract_human_objective_lines(
            live_env_text,
            live_summary,
        )
        if objective_line or win_condition_line:
            return objective_line, win_condition_line

    return _extract_human_objective_lines(
        trajectory_env_text,
        trajectory_env_summary,
    )


@dataclass  # pylint: disable=too-many-instance-attributes
class PlaySessionState:
    """Hold mutable reward diagnostics for one interactive session.

    This data object groups dense/sparse totals and component histories so
    interactive stepping can update state through a single well-typed object. It
    is needed because the play loop performs incremental updates on every
    keypress, and it differs from ad-hoc local variables by making state
    transitions explicit and testable.
    """

    dense_total: float
    sparse_total: float
    component_order: tuple[str, ...]
    component_totals: dict[str, float]
    component_series: dict[str, list[float]]
    last_dense_reward: float
    last_sparse_reward: float
    last_components: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for interactive level play.

    This parser defines the user-facing interface for selecting a candidate run,
    controlling display cadence, and setting output trace paths. It is needed so
    debugging sessions can be reproduced from the same artifacts used by the
    video renderer, and it differs from `generate_training_video` argument
    parsing by emphasizing live control-loop settings (`window_size`, `fps`) and
    by exposing a play-specific trace artifact name.

    Args:
        argv: Optional explicit argument sequence for testability. When omitted,
            arguments are read from `sys.argv`.

    Returns:
        Parsed argparse namespace with normalized typed fields.
    """
    parser = argparse.ArgumentParser(
        description="Play a saved eval trajectory level with keyboard actions"
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_STATE_ROOT,
        help="GEPA state root (default: %(default)s)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory containing eval_trajectory.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Reserved for future gameplay recording; currently unused.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Output trace JSON path (default: <run_dir>/play_level_trace.json)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on total human actions across the session",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Square pygame window size in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Render refresh cap for interactive display (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _resolve_trace_output(run_dir: Path, requested_path: Path | None) -> Path:
    """Resolve and create the trace output path used by interactive play.

    This helper centralizes trace-path resolution so success and failure flows
    always serialize to the same location. It is needed because manual sessions
    can terminate via normal quit, exceptions, or window-close events, and it
    differs from inline path handling by ensuring parent directories exist
    before the event loop starts.

    Args:
        run_dir: Candidate run directory used as the default output root.
        requested_path: Optional user-provided output location.

    Returns:
        Absolute filesystem path where the trace JSON should be written.
    """
    resolved = requested_path or (run_dir / DEFAULT_TRACE_NAME)
    output = resolved.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _resolve_initial_state(trajectory: Mapping[str, Any], benchmark: Any) -> tuple[Any, Any]:
    """Resolve deterministic ruleset and reset key from saved trajectory data.

    This helper reconstructs the exact initial rollout state from
    `eval_trajectory.json` so human play starts from the same instance used by
    policy evaluation. It is needed because reward debugging relies on
    apples-to-apples comparison between policy behavior and human behavior, and
    it differs from fresh random environment sampling by validating and using
    serialized RNG material (`reset_key`) plus ruleset selection metadata.

    Args:
        trajectory: Parsed evaluation trajectory payload.
        benchmark: Loaded XLand benchmark object used for ruleset selection.

    Returns:
        Tuple `(ruleset, reset_key)` ready for environment parameter replacement
        and deterministic reset execution.

    Raises:
        ValueError: If the reset key is missing or malformed.
    """
    if "reset_key" not in trajectory:
        raise ValueError(
            "Trajectory is missing reset_key required for deterministic play "
            f"({TRAJECTORY_FILENAME})"
        )
    reset_key_raw = trajectory.get("reset_key")
    if not isinstance(reset_key_raw, list):
        raise ValueError("Trajectory reset_key must be a two-item integer list")

    reset_key = _coerce_key(reset_key_raw, name="reset_key")
    ruleset = _resolve_ruleset(dict(trajectory), benchmark)
    return ruleset, reset_key


def _initialize_accumulators(dense_reward_fn: Any) -> PlaySessionState:
    """Initialize session accumulator state for rewards and component history.

    This helper prepares mutable counters used throughout a human-controlled
    episode. It is needed because dense/sparse totals and component trend-series
    must survive many per-keypress updates, and it differs from ad-hoc local
    initialization by deriving stable component ordering from the synthesized
    reward function's declared keys when available.

    Args:
        dense_reward_fn: Compiled dense reward callable loaded from run
            artifacts.

    Returns:
        Dict containing initialized totals, component maps, and history buffers.
    """
    component_order = tuple(getattr(dense_reward_fn, "__reward_component_keys__", ()))
    return PlaySessionState(
        dense_total=0.0,
        sparse_total=0.0,
        component_order=component_order,
        component_totals={name: 0.0 for name in component_order},
        component_series={name: [] for name in component_order},
        last_dense_reward=0.0,
        last_sparse_reward=0.0,
        last_components={name: 0.0 for name in component_order},
    )


def _reset_episode(
    reset_fn: Callable[[Any, Any], Any],
    *,
    env_params: Any,
    reset_key: Any,
) -> Any:
    """Reset the environment to the same deterministic initial state.

    This helper wraps reset invocation to make deterministic restart semantics
    explicit and easy to test. It is needed because the `R` key must recreate
    the exact saved start-state every time, and it differs from directly calling
    `env.reset` by preserving one clear boundary for replay-deterministic
    behavior checks.

    Args:
        reset_fn: Prepared reset callable from `_build_replay_step_fns`.
        env_params: Environment parameters with selected ruleset installed.
        reset_key: Serialized reset key reconstructed from trajectory metadata.

    Returns:
        Environment timestep object returned by `reset_fn`.
    """
    return reset_fn(env_params, reset_key)


def _apply_action_and_collect(
    *,
    step_fn: Callable[[Any, Any, Any], Any],
    env_params: Any,
    timestep: Any,
    action_value: int,
    step_index: int,
    episode_index: int,
    session_state: PlaySessionState,
) -> tuple[Any, dict[str, Any], PlaySessionState]:  # pylint: disable=too-many-arguments,too-many-locals
    """Step the environment once and update reward diagnostics for tracing.

    This helper executes exactly one environment step for one keyboard action,
    then normalizes reward outputs into trace-friendly Python scalars. It is
    needed because interactive loops require synchronized updates across totals,
    per-component series, and user-visible overlay values, and it differs from
    the video renderer's inline loop by returning all updated accumulators as a
    single deterministic state transition that unit tests can validate in
    isolation.

    Args:
        step_fn: Prepared environment step callable from replay setup.
        env_params: Active environment params including selected ruleset.
        timestep: Current timestep before action execution.
        action_value: Integer action id selected from keyboard input.
        step_index: Zero-based step index within the current episode.
        episode_index: Zero-based episode index incremented on resets.
        session_state: Mutable session-level reward/component diagnostics.

    Returns:
        Tuple containing new timestep, emitted trace row, and updated session
        diagnostics object.
    """
    action = jnp.asarray(int(action_value))
    timestep_next = step_fn(env_params, timestep, action)

    extras = getattr(timestep_next, "extras", None)
    dense_reward_value = float(jnp.asarray(timestep_next.reward))
    sparse_reward_value = dense_reward_value
    reward_components: dict[str, Any] = {}
    if extras is not None:
        sparse_reward_value = float(
            jnp.asarray(extras.get("ground_truth_reward", dense_reward_value))
        )
        reward_components = extras.get("reward_components") or {}

    session_state.dense_total += dense_reward_value
    session_state.sparse_total += sparse_reward_value

    if not session_state.component_order and reward_components:
        session_state.component_order = tuple(sorted(reward_components.keys()))
        session_state.component_totals = {name: 0.0 for name in session_state.component_order}
        session_state.component_series = {
            name: [0.0] * step_index for name in session_state.component_order
        }

    component_values = _normalize_component_map(reward_components, session_state.component_order)
    for name, value in component_values.items():
        session_state.component_totals[name] = session_state.component_totals.get(name, 0.0) + value
        session_state.component_series.setdefault(name, []).append(value)

    session_state.last_dense_reward = dense_reward_value
    session_state.last_sparse_reward = sparse_reward_value
    session_state.last_components = dict(component_values)

    trace_row = {
        "episode_index": int(episode_index),
        "step": int(step_index),
        "action": int(action_value),
        "action_name": ACTION_NAMES.get(int(action_value), f"action_{int(action_value)}"),
        "dense_reward": dense_reward_value,
        "sparse_reward": sparse_reward_value,
        "dense_total": session_state.dense_total,
        "sparse_total": session_state.sparse_total,
        "reward_components": component_values,
        "reward_component_totals": dict(session_state.component_totals),
        "episode_done": bool(timestep_next.last()),
    }

    return timestep_next, trace_row, session_state


def _render_overlay_frame(
    frame: Any,
    lines: list[str],
    *,
    component_series: Mapping[str, list[float]],
    component_order: Iterable[str],
) -> np.ndarray:
    """Render gameplay frame plus diagnostics sidebar into one RGB array.

    This helper delegates to the shared overlay renderer so interactive play and
    offline video generation present the same textual and chart diagnostics. It
    is needed because visual parity is a core debugging requirement, and it
    differs from direct `_draw_overlay` calls by defining a stable testable
    boundary for verifying component-history forwarding.

    Args:
        frame: Raw environment-rendered frame.
        lines: Already formatted text rows for the diagnostics panel.
        component_series: Component reward history by name.
        component_order: Stable component display order.

    Returns:
        Composed RGB image array containing map viewport and diagnostics panel.
    """
    return _draw_overlay(
        frame,
        lines,
        component_series=dict(component_series),
        component_order=tuple(component_order),
    )


def _update_display(
    *,
    pygame_mod: Any,
    window: Any,
    composed: np.ndarray,
    window_size: int,
) -> None:
    """Blit a composed RGB frame into the pygame window with scaling.

    This helper converts NumPy image data into a pygame surface and scales it to
    the configured square window size. It is needed because the composed map +
    sidebar dimensions vary by environment and text length, and it differs from
    direct per-loop blitting by encapsulating the axis transpose and scaling
    transformations in one reusable location.

    Args:
        pygame_mod: Imported `pygame` module, injected for testability.
        window: Active pygame display surface.
        composed: RGB image with shape `[height, width, 3]`.
        window_size: Square target side-length in pixels.
    """
    surface = pygame_mod.surfarray.make_surface(np.transpose(composed, (1, 0, 2)))
    scaled = pygame_mod.transform.smoothscale(surface, (window_size, window_size))
    window.blit(scaled, (0, 0))
    pygame_mod.display.flip()


def _build_trace_payload(
    *,
    trajectory: Mapping[str, Any],
    env_seed: Any,
    env_text: Any,
    env_summary: str | None,
    run_dir: Path,
    trace_steps: list[dict[str, Any]],
    replay_error: str | None,
) -> dict[str, Any]:  # pylint: disable=too-many-arguments
    """Assemble trace JSON payload for both success and failure paths.

    This helper ensures that interactive play traces follow one schema
    regardless of exit path. It is needed because debugging sessions may end via
    normal quit or exceptions and should still remain comparable, and it differs
    from inlined dict construction by centralizing metadata fields specific to
    keyboard-driven sessions.

    Args:
        trajectory: Original trajectory metadata loaded from run artifacts.
        env_seed: Environment seed metadata from the trajectory payload.
        env_text: Full environment description text from trajectory payload.
        env_summary: Normalized environment summary string for UI display.
        run_dir: Candidate run directory used for this session.
        trace_steps: Per-action diagnostics collected during play.
        replay_error: Optional stringified exception summary.

    Returns:
        JSON-serializable trace payload dictionary.
    """
    return {
        "trajectory": dict(trajectory),
        "env_seed": env_seed,
        "env_text": env_text,
        "env_summary": env_summary,
        "run_dir": str(run_dir),
        "dense_reward_path": str(run_dir / DENSE_REWARD_FILENAME),
        "input_mode": "human_keyboard",
        "keymap": {
            "up|w": "move_forward",
            "left|a": "turn_left",
            "right|d": "turn_right",
            "tab|e": "pick_up",
            "left shift|q": "put_down",
            "space": "toggle",
            "r": "reset to initial saved state",
            "esc": "quit",
        },
        "steps": trace_steps,
        "replay_error": replay_error,
        "replay_complete": replay_error is None,
    }


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Run the interactive keyboard play loop for one saved run artifact.

    This entry point coordinates deterministic environment reconstruction, live
    keyboard action stepping, diagnostics overlay rendering, and durable trace
    writing. It is needed as a one-command debugging tool for dense reward
    behavior, and it differs from policy replay by allowing humans to test
    intentional trajectories while preserving the same reward decomposition data
    model used elsewhere in the project.
    """
    _configure_replay_jax_runtime()
    args = parse_args()

    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0 when provided")
    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    state_root = args.state_root.expanduser().resolve()
    run_dir = _resolve_run_dir(state_root, args.run_dir)
    trace_output = _resolve_trace_output(run_dir, args.trace_output)

    trajectory_path = run_dir / TRAJECTORY_FILENAME
    trajectory = _load_json(trajectory_path)
    env_text = trajectory.get("env_text")
    env_seed = trajectory.get("env_seed", trajectory.get("eval_seed"))
    env_summary = _summarize_env_text(env_text)

    dense_reward_fn = _load_dense_reward(run_dir)
    env, env_params, benchmark = _build_env(trajectory)
    ruleset, reset_key = _resolve_initial_state(trajectory, benchmark)
    env_params = env_params.replace(ruleset=ruleset)
    objective_line, win_condition_line = _resolve_human_overlay_objective_lines(
        env=env,
        env_params=env_params,
        trajectory_env_text=env_text,
        trajectory_env_summary=env_summary,
    )
    env = _wrap_env_with_dense_reward(env, dict(trajectory), dense_reward_fn)
    reset_fn, step_fn = _build_replay_step_fns(env)

    session_state = _initialize_accumulators(dense_reward_fn)
    trace_steps: list[dict[str, Any]] = []

    timestep = _reset_episode(reset_fn, env_params=env_params, reset_key=reset_key)
    episode_index = 0
    episode_step = 0

    replay_error: str | None = None
    if pygame is None:
        raise ImportError("pygame is required for interactive play_level sessions")

    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode((args.window_size, args.window_size))
    pygame.display.set_caption("llm-desparsifier play_level")
    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type != pygame.KEYDOWN:
                    continue

                key_name = pygame.key.name(int(event.key))
                if key_name == "escape":
                    running = False
                    break

                if key_name == "r":
                    timestep = _reset_episode(reset_fn, env_params=env_params, reset_key=reset_key)
                    session_state = _initialize_accumulators(dense_reward_fn)
                    episode_index += 1
                    episode_step = 0
                    continue

                action_value = KEY_TO_ACTION.get(key_name)
                if action_value is None:
                    continue

                if bool(timestep.last()):
                    continue
                if args.max_steps is not None and len(trace_steps) >= args.max_steps:
                    continue

                timestep, trace_row, session_state = _apply_action_and_collect(
                    step_fn=step_fn,
                    env_params=env_params,
                    timestep=timestep,
                    action_value=action_value,
                    step_index=episode_step,
                    episode_index=episode_index,
                    session_state=session_state,
                )
                trace_steps.append(trace_row)
                episode_step += 1

            frame = env.render(env_params, timestep)
            if frame is None:
                raise RuntimeError("Environment render returned None")

            max_steps_for_overlay = (
                args.max_steps if args.max_steps is not None else max(1, episode_step + 1)
            )
            lines = _build_human_overlay_context_lines(
                objective_line=objective_line,
                win_condition_line=win_condition_line,
            )
            lines.extend(
                _format_overlay_lines(
                    env_summary=None,
                    step_index=min(episode_step, max_steps_for_overlay - 1),
                    total_steps=max_steps_for_overlay,
                    dense_reward=session_state.last_dense_reward,
                    dense_total=session_state.dense_total,
                    sparse_reward=session_state.last_sparse_reward,
                    sparse_total=session_state.sparse_total,
                    component_values=session_state.last_components,
                    component_totals=session_state.component_totals,
                    component_order=session_state.component_order,
                )
            )
            if bool(timestep.last()):
                lines.append("episode complete: press R to reset")
            if args.max_steps is not None and len(trace_steps) >= args.max_steps:
                lines.append("max steps reached: press Esc to quit or R to reset")

            composed = _render_overlay_frame(
                frame,
                lines,
                component_series=session_state.component_series,
                component_order=session_state.component_order,
            )
            _update_display(
                pygame_mod=pygame,
                window=window,
                composed=composed,
                window_size=args.window_size,
            )
            clock.tick(args.fps)

    except Exception as exc:
        replay_error = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        payload = _build_trace_payload(
            trajectory=trajectory,
            env_seed=env_seed,
            env_text=env_text,
            env_summary=env_summary,
            run_dir=run_dir,
            trace_steps=trace_steps,
            replay_error=replay_error,
        )
        _write_trace_payload(trace_output, payload)
        print(f"[play_level] wrote {trace_output}")
        pygame.quit()


if __name__ == "__main__":
    main()
