#!/usr/bin/env python3
"""Generate an MP4 training video by replaying a saved evaluation trajectory."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from llm_desparsifier.search.astar import (
    AStarPlanResult as SharedAStarPlanResult,
)
from llm_desparsifier.search.astar import (
    estimate_dense_qmax,
    extract_step_reward_details,
    is_sparse_success,
    plan_with_astar,
    state_cache_key_from_timestep,
)

DEFAULT_STATE_ROOT = Path("artifacts/gepa_state")
TRAJECTORY_FILENAME = "eval_trajectory.json"
DENSE_REWARD_FILENAME = "dense_reward_synthesized.py"
TASK_INSTANCE_FILENAME = "task_instance.json"
ASTAR_PLAN_FILENAME = "astar_plan.json"
DEFAULT_VIDEO_NAME = "training_video.mp4"
DEFAULT_TRACE_NAME = "training_video_trace.json"
DEFAULT_ASTAR_HEURISTIC_VIDEO_NAME = "training_video_astar_heuristic.mp4"
DEFAULT_ASTAR_HEURISTIC_TRACE_NAME = "training_video_astar_heuristic_trace.json"
DEFAULT_ASTAR_NO_HEURISTIC_VIDEO_NAME = "training_video_astar_no_heuristic.mp4"
DEFAULT_ASTAR_NO_HEURISTIC_TRACE_NAME = "training_video_astar_no_heuristic_trace.json"
ROLLOUT_MODE_REPLAY = "trajectory_replay"
ROLLOUT_MODE_ASTAR_HEURISTIC = "astar_dense_heuristic"
ROLLOUT_MODE_ASTAR_NO_HEURISTIC = "astar_no_heuristic"
ASTAR_TIE_BREAK = "lowest_action_id"
DEFAULT_ASTAR_MAX_NODES = 200_000
DEFAULT_ASTAR_MAX_EXPANSIONS = 100_000
DEFAULT_VIEWPORT_SCALE = 2
OVERLAY_PANEL_PADDING = 8
OVERLAY_LINE_SPACING = 2
OVERLAY_MAP_PANEL_GAP = 10
OVERLAY_PANEL_MIN_WIDTH = 320
OVERLAY_GOAL_WRAP_CHARS = 58
OVERLAY_PLOT_HEIGHT = 180
OVERLAY_PLOT_MARGIN = 8
OVERLAY_PLOT_LINE_WIDTH = 2
OVERLAY_PLOT_LABEL_PAD = 4
CPU_FALLBACK_REEXEC_FLAG = "LLM_DESPARSIFIER_VIDEO_CPU_FALLBACK_DONE"


@dataclass(frozen=True)
class _HeuristicOnlyNode:
    """Store one explored node for video-local `h_only` A* planning.

    This record keeps the minimal ancestry and score data required to run the
    generate-video planner with a state-only dense heuristic. It is needed
    because the video workflow now intentionally diverges from the shared A*
    implementation by using dense reward only as `h(state)` while preserving
    deterministic path reconstruction, and it differs from the shared search
    node type by fixing `g` to zero for all nodes instead of tracking path
    length.
    """

    key: bytes
    timestep: Any
    parent_key: bytes | None
    parent_action: int | None
    h_cost: float
    f_cost: float


def _is_cuda_backend_init_error(exc: BaseException) -> bool:
    """Return whether an exception represents JAX CUDA backend-init failure.

    This helper classifies runtime failures that happen while JAX attempts to
    initialize CUDA devices (including out-of-memory during StreamExecutor
    creation). It is needed because replay can continue on CPU when GPU backend
    initialization fails, and it differs from broad exception handling by
    matching only known CUDA initialization signatures rather than masking
    unrelated runtime errors.

    Args:
        exc: Raised exception from replay setup or execution.

    Returns:
        True when the error text indicates CUDA backend initialization failed;
        False otherwise.
    """
    message = f"{exc.__class__.__name__}: {exc}".lower()
    indicators = (
        "unable to initialize backend 'cuda'",
        "no supported devices found for platform cuda",
        "cuda_error_out_of_memory",
        "unable to create streamexecutor for cuda",
        "failed call to cuinit",
    )
    return any(token in message for token in indicators)


def _reexec_with_cpu_fallback() -> None:
    """Re-exec this script with JAX forced to CPU exactly once.

    This helper replaces the current process so all future imports happen in a
    clean interpreter with CPU backend selection applied before JAX is touched.
    It is needed because changing JAX platform selection after failed backend
    initialization in-process is not reliable, and it differs from in-process
    retries by guaranteeing a fresh startup path with deterministic environment
    variables.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env[CPU_FALLBACK_REEXEC_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _configure_replay_jax_runtime() -> None:
    """Set conservative JAX threading defaults for replay/video workflows.

    This helper configures XLA CPU threading before the first JAX import so the
    video writer's ffmpeg subprocess launch avoids fork-with-many-threads
    deadlock scenarios that can hang long replay jobs. It is needed because this
    script combines JAX-based environment stepping with subprocess-backed video
    encoding in a single process, and it differs from global shell-level tuning
    by applying a safe default only for this CLI while respecting user-provided
    `XLA_FLAGS` values when they are already set.
    """
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    # Respect explicit user preference while keeping replay resilient by default:
    # if a previous attempt failed CUDA init and re-exec'd, force CPU backend.
    if os.environ.get(CPU_FALLBACK_REEXEC_FLAG) == "1":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training video generation.

    This parser defines how users select a run directory (or state root), output
    file locations, and replay limits. It is needed because the video generator
    can operate on many GEPA run outputs, and it differs from training CLI
    parsing by focusing solely on replay/render settings rather than GEPA or PPO
    configuration.
    """
    parser = argparse.ArgumentParser(
        description="Replay a saved eval trajectory and render a training video"
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
        "--latest-candidates",
        type=int,
        default=None,
        help=(
            "Process the most recently modified candidate run directories "
            "that contain replay artifacts (default: disabled)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: <run_dir>/training_video.mp4)",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Output trace JSON path (default: <run_dir>/training_video_trace.json)",
    )
    parser.add_argument(
        "--astar",
        action="store_true",
        help="Enable A* rollout video/trace generation",
    )
    parser.add_argument(
        "--astar-heuristic-output",
        type=Path,
        default=None,
        help=(
            "Output A* heuristic MP4 path "
            "(default: <run_dir>/training_video_astar_heuristic.mp4)"
        ),
    )
    parser.add_argument(
        "--astar-heuristic-trace-output",
        type=Path,
        default=None,
        help=(
            "Output A* heuristic trace JSON path "
            "(default: <run_dir>/training_video_astar_heuristic_trace.json)"
        ),
    )
    parser.add_argument(
        "--astar-no-heuristic-output",
        type=Path,
        default=None,
        help=(
            "Output A* no-heuristic MP4 path "
            "(default: <run_dir>/training_video_astar_no_heuristic.mp4)"
        ),
    )
    parser.add_argument(
        "--astar-no-heuristic-trace-output",
        type=Path,
        default=None,
        help=(
            "Output A* no-heuristic trace JSON path "
            "(default: <run_dir>/training_video_astar_no_heuristic_trace.json)"
        ),
    )
    parser.add_argument(
        "--astar-max-nodes",
        type=int,
        default=DEFAULT_ASTAR_MAX_NODES,
        help=(
            "Maximum unique states A* may generate "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--astar-max-expansions",
        type=int,
        default=DEFAULT_ASTAR_MAX_EXPANSIONS,
        help=(
            "Maximum states A* may expand from the open set "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for the output video (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on replay steps (default: no cap)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    """Load a JSON file into a Python dict with basic validation.

    This helper centralizes JSON loading so trajectory parsing errors can be
    surfaced with clear file context. It is needed because trajectory payloads
    are shared between training and replay tooling, and it differs from inline
    `json.loads` usage by attaching the source path to any raised exceptions.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read JSON from {path}") from exc


def _load_replay_payload(run_dir: Path) -> dict[str, Any]:
    """Load replay input from either the legacy or heuristic-only artifact set.

    This helper preserves backwards compatibility with legacy
    `eval_trajectory.json` runs while allowing the refactored heuristic pipeline
    to replay directly from `task_instance.json` plus `astar_plan.json`. It is
    needed because the renderer must support both artifact generations during the
    transition, and it differs from direct trajectory loading by synthesizing the
    legacy replay shape when only the new files are present.
    """

    trajectory_path = run_dir / TRAJECTORY_FILENAME
    try:
        return _load_json(trajectory_path)
    except (FileNotFoundError, ValueError):
        pass
    task_instance_path = run_dir / TASK_INSTANCE_FILENAME
    astar_plan_path = run_dir / ASTAR_PLAN_FILENAME
    try:
        task_instance = _load_json(task_instance_path)
        astar_plan = _load_json(astar_plan_path)
        reset_payload = task_instance.get("reset_payload", {})
        return {
            "version": 2,
            "env_id": task_instance["env_id"],
            "benchmark_id": task_instance["benchmark_id"],
            "deterministic_rulesets": True,
            "fixed_ruleset_seed": task_instance.get("ruleset_seed"),
            "ruleset_index": None,
            "ruleset_key": None,
            "reset_key": reset_payload.get("reset_key", task_instance.get("reset_key", [0, 0])),
            "episode_index": 0,
            "episode_length": len(astar_plan.get("actions", [])),
            "episode_return": 0.0,
            "actions": astar_plan.get("actions", []),
            "num_eval_episodes": 1,
            "eval_seed": task_instance.get("seed"),
            "env_seed": task_instance.get("seed"),
            "env_text": task_instance.get("ruleset_text", task_instance.get("goal_description")),
            "img_obs": False,
            "search_stats": _load_json(run_dir / "astar_search_stats.json").get("aggregate_stats", {}),
        }
    except (FileNotFoundError, ValueError):
        pass
    raise FileNotFoundError(
        f"Missing replay artifacts in {run_dir}; expected either {TRAJECTORY_FILENAME} "
        f"or {TASK_INSTANCE_FILENAME} + {ASTAR_PLAN_FILENAME}"
    )


def _list_candidate_runs(state_root: Path) -> list[Path]:
    """List replayable GEPA candidate run directories ordered by recency.

    This helper finds candidate run directories that contain both the replay
    trajectory and synthesized dense reward needed for video generation. It is
    needed because the CLI now supports both single-run replay and "latest N"
    batch processing over recent candidates, and it differs from simple globbing
    by filtering out non-candidate directories such as sparse baselines and
    holdout outputs that do not match the per-candidate replay contract.

    Args:
        state_root: GEPA state root containing the `gepa_runs` directory.

    Returns:
        Candidate run directories sorted from most recently modified to oldest.
    """
    candidate_roots = [
        state_root / "gepa_runs",
        state_root / "heuristic_runs",
    ]
    candidates: list[Path] = []
    for runs_root in candidate_roots:
        if not runs_root.exists():
            continue
        for path in runs_root.rglob(TRAJECTORY_FILENAME):
            run_dir = path.parent
            if run_dir.name.startswith("candidate-"):
                candidates.append(run_dir)
        for path in runs_root.rglob(ASTAR_PLAN_FILENAME):
            run_dir = path.parent
            if run_dir.name.startswith("candidate-"):
                candidates.append(run_dir)
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No replayable candidate runs found under {state_root}"
        )
    return candidates


def _select_latest_run(state_root: Path) -> Path:
    """Find the most recent GEPA candidate run directory with replay artifacts.

    This helper provides the legacy single-run default when users omit
    `--run-dir`. It is needed because batch selection is now handled separately,
    and it differs from `_list_candidate_runs` by returning only the newest
    eligible candidate directory.
    """
    return _list_candidate_runs(state_root)[0]


def _select_latest_candidate_runs(state_root: Path, count: int) -> list[Path]:
    """Return the most recent replayable candidate run directories.

    This helper implements the batch-selection behavior for `--latest-candidates`
    by slicing the recency-ordered candidate list to the requested size. It is
    needed because users often want to compare recent GEPA proposals without
    manually typing each run directory, and it differs from `_select_latest_run`
    by returning a validated list while enforcing a strictly positive count.

    Args:
        state_root: GEPA state root containing candidate replay artifacts.
        count: Number of recent candidate directories to select.

    Returns:
        Up to `count` candidate run directories, newest first.
    """
    if count <= 0:
        raise ValueError("--latest-candidates must be > 0")
    return _list_candidate_runs(state_root)[:count]


def _resolve_run_dir(state_root: Path, run_dir: Path | None) -> Path:
    """Resolve the run directory to replay for video generation.

    This helper chooses either the user-specified run directory or the newest
    available candidate under the state root. It is needed to keep the CLI
    concise, and it differs from `_select_latest_run` by handling explicit paths
    and validating required artifacts.
    """
    resolved = (
        run_dir.expanduser().resolve() if run_dir else _select_latest_run(state_root)
    )
    has_legacy = (resolved / TRAJECTORY_FILENAME).exists()
    has_heuristic = (resolved / TASK_INSTANCE_FILENAME).exists() and (
        resolved / ASTAR_PLAN_FILENAME
    ).exists()
    if not has_legacy and not has_heuristic:
        raise FileNotFoundError(
            f"Missing replay artifacts in {resolved}; expected {TRAJECTORY_FILENAME} "
            f"or {TASK_INSTANCE_FILENAME} + {ASTAR_PLAN_FILENAME}"
        )
    return resolved


def _coerce_key(values: Iterable[int], *, name: str) -> Any:
    """Convert stored key material into a replay-ready JAX PRNG key.

    This helper reconstructs PRNG keys saved in trajectory payloads so replay
    uses the exact same ruleset and reset randomness as evaluation-time policy
    inference. It is needed because trajectory JSON stores raw key data as
    integer lists, while modern JAX environments often expect typed keys
    (`key<fry>`) during reset/sample calls. It differs from `jax.random.key`
    by consuming previously captured key bits instead of deriving a new key
    from a seed, and it differs from a plain uint32 array conversion by
    upgrading to a typed key through `jax.random.wrap_key_data` when possible.

    Args:
        values: Sequence containing the two uint32 key words captured in JSON.
        name: Field label used in validation error messages.

    Returns:
        A key object that can be passed directly to environment and benchmark
        APIs (typed key on modern JAX, uint32 fallback otherwise).
    """
    import jax
    import jax.numpy as jnp

    arr = jnp.asarray(list(values), dtype=jnp.uint32)
    if arr.shape != (2,):
        raise ValueError(f"{name} must contain exactly two uint32 values")
    try:
        return jax.random.wrap_key_data(arr)
    except Exception:
        return arr


def _build_env(trajectory: dict) -> tuple[Any, Any, Any]:
    """Construct the environment stack needed for trajectory replay.

    This helper mirrors the ground-truth evaluation wrappers so the replayed
    episode uses the same environment API as training-time evaluation. It is
    needed because video generation must re-step the environment deterministically,
    and it differs from training-time env setup by omitting policy initialization
    and focusing on replay-only wrappers.
    """
    import xminigrid
    from xminigrid.wrappers import GymAutoResetWrapper

    env_id = str(trajectory["env_id"])
    benchmark_id = str(trajectory["benchmark_id"])
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    if trajectory.get("img_obs"):
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)
    benchmark = xminigrid.load_benchmark(benchmark_id)
    return env, env_params, benchmark


def _resolve_ruleset(trajectory: dict, benchmark: Any) -> Any:
    """Resolve the ruleset for trajectory replay.

    This helper reconstructs the ruleset used during the saved evaluation so
    action replay is deterministic. It is needed because XLand benchmarks can
    sample rulesets, and it differs from training-time selection by consuming
    stored ruleset keys or indices instead of sampling fresh randomness.
    """
    import jax

    from llm_desparsifier.rl.eval import DEFAULT_RULESET_INDEX

    if trajectory.get("deterministic_rulesets"):
        fixed_ruleset_seed = trajectory.get("fixed_ruleset_seed")
        if fixed_ruleset_seed is not None:
            return benchmark.sample_ruleset(jax.random.key(int(fixed_ruleset_seed)))
        index = trajectory.get("ruleset_index")
        if index is None:
            index = int(DEFAULT_RULESET_INDEX)
        return benchmark.get_ruleset(int(index))
    ruleset_key = trajectory.get("ruleset_key")
    if ruleset_key is None:
        raise ValueError("Trajectory is missing ruleset_key for stochastic replay")
    return benchmark.sample_ruleset(_coerce_key(ruleset_key, name="ruleset_key"))


def _normalize_frame(frame: Any) -> np.ndarray:
    """Normalize a rendered frame into a 3-channel uint8 array.

    This helper ensures the video encoder always receives an RGB frame even if
    the environment render returns grayscale or single-channel arrays. It is
    needed because imageio expects a consistent shape, and it differs from ad-hoc
    casting by explicitly expanding grayscale frames to RGB.
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.uint8, copy=False)


def _format_overlay_lines(
    *,
    env_summary: str | None,
    rollout_status_lines: Iterable[str] | None = None,
    step_index: int,
    total_steps: int,
    dense_reward: float,
    dense_total: float,
    sparse_reward: float,
    sparse_total: float,
    component_values: dict[str, float],
    component_totals: dict[str, float],
    component_order: Iterable[str],
) -> list[str]:
    """Format the overlay text lines for a single replay frame.

    This helper centralizes overlay formatting so the video text remains
    consistent across frames. It is needed because both per-step and cumulative
    values must be displayed and users often need quick goal context while
    watching a replay, and it differs from inline formatting by enforcing a
    stable component ordering, optional goal-summary inclusion, and consistent
    numeric precision.
    """
    lines: list[str] = []
    if env_summary:
        lines.extend(
            _wrap_overlay_goal_line(
                env_summary,
                max_chars=OVERLAY_GOAL_WRAP_CHARS,
            )
        )
    if rollout_status_lines:
        lines.extend(str(line) for line in rollout_status_lines)
    lines.extend(
        [
            f"step {step_index + 1}/{total_steps}",
            f"dense {dense_reward:+.3f} | total {dense_total:+.3f}",
            f"sparse {sparse_reward:+.3f} | total {sparse_total:+.3f}",
        ]
    )
    for name in component_order:
        value = component_values.get(name, 0.0)
        total = component_totals.get(name, 0.0)
        lines.append(f"{name}: {value:+.3f} | total {total:+.3f}")
    return lines


def _format_heuristic_comparison_verdict(verdict: str) -> str:
    """Convert a structured heuristic-comparison verdict into sidebar text.

    This helper maps the JSON-stable verdict codes stored in
    `heuristic_comparison` onto concise human-readable strings for overlays and
    logs. It is needed because the comparison payload should remain structured
    and machine-friendly while the rendered video should read naturally, and it
    differs from inlining string replacements by keeping the wording consistent
    everywhere the verdict is shown.

    Args:
        verdict: Structured comparison verdict emitted by
            `_build_heuristic_comparison_payload`.

    Returns:
        Short human-readable label suitable for the video sidebar.
    """
    verdict_map = {
        "heuristic_faster": "heuristic faster",
        "heuristic_slower": "heuristic slower",
        "same_search_outcome": "same search outcome",
    }
    return verdict_map.get(verdict, verdict.replace("_", " "))


def _classify_heuristic_comparison_verdict(
    *,
    baseline_solved: bool,
    heuristic_solved: bool,
    baseline_expanded: int,
    heuristic_expanded: int,
) -> str:
    """Classify whether heuristic A* did better, worse, or the same.

    This helper implements the comparison contract shared by the video sidebar,
    trace payload, and batch stdout summary. It is needed because the project
    now wants one explicit verdict derived from solve status first and then
    `expanded_states`, and it differs from `_format_heuristic_comparison_verdict`
    by returning the structured machine-readable code rather than display text.

    Args:
        baseline_solved: Whether the no-heuristic baseline solved the task.
        heuristic_solved: Whether the dense-heuristic A* run solved the task.
        baseline_expanded: Expanded-state count for the no-heuristic baseline.
        heuristic_expanded: Expanded-state count for the dense-heuristic run.

    Returns:
        Structured verdict string: `heuristic_faster`,
        `heuristic_slower`, or `same_search_outcome`.
    """
    if heuristic_solved != baseline_solved:
        return "heuristic_faster" if heuristic_solved else "heuristic_slower"
    if heuristic_expanded < baseline_expanded:
        return "heuristic_faster"
    if heuristic_expanded > baseline_expanded:
        return "heuristic_slower"
    return "same_search_outcome"


def _build_astar_overlay_status_lines(
    search_stats: Mapping[str, Any],
    *,
    heuristic_comparison: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build concise overlay lines summarizing A* search status and comparison.

    This helper converts planner-level `search_stats` into short text rows that
    can be rendered on every video frame. It is needed because users should see
    directly in the MP4 whether A* solved the task or stopped due to budget
    limits, and now also whether the heuristic beat the no-heuristic baseline.
    It differs from trace-only diagnostics by exposing both completion state and
    the shared cross-rollout verdict directly in the visual artifact itself.

    Args:
        search_stats: Planner summary dictionary attached by A* selector factory.
        heuristic_comparison: Optional cross-rollout comparison payload shared by
            the heuristic and no-heuristic A* videos for the same candidate.

    Returns:
        Ordered status lines describing the heuristic comparison verdict, solve
        state, and generated/expanded state counts.
    """
    solved = bool(search_stats.get("solved", False))
    reason = str(search_stats.get("terminated_reason", "unknown"))
    generated = int(search_stats.get("generated_states", 0))
    expanded = int(search_stats.get("expanded_states", 0))
    searched = generated
    status = f"solved ({searched} searched)" if solved else "terminated_early"

    lines: list[str] = []
    if isinstance(heuristic_comparison, Mapping):
        verdict = _format_heuristic_comparison_verdict(
            str(heuristic_comparison.get("comparison_verdict", "same_search_outcome"))
        )
        heuristic_expanded = int(heuristic_comparison.get("heuristic_expanded_states", 0))
        baseline_expanded = int(heuristic_comparison.get("baseline_expanded_states", 0))
        lines.extend(
            [
                f"compare {verdict}",
                f"expanded heur={heuristic_expanded} base={baseline_expanded}",
            ]
        )
    lines.extend(
        [
            f"astar {status} ({reason})",
            f"states gen={generated} exp={expanded}",
        ]
    )
    return lines


def _wrap_overlay_goal_line(env_summary: str, *, max_chars: int) -> list[str]:
    """Wrap the goal summary into readable overlay lines.

    This helper converts the environment goal summary into one or more lines that
    comfortably fit the diagnostics panel. It is needed because even compact
    goal summaries can become too wide for a small default font, which previously
    caused the black diagnostics panel to expand and obscure gameplay. It differs
    from plain `textwrap.wrap` by preserving a stable `goal ` prefix on the first
    line and aligning continuation lines underneath that prefix for easier visual
    scanning while stepping through video frames.

    Args:
        env_summary: Single-line goal summary text to show in the overlay.
        max_chars: Maximum visible characters per wrapped chunk, not counting the
            first-line prefix indentation.

    Returns:
        A non-empty list of formatted overlay lines beginning with `goal `.
    """
    prefix = "goal "
    continuation_prefix = " " * len(prefix)
    chunks = textwrap.wrap(
        env_summary,
        width=max(8, max_chars),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not chunks:
        return [prefix.rstrip()]
    wrapped: list[str] = []
    for idx, chunk in enumerate(chunks):
        line_prefix = prefix if idx == 0 else continuation_prefix
        wrapped.append(f"{line_prefix}{chunk}")
    return wrapped


def _summarize_env_text(env_text: Any) -> str | None:
    """Normalize the full environment text for overlay rendering.

    This helper keeps the complete saved `env_text` while collapsing whitespace
    so it can be wrapped cleanly in the side diagnostics panel. It is needed
    because viewers now have enough horizontal space to see full task context,
    and it differs from the previous summary behavior by intentionally avoiding
    character-limit truncation.

    Args:
        env_text: Raw trajectory environment description; expected to be a string.

    Returns:
        The full normalized environment text, or None when no valid text is
        provided.
    """
    if not isinstance(env_text, str):
        return None
    normalized = " ".join(env_text.split())
    if not normalized:
        return None
    return normalized


def _draw_component_plot(
    draw: ImageDraw.ImageDraw,
    *,
    panel_width: int,
    y_top: int,
    component_series: dict[str, list[float]],
    component_order: Iterable[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    """Render a multi-line instantaneous component-reward chart in the panel.

    This helper draws an XY line chart where the x-axis is replay timestep and
    each line corresponds to one reward component's instantaneous reward at each
    step. It is needed because tabular text alone makes temporal behavior hard
    to spot during fast video playback, and it differs from cumulative totals by
    explicitly visualizing per-step fluctuations and sign changes over time.

    Args:
        draw: Mutable PIL drawing context bound to the diagnostics panel.
        panel_width: Total width in pixels of the diagnostics panel.
        y_top: Top pixel row where the plot area should begin.
        component_series: Mapping from component name to per-step instantaneous
            reward history aligned by timestep index.
        component_order: Stable ordered component names used for rendering.
        font: Font object used for axis labels and component legends.

    Returns:
        The vertical pixel position immediately after the plotted region so
        callers can continue rendering any additional content below the chart.
    """
    names = [name for name in component_order if component_series.get(name)]
    if not names:
        return y_top

    x0 = OVERLAY_PANEL_PADDING
    y0 = y_top
    x1 = panel_width - OVERLAY_PANEL_PADDING
    y1 = y0 + OVERLAY_PLOT_HEIGHT
    if x1 - x0 < 40:
        return y_top

    draw.rectangle([(x0, y0), (x1, y1)], outline=(80, 80, 80), fill=(12, 12, 12))

    all_values = [value for name in names for value in component_series.get(name, [])]
    y_min = min(all_values) if all_values else -1.0
    y_max = max(all_values) if all_values else 1.0
    if abs(y_max - y_min) < 1e-6:
        span_pad = max(abs(y_max), 1.0) * 0.25
        y_min -= span_pad
        y_max += span_pad

    max_len = max(len(component_series.get(name, [])) for name in names)
    if max_len < 1:
        return y1 + OVERLAY_PLOT_MARGIN

    inner_left = x0 + OVERLAY_PLOT_MARGIN
    inner_right = x1 - OVERLAY_PLOT_MARGIN
    inner_top = y0 + OVERLAY_PLOT_MARGIN
    inner_bottom = y1 - OVERLAY_PLOT_MARGIN
    if inner_right <= inner_left or inner_bottom <= inner_top:
        return y1 + OVERLAY_PLOT_MARGIN

    def x_at(index: int) -> float:
        if max_len == 1:
            return float(inner_left)
        return inner_left + (inner_right - inner_left) * (index / (max_len - 1))

    def y_at(value: float) -> float:
        ratio = (value - y_min) / (y_max - y_min)
        return inner_bottom - ratio * (inner_bottom - inner_top)

    if y_min <= 0.0 <= y_max:
        y_zero = y_at(0.0)
        draw.line([(inner_left, y_zero), (inner_right, y_zero)], fill=(110, 110, 110))

    palette = (
        (255, 99, 71),  # tomato
        (80, 180, 255),  # bright blue
        (255, 206, 86),  # yellow
        (99, 255, 132),  # green
        (255, 127, 214),  # pink
        (192, 164, 255),  # lavender
        (255, 165, 64),  # orange
        (118, 255, 245),  # cyan
    )
    for color_idx, name in enumerate(names):
        series = component_series.get(name, [])
        if not series:
            continue
        color = palette[color_idx % len(palette)]
        if len(series) == 1:
            cx = x_at(0)
            cy = y_at(series[0])
            draw.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)], fill=color)
        else:
            points = [(x_at(idx), y_at(value)) for idx, value in enumerate(series)]
            draw.line(points, fill=color, width=OVERLAY_PLOT_LINE_WIDTH)

    y_min_label = f"{y_min:+.2f}"
    y_max_label = f"{y_max:+.2f}"
    x_end_label = f"t={max_len - 1}"
    draw.text((inner_left, inner_top), y_max_label, fill=(200, 200, 200), font=font)
    draw.text(
        (inner_left, inner_bottom - 10),
        y_min_label,
        fill=(200, 200, 200),
        font=font,
    )
    x_end_bbox = draw.textbbox((0, 0), x_end_label, font=font)
    x_end_width = x_end_bbox[2] - x_end_bbox[0]
    draw.text(
        (inner_right - x_end_width, inner_bottom - 10),
        x_end_label,
        fill=(200, 200, 200),
        font=font,
    )

    legend_y = y1 + OVERLAY_PLOT_LABEL_PAD
    for color_idx, name in enumerate(names):
        color = palette[color_idx % len(palette)]
        label = str(name)
        label_bbox = draw.textbbox((0, 0), label, font=font)
        swatch_right = x0 + 10
        swatch_left = x0 + 2
        draw.rectangle(
            [(swatch_left, legend_y + 2), (swatch_right, legend_y + 9)],
            fill=color,
        )
        draw.text((x0 + 14, legend_y), label, fill=(230, 230, 230), font=font)
        legend_y += int(max(10, label_bbox[3] - label_bbox[1]) + OVERLAY_PLOT_LABEL_PAD)
        if legend_y > y1 + OVERLAY_PLOT_HEIGHT:
            break

    return legend_y


def _draw_overlay(
    frame: Any,
    lines: list[str],
    *,
    component_series: dict[str, list[float]] | None = None,
    component_order: Iterable[str] = (),
) -> np.ndarray:
    """Compose an enlarged map frame plus diagnostics panel with a trend chart.

    This helper renders textual replay diagnostics and a per-component
    instantaneous-reward line plot in a dedicated panel to the right of the map
    viewport. It is needed because dense reward debugging depends on both exact
    numeric values and temporal context, and it differs from the prior text-only
    overlay by appending an inline chart that shows each component's reward
    fluctuations over timestep index while preserving full map visibility.
    """
    base = Image.fromarray(_normalize_frame(frame))
    if DEFAULT_VIEWPORT_SCALE > 1:
        base = base.resize(
            (base.width * DEFAULT_VIEWPORT_SCALE, base.height * DEFAULT_VIEWPORT_SCALE),
            Image.Resampling.NEAREST,
        )

    panel = Image.new("RGB", (1, 1), color=(0, 0, 0))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    padding = OVERLAY_PANEL_PADDING
    spacing = OVERLAY_LINE_SPACING
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    max_width = max(widths) if widths else 0
    line_height = max(heights) if heights else 0
    panel_width = int(max(OVERLAY_PANEL_MIN_WIDTH, max_width + padding * 2))
    text_block_height = (line_height * len(lines)) + spacing * (len(lines) - 1)
    panel_height = int(text_block_height + padding * 2)
    component_series = component_series or {}
    names_with_history = [
        name for name in component_order if component_series.get(name) is not None
    ]
    if names_with_history:
        legend_rows = len(names_with_history)
        reserved_plot_height = (
            OVERLAY_PLOT_MARGIN
            + OVERLAY_PLOT_HEIGHT
            + OVERLAY_PLOT_LABEL_PAD
            + legend_rows * (10 + OVERLAY_PLOT_LABEL_PAD)
            + OVERLAY_PANEL_PADDING
        )
        panel_height = int(
            max(panel_height, padding + text_block_height + reserved_plot_height)
        )

    panel = Image.new("RGB", (panel_width, panel_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(panel)
    y = int(padding)
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += int(line_height + spacing)

    if names_with_history:
        plot_start_y = int(y + OVERLAY_PLOT_MARGIN)
        _draw_component_plot(
            draw,
            panel_width=panel_width,
            y_top=plot_start_y,
            component_series=component_series,
            component_order=component_order,
            font=font,
        )

    canvas_width = base.width + OVERLAY_MAP_PANEL_GAP + panel.width
    canvas_height = max(base.height, panel.height)
    composed = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    composed.paste(base, (0, 0))
    composed.paste(panel, (base.width + OVERLAY_MAP_PANEL_GAP, 0))
    return np.asarray(composed)


def _load_dense_reward(run_dir: Path) -> Any:
    """Load and compile the dense reward function from a run directory.

    This helper reads the synthesized reward code and compiles it with the
    sanitizer so replay uses the exact same dense reward logic. It is needed
    because the reward function is not serialized as bytecode, and it differs
    from training-time reward generation by skipping LLM synthesis and relying
    on the cached code artifact.
    """
    reward_path = run_dir / DENSE_REWARD_FILENAME
    if not reward_path.exists():
        return None
    from llm_desparsifier.rewards.sanitizer import sanitize_and_compile

    code = reward_path.read_text(encoding="utf-8")
    return sanitize_and_compile(code)


def _wrap_env_with_dense_reward(
    env: Any, trajectory: dict, dense_reward_fn: Any
) -> Any:
    """Apply the dense-reward wrapper stack used during replay.

    This helper encapsulates wrapper construction so replay logic and tests have
    a single boundary for environment instrumentation. It is needed because
    wrapper setup depends on additional runtime imports and optional XLand
    context extraction, and it differs from inlined wrapper calls by enabling
    targeted monkeypatching of wrapper behavior in unit tests.

    Args:
        env: Base replay environment instance.
        trajectory: Replay payload used to infer environment-specific context.
        dense_reward_fn: Compiled dense reward callable.

    Returns:
        The wrapped replay environment with dense diagnostics enabled.
    """
    if dense_reward_fn is None:
        return env
    from llm_desparsifier.rl.wrappers import DesparsifyRewardWrapper
    from llm_desparsifier.utils import extract_xland_ctx

    ctx_fn = extract_xland_ctx if "XLand" in str(trajectory.get("env_id", "")) else None
    return DesparsifyRewardWrapper(env, dense_fn=dense_reward_fn, ctx_fn=ctx_fn)


def _build_replay_step_fns(env: Any) -> tuple[Any, Any]:
    """Build reset/step callables for efficient replay execution.

    This helper prepares environment transition functions used by the replay
    loop. It is needed because per-step Python dispatch can be slow for long
    trajectories, and JIT-compiling `reset`/`step` substantially reduces replay
    wall time on real XLand runs. It differs from direct `jax.jit` calls by
    providing a robust fallback to raw methods when a custom test stub or a
    non-JAX-compatible wrapper cannot be jitted.

    Args:
        env: Replay environment object exposing `reset` and `step` methods.

    Returns:
        A tuple `(reset_fn, step_fn)` where each callable has the same signature
        as the corresponding environment method.
    """
    import jax

    try:
        jitted_reset = jax.jit(env.reset)
        jitted_step = jax.jit(env.step)
    except Exception:
        return env.reset, env.step

    use_jit = {"enabled": True}

    def reset_fn(env_params: Any, reset_key: Any) -> Any:
        """Run reset with JIT when compatible, else transparently fallback."""

        if use_jit["enabled"]:
            try:
                return jitted_reset(env_params, reset_key)
            except Exception:
                use_jit["enabled"] = False
        return env.reset(env_params, reset_key)

    def step_fn(env_params: Any, timestep: Any, action: Any) -> Any:
        """Run step with JIT when compatible, else transparently fallback."""

        if use_jit["enabled"]:
            try:
                return jitted_step(env_params, timestep, action)
            except Exception:
                use_jit["enabled"] = False
        return env.step(env_params, timestep, action)

    return reset_fn, step_fn


def _normalize_component_map(
    reward_components: Any,
    component_order: Iterable[str],
) -> dict[str, float]:
    """Normalize reward component mappings into plain floats.

    This helper converts the component dict emitted by the wrapper into a
    JSON-friendly mapping of floats. It is needed because JAX arrays and frozen
    dicts cannot be serialized directly, and it differs from naive casting by
    supplying zeros for missing components to preserve ordering.
    """
    import jax.numpy as jnp

    mapping = reward_components or {}
    normalized: dict[str, float] = {}
    for name in component_order:
        value = mapping.get(name, 0.0)
        normalized[name] = float(jnp.asarray(value))
    return normalized


@dataclass
class _RolloutAccumulators:
    """Mutable reward diagnostics accumulated during one rollout render pass.

    This state object tracks dense/sparse running totals, per-component running
    totals, and per-component instantaneous time series for charting. It is
    needed because both trajectory replay and A* rollouts must emit identical
    diagnostics layouts and trace fields, and it differs from local variables by
    packaging related mutable values so helper functions can update them in a
    testable and mode-agnostic way.
    """

    dense_total: float
    sparse_total: float
    component_order: tuple[str, ...]
    component_totals: dict[str, float]
    component_series: dict[str, list[float]]


@dataclass
class _RolloutContext:
    """Shared replay context used by each rollout mode execution.

    This context bundles immutable setup outputs needed by the render loop:
    environment callables, normalized metadata, diagnostics configuration, and
    the dense reward function. It is needed because each rollout mode should run
    against a fresh reconstructed environment while reusing the same setup
    procedure, and it differs from ad-hoc tuple passing by naming each required
    field explicitly for readability and maintenance.
    """

    env: Any
    env_params: Any
    reset_fn: Any
    step_fn: Any
    dense_reward_fn: Any
    reset_key: Any
    env_text: Any
    env_seed: Any
    env_summary: str | None
    reward_object_key_diagnostics: dict[str, Any]
    trace_steps_cap: int


@dataclass
class _ActionSelectorBundle:
    """Container for rollout action-selection callable plus trace metadata.

    This wrapper keeps rollout execution generic while allowing planning-based
    selectors to surface high-level diagnostics in the trace root. It is needed
    because replay selectors only provide per-step actions, whereas A* selectors
    also produce search summary metrics that should be attached once per rollout.
    It differs from returning bare callables by explicitly carrying optional
    trace metadata that `_run_rollout_video` can serialize without mode-specific
    branching.
    """

    selector: Callable[[int, Any, Any, Any], tuple[int, dict[str, Any] | None]]
    trace_metadata: dict[str, Any] | None = None


@dataclass
class _AStarPlanResult:
    """Video-local wrapper around the shared A* plan result.

    This wrapper preserves the rollout-script-specific per-step selection
    annotations while delegating the actual search to the shared planner. It is
    needed because the video trace schema records one annotation per replayed
    action, and it differs from the shared planner result by adding only these
    render-time annotations.
    """

    actions: list[int]
    per_step_selection: list[dict[str, Any]]
    search_stats: dict[str, Any]


@dataclass
class _RolloutRunResult:
    """Outcome of one rollout execution, including emitted trace payload.

    This wrapper makes post-processing (like cross-rollout heuristic comparison)
    straightforward by returning both the optional runtime error and the exact
    trace payload written to disk. It is needed because `main()` now computes
    comparison metrics across two separate A* traces, and it differs from the
    previous string-only return value by preserving structured rollout metadata.
    """

    replay_error: str | None
    trace_output: Path
    trace_payload: dict[str, Any]


def _initialize_rollout_accumulators(dense_reward_fn: Any) -> _RolloutAccumulators:
    """Create zero-initialized reward bookkeeping structures for one rollout.

    This helper initializes per-rollout totals and component histories based on
    dense reward metadata so both rollout modes start from identical diagnostics
    state. It is needed because dense reward components may be declared up front
    or discovered dynamically from wrapper extras, and it differs from inline
    dict initialization by centralizing this contract in one reusable place.

    Args:
        dense_reward_fn: Compiled dense reward function, optionally declaring
            `__reward_component_keys__`.

    Returns:
        Initialized `_RolloutAccumulators` with zero totals and empty histories.
    """
    component_order = tuple(getattr(dense_reward_fn, "__reward_component_keys__", ()))
    return _RolloutAccumulators(
        dense_total=0.0,
        sparse_total=0.0,
        component_order=component_order,
        component_totals={name: 0.0 for name in component_order},
        component_series={name: [] for name in component_order},
    )


def _extract_step_reward_details(timestep: Any) -> tuple[float, float, dict[str, Any]]:
    """Extract dense/sparse rewards and component diagnostics from a timestep.

    This helper normalizes reward fields emitted by wrapped environments into a
    uniform tuple consumed by trace and overlay formatting. It is needed because
    some timesteps omit extras while others include sparse baseline and component
    decompositions, and it differs from direct attribute access by providing one
    defensive normalization path shared across rollout modes.

    Args:
        timestep: Environment timestep output after applying an action.

    Returns:
        Tuple of `(dense_reward, sparse_reward, reward_components_raw)` where
        sparse defaults to dense and components default to an empty mapping.
    """
    return extract_step_reward_details(timestep)


def _update_rollout_accumulators(
    *,
    acc: _RolloutAccumulators,
    reward_components: Mapping[str, Any],
    dense_reward_value: float,
    sparse_reward_value: float,
    step_idx: int,
) -> dict[str, float]:
    """Update cumulative totals and component histories for one replay step.

    This helper mutates rollout diagnostics state after each environment step
    and returns normalized component values for the current timestep. It is
    needed because replay can discover reward-component keys lazily in timestep
    extras, and it differs from simple accumulation by backfilling zero-history
    entries for newly discovered components to keep chart series aligned.

    Args:
        acc: Mutable rollout accumulators to update in-place.
        reward_components: Raw per-step component rewards from environment extras.
        dense_reward_value: Dense reward observed at the current timestep.
        sparse_reward_value: Sparse baseline reward observed at current timestep.
        step_idx: Zero-based replay step index.

    Returns:
        Mapping of normalized float component values for the current step.
    """
    acc.dense_total += dense_reward_value
    acc.sparse_total += sparse_reward_value

    if not acc.component_order and reward_components:
        acc.component_order = tuple(sorted(reward_components.keys()))
        acc.component_totals = {name: 0.0 for name in acc.component_order}
        acc.component_series = {
            name: [0.0] * step_idx for name in acc.component_order
        }

    component_values = _normalize_component_map(reward_components, acc.component_order)
    for name, value in component_values.items():
        acc.component_totals[name] = acc.component_totals.get(name, 0.0) + value
        acc.component_series.setdefault(name, []).append(value)
    return component_values


def _trajectory_action_selector(
    *,
    actions: list[Any],
) -> _ActionSelectorBundle:
    """Build an action selector that replays recorded trajectory actions.

    This factory returns a selector callable compatible with the shared rollout
    loop so trajectory replay and A* rollouts can use a common execution path.
    It is needed because the render loop should not hardcode mode-specific
    action logic, and it differs from planner selectors by reading fixed action
    values from the captured trajectory rather than scoring alternatives.

    Args:
        actions: Serialized action sequence from `eval_trajectory.json`.

    Returns:
        Callable that maps `(step_idx, timestep, step_fn, env_params)` to the
        selected action plus optional selection diagnostics (always `None` here).
    """

    def selector(
        step_idx: int,
        _timestep: Any,
        _step_fn: Any,
        _env_params: Any,
    ) -> tuple[int, dict[str, Any] | None]:
        """Return the recorded action for this step index."""
        return int(actions[step_idx]), None

    return _ActionSelectorBundle(selector=selector, trace_metadata=None)


def _plan_with_astar(
    *,
    env: Any,
    env_params: Any,
    step_fn: Any,
    root_timestep: Any,
    use_dense_heuristic: bool,
    max_nodes: int,
    max_expansions: int,
) -> _AStarPlanResult:
    """Plan an action sequence with deterministic A* and local trace metadata.

    This wrapper uses the shared planner for the no-heuristic baseline but runs
    a video-local `h_only` planner when dense guidance is enabled so the search
    priority reflects state-only dense desirability rather than path-length-plus
    proxy cost. It is needed because the user wants to inspect `h_only`
    behavior specifically in generate-video diagnostics, and it differs from
    `llm_desparsifier.search.plan_with_astar` by swapping in that local
    heuristic-only formulation while still returning the same trace shape.
    """
    shared_plan = (
        _plan_with_h_only_astar(
            env=env,
            env_params=env_params,
            step_fn=step_fn,
            root_timestep=root_timestep,
            max_nodes=max_nodes,
            max_expansions=max_expansions,
        )
        if use_dense_heuristic
        else plan_with_astar(
            env=env,
            env_params=env_params,
            step_fn=step_fn,
            root_timestep=root_timestep,
            use_dense_heuristic=False,
            max_nodes=max_nodes,
            max_expansions=max_expansions,
        )
    )
    per_step_selection = [
        {
            "policy": "astar_plan",
            "mode": (
                ROLLOUT_MODE_ASTAR_HEURISTIC
                if use_dense_heuristic
                else ROLLOUT_MODE_ASTAR_NO_HEURISTIC
            ),
            "planned_step_index": idx,
            "selected_action": int(action_value),
            "tie_break": ASTAR_TIE_BREAK,
            "source": "planned_path",
        }
        for idx, action_value in enumerate(shared_plan.actions)
    ]
    return _AStarPlanResult(
        actions=list(shared_plan.actions),
        per_step_selection=per_step_selection,
        search_stats=dict(shared_plan.search_stats),
    )


def _reconstruct_h_only_path(
    nodes: dict[bytes, _HeuristicOnlyNode], leaf_key: bytes
) -> list[int]:
    """Reconstruct the chosen action sequence for the video-local `h_only` run.

    This helper walks parent links emitted by `_plan_with_h_only_astar` so the
    video overlay and trace can reuse the exact same planned action list format
    used everywhere else in the script. It is needed because the local
    heuristic-only planner stores ancestry incrementally instead of copying
    prefixes at each node, and it differs from the shared search helper by
    operating on the video-local node type.
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


def _plan_with_h_only_astar(
    *,
    env: Any,
    env_params: Any,
    step_fn: Any,
    root_timestep: Any,
    max_nodes: int,
    max_expansions: int,
) -> SharedAStarPlanResult:
    """Plan with dense reward used only as state heuristic `h(state)`.

    This helper implements the generate-video-specific planner variant requested
    by the user: zero path cost and `h(state) = -max_a dense_reward(state, a)`.
    It is needed because the video diagnostics should now visualize that
    heuristic-only formulation directly, and it differs from the shared planner
    by removing path-length accumulation entirely from the heuristic-enabled
    path while leaving sparse success checks and replay determinism unchanged.
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
    root_h = -float(
        estimate_dense_qmax(
            env=env,
            timestep=root_timestep,
            step_fn=step_fn,
            env_params=env_params,
        )
    )
    nodes: dict[bytes, _HeuristicOnlyNode] = {
        root_key: _HeuristicOnlyNode(
            key=root_key,
            timestep=root_timestep,
            parent_key=None,
            parent_action=None,
            h_cost=root_h,
            f_cost=root_h,
        )
    }
    seen_keys: set[bytes] = {root_key}
    heuristic_cache: dict[bytes, float] = {root_key: root_h}
    open_heap: list[tuple[float, int, bytes]] = [(root_h, 0, root_key)]
    tie_counter = 1
    expanded_states = 0
    termination_reason = "open_set_exhausted"
    solved_key: bytes | None = None
    best_fallback_key = root_key

    while open_heap:
        current_f, _, current_key = heapq.heappop(open_heap)
        current_node = nodes[current_key]
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
            if next_key in seen_keys:
                continue
            if len(nodes) >= max_nodes:
                termination_reason = "max_nodes_reached"
                break

            h_value = heuristic_cache.get(next_key)
            if h_value is None:
                h_value = -float(
                    estimate_dense_qmax(
                        env=env,
                        timestep=next_ts,
                        step_fn=step_fn,
                        env_params=env_params,
                    )
                )
                heuristic_cache[next_key] = h_value
            nodes[next_key] = _HeuristicOnlyNode(
                key=next_key,
                timestep=next_ts,
                parent_key=current_key,
                parent_action=action_value,
                h_cost=h_value,
                f_cost=h_value,
            )
            seen_keys.add(next_key)
            heapq.heappush(open_heap, (h_value, tie_counter, next_key))
            tie_counter += 1
            if is_sparse_success(next_ts):
                solved_key = next_key
                termination_reason = "solved"
                break

        if solved_key is not None or termination_reason == "max_nodes_reached":
            break

    final_key = solved_key if solved_key is not None else best_fallback_key
    planned_actions = _reconstruct_h_only_path(nodes, final_key)
    final_node = nodes[final_key]
    final_dense, final_sparse, _ = extract_step_reward_details(final_node.timestep)
    return SharedAStarPlanResult(
        actions=planned_actions,
        search_stats={
            "planner": "python_astar_h_only_proxy",
            "solved": solved_key is not None,
            "terminated_reason": termination_reason,
            "generated_states": int(len(nodes)),
            "expanded_states": int(expanded_states),
            "max_nodes": int(max_nodes),
            "max_expansions": int(max_expansions),
            "solution_length": int(len(planned_actions)),
            "solution_cost": 0.0,
            "final_dense_reward": float(final_dense),
            "final_sparse_reward": float(final_sparse),
            "use_dense_heuristic": True,
            "heuristic_mode": "h_only",
        },
    )


def _planned_action_selector(
    *,
    actions: list[int],
    per_step_selection: list[dict[str, Any]],
    mode_name: str,
) -> Callable[[int, Any, Any, Any], tuple[int, dict[str, Any] | None]]:
    """Create a selector that replays planned actions with step annotations.

    This helper adapts precomputed planner output to the shared rollout-loop
    selector protocol. It is needed because replay rendering expects a pull-style
    action callback, and it differs from trajectory selectors by attaching
    planner-derived metadata to each step for trace inspection.
    """

    def selector(
        step_idx: int,
        _timestep: Any,
        _step_fn: Any,
        _env_params: Any,
    ) -> tuple[int, dict[str, Any] | None]:
        if step_idx >= len(actions):
            raise IndexError(
                f"{mode_name} requested step {step_idx} beyond planned horizon {len(actions)}"
            )
        payload = per_step_selection[step_idx] if step_idx < len(per_step_selection) else None
        return int(actions[step_idx]), payload

    return selector


def _build_astar_action_selector_bundle(
    *,
    context: _RolloutContext,
    use_dense_heuristic: bool,
    max_nodes: int,
    max_expansions: int,
    heuristic_comparison: Mapping[str, Any] | None = None,
) -> _ActionSelectorBundle:
    """Plan one A* rollout immediately and package replay-ready metadata.

    This helper performs the deterministic A* search up front and converts the
    result into the shared selector-bundle shape consumed by `_run_rollout_video`.
    It is needed because the video script must now compare heuristic and
    no-heuristic search outcomes before rendering either A* video so the shared
    verdict can appear in both sidebars. It differs from
    `_astar_action_selector_factory` by executing planning immediately instead
    of deferring it until the render loop requests a selector.

    Args:
        context: Prepared rollout context describing the deterministic task
            instance to plan against.
        use_dense_heuristic: Whether the dense reward should define the A*
            heuristic term.
        max_nodes: Maximum number of unique states A* may generate.
        max_expansions: Maximum number of states A* may expand from the open set.
        heuristic_comparison: Optional shared comparison payload that should be
            embedded into the resulting trace metadata and sidebar text.

    Returns:
        `_ActionSelectorBundle` containing a deterministic replay selector plus
        planner search stats and optional cross-rollout comparison metadata.
    """
    initial_timestep = context.reset_fn(context.env_params, context.reset_key)
    plan = _plan_with_astar(
        env=context.env,
        env_params=context.env_params,
        step_fn=context.step_fn,
        root_timestep=initial_timestep,
        use_dense_heuristic=use_dense_heuristic,
        max_nodes=max_nodes,
        max_expansions=max_expansions,
    )
    mode_name = (
        ROLLOUT_MODE_ASTAR_HEURISTIC
        if use_dense_heuristic
        else ROLLOUT_MODE_ASTAR_NO_HEURISTIC
    )
    selector = _planned_action_selector(
        actions=plan.actions,
        per_step_selection=plan.per_step_selection,
        mode_name=mode_name,
    )
    trace_metadata: dict[str, Any] = {"search_stats": dict(plan.search_stats)}
    if heuristic_comparison is not None:
        trace_metadata["heuristic_comparison"] = dict(heuristic_comparison)
    return _ActionSelectorBundle(
        selector=selector,
        trace_metadata=trace_metadata,
    )


def _astar_action_selector_factory(
    *,
    use_dense_heuristic: bool,
    max_nodes: int,
    max_expansions: int,
) -> Callable[[_RolloutContext], _ActionSelectorBundle]:
    """Build an action-selector factory that plans once with deterministic A*.

    This factory defers planning until rollout context is built so search uses
    the exact deterministic reset/ruleset used by rendering. It is needed
    because A* must plan against the wrapped dense-reward environment and then
    feed actions back into the same loop, and it differs from per-step selectors
    logic by solving a full search problem up front and exposing search-level
    diagnostics in trace metadata. It remains useful for single-plan execution,
    while `_build_astar_action_selector_bundle` is used when callers need both
    plans before rendering to derive a shared comparison verdict.
    """

    def builder(context: _RolloutContext) -> _ActionSelectorBundle:
        return _build_astar_action_selector_bundle(
            context=context,
            use_dense_heuristic=use_dense_heuristic,
            max_nodes=max_nodes,
            max_expansions=max_expansions,
        )

    return builder


def _preplanned_action_selector_factory(
    bundle: _ActionSelectorBundle,
) -> Callable[[_RolloutContext], _ActionSelectorBundle]:
    """Return a selector factory that reuses one precomputed selector bundle.

    This helper adapts an already planned selector bundle back into the
    factory-shaped interface expected by `_run_rollout_video`. It is needed
    because A* videos now precompute both plans before rendering so the shared
    heuristic comparison can be shown in the sidebar, and it differs from the
    normal planner factory by ignoring the render-time context and reusing the
    exact preplanned actions and trace metadata.

    Args:
        bundle: Precomputed selector bundle to reuse during rendering.

    Returns:
        Factory callable that always returns `bundle`.
    """

    def builder(_context: _RolloutContext) -> _ActionSelectorBundle:
        """Return the precomputed selector bundle unchanged."""
        return bundle

    return builder


def _build_rollout_context(run_dir: Path, trajectory: dict[str, Any]) -> _RolloutContext:
    """Construct all environment and metadata dependencies for one rollout mode.

    This helper performs deterministic replay setup, dense reward compilation,
    and metadata normalization so each rollout mode can run independently on a
    fresh environment instance. It is needed because running both rollout modes
    requires isolated state while preserving identical task initialization, and
    it differs from the previous monolithic `main()` setup by returning a typed
    context object used by shared rollout execution code.

    Args:
        run_dir: Candidate run directory containing trajectory and reward files.
        trajectory: Parsed trajectory payload loaded from `eval_trajectory.json`.

    Returns:
        `_RolloutContext` containing wrapped env, replay callables, and metadata.
    """
    env_text = trajectory.get("env_text")
    env_seed = trajectory.get("env_seed", trajectory.get("eval_seed"))
    env_summary = _summarize_env_text(env_text)
    reward_object_key_diagnostics = _build_replay_reward_key_diagnostics(
        run_dir, trajectory
    )
    dense_reward_fn = _load_dense_reward(run_dir)
    env, env_params, benchmark = _build_env(trajectory)
    ruleset = _resolve_ruleset(trajectory, benchmark)
    env_params = env_params.replace(ruleset=ruleset)
    env = _wrap_env_with_dense_reward(env, trajectory, dense_reward_fn)
    reset_fn, step_fn = _build_replay_step_fns(env)
    reset_key = _coerce_key(trajectory["reset_key"], name="reset_key")

    actions = trajectory.get("actions", [])
    trace_steps_cap = len(actions)
    return _RolloutContext(
        env=env,
        env_params=env_params,
        reset_fn=reset_fn,
        step_fn=step_fn,
        dense_reward_fn=dense_reward_fn,
        reset_key=reset_key,
        env_text=env_text,
        env_seed=env_seed,
        env_summary=env_summary,
        reward_object_key_diagnostics=reward_object_key_diagnostics,
        trace_steps_cap=trace_steps_cap,
    )


def _write_trace_payload(trace_output: Path, payload: dict[str, Any]) -> None:
    """Write replay diagnostics to JSON in a single durable location.

    This helper centralizes trace serialization so both success and failure
    paths emit the same structured payload. It is needed because replay can fail
    after expensive setup (for example writer subprocess or environment stepping)
    and users still need the partial step log plus error details for debugging.
    It differs from inline `write_text` calls by enforcing a single output
    format and allowing callers to write traces before re-raising exceptions.

    Args:
        trace_output: Destination path for the trace JSON artifact.
        payload: Fully assembled trace payload containing trajectory metadata,
            per-step diagnostics, and optional replay error details.
    """
    trace_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _build_replay_reward_key_diagnostics(run_dir: Path, trajectory: dict) -> dict[str, Any]:
    """Compare reward object lookups against the captured replay task text.

    This helper runs the same reward/object-key alignment analysis used by the
    GEPA reflection pipeline, but anchors it to the replay trajectory's saved
    `env_text`. It is needed because video debugging often happens long after a
    run finishes, and seeing `reward_components` stay at zero is hard to
    interpret without an explicit explanation of whether the reward is querying
    objects that do not exist in that replayed task. It differs from the
    training-time diagnostics path by reading the cached reward source directly
    from the run directory and by returning a JSON-ready payload for inclusion
    in `training_video_trace.json`.

    Args:
        run_dir: Candidate run directory containing the synthesized reward code.
        trajectory: Parsed `eval_trajectory.json` payload for the replayed run.

    Returns:
        A JSON-serializable diagnostics dictionary containing referenced object
        keys, task-described object keys, and any missing keys. On failure, the
        payload contains `diagnostics_error` so replay can proceed.
    """
    if not (run_dir / DENSE_REWARD_FILENAME).exists():
        return {
            "referenced_object_keys": [],
            "task_object_keys": [],
            "missing_from_task": [],
        }
    from llm_desparsifier.rewards import build_reward_object_key_diagnostics

    try:
        reward_code = (run_dir / DENSE_REWARD_FILENAME).read_text(encoding="utf-8")
        diagnostics = build_reward_object_key_diagnostics(
            reward_code=reward_code,
            env_description=trajectory.get("env_text"),
        )
        return {
            "referenced_object_keys": list(diagnostics.referenced_object_keys),
            "task_object_keys": list(diagnostics.task_object_keys),
            "missing_from_task": list(diagnostics.missing_from_task),
        }
    except Exception as exc:
        return {
            "referenced_object_keys": [],
            "task_object_keys": [],
            "missing_from_task": [],
            "diagnostics_error": f"{exc.__class__.__name__}: {exc}",
        }


def _build_rollout_trace_payload(
    *,
    rollout_mode: str,
    trajectory: dict[str, Any],
    context: _RolloutContext,
    run_dir: Path,
    output_path: Path,
    trace_steps: list[dict[str, Any]],
    replay_error: str | None,
    rollout_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the trace payload schema shared by replay and A* rollouts.

    This helper creates a stable JSON payload structure for all rollout modes so
    downstream analysis tools can consume traces uniformly while still seeing
    mode-specific metadata. It is needed because this script now emits multiple
    rollout traces per run, and it differs from inline dict assembly by
    centralizing required fields and explicit mode labeling.

    Args:
        rollout_mode: Rollout strategy identifier for this payload.
        trajectory: Original parsed trajectory payload.
        context: Rollout setup context with normalized replay metadata.
        run_dir: Candidate run directory used for artifact resolution.
        output_path: Video path corresponding to this trace.
        trace_steps: Per-step diagnostics captured during rollout execution.
        replay_error: Optional serialized exception string from rollout failure.
        rollout_metadata: Optional mode-specific top-level metadata.

    Returns:
        JSON-serializable trace payload dictionary.
    """
    payload = {
        "rollout_mode": rollout_mode,
        "trajectory": trajectory,
        "env_seed": context.env_seed,
        "env_text": context.env_text,
        "env_summary": context.env_summary,
        "run_dir": str(run_dir),
        "dense_reward_path": str(run_dir / DENSE_REWARD_FILENAME),
        "video_path": str(output_path),
        "steps": trace_steps,
        "reward_object_key_diagnostics": context.reward_object_key_diagnostics,
        "replay_error": replay_error,
        "replay_complete": replay_error is None,
    }
    if rollout_metadata:
        payload.update(rollout_metadata)
    return payload


def _run_rollout_video(
    *,
    run_dir: Path,
    trajectory: dict[str, Any],
    rollout_mode: str,
    output_path: Path,
    trace_output: Path,
    fps: int,
    max_steps: int | None,
    action_selector_factory: Callable[
        [_RolloutContext],
        _ActionSelectorBundle,
    ],
) -> _RolloutRunResult:
    """Execute one rollout mode and emit its video and trace artifacts.

    This helper runs full replay rendering for one rollout strategy while
    handling trace writes on both success and failure. It is needed because the
    script now supports multiple rollout modes with identical rendering logic,
    including preplanned A* bundles that already contain shared comparison
    metadata for sidebar display. It differs from the previous single-mode
    monolith by accepting a pluggable action-selection function and returning an
    optional error for aggregated exit handling in `main()`.

    Args:
        run_dir: Candidate run directory containing replay artifacts.
        trajectory: Parsed trajectory payload.
        rollout_mode: Mode label used in logs and trace metadata.
        output_path: Destination MP4 path for this rollout.
        trace_output: Destination trace JSON path for this rollout.
        fps: Output frame-rate passed to `imageio`.
        max_steps: Optional step cap applied on top of default trace-length cap.
        action_selector_factory: Callable that builds the rollout-mode-specific
            action selector bundle using the prepared context.

    Returns:
        `_RolloutRunResult` containing optional error plus written trace payload.
    """
    import jax.numpy as jnp

    context = _build_rollout_context(run_dir, trajectory)
    selector_bundle = action_selector_factory(context)
    action_selector = selector_bundle.selector
    rollout_status_lines: list[str] | None = None
    planned_length = None
    trace_metadata = selector_bundle.trace_metadata or {}
    search_stats = trace_metadata.get("search_stats")
    heuristic_comparison = trace_metadata.get("heuristic_comparison")
    if isinstance(search_stats, Mapping):
        maybe_length = search_stats.get("solution_length")
        if isinstance(maybe_length, int) and maybe_length >= 0:
            planned_length = maybe_length
        rollout_status_lines = _build_astar_overlay_status_lines(
            search_stats,
            heuristic_comparison=(
                heuristic_comparison if isinstance(heuristic_comparison, Mapping) else None
            ),
        )
    acc = _initialize_rollout_accumulators(context.dense_reward_fn)
    trace_steps: list[dict[str, Any]] = []
    replay_error: str | None = None
    base_step_cap = planned_length if planned_length is not None else context.trace_steps_cap
    step_cap = min(base_step_cap, max_steps) if max_steps is not None else base_step_cap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer_obj:
            writer = cast(Any, writer_obj)
            timestep = context.reset_fn(context.env_params, context.reset_key)

            for step_idx in range(step_cap):
                if bool(timestep.last()):
                    break

                action_value, selection_payload = action_selector(
                    step_idx,
                    timestep,
                    context.step_fn,
                    context.env_params,
                )
                action = jnp.asarray(int(action_value))
                timestep = context.step_fn(context.env_params, timestep, action)

                dense_reward_value, sparse_reward_value, reward_components = (
                    _extract_step_reward_details(timestep)
                )
                component_values = _update_rollout_accumulators(
                    acc=acc,
                    reward_components=reward_components,
                    dense_reward_value=dense_reward_value,
                    sparse_reward_value=sparse_reward_value,
                    step_idx=step_idx,
                )

                lines = _format_overlay_lines(
                    env_summary=context.env_summary,
                    rollout_status_lines=rollout_status_lines,
                    step_index=step_idx,
                    total_steps=step_cap,
                    dense_reward=dense_reward_value,
                    dense_total=acc.dense_total,
                    sparse_reward=sparse_reward_value,
                    sparse_total=acc.sparse_total,
                    component_values=component_values,
                    component_totals=acc.component_totals,
                    component_order=acc.component_order,
                )
                frame = context.env.render(context.env_params, timestep)
                if frame is None:
                    raise RuntimeError("Environment render returned None")
                writer.append_data(
                    _draw_overlay(
                        frame,
                        lines,
                        component_series=acc.component_series,
                        component_order=acc.component_order,
                    )
                )

                row = {
                    "step": step_idx,
                    "action": int(action_value),
                    "dense_reward": dense_reward_value,
                    "sparse_reward": sparse_reward_value,
                    "dense_total": acc.dense_total,
                    "sparse_total": acc.sparse_total,
                    "reward_components": component_values,
                    "reward_component_totals": dict(acc.component_totals),
                }
                if selection_payload is not None:
                    row["selection"] = selection_payload
                trace_steps.append(row)
    except Exception as exc:  # pragma: no cover - covered via failure-path unit tests
        if (
            os.environ.get(CPU_FALLBACK_REEXEC_FLAG) != "1"
            and _is_cuda_backend_init_error(exc)
        ):
            print(
                "[generate_training_video] CUDA backend initialization failed; "
                "retrying with JAX_PLATFORMS=cpu"
            )
            _reexec_with_cpu_fallback()
        replay_error = f"{exc.__class__.__name__}: {exc}"
    finally:
        trace_payload = _build_rollout_trace_payload(
            rollout_mode=rollout_mode,
            trajectory=trajectory,
            context=context,
            run_dir=run_dir,
            output_path=output_path,
            trace_steps=trace_steps,
            replay_error=replay_error,
            rollout_metadata=selector_bundle.trace_metadata,
        )
        _write_trace_payload(trace_output, trace_payload)
        if replay_error is None:
            print(f"[generate_training_video] wrote {output_path}")
        print(f"[generate_training_video] wrote {trace_output}")
    return _RolloutRunResult(
        replay_error=replay_error,
        trace_output=trace_output,
        trace_payload=trace_payload,
    )


def _build_heuristic_comparison_payload(
    *,
    baseline_search_stats: Mapping[str, Any],
    heuristic_search_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute heuristic-versus-baseline search metrics for one environment.

    This helper compares no-heuristic and dense-heuristic A* search statistics
    and derives stable scalar metrics for user inspection. It is needed because
    heuristic impact should be explicit rather than inferred manually from two
    separate traces or videos. It differs from per-run search stats by
    producing direct deltas, a human-readable verdict based on solve status and
    expanded-state count, and shared-baseline comparisons that can be rendered
    inside both A* videos and aggregated across many candidate runs.
    """
    baseline_generated = int(baseline_search_stats.get("generated_states", 0))
    heuristic_generated = int(heuristic_search_stats.get("generated_states", 0))
    reduction_abs = baseline_generated - heuristic_generated
    reduction_pct = (
        (100.0 * reduction_abs / baseline_generated) if baseline_generated > 0 else 0.0
    )
    baseline_expanded = int(baseline_search_stats.get("expanded_states", 0))
    heuristic_expanded = int(heuristic_search_stats.get("expanded_states", 0))
    baseline_cost = baseline_search_stats.get("solution_cost")
    heuristic_cost = heuristic_search_stats.get("solution_cost")
    baseline_len = baseline_search_stats.get("solution_length")
    heuristic_len = heuristic_search_stats.get("solution_length")
    baseline_solved = bool(baseline_search_stats.get("solved", False))
    heuristic_solved = bool(heuristic_search_stats.get("solved", False))
    comparison_verdict = _classify_heuristic_comparison_verdict(
        baseline_solved=baseline_solved,
        heuristic_solved=heuristic_solved,
        baseline_expanded=baseline_expanded,
        heuristic_expanded=heuristic_expanded,
    )

    heuristic_converged_faster = comparison_verdict == "heuristic_faster"
    return {
        "comparison_basis": "expanded_states",
        "comparison_verdict": comparison_verdict,
        "baseline_generated_states": baseline_generated,
        "heuristic_generated_states": heuristic_generated,
        "baseline_expanded_states": baseline_expanded,
        "heuristic_expanded_states": heuristic_expanded,
        "generated_state_reduction_abs": int(reduction_abs),
        "generated_state_reduction_pct": float(reduction_pct),
        "baseline_solved": baseline_solved,
        "heuristic_solved": heuristic_solved,
        "heuristic_converged_faster": heuristic_converged_faster,
        "solution_cost_match": baseline_cost == heuristic_cost,
        "solution_length_match": baseline_len == heuristic_len,
    }


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before any replay work begins.

    This helper centralizes command-line validation so both single-run and
    multi-run modes fail fast with consistent error messages. It is needed
    because batch selection introduces argument combinations that do not make
    sense with single-output overrides, and it differs from ad-hoc checks in
    `main()` by grouping all CLI contract enforcement in one place.

    Args:
        args: Parsed CLI namespace from `parse_args()`.
    """
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.astar_max_nodes <= 0:
        raise ValueError("--astar-max-nodes must be > 0")
    if args.astar_max_expansions <= 0:
        raise ValueError("--astar-max-expansions must be > 0")
    if args.latest_candidates is not None and args.run_dir is not None:
        raise ValueError("--run-dir cannot be combined with --latest-candidates")
    if args.latest_candidates is not None and args.latest_candidates <= 0:
        raise ValueError("--latest-candidates must be > 0")
    if args.latest_candidates is not None:
        output_overrides = (
            args.output,
            args.trace_output,
            args.astar_heuristic_output,
            args.astar_heuristic_trace_output,
            args.astar_no_heuristic_output,
            args.astar_no_heuristic_trace_output,
        )
        if any(path is not None for path in output_overrides):
            raise ValueError(
                "Explicit output-path overrides are only supported for single-run mode"
            )
    if not args.astar:
        astar_output_overrides = (
            args.astar_heuristic_output,
            args.astar_heuristic_trace_output,
            args.astar_no_heuristic_output,
            args.astar_no_heuristic_trace_output,
        )
        if any(path is not None for path in astar_output_overrides):
            raise ValueError("A* output-path overrides require --astar")


def _resolve_target_run_dirs(args: argparse.Namespace, state_root: Path) -> list[Path]:
    """Resolve the list of run directories that the CLI invocation should process.

    This helper maps the user-facing selection options onto concrete candidate
    directories. It is needed because `main()` now orchestrates a loop over one
    or more runs, and it differs from `_resolve_run_dir` by preserving the
    invocation order for batch mode instead of collapsing to a single path.

    Args:
        args: Parsed CLI namespace.
        state_root: Resolved state-root path.

    Returns:
        Non-empty ordered list of run directories to process.
    """
    if args.latest_candidates is not None:
        return _select_latest_candidate_runs(state_root, args.latest_candidates)
    return [_resolve_run_dir(state_root, args.run_dir)]


def _run_single_candidate(
    *,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run replay and optional A* rollouts for one candidate directory.

    This helper contains the per-run orchestration that was previously embedded
    in `main()`. It is needed because batch mode must reuse the exact same
    rollout behavior, trace writing, and error semantics for each candidate,
    including precomputing both A* plans before rendering so their shared
    comparison verdict can appear in both sidebars. It differs from
    `_run_rollout_video` by coordinating all rollout modes plus per-run
    heuristic-comparison planning and aggregation.

    Args:
        run_dir: Candidate directory containing replay artifacts.
        args: Parsed CLI namespace controlling render/planner behavior.

    Returns:
        A summary dictionary containing the processed run path, any rollout
        errors, and optional heuristic comparison metadata for aggregation.
    """
    trajectory = _load_replay_payload(run_dir)

    actions = trajectory.get("actions", [])
    if not actions:
        raise ValueError(f"Replay payload in {run_dir} contains no actions")

    reward_key_diagnostics = _build_replay_reward_key_diagnostics(run_dir, trajectory)
    missing_keys = reward_key_diagnostics.get("missing_from_task") or []
    if isinstance(missing_keys, list) and missing_keys:
        print(
            "[generate_training_video] reward/task object-key mismatch detected: "
            f"run_dir={run_dir} missing_from_task={missing_keys}"
        )

    output_path = args.output or (run_dir / DEFAULT_VIDEO_NAME)
    trace_output = args.trace_output or (run_dir / DEFAULT_TRACE_NAME)
    astar_heuristic_output_path = args.astar_heuristic_output or (
        run_dir / DEFAULT_ASTAR_HEURISTIC_VIDEO_NAME
    )
    astar_heuristic_trace_output = args.astar_heuristic_trace_output or (
        run_dir / DEFAULT_ASTAR_HEURISTIC_TRACE_NAME
    )
    astar_no_heuristic_output_path = args.astar_no_heuristic_output or (
        run_dir / DEFAULT_ASTAR_NO_HEURISTIC_VIDEO_NAME
    )
    astar_no_heuristic_trace_output = args.astar_no_heuristic_trace_output or (
        run_dir / DEFAULT_ASTAR_NO_HEURISTIC_TRACE_NAME
    )

    errors: list[str] = []
    comparison: dict[str, Any] | None = None
    replay_result = _run_rollout_video(
        run_dir=run_dir,
        trajectory=trajectory,
        rollout_mode=ROLLOUT_MODE_REPLAY,
        output_path=output_path,
        trace_output=trace_output,
        fps=args.fps,
        max_steps=args.max_steps,
        action_selector_factory=lambda _ctx: _trajectory_action_selector(actions=list(actions)),
    )
    if replay_result.replay_error is not None:
        errors.append(f"{ROLLOUT_MODE_REPLAY}: {replay_result.replay_error}")

    if replay_result.replay_error is None and args.astar:
        planning_context = _build_rollout_context(run_dir, trajectory)
        astar_baseline_bundle = _build_astar_action_selector_bundle(
            context=planning_context,
            use_dense_heuristic=False,
            max_nodes=args.astar_max_nodes,
            max_expansions=args.astar_max_expansions,
        )
        astar_heuristic_bundle = _build_astar_action_selector_bundle(
            context=planning_context,
            use_dense_heuristic=True,
            max_nodes=args.astar_max_nodes,
            max_expansions=args.astar_max_expansions,
        )
        baseline_metadata = astar_baseline_bundle.trace_metadata or {}
        heuristic_metadata = astar_heuristic_bundle.trace_metadata or {}
        baseline_stats = baseline_metadata.get("search_stats")
        heuristic_stats = heuristic_metadata.get("search_stats")
        if isinstance(baseline_stats, Mapping) and isinstance(heuristic_stats, Mapping):
            comparison = _build_heuristic_comparison_payload(
                baseline_search_stats=baseline_stats,
                heuristic_search_stats=heuristic_stats,
            )
            baseline_metadata["heuristic_comparison"] = dict(comparison)
            heuristic_metadata["heuristic_comparison"] = dict(comparison)
            astar_baseline_bundle.trace_metadata = baseline_metadata
            astar_heuristic_bundle.trace_metadata = heuristic_metadata
            print(
                "[generate_training_video] heuristic search comparison: "
                f"run_dir={run_dir} "
                f"verdict={comparison['comparison_verdict']} "
                f"basis={comparison['comparison_basis']} "
                f"baseline_expanded={comparison['baseline_expanded_states']} "
                f"heuristic_expanded={comparison['heuristic_expanded_states']} "
                f"baseline_generated={comparison['baseline_generated_states']} "
                f"heuristic_generated={comparison['heuristic_generated_states']}"
            )
        astar_baseline_result = _run_rollout_video(
            run_dir=run_dir,
            trajectory=trajectory,
            rollout_mode=ROLLOUT_MODE_ASTAR_NO_HEURISTIC,
            output_path=astar_no_heuristic_output_path,
            trace_output=astar_no_heuristic_trace_output,
            fps=args.fps,
            max_steps=args.max_steps,
            action_selector_factory=_preplanned_action_selector_factory(astar_baseline_bundle),
        )
        if astar_baseline_result.replay_error is not None:
            errors.append(
                f"{ROLLOUT_MODE_ASTAR_NO_HEURISTIC}: {astar_baseline_result.replay_error}"
            )
        astar_heuristic_result = _run_rollout_video(
            run_dir=run_dir,
            trajectory=trajectory,
            rollout_mode=ROLLOUT_MODE_ASTAR_HEURISTIC,
            output_path=astar_heuristic_output_path,
            trace_output=astar_heuristic_trace_output,
            fps=args.fps,
            max_steps=args.max_steps,
            action_selector_factory=_preplanned_action_selector_factory(astar_heuristic_bundle),
        )
        if astar_heuristic_result.replay_error is not None:
            errors.append(
                f"{ROLLOUT_MODE_ASTAR_HEURISTIC}: {astar_heuristic_result.replay_error}"
            )

    return {
        "run_dir": run_dir,
        "errors": errors,
        "heuristic_comparison": comparison,
    }


def _print_batch_summary(run_summaries: list[dict[str, Any]]) -> None:
    """Print aggregate heuristic-search metrics across processed candidates.

    This helper reduces per-run comparison payloads into a small stdout summary
    for batch invocations. It is needed because users requested a direct answer
    to how often the dense heuristic converged faster than baseline over the
    latest candidate set, and now also whether it was slower or tied. It differs
    from per-run trace payloads by aggregating verdict counts across
    environments in the current CLI execution.

    Args:
        run_summaries: Per-run orchestration summaries returned by
            `_run_single_candidate`.
    """
    comparisons = [
        summary["heuristic_comparison"]
        for summary in run_summaries
        if isinstance(summary.get("heuristic_comparison"), Mapping)
    ]
    if not comparisons:
        return
    verdict_counts = {
        "heuristic_faster": 0,
        "heuristic_slower": 0,
        "same_search_outcome": 0,
    }
    for comparison in comparisons:
        verdict = str(comparison.get("comparison_verdict", "same_search_outcome"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    print(
        "[generate_training_video] batch heuristic summary: "
        f"heuristic_faster={verdict_counts['heuristic_faster']} "
        f"heuristic_slower={verdict_counts['heuristic_slower']} "
        f"same_search_outcome={verdict_counts['same_search_outcome']} "
        f"total={len(comparisons)}"
    )


def main() -> None:
    """Generate trajectory replay plus optional A* diagnostic rollouts.

    This entry point resolves run artifacts, validates CLI limits, executes the
    existing policy trajectory replay, and optionally executes two A* rollout
    variants (with and without dense heuristic) when `--astar` is enabled.
    It is needed so users can compare heuristic search efficiency directly, and
    it differs from earlier behavior by emitting three independent rollout
    artifact pairs plus explicit cross-run heuristic comparison metrics.
    """
    _configure_replay_jax_runtime()
    args = parse_args()
    _validate_cli_args(args)

    state_root = args.state_root.expanduser().resolve()
    run_dirs = _resolve_target_run_dirs(args, state_root)
    run_summaries: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for run_dir in run_dirs:
        print(f"[generate_training_video] processing run_dir={run_dir}")
        summary = _run_single_candidate(run_dir=run_dir, args=args)
        run_summaries.append(summary)
        run_errors = [f"{run_dir}: {error}" for error in summary["errors"]]
        all_errors.extend(run_errors)

    if len(run_dirs) > 1:
        _print_batch_summary(run_summaries)

    if all_errors:
        raise RuntimeError("One or more rollouts failed: " + " | ".join(all_errors))


if __name__ == "__main__":
    main()
