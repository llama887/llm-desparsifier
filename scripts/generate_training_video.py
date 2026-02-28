#!/usr/bin/env python3
"""Generate an MP4 training video by replaying a saved evaluation trajectory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_STATE_ROOT = Path("artifacts/gepa_state")
TRAJECTORY_FILENAME = "eval_trajectory.json"
DENSE_REWARD_FILENAME = "dense_reward_synthesized.py"
DEFAULT_VIDEO_NAME = "training_video.mp4"
DEFAULT_TRACE_NAME = "training_video_trace.json"
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


def _select_latest_run(state_root: Path) -> Path:
    """Find the most recent GEPA run directory with a saved trajectory.

    This routine scans the `gepa_runs` directory for candidates containing
    `eval_trajectory.json` so users can omit an explicit run path. It is needed
    to make video generation ergonomic after long GEPA runs, and it differs from
    direct glob usage by enforcing the presence of required artifacts.
    """
    runs_root = state_root / "gepa_runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Missing gepa_runs directory under {state_root}")
    candidates = []
    for path in runs_root.rglob(TRAJECTORY_FILENAME):
        run_dir = path.parent
        if (run_dir / DENSE_REWARD_FILENAME).exists():
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with {TRAJECTORY_FILENAME} found under {runs_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
    trajectory_path = resolved / TRAJECTORY_FILENAME
    reward_path = resolved / DENSE_REWARD_FILENAME
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing {TRAJECTORY_FILENAME} in {resolved}")
    if not reward_path.exists():
        raise FileNotFoundError(f"Missing {DENSE_REWARD_FILENAME} in {resolved}")
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
    from llm_desparsifier.rewards.sanitizer import sanitize_and_compile

    reward_path = run_dir / DENSE_REWARD_FILENAME
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


def main() -> None:
    """Replay a saved trajectory and emit a diagnostic training video.

    This entry point orchestrates trajectory loading, environment reconstruction,
    dense reward replay, overlay formatting, and MP4 writing. It is needed as a
    one-shot CLI so users can render videos from selected runs without touching
    the training pipeline, and it differs from lower-level replay helpers by
    owning end-to-end file IO and video encoding.
    """
    _configure_replay_jax_runtime()
    args = parse_args()
    state_root = args.state_root.expanduser().resolve()
    run_dir = _resolve_run_dir(state_root, args.run_dir)
    trajectory_path = run_dir / TRAJECTORY_FILENAME
    trajectory = _load_json(trajectory_path)
    reward_object_key_diagnostics = _build_replay_reward_key_diagnostics(
        run_dir, trajectory
    )
    env_text = trajectory.get("env_text")
    env_seed = trajectory.get("env_seed", trajectory.get("eval_seed"))
    env_summary = _summarize_env_text(env_text)
    missing_keys = reward_object_key_diagnostics.get("missing_from_task") or []
    if isinstance(missing_keys, list) and missing_keys:
        print(
            "[generate_training_video] reward/task object-key mismatch detected: "
            f"missing_from_task={missing_keys}"
        )

    actions = trajectory.get("actions", [])
    if not actions:
        raise ValueError(f"Trajectory {trajectory_path} contains no actions")

    max_steps = args.max_steps
    total_steps = min(len(actions), max_steps) if max_steps else len(actions)
    output_path = args.output or (run_dir / DEFAULT_VIDEO_NAME)
    trace_output = args.trace_output or (run_dir / DEFAULT_TRACE_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_output.parent.mkdir(parents=True, exist_ok=True)

    dense_total = 0.0
    sparse_total = 0.0
    component_order: tuple[str, ...] = ()
    component_totals: dict[str, float] = {}
    component_series: dict[str, list[float]] = {}
    trace_steps: list[dict[str, Any]] = []

    replay_error: str | None = None
    try:
        # Open the video writer before expensive JAX replay work so ffmpeg's
        # subprocess launch happens early and avoids fork-after-heavy-threading
        # deadlock scenarios on some systems.
        with imageio.get_writer(
            str(output_path), fps=args.fps, codec="libx264", quality=8
        ) as writer_obj:
            import jax.numpy as jnp

            writer = cast(Any, writer_obj)

            dense_reward_fn = _load_dense_reward(run_dir)
            component_order = tuple(
                getattr(dense_reward_fn, "__reward_component_keys__", ())
            )
            component_totals = {name: 0.0 for name in component_order}
            component_series = {name: [] for name in component_order}

            env, env_params, benchmark = _build_env(trajectory)
            ruleset = _resolve_ruleset(trajectory, benchmark)
            env_params = env_params.replace(ruleset=ruleset)
            env = _wrap_env_with_dense_reward(env, trajectory, dense_reward_fn)
            reset_fn, step_fn = _build_replay_step_fns(env)

            reset_key = _coerce_key(trajectory["reset_key"], name="reset_key")
            timestep = reset_fn(env_params, reset_key)

            for step_idx, action_value in enumerate(actions[:total_steps]):
                action = jnp.asarray(int(action_value))
                timestep = step_fn(env_params, timestep, action)

                extras = getattr(timestep, "extras", None)
                dense_reward_value = float(jnp.asarray(timestep.reward))
                sparse_reward = dense_reward_value
                reward_components: dict[str, Any] = {}
                if extras is not None:
                    sparse_reward = float(
                        jnp.asarray(
                            extras.get("ground_truth_reward", dense_reward_value)
                        )
                    )
                    reward_components = extras.get("reward_components") or {}

                dense_total += dense_reward_value
                sparse_total += sparse_reward

                if not component_order and reward_components:
                    component_order = tuple(sorted(reward_components.keys()))
                    component_totals = {name: 0.0 for name in component_order}
                    component_series = {
                        name: [0.0] * step_idx for name in component_order
                    }

                component_values = _normalize_component_map(
                    reward_components, component_order
                )
                for name, value in component_values.items():
                    component_totals[name] = component_totals.get(name, 0.0) + value
                    component_series.setdefault(name, []).append(value)

                lines = _format_overlay_lines(
                    env_summary=env_summary,
                    step_index=step_idx,
                    total_steps=total_steps,
                    dense_reward=dense_reward_value,
                    dense_total=dense_total,
                    sparse_reward=sparse_reward,
                    sparse_total=sparse_total,
                    component_values=component_values,
                    component_totals=component_totals,
                    component_order=component_order,
                )

                frame = env.render(env_params, timestep)
                if frame is None:
                    raise RuntimeError("Environment render returned None")
                writer.append_data(
                    _draw_overlay(
                        frame,
                        lines,
                        component_series=component_series,
                        component_order=component_order,
                    )
                )

                trace_steps.append(
                    {
                        "step": step_idx,
                        "action": int(action_value),
                        "dense_reward": dense_reward_value,
                        "sparse_reward": sparse_reward,
                        "dense_total": dense_total,
                        "sparse_total": sparse_total,
                        "reward_components": component_values,
                        "reward_component_totals": dict(component_totals),
                    }
                )

                if bool(timestep.last()):
                    break
    except Exception as exc:
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
        trace_payload = {
            "trajectory": trajectory,
            "env_seed": env_seed,
            "env_text": env_text,
            "env_summary": env_summary,
            "run_dir": str(run_dir),
            "dense_reward_path": str(run_dir / DENSE_REWARD_FILENAME),
            "video_path": str(output_path),
            "steps": trace_steps,
            "reward_object_key_diagnostics": reward_object_key_diagnostics,
            "replay_error": replay_error,
            "replay_complete": False,
        }
        _write_trace_payload(trace_output, trace_payload)
        print(f"[generate_training_video] wrote {trace_output}")
        raise

    trace_payload = {
        "trajectory": trajectory,
        "env_seed": env_seed,
        "env_text": env_text,
        "env_summary": env_summary,
        "run_dir": str(run_dir),
        "dense_reward_path": str(run_dir / DENSE_REWARD_FILENAME),
        "video_path": str(output_path),
        "steps": trace_steps,
        "reward_object_key_diagnostics": reward_object_key_diagnostics,
        "replay_error": replay_error,
        "replay_complete": True,
    }
    _write_trace_payload(trace_output, trace_payload)
    print(f"[generate_training_video] wrote {output_path}")
    print(f"[generate_training_video] wrote {trace_output}")


if __name__ == "__main__":
    main()
