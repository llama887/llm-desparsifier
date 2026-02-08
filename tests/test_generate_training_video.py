from __future__ import annotations

import json
import importlib.util
from argparse import Namespace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image


def _load_video_module():
    """Load the training video script module directly from its file path.

    This helper imports `scripts/generate_training_video.py` without requiring
    the `scripts` directory to be a Python package. It is needed because pytest
    module discovery runs with a package-oriented import path, and it differs
    from standard `import` usage by building an explicit module spec from the
    script file location.
    """

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_training_video.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_training_video_module", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


video_mod = _load_video_module()


def test_configure_replay_jax_runtime_sets_safe_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure replay runtime applies safe XLA defaults when unset.

    This test validates that the script configures a conservative thread setting
    before JAX import when `XLA_FLAGS` is absent. It is needed because replay
    video generation mixes JAX with subprocess-backed encoders, and it differs
    from integration tests by directly asserting environment-variable behavior
    without invoking external encoders.
    """

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    video_mod._configure_replay_jax_runtime()
    assert "intra_op_parallelism_threads=1" in str(
        video_mod.os.environ.get("XLA_FLAGS")
    )


def test_coerce_key_round_trip_uses_stored_key_data() -> None:
    """Ensure replay key coercion preserves exact captured key bits.

    This test verifies that `_coerce_key` reconstructs a key object whose raw
    key data exactly matches the two uint32 words stored in trajectory JSON. It
    is needed because replay determinism depends on preserving these values
    verbatim across serialization boundaries, and it differs from end-to-end
    replay tests by validating the low-level key conversion primitive in
    isolation for fast, targeted failure diagnosis.
    """

    coerced = video_mod._coerce_key([123, 456], name="reset_key")
    assert jax.random.key_data(coerced).reshape(-1).tolist() == [123, 456]


def test_main_writes_trace_even_when_replay_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify trace JSON is emitted before replay exceptions are re-raised.

    This test forces a replay-time failure in `env.step` and asserts that
    `training_video_trace.json` is still written with partial diagnostic data
    and error metadata. It is needed because users rely on trace artifacts to
    debug why video generation failed, and it differs from successful replay
    tests by exercising the error-handling path that previously dropped traces.
    """

    class DummyEnvParams:
        """Minimal env params stub supporting the `replace` API."""

        def replace(self, **_kwargs: Any) -> "DummyEnvParams":
            """Return self to mimic immutable env params replacement."""

            return self

    class DummyTimestep:
        """Minimal timestep carrying reward and last-step signal methods."""

        def __init__(self) -> None:
            """Initialize with default zero reward and empty extras map."""

            self.reward = jnp.asarray(0.0)
            self.extras = {}

        def last(self) -> jax.Array:
            """Always report non-terminal state for this failure test."""

            return jnp.asarray(False)

    class DummyEnv:
        """Environment stub that fails during stepping after reset succeeds."""

        def reset(self, _env_params: Any, _reset_key: Any) -> DummyTimestep:
            """Return a valid initial timestep so replay enters the loop."""

            return DummyTimestep()

        def step(self, _env_params: Any, _ts: Any, _action: Any) -> DummyTimestep:
            """Raise the injected replay error used to validate trace writes."""

            raise RuntimeError("step failed intentionally")

        def render(self, _env_params: Any, _ts: Any) -> np.ndarray:
            """Return a small RGB frame, though this path is never reached."""

            return np.zeros((8, 8, 3), dtype=np.uint8)

    class DummyWriter:
        """No-op video writer context manager used to avoid ffmpeg subprocesses."""

        def __enter__(self) -> "DummyWriter":
            """Return self to satisfy context manager contract."""

            return self

        def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
            """Do not suppress exceptions from replay execution."""

            return False

        def append_data(self, _frame: Any) -> None:
            """Accept appended frames without side effects."""

            return None

    run_dir = tmp_path / "candidate-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    trajectory = {
        "env_id": "XLand-MiniGrid-R1-9x9",
        "benchmark_id": "trivial-1m",
        "deterministic_rulesets": True,
        "ruleset_index": 42,
        "reset_key": [1, 2],
        "actions": [0, 1],
        "env_text": "Synthetic environment text for failure-path trace test.",
        "eval_seed": 7,
    }

    def _dense_reward(*_args: Any, **_kwargs: Any):
        """Return a valid dense reward tuple for interface completeness."""

        return jnp.asarray(0.0), {}

    setattr(_dense_reward, "__reward_component_keys__", ())

    monkeypatch.setattr(
        video_mod,
        "parse_args",
        lambda: Namespace(
            state_root=tmp_path,
            run_dir=run_dir,
            output=None,
            trace_output=None,
            fps=8,
            max_steps=None,
        ),
    )
    monkeypatch.setattr(video_mod, "_resolve_run_dir", lambda _s, _r: run_dir)
    monkeypatch.setattr(video_mod, "_load_json", lambda _p: trajectory)
    monkeypatch.setattr(video_mod, "_load_dense_reward", lambda _d: _dense_reward)
    monkeypatch.setattr(
        video_mod,
        "_build_env",
        lambda _t: (DummyEnv(), DummyEnvParams(), object()),
    )
    monkeypatch.setattr(video_mod, "_resolve_ruleset", lambda _t, _b: object())
    monkeypatch.setattr(
        video_mod,
        "_wrap_env_with_dense_reward",
        lambda env, trajectory_payload, dense_fn: env,
    )
    monkeypatch.setattr(
        video_mod.imageio, "get_writer", lambda *_a, **_k: DummyWriter()
    )

    with pytest.raises(RuntimeError, match="step failed intentionally"):
        video_mod.main()

    trace_path = run_dir / video_mod.DEFAULT_TRACE_NAME
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["replay_complete"] is False
    assert "step failed intentionally" in payload["replay_error"]
    assert payload["steps"] == []


def test_format_overlay_lines_wraps_long_goal_summary() -> None:
    """Ensure long goals wrap into multiple overlay lines.

    This test validates that long `env_summary` strings no longer produce a
    single oversized text row in the diagnostics panel. It is needed because the
    video overlay must remain readable while preserving goal context, and it
    differs from rendering-level tests by checking the text-formatting contract
    directly at `_format_overlay_lines` output.
    """

    lines = video_mod._format_overlay_lines(
        env_summary=(
            "Collect the red key, open the matching door, and reach the goal "
            "while avoiding lava tiles in narrow corridors."
        ),
        step_index=0,
        total_steps=5,
        dense_reward=0.25,
        dense_total=0.25,
        sparse_reward=0.0,
        sparse_total=0.0,
        component_values={},
        component_totals={},
        component_order=(),
    )

    step_line_index = lines.index("step 1/5")
    goal_lines = lines[:step_line_index]
    assert len(goal_lines) >= 2
    assert goal_lines[0].startswith("goal ")


def test_summarize_env_text_keeps_full_text_without_truncation() -> None:
    """Ensure full environment text is preserved for overlay display.

    This test verifies that `_summarize_env_text` no longer truncates long
    descriptions and instead returns the complete whitespace-normalized text. It
    is needed because the renderer now places diagnostics in a side panel with
    enough space for wrapped full-goal display, and it differs from wrapping
    tests by directly validating the text-preparation step.
    """

    env_text = (
        "Collect the red key before touching lava. "
        "Then open the locked door and reach the green goal tile. "
        "Avoid dead ends and preserve optional bonus pickup timing."
    )
    expected = " ".join(env_text.split())
    assert video_mod._summarize_env_text(env_text) == expected


def test_draw_overlay_keeps_map_uncovered_and_scales_viewport() -> None:
    """Verify diagnostics render in a side panel, not over map pixels.

    This test ensures `_draw_overlay` returns a composed frame where the map
    viewport is upscaled and preserved exactly in the left region, while overlay
    text is drawn in a separate right-side panel. It is needed because users must
    see the full map during replay even with many diagnostics lines, and it
    differs from text-wrapping tests by asserting pixel-level layout behavior.
    """

    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    frame[..., 0] = 30
    frame[..., 1] = 90
    frame[..., 2] = 140

    lines = [
        "goal short objective",
        "step 1/3",
        "dense +0.100 | total +0.100",
    ]
    rendered = video_mod._draw_overlay(frame, lines)

    expected_map = np.asarray(
        Image.fromarray(video_mod._normalize_frame(frame)).resize(
            (
                frame.shape[1] * video_mod.DEFAULT_VIEWPORT_SCALE,
                frame.shape[0] * video_mod.DEFAULT_VIEWPORT_SCALE,
            ),
            Image.Resampling.NEAREST,
        )
    )

    assert rendered.shape[0] >= expected_map.shape[0]
    assert rendered.shape[1] > expected_map.shape[1]
    assert np.array_equal(
        rendered[: expected_map.shape[0], : expected_map.shape[1]],
        expected_map,
    )
