from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest


def _load_play_module():
    """Load the play-level script as a module from its filesystem path.

    This helper imports `scripts/play_level.py` without requiring `scripts/` to
    be a package. It is needed because pytest collection imports package modules
    by default, and it differs from normal imports by constructing an explicit
    module spec tied to the script file.
    """

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "play_level.py"
    if str(script_path.parent) not in sys.path:
        sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location("play_level_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


play_mod = _load_play_module()


def test_parse_args_defaults_trace_path(tmp_path: Path) -> None:
    """Ensure trace output defaults to `play_level_trace.json` under run dir.

    This test verifies that trace path resolution keeps the default file name
    stable and creates parent directories under the selected run directory. It
    is needed so users can run the tool without specifying `--trace-output`, and
    it differs from CLI integration tests by validating the path helper in
    isolation.
    """

    run_dir = tmp_path / "candidate"
    run_dir.mkdir(parents=True)

    resolved = play_mod._resolve_trace_output(run_dir, None)

    assert resolved == (run_dir / play_mod.DEFAULT_TRACE_NAME).resolve()
    assert resolved.parent.exists()


def test_key_to_action_mapping_includes_wasd_and_arrows() -> None:
    """Ensure key bindings include both WASD and arrow-key movement aliases.

    This test validates that the fixed keymap supports both keyboard styles,
    which is needed for ergonomic manual debugging sessions. It differs from
    event-loop tests by checking the binding contract directly as data.
    """

    assert play_mod.KEY_TO_ACTION["up"] == 0
    assert play_mod.KEY_TO_ACTION["w"] == 0
    assert play_mod.KEY_TO_ACTION["right"] == 1
    assert play_mod.KEY_TO_ACTION["d"] == 1
    assert play_mod.KEY_TO_ACTION["left"] == 2
    assert play_mod.KEY_TO_ACTION["a"] == 2


def test_action_names_match_xminigrid_turn_convention() -> None:
    """Ensure action-id labels align with XMiniGrid's clockwise conventions.

    This test validates the canonical action-name table used by play traces and
    behavior summaries. It is needed because left/right label inversions can make
    human debugging and reflection feedback misleading, and it differs from key
    binding tests by checking semantic action-id labeling directly.
    """

    assert play_mod.ACTION_NAMES[0] == "move_forward"
    assert play_mod.ACTION_NAMES[1] == "turn_right"
    assert play_mod.ACTION_NAMES[2] == "turn_left"


def test_key_mapping_matches_environment_turn_directions() -> None:
    """Ensure left/right key bindings match environment rotation semantics.

    This test executes one right-turn and one left-turn action in a real
    XMiniGrid environment using the action ids configured by `play_level` key
    bindings. It is needed because mismatched key-to-action mapping can create
    an apparent rendering inversion and silently poison behavior diagnostics, and
    it differs from static mapping tests by validating end-to-end directional
    effects against environment dynamics.
    """

    xminigrid = pytest.importorskip("xminigrid")
    jax_mod = pytest.importorskip("jax")

    env, env_params = xminigrid.make("MiniGrid-Empty-5x5")
    timestep = env.reset(env_params, jax_mod.random.PRNGKey(0))
    start_direction = int(jnp.asarray(timestep.state.agent.direction))

    timestep_right = env.step(env_params, timestep, jnp.asarray(play_mod.KEY_TO_ACTION["right"]))
    timestep_left = env.step(env_params, timestep, jnp.asarray(play_mod.KEY_TO_ACTION["left"]))

    assert int(jnp.asarray(timestep_right.state.agent.direction)) == ((start_direction + 1) % 4)
    assert int(jnp.asarray(timestep_left.state.agent.direction)) == ((start_direction - 1) % 4)


def test_extract_human_objective_lines_prefers_task_and_success_clauses() -> None:
    """Ensure manual overlay objective parsing focuses on human-relevant clauses.

    This test verifies that `_extract_human_objective_lines` pulls only the
    `Your task is ...` and `Success when ...` sentences from a verbose
    environment description. It is needed because manual play should highlight
    concise objective guidance rather than implementation-heavy context, and it
    differs from full overlay tests by validating parser behavior in isolation.
    """

    env_text = (
        "This level uses layout R1. "
        "Your task is to place the blue key immediately down of the red star. "
        "Success when the blue key is exactly one cell down of the red star. "
        "Use distances and spatial relations; avoid Python-side branching."
    )

    objective_line, win_condition_line = play_mod._extract_human_objective_lines(
        env_text,
        None,
    )

    assert objective_line == "Your task is to place the blue key immediately down of the red star."
    assert (
        win_condition_line == "Success when the blue key is exactly one cell down of the red star."
    )


def test_build_human_overlay_context_lines_includes_objective_and_controls() -> None:
    """Ensure human overlay block includes objective, win condition, and controls.

    This test validates that `_build_human_overlay_context_lines` emits the
    expected labeled sections used in manual play (`OBJECTIVE`, `WIN CONDITION`,
    and `CONTROLS`). It is needed because the play HUD must communicate both the
    level target and keyboard mapping directly on screen, and it differs from
    event-loop tests by asserting deterministic line content from helper output.
    """

    lines = play_mod._build_human_overlay_context_lines(
        objective_line="Your task is to place the blue key next to the red star.",
        win_condition_line="Success when the blue key is adjacent to the red star.",
    )
    joined = " ".join(lines)

    assert any(line.startswith("OBJECTIVE:") for line in lines)
    assert any(line.startswith("WIN CONDITION:") for line in lines)
    assert any(line.startswith("CONTROLS:") for line in lines)
    assert "Up/W=move_forward" in joined
    assert "Esc=quit" in joined


def test_resolve_human_overlay_objective_prefers_live_ruleset_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure manual play objective lines prefer fresh ruleset-derived wording.

    This test validates that `_resolve_human_overlay_objective_lines` uses
    `describe_ruleset` output when available, even if the trajectory metadata
    contains stale objective language. It is needed because old artifacts may
    embed previously incorrect direction text, and it differs from extraction
    helper tests by checking source-selection precedence.
    """

    monkeypatch.setattr(
        play_mod,
        "describe_ruleset",
        lambda _env, _env_params: (
            "Your task is to place the blue square immediately right of the green pyramid. "
            "Success when the blue square is exactly one cell right of the green pyramid."
        ),
    )

    objective_line, win_condition_line = play_mod._resolve_human_overlay_objective_lines(
        env=object(),
        env_params=object(),
        trajectory_env_text=(
            "Your task is to place the blue square immediately left of the green pyramid. "
            "Success when the blue square is exactly one cell left of the green pyramid."
        ),
        trajectory_env_summary=None,
    )

    assert (
        objective_line
        == "Your task is to place the blue square immediately right of the green pyramid."
    )
    assert (
        win_condition_line
        == "Success when the blue square is exactly one cell right of the green pyramid."
    )


def test_resolve_human_overlay_objective_falls_back_to_trajectory_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure manual play objective lines fall back when live description fails.

    This test confirms that play overlay objective extraction remains robust when
    `describe_ruleset` raises, by reusing trajectory metadata text as a fallback.
    It is needed because interactive sessions should still display objective
    guidance even under optional dependency failures, and it differs from source
    preference tests by covering the error path.
    """

    def _raise(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(play_mod, "describe_ruleset", _raise)

    objective_line, win_condition_line = play_mod._resolve_human_overlay_objective_lines(
        env=object(),
        env_params=object(),
        trajectory_env_text=(
            "Your task is to place the blue square immediately right of the green pyramid. "
            "Success when the blue square is exactly one cell right of the green pyramid."
        ),
        trajectory_env_summary=None,
    )

    assert (
        objective_line
        == "Your task is to place the blue square immediately right of the green pyramid."
    )
    assert (
        win_condition_line
        == "Success when the blue square is exactly one cell right of the green pyramid."
    )


def test_apply_action_updates_dense_sparse_and_components() -> None:
    """Verify one action step updates totals and component histories correctly.

    This test exercises `_apply_action_and_collect` with a controlled timestep
    stub so dense reward, sparse reward, and component accounting can be
    asserted deterministically. It is needed because these values are central to
    reward-debugging overlays and trace logs, and it differs from end-to-end
    tests by isolating one state transition.
    """

    class DummyStep:
        def __init__(self) -> None:
            self.reward = jnp.asarray(0.3)
            self.extras = {
                "ground_truth_reward": jnp.asarray(0.2),
                "reward_components": {"progress": jnp.asarray(1.0)},
            }

        def last(self) -> jnp.ndarray:
            return jnp.asarray(False)

    def step_fn(_env_params: Any, _ts: Any, _action: Any) -> DummyStep:
        return DummyStep()

    session_state = play_mod.PlaySessionState(
        dense_total=0.0,
        sparse_total=0.0,
        component_order=("progress",),
        component_totals={"progress": 0.0},
        component_series={"progress": []},
        last_dense_reward=0.0,
        last_sparse_reward=0.0,
        last_components={"progress": 0.0},
    )

    _next_ts, trace_row, updated_state = play_mod._apply_action_and_collect(
        step_fn=step_fn,
        env_params=object(),
        timestep=object(),
        action_value=0,
        step_index=0,
        episode_index=0,
        session_state=session_state,
    )

    assert updated_state.component_order == ("progress",)
    assert trace_row["action"] == 0
    assert trace_row["action_name"] == "move_forward"
    assert trace_row["dense_reward"] == pytest.approx(0.3)
    assert trace_row["sparse_reward"] == pytest.approx(0.2)
    assert updated_state.dense_total == pytest.approx(0.3)
    assert updated_state.sparse_total == pytest.approx(0.2)
    assert updated_state.component_totals["progress"] == pytest.approx(1.0)
    assert updated_state.component_series["progress"] == [pytest.approx(1.0)]


def test_trace_written_on_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure trace JSON is persisted when the interactive loop raises errors.

    This test forces a runtime failure during render and asserts that
    `play_level_trace.json` is still emitted with error metadata and
    `replay_complete=False`. It is needed because debugging sessions should not
    lose partial diagnostic context when rendering fails, and it differs from
    success-path tests by validating exception cleanup behavior.
    """

    class DummyEnvParams:
        def replace(self, **_kwargs: Any) -> "DummyEnvParams":
            return self

    class DummyTimestep:
        reward = jnp.asarray(0.0)
        extras: dict[str, Any] = {}

        def last(self) -> jnp.ndarray:
            return jnp.asarray(False)

    class DummyEnv:
        def reset(self, _env_params: Any, _key: Any) -> DummyTimestep:
            return DummyTimestep()

        def step(self, _env_params: Any, _ts: Any, _action: Any) -> DummyTimestep:
            return DummyTimestep()

        def render(self, _env_params: Any, _ts: Any) -> np.ndarray:
            raise RuntimeError("render failed intentionally")

    class DummyClock:
        def tick(self, _fps: int) -> None:
            return None

    class DummyWindow:
        def blit(self, _surf: Any, _xy: tuple[int, int]) -> None:
            return None

    fake_pygame = types.SimpleNamespace(
        QUIT=12,
        KEYDOWN=2,
        init=lambda: None,
        quit=lambda: None,
        display=types.SimpleNamespace(
            init=lambda: None,
            set_mode=lambda _size: DummyWindow(),
            set_caption=lambda _title: None,
            flip=lambda: None,
        ),
        event=types.SimpleNamespace(get=lambda: []),
        key=types.SimpleNamespace(name=lambda _k: "escape"),
        surfarray=types.SimpleNamespace(make_surface=lambda _arr: np.zeros((4, 4, 3))),
        transform=types.SimpleNamespace(smoothscale=lambda surf, _size: surf),
        time=types.SimpleNamespace(Clock=lambda: DummyClock()),
    )

    run_dir = tmp_path / "candidate"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(play_mod, "pygame", fake_pygame)
    monkeypatch.setattr(
        play_mod,
        "parse_args",
        lambda: Namespace(
            state_root=tmp_path,
            run_dir=run_dir,
            output=None,
            trace_output=None,
            max_steps=None,
            window_size=320,
            fps=10,
        ),
    )
    monkeypatch.setattr(play_mod, "_resolve_run_dir", lambda _s, _r: run_dir)
    monkeypatch.setattr(
        play_mod,
        "_load_json",
        lambda _p: {
            "env_id": "XLand-MiniGrid-R1-9x9",
            "benchmark_id": "trivial-1m",
            "deterministic_rulesets": True,
            "ruleset_index": 42,
            "reset_key": [1, 2],
            "env_text": "Synthetic env text",
            "eval_seed": 7,
        },
    )

    def _dense_reward(*_args: Any, **_kwargs: Any):
        return jnp.asarray(0.0), {}

    setattr(_dense_reward, "__reward_component_keys__", ())

    monkeypatch.setattr(play_mod, "_load_dense_reward", lambda _d: _dense_reward)
    monkeypatch.setattr(play_mod, "_build_env", lambda _t: (DummyEnv(), DummyEnvParams(), object()))
    monkeypatch.setattr(
        play_mod,
        "_resolve_initial_state",
        lambda _t, _b: (object(), jnp.asarray([1, 2], dtype=jnp.uint32)),
    )
    monkeypatch.setattr(play_mod, "_wrap_env_with_dense_reward", lambda env, _t, _fn: env)
    monkeypatch.setattr(play_mod, "_build_replay_step_fns", lambda env: (env.reset, env.step))

    with pytest.raises(RuntimeError, match="render failed intentionally"):
        play_mod.main()

    trace_path = run_dir / play_mod.DEFAULT_TRACE_NAME
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["replay_complete"] is False
    assert "render failed intentionally" in payload["replay_error"]


def test_reset_reuses_saved_initial_state() -> None:
    """Ensure reset helper always calls reset with the same saved reset key.

    This test validates deterministic reset behavior used by the `R` hotkey.
    It is needed because manual episodes must restart from the same initial
    state for consistent policy-vs-human comparisons, and it differs from full
    loop tests by asserting call arguments directly.
    """

    seen: list[tuple[Any, Any]] = []

    def reset_fn(env_params: Any, reset_key: Any) -> str:
        seen.append((env_params, reset_key))
        return "ts"

    env_params = object()
    reset_key = (123, 456)

    ts1 = play_mod._reset_episode(reset_fn, env_params=env_params, reset_key=reset_key)
    ts2 = play_mod._reset_episode(reset_fn, env_params=env_params, reset_key=reset_key)

    assert ts1 == "ts"
    assert ts2 == "ts"
    assert seen == [(env_params, reset_key), (env_params, reset_key)]


def test_overlay_render_called_with_component_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure overlay helper forwards component histories into renderer.

    This test verifies that interactive rendering passes component-series data
    to the shared overlay compositor. It is needed so line plots remain visible
    during manual play, and it differs from pixel assertions by validating the
    function-call contract directly.
    """

    captured: dict[str, Any] = {}

    def fake_draw_overlay(frame: Any, lines: list[str], **kwargs: Any) -> np.ndarray:
        captured["frame"] = frame
        captured["lines"] = lines
        captured["kwargs"] = kwargs
        return np.zeros((6, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(play_mod, "_draw_overlay", fake_draw_overlay)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rendered = play_mod._render_overlay_frame(
        frame,
        ["step 1/1"],
        component_series={"progress": [0.1, 0.2]},
        component_order=("progress",),
    )

    assert rendered.shape == (6, 6, 3)
    assert captured["kwargs"]["component_series"] == {"progress": [0.1, 0.2]}
    assert captured["kwargs"]["component_order"] == ("progress",)
