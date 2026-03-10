from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

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

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_training_video.py"
    spec = importlib.util.spec_from_file_location("generate_training_video_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    assert "intra_op_parallelism_threads=1" in str(video_mod.os.environ.get("XLA_FLAGS"))


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
            self.extras: dict[str, Any] = {}

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

        def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> Literal[False]:
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
            latest_candidates=None,
            output=None,
            trace_output=None,
            astar_heuristic_output=None,
            astar_heuristic_trace_output=None,
            astar_no_heuristic_output=None,
            astar_no_heuristic_trace_output=None,
            no_astar_video=False,
            astar_max_nodes=64,
            astar_max_expansions=64,
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
    monkeypatch.setattr(video_mod.imageio, "get_writer", lambda *_a, **_k: DummyWriter())

    with pytest.raises(RuntimeError, match="step failed intentionally"):
        video_mod.main()

    trace_path = run_dir / video_mod.DEFAULT_TRACE_NAME
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["replay_complete"] is False
    assert "step failed intentionally" in payload["replay_error"]
    assert payload["steps"] == []


def test_astar_planner_prefers_progressing_path_with_dense_heuristic() -> None:
    """Validate dense-heuristic A* solves a deterministic toy graph.

    This test exercises `_plan_with_astar` directly on a compact environment
    with one optimal progression branch and one distracting dead-end branch. It
    is needed because rollout behavior now depends on planner-generated action
    sequences and planner metrics, and it differs from orchestration tests by
    validating search behavior without video I/O.
    """

    class DummyTimestep:
        """Minimal timestep carrying state, rewards, extras, and terminal flag."""

        def __init__(self, pos: int, dense_reward: float, sparse_reward: float, done: bool) -> None:
            self.state = {"pos": jnp.asarray(pos, dtype=jnp.int32)}
            self.reward = jnp.asarray(dense_reward, dtype=jnp.float32)
            self.extras = {"ground_truth_reward": jnp.asarray(sparse_reward, dtype=jnp.float32)}
            self._done = done

        def last(self) -> jax.Array:
            return jnp.asarray(self._done)

    class DummyEnv:
        """Two-action deterministic environment used for A* unit tests."""

        def num_actions(self, _env_params: Any) -> int:
            return 2

    def step_fn(_env_params: Any, timestep: DummyTimestep, action: Any) -> DummyTimestep:
        pos = int(jnp.asarray(timestep.state["pos"]))
        action_id = int(jnp.asarray(action))
        if pos >= 3:
            return DummyTimestep(pos, dense_reward=0.0, sparse_reward=1.0, done=True)
        if action_id == 0:
            next_pos = pos + 1
            solved = next_pos >= 3
            return DummyTimestep(
                next_pos,
                dense_reward=1.0 if not solved else 2.0,
                sparse_reward=1.0 if solved else 0.0,
                done=solved,
            )
        return DummyTimestep(pos + 10, dense_reward=-1.0, sparse_reward=0.0, done=False)

    root = DummyTimestep(pos=0, dense_reward=0.0, sparse_reward=0.0, done=False)
    plan = video_mod._plan_with_astar(
        env=DummyEnv(),
        env_params=object(),
        step_fn=step_fn,
        root_timestep=root,
        use_dense_heuristic=True,
        max_nodes=128,
        max_expansions=128,
    )

    assert plan.actions == [0, 0, 0]
    assert plan.search_stats["solved"] is True
    assert plan.search_stats["terminated_reason"] == "solved"
    assert plan.search_stats["generated_states"] >= 1
    assert plan.search_stats["expanded_states"] >= 1


def test_main_generates_replay_and_astar_outputs_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify default CLI behavior invokes replay and both A* rollout modes.

    This test ensures `main()` runs replay plus both A* modes when
    `--no-astar-video` is not set. It is needed because the script now produces
    three rollout artifacts by default, and it differs from planner tests by
    validating orchestration and output defaults.
    """

    run_dir = tmp_path / "candidate-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory = {"actions": [0, 1], "reset_key": [1, 2], "env_id": "x", "benchmark_id": "y"}
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        video_mod,
        "parse_args",
        lambda: Namespace(
            state_root=tmp_path,
            run_dir=run_dir,
            latest_candidates=None,
            output=None,
            trace_output=None,
            astar_heuristic_output=None,
            astar_heuristic_trace_output=None,
            astar_no_heuristic_output=None,
            astar_no_heuristic_trace_output=None,
            no_astar_video=False,
            astar_max_nodes=32,
            astar_max_expansions=32,
            fps=8,
            max_steps=None,
        ),
    )
    monkeypatch.setattr(video_mod, "_resolve_run_dir", lambda _s, _r: run_dir)
    monkeypatch.setattr(video_mod, "_load_json", lambda _p: trajectory)
    monkeypatch.setattr(
        video_mod,
        "_build_replay_reward_key_diagnostics",
        lambda _run_dir, _trajectory: {"missing_from_task": []},
    )

    def fake_run_rollout_video(**kwargs: Any):
        """Capture rollout invocation arguments without running replay logic."""

        calls.append(kwargs)
        mode = str(kwargs["rollout_mode"])
        payload: dict[str, Any] = {"rollout_mode": mode}
        if mode != video_mod.ROLLOUT_MODE_REPLAY:
            payload["search_stats"] = {
                "generated_states": 8
                if mode == video_mod.ROLLOUT_MODE_ASTAR_NO_HEURISTIC
                else 5,
                "solution_cost": 3,
                "solution_length": 3,
            }
        return video_mod._RolloutRunResult(
            replay_error=None,
            trace_output=Path(kwargs["trace_output"]),
            trace_payload=payload,
        )

    monkeypatch.setattr(video_mod, "_run_rollout_video", fake_run_rollout_video)
    monkeypatch.setattr(video_mod, "_write_trace_payload", lambda *_a, **_k: None)
    video_mod.main()

    assert len(calls) == 3
    assert calls[0]["rollout_mode"] == video_mod.ROLLOUT_MODE_REPLAY
    assert calls[1]["rollout_mode"] == video_mod.ROLLOUT_MODE_ASTAR_NO_HEURISTIC
    assert calls[2]["rollout_mode"] == video_mod.ROLLOUT_MODE_ASTAR_HEURISTIC
    assert calls[0]["output_path"] == run_dir / video_mod.DEFAULT_VIDEO_NAME
    assert calls[1]["output_path"] == run_dir / video_mod.DEFAULT_ASTAR_NO_HEURISTIC_VIDEO_NAME
    assert calls[2]["output_path"] == run_dir / video_mod.DEFAULT_ASTAR_HEURISTIC_VIDEO_NAME
    assert calls[0]["trace_output"] == run_dir / video_mod.DEFAULT_TRACE_NAME
    assert calls[1]["trace_output"] == run_dir / video_mod.DEFAULT_ASTAR_NO_HEURISTIC_TRACE_NAME
    assert calls[2]["trace_output"] == run_dir / video_mod.DEFAULT_ASTAR_HEURISTIC_TRACE_NAME


def test_main_respects_no_astar_video_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure the opt-out flag disables A* rollout orchestration.

    This test validates that `--no-astar-video` preserves legacy behavior by
    running only the trajectory replay rollout. It is needed because users may
    disable extra outputs on constrained systems, and it differs from default
    orchestration tests by asserting selective mode suppression.
    """

    run_dir = tmp_path / "candidate-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory = {"actions": [0], "reset_key": [1, 2], "env_id": "x", "benchmark_id": "y"}
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        video_mod,
        "parse_args",
        lambda: Namespace(
            state_root=tmp_path,
            run_dir=run_dir,
            latest_candidates=None,
            output=None,
            trace_output=None,
            astar_heuristic_output=None,
            astar_heuristic_trace_output=None,
            astar_no_heuristic_output=None,
            astar_no_heuristic_trace_output=None,
            no_astar_video=True,
            astar_max_nodes=32,
            astar_max_expansions=32,
            fps=8,
            max_steps=None,
        ),
    )
    monkeypatch.setattr(video_mod, "_resolve_run_dir", lambda _s, _r: run_dir)
    monkeypatch.setattr(video_mod, "_load_json", lambda _p: trajectory)
    monkeypatch.setattr(
        video_mod,
        "_build_replay_reward_key_diagnostics",
        lambda _run_dir, _trajectory: {"missing_from_task": []},
    )
    def _record_rollout_call(**kwargs: Any):
        """Append rollout invocation args to a list for assertion checks."""

        calls.append(kwargs)
        return video_mod._RolloutRunResult(
            replay_error=None,
            trace_output=Path(kwargs["trace_output"]),
            trace_payload={"rollout_mode": str(kwargs["rollout_mode"])},
        )

    monkeypatch.setattr(
        video_mod,
        "_run_rollout_video",
        _record_rollout_call,
    )

    video_mod.main()
    assert len(calls) == 1
    assert calls[0]["rollout_mode"] == video_mod.ROLLOUT_MODE_REPLAY


def test_build_astar_overlay_status_lines_include_searched_count() -> None:
    """Ensure solved overlays report the searched-state count inline.

    This test validates the user-visible overlay string requested for solved A*
    runs. It is needed because the generated MP4 should answer how much search
    work was required without opening the JSON trace, and it differs from trace
    schema tests by checking the exact overlay-formatting contract.
    """

    lines = video_mod._build_astar_overlay_status_lines(
        {
            "solved": True,
            "terminated_reason": "solved",
            "generated_states": 17,
            "expanded_states": 9,
        }
    )

    assert lines[0] == "astar solved (17 searched) (solved)"
    assert lines[1] == "states gen=17 exp=9"


def test_build_rollout_trace_payload_includes_astar_selection_schema(tmp_path: Path) -> None:
    """Confirm A* traces carry rollout mode and per-step selection metadata.

    This test checks that trace payload assembly includes the new top-level
    `rollout_mode` field and preserves selection diagnostics nested within step
    rows. It is needed because downstream trace consumers depend on schema
    stability, and it differs from orchestration tests by validating payload
    structure directly.
    """

    context = video_mod._RolloutContext(
        env=object(),
        env_params=object(),
        reset_fn=lambda *_a: None,
        step_fn=lambda *_a: None,
        dense_reward_fn=lambda *_a: None,
        reset_key=None,
        env_text="synthetic env",
        env_seed=11,
        env_summary="goal synthetic",
        reward_object_key_diagnostics={"missing_from_task": []},
        trace_steps_cap=3,
    )
    step_row = {
        "step": 0,
        "action": 1,
        "dense_reward": 0.2,
        "sparse_reward": 0.1,
        "dense_total": 0.2,
        "sparse_total": 0.1,
        "reward_components": {},
        "reward_component_totals": {},
        "selection": {
            "policy": "astar_plan",
            "selected_action": 1,
            "tie_break": video_mod.ASTAR_TIE_BREAK,
        },
    }
    payload = video_mod._build_rollout_trace_payload(
        rollout_mode=video_mod.ROLLOUT_MODE_ASTAR_HEURISTIC,
        trajectory={"actions": [0]},
        context=context,
        run_dir=tmp_path,
        output_path=tmp_path / "training_video_astar_heuristic.mp4",
        trace_steps=[step_row],
        replay_error=None,
        rollout_metadata={"search_stats": {"generated_states": 10}},
    )

    assert payload["rollout_mode"] == video_mod.ROLLOUT_MODE_ASTAR_HEURISTIC
    assert payload["replay_complete"] is True
    assert payload["search_stats"]["generated_states"] == 10
    assert payload["steps"][0]["selection"]["selected_action"] == 1
    assert payload["steps"][0]["selection"]["policy"] == "astar_plan"


def test_main_raises_when_astar_fails_after_replay_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure rollout failures are isolated but still produce non-zero exit.

    This test simulates a successful replay rollout followed by a heuristic A*
    rollout failure and verifies that `main()` raises while preserving mode
    ordering and execution of both rollout attempts. It is needed because
    the command must report partial failure without skipping remaining rollouts.
    """

    run_dir = tmp_path / "candidate-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory = {"actions": [0, 1], "reset_key": [1, 2], "env_id": "x", "benchmark_id": "y"}
    mode_calls: list[str] = []

    monkeypatch.setattr(
        video_mod,
        "parse_args",
        lambda: Namespace(
            state_root=tmp_path,
            run_dir=run_dir,
            latest_candidates=None,
            output=None,
            trace_output=None,
            astar_heuristic_output=None,
            astar_heuristic_trace_output=None,
            astar_no_heuristic_output=None,
            astar_no_heuristic_trace_output=None,
            no_astar_video=False,
            astar_max_nodes=32,
            astar_max_expansions=32,
            fps=8,
            max_steps=None,
        ),
    )
    monkeypatch.setattr(video_mod, "_resolve_run_dir", lambda _s, _r: run_dir)
    monkeypatch.setattr(video_mod, "_load_json", lambda _p: trajectory)
    monkeypatch.setattr(
        video_mod,
        "_build_replay_reward_key_diagnostics",
        lambda _run_dir, _trajectory: {"missing_from_task": []},
    )

    def fake_run_rollout_video(**kwargs: Any):
        """Return a synthetic failure only for heuristic A* mode."""

        mode = str(kwargs["rollout_mode"])
        mode_calls.append(mode)
        replay_error = (
            "RuntimeError: astar failed"
            if mode == video_mod.ROLLOUT_MODE_ASTAR_HEURISTIC
            else None
        )
        payload: dict[str, Any] = {"rollout_mode": mode}
        if mode != video_mod.ROLLOUT_MODE_REPLAY:
            payload["search_stats"] = {
                "generated_states": 7,
                "solution_cost": 3,
                "solution_length": 3,
            }
        return video_mod._RolloutRunResult(
            replay_error=replay_error,
            trace_output=Path(kwargs["trace_output"]),
            trace_payload=payload,
        )

    monkeypatch.setattr(video_mod, "_run_rollout_video", fake_run_rollout_video)
    monkeypatch.setattr(video_mod, "_write_trace_payload", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="astar failed"):
        video_mod.main()
    assert mode_calls == [
        video_mod.ROLLOUT_MODE_REPLAY,
        video_mod.ROLLOUT_MODE_ASTAR_NO_HEURISTIC,
        video_mod.ROLLOUT_MODE_ASTAR_HEURISTIC,
    ]


def test_resolve_target_run_dirs_uses_latest_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure batch mode resolves the requested number of recent candidates.

    This test validates the new CLI selection path used for "last N candidates".
    It is needed because batch replay should bypass single-run resolution when
    `--latest-candidates` is provided, and it differs from `main()` orchestration
    tests by targeting the run-selection helper directly.
    """

    expected = [tmp_path / "candidate-a", tmp_path / "candidate-b"]
    monkeypatch.setattr(
        video_mod,
        "_select_latest_candidate_runs",
        lambda _state_root, count: expected[:count],
    )

    resolved = video_mod._resolve_target_run_dirs(
        Namespace(latest_candidates=2, run_dir=None),
        tmp_path,
    )

    assert resolved == expected


def test_main_prints_batch_heuristic_summary_for_latest_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ensure batch mode reports how often the heuristic converged faster.

    This test drives the new multi-run entry path and checks the aggregate
    stdout summary requested by the user. It is needed because the batch CLI is
    useful only if it surfaces a cross-environment heuristic win count, and it
    differs from per-run tests by asserting aggregated reporting across two
    candidates in one invocation.
    """

    run_dirs = [tmp_path / "candidate-1", tmp_path / "candidate-2"]
    for run_dir in run_dirs:
        run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        video_mod,
        "parse_args",
        lambda: Namespace(
            state_root=tmp_path,
            run_dir=None,
            latest_candidates=2,
            output=None,
            trace_output=None,
            astar_heuristic_output=None,
            astar_heuristic_trace_output=None,
            astar_no_heuristic_output=None,
            astar_no_heuristic_trace_output=None,
            no_astar_video=False,
            astar_max_nodes=32,
            astar_max_expansions=32,
            fps=8,
            max_steps=None,
        ),
    )
    monkeypatch.setattr(video_mod, "_resolve_target_run_dirs", lambda _args, _state_root: run_dirs)
    run_summaries = iter(
        [
            {
                "run_dir": run_dirs[0],
                "errors": [],
                "heuristic_comparison": {"heuristic_converged_faster": True},
            },
            {
                "run_dir": run_dirs[1],
                "errors": [],
                "heuristic_comparison": {"heuristic_converged_faster": False},
            },
        ]
    )
    monkeypatch.setattr(
        video_mod,
        "_run_single_candidate",
        lambda **_kwargs: next(run_summaries),
    )

    video_mod.main()

    stdout = capsys.readouterr().out
    assert "heuristic_converged_faster=1/2" in stdout


def test_build_heuristic_comparison_payload_marks_heuristic_faster_when_baseline_unsolved() -> None:
    """Ensure faster convergence treats heuristic-only solves as an improvement.

    This test captures the intended comparison semantics for batch summaries. It
    is needed because a heuristic that solves within budget while baseline does
    not should count as converging faster, and it differs from generated-state
    reduction checks by covering the asymmetric solved/unsolved case directly.
    """

    comparison = video_mod._build_heuristic_comparison_payload(
        baseline_search_stats={"generated_states": 100, "solved": False},
        heuristic_search_stats={"generated_states": 80, "solved": True},
    )

    assert comparison["heuristic_converged_faster"] is True


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


def test_format_overlay_lines_never_includes_human_control_legend() -> None:
    """Ensure policy-rollout overlays exclude human-play control hints.

    This test guards the separation between manual-play UX and policy replay
    rendering. It is needed because human-only controls guidance should not
    appear in generated rollout videos, and it differs from wrapping tests by
    asserting the absence of `CONTROLS` text in the shared rollout formatter.
    """

    lines = video_mod._format_overlay_lines(
        env_summary="Place the blue key below the red star.",
        step_index=2,
        total_steps=10,
        dense_reward=0.1,
        dense_total=0.4,
        sparse_reward=0.0,
        sparse_total=0.0,
        component_values={},
        component_totals={},
        component_order=(),
    )

    assert all("controls" not in line.lower() for line in lines)


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


def test_draw_overlay_adds_component_reward_line_plot() -> None:
    """Ensure the overlay can render per-component instantaneous reward trends.

    This test validates that `_draw_overlay` draws chart primitives in the
    diagnostics panel when component histories are provided. It is needed because
    replay videos now visualize per-step component rewards over time, and it
    differs from map-preservation tests by asserting that chart pixels are
    present in the panel region even when no text lines are supplied.
    """

    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    component_order = ("pickup_bonus", "goal_progress")
    component_series = {
        "pickup_bonus": [0.0, 0.1, -0.2, 0.3],
        "goal_progress": [0.0, 0.05, 0.2, 0.4],
    }

    rendered = video_mod._draw_overlay(
        frame,
        [],
        component_series=component_series,
        component_order=component_order,
    )

    expected_map = np.asarray(
        Image.fromarray(video_mod._normalize_frame(frame)).resize(
            (
                frame.shape[1] * video_mod.DEFAULT_VIEWPORT_SCALE,
                frame.shape[0] * video_mod.DEFAULT_VIEWPORT_SCALE,
            ),
            Image.Resampling.NEAREST,
        )
    )
    panel_start = expected_map.shape[1] + video_mod.OVERLAY_MAP_PANEL_GAP
    panel_pixels = rendered[:, panel_start:, :]

    assert panel_pixels.size > 0
    assert np.any(panel_pixels != 0)
