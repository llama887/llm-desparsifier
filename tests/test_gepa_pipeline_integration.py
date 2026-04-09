from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy
import pytest


def _load_run_reward_batch():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "run_reward_batch.py"
    spec = importlib.util.spec_from_file_location("run_reward_batch", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _DummyRewardGenerator:
    def __init__(self, *args, **kwargs):
        self.last_env_description = "dummy-env"
        self.last_attempt_history = []

    def _build_feedback_block(self, _attempts):
        return "sanitizer feedback"


class _DummyGEPA:
    def __init__(self, metric, *args, **kwargs):
        self.metric = metric
        self.stats = {"dummy": True}

    def compile(self, program, trainset=None, **_kwargs):
        for example in trainset or []:
            prediction = dspy.Prediction(prompt_text="dummy-prompt")
            self.metric(
                example,
                prediction,
                trace=[("predictor", {}, {})],
                pred_name=None,
                pred_trace=None,
            )
            self.metric(
                example,
                prediction,
                trace=[("predictor", {}, {})],
                pred_name="prompt_generator",
                pred_trace=[("predictor", {}, {})],
            )
        class _DummyPromptGenerator:
            def __call__(self, **_kwargs):
                return "dummy-prompt"

            def dump_state(self):
                return {}

        program.prompt_generator = _DummyPromptGenerator()
        return program


def _dummy_training_result(env_id: str, benchmark_id: str):
    config = SimpleNamespace(env_id=env_id, benchmark_id=benchmark_id)
    train_info = {
        "loss_info": {"eval/ground_truth_returns_mean": [0.0]},
        "component_logs": {},
    }
    return SimpleNamespace(
        config=config,
        train_info=train_info,
        artifacts={},
        final_metrics={},
        emitted_reward_code="def dense_reward(...): pass",
        ground_truth_eval={"returns": [1.0], "successes": 1},
    )


def _dummy_astar_result(env_id: str, benchmark_id: str):
    config = SimpleNamespace(
        env_id=env_id,
        benchmark_id=benchmark_id,
        eval_seed=2,
        deterministic_rulesets=True,
        fixed_ruleset_seed=123,
    )
    return SimpleNamespace(
        config=config,
        score=0.875,
        solved=True,
        search_stats={
            "solved": True,
            "expanded_states": 25,
            "generated_states": 40,
            "terminated_reason": "solved",
            "solution_length": 4,
        },
        final_metrics={
            "score": 0.875,
            "solve_rate": 1.0,
            "expanded_states": 25.0,
            "generated_states": 40.0,
        },
        artifacts={"eval_trajectory": "/tmp/eval_trajectory.json"},
        emitted_reward_code="def dense_reward(...): pass",
        reward_mode="dense",
        env_description="dummy-env",
    )


def test_gepa_pipeline_runs_with_mocked_training(tmp_path, monkeypatch):
    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 1\n"
        "  eval_seed: 2\n",
        encoding="utf-8",
    )

    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"

    def _fake_run_training_with_reward(_reward_generator, output_dir=None, **_kwargs):
        return _dummy_training_result("XLand-MiniGrid-R1-9x9", "trivial-1m")

    def _fake_sparse_baseline(jobs, **_kwargs):
        return ({job.name: {"solve_rate": 0.0} for job in jobs}, 0.0)

    monkeypatch.setattr(run_reward_batch, "RewardGenerator", _DummyRewardGenerator)
    monkeypatch.setattr(
        run_reward_batch, "run_training_with_reward", _fake_run_training_with_reward
    )
    monkeypatch.setattr(
        run_reward_batch, "ensure_sparse_baseline", _fake_sparse_baseline
    )
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_holdout_sparse_baselines",
        lambda holdout_jobs, **_kwargs: (
            {job.name: {"solve_rate": 0.0} for job in holdout_jobs},
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        lambda jobs, **_kwargs: (
            {
                job.name: {
                    "solve_rate_mean": 0.0,
                    "solve_rate_std": 0.0,
                    "solve_rates": [0.0],
                }
                for job in jobs
            },
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "build_reward_reflection",
        lambda *_args, **_kwargs: "reflection",
    )
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _DummyGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    active_path = run_reward_batch.get_active_prompt_path(state_root)
    assert active_path.exists()


def test_run_batch_uses_astar_backend_for_metric_and_holdout(tmp_path, monkeypatch):
    """Ensure `--inner-loop=astar` routes the pipeline through search helpers.

    This integration test verifies the top-level runner flips all relevant
    orchestration paths when A* mode is selected. It is needed because the
    backend switch must cover baselines, GEPA metric evaluation, and holdouts
    together, and it differs from unit tests by driving the full `run_batch`
    entrypoint with a mocked GEPA compile loop.
    """

    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 1\n"
        "  eval_seed: 2\n",
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"
    captured = {"astar_metric_calls": 0, "holdout_called": False, "baseline_called": False}

    def _fake_run_astar_with_reward(_reward_generator, output_dir=None, **_kwargs):
        output_path = Path(str(output_dir))
        output_path.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_path / "eval_trajectory.json"
        trajectory_path.write_text(
            json.dumps({"actions": [0, 1], "env_text": "dummy-env"}),
            encoding="utf-8",
        )
        result = _dummy_astar_result("XLand-MiniGrid-R1-9x9", "trivial-1m")
        result.artifacts = {"eval_trajectory": str(trajectory_path)}
        captured["astar_metric_calls"] += 1
        return result

    def _fake_astar_baseline(jobs, **_kwargs):
        captured["baseline_called"] = True
        return ({job.name: {"score": 0.25, "solve_rate": 0.0} for job in jobs}, 0.25)

    def _fake_astar_holdout(jobs, **_kwargs):
        captured["holdout_called"] = True
        return (
            {
                job.name: {
                    "score_mean": 0.5,
                    "score_std": 0.0,
                    "solve_rate_mean": 1.0,
                    "solve_rate_std": 0.0,
                    "scores": [0.5],
                    "solve_rates": [1.0],
                }
                for job in jobs
            },
            0.5,
        )

    monkeypatch.setattr(run_reward_batch, "RewardGenerator", _DummyRewardGenerator)
    monkeypatch.setattr(run_reward_batch, "run_astar_with_reward", _fake_run_astar_with_reward)
    monkeypatch.setattr(run_reward_batch, "ensure_astar_sparse_baseline", _fake_astar_baseline)
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_astar_holdout_sparse_baselines",
        lambda holdout_jobs, **_kwargs: (
            {job.name: {"score": 0.2, "solve_rate": 0.0} for job in holdout_jobs},
            0.2,
        ),
    )
    monkeypatch.setattr(run_reward_batch, "evaluate_astar_on_jobs", _fake_astar_holdout)
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_sparse_baseline",
        lambda *_args, **_kwargs: pytest.fail("RL sparse baseline should not run in astar mode"),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        lambda *_args, **_kwargs: pytest.fail("RL holdout eval should not run in astar mode"),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "build_reward_reflection",
        lambda *_args, **_kwargs: pytest.fail("RL reflection builder should not run in astar mode"),
    )
    monkeypatch.setattr(run_reward_batch, "build_astar_feedback", lambda *_args, **_kwargs: "astar reflection")
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _DummyGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
        "--inner-loop",
        "astar",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    assert captured["baseline_called"] is True
    assert captured["holdout_called"] is True
    assert captured["astar_metric_calls"] >= 1


def test_run_astar_sparse_baseline_disables_dense_heuristic(tmp_path, monkeypatch):
    """Verify A* sparse baselines always run without the dense heuristic.

    This test checks the baseline helper's call contract with the shared A*
    evaluator. It is needed because the search baseline defines the comparison
    target for dense heuristics, and it differs from pipeline tests by
    asserting the exact `reward_mode` and `use_dense_heuristic` values passed to
    `run_astar_with_reward`.
    """

    run_reward_batch = _load_run_reward_batch()
    calls: list[dict[str, Any]] = []

    def _capture_run_astar_with_reward(_reward_generator, output_dir=None, **kwargs):
        calls.append(dict(kwargs))
        return _dummy_astar_result("XLand-MiniGrid-R1-9x9", "trivial-1m")

    monkeypatch.setattr(run_reward_batch, "run_astar_with_reward", _capture_run_astar_with_reward)

    job = run_reward_batch.EnvJob(
        name="job-1",
        env_id="XLand-MiniGrid-R1-9x9",
        benchmark_id="trivial-1m",
        total_timesteps=1,
        train_seed=1,
        eval_seed=2,
    )
    baselines, baseline_mean = run_reward_batch.run_astar_sparse_baseline(
        [job],
        logs_root=tmp_path,
        log_wandb=lambda *_args, **_kwargs: None,
        config_clamper=lambda cfg: (dict(cfg), []),
        max_nodes=32,
        max_expansions=32,
    )

    assert calls
    assert calls[0]["reward_mode"] == "sparse"
    assert calls[0]["use_dense_heuristic"] is False
    assert baselines["job-1"]["score"] == pytest.approx(0.875)
    assert baseline_mean == pytest.approx(0.875)


def test_room_count_filters_training_and_holdout_jobs(tmp_path, monkeypatch):
    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "jobs:\n"
        "  - name: train-r1\n"
        "    env_id: XLand-MiniGrid-R1-11x11\n"
        "    benchmark_id: trivial-1m\n"
        "    total_timesteps: 10\n"
        "    train_seed: 1\n"
        "    eval_seed: 2\n"
        "  - name: train-r4\n"
        "    env_id: XLand-MiniGrid-R4-11x11\n"
        "    benchmark_id: trivial-1m\n"
        "    total_timesteps: 10\n"
        "    train_seed: 3\n"
        "    eval_seed: 4\n"
        "eval_jobs:\n"
        "  - name: eval-r1\n"
        "    env_id: XLand-MiniGrid-R1-9x9\n"
        "    benchmark_id: trivial-1m\n"
        "    total_timesteps: 10\n"
        "    train_seed: 5\n"
        "    eval_seed: 6\n"
        "  - name: eval-r4\n"
        "    env_id: XLand-MiniGrid-R4-9x9\n"
        "    benchmark_id: trivial-1m\n"
        "    total_timesteps: 10\n"
        "    train_seed: 7\n"
        "    eval_seed: 8\n",
        encoding="utf-8",
    )

    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"
    captured = {"train_env_ids": [], "holdout_env_ids": []}

    def _fake_run_training_with_reward(_reward_generator, output_dir=None, **kwargs):
        cfg = kwargs.get("config_override", {})
        return _dummy_training_result(
            cfg.get("env_id", "XLand-MiniGrid-R1-9x9"), "trivial-1m"
        )

    def _fake_sparse_baseline(jobs, **_kwargs):
        return ({job.name: {"solve_rate": 0.0} for job in jobs}, 0.0)

    def _fake_holdout_sparse_baselines(holdout_jobs, **_kwargs):
        captured["holdout_env_ids"] = [job.env_id for job in holdout_jobs]
        return ({job.name: {"solve_rate": 0.0} for job in holdout_jobs}, 0.0)

    def _fake_evaluate_dense_on_jobs(*, jobs, **_kwargs):
        captured["holdout_env_ids"] = [job.env_id for job in jobs]
        return (
            {
                job.name: {
                    "solve_rate_mean": 0.0,
                    "solve_rate_std": 0.0,
                    "solve_rates": [0.0],
                }
                for job in jobs
            },
            0.0,
        )

    class _CapturingGEPA(_DummyGEPA):
        def compile(self, program, trainset=None, **_kwargs):
            captured["train_env_ids"] = [
                example.job_config["env_id"] for example in trainset or []
            ]
            return super().compile(program, trainset=trainset, **_kwargs)

    monkeypatch.setattr(run_reward_batch, "RewardGenerator", _DummyRewardGenerator)
    monkeypatch.setattr(
        run_reward_batch, "run_training_with_reward", _fake_run_training_with_reward
    )
    monkeypatch.setattr(
        run_reward_batch, "ensure_sparse_baseline", _fake_sparse_baseline
    )
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_holdout_sparse_baselines",
        _fake_holdout_sparse_baselines,
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        _fake_evaluate_dense_on_jobs,
    )
    monkeypatch.setattr(
        run_reward_batch,
        "build_reward_reflection",
        lambda *_args, **_kwargs: "reflection",
    )
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _CapturingGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
        "--room-count",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    assert captured["train_env_ids"] == ["XLand-MiniGrid-R1-11x11"]
    assert captured["holdout_env_ids"] == ["XLand-MiniGrid-R1-9x9"]


def test_room_count_raises_when_no_matching_training_jobs(tmp_path, monkeypatch):
    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-r1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 1\n"
        "  eval_seed: 2\n",
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
        "--room-count",
        "99",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match=r"--room-count filter removed all jobs"):
        run_reward_batch.run_batch()


def test_gepa_metric_hard_fails_reward_task_mismatch_before_reflection(
    tmp_path, monkeypatch
):
    """Reject task-mismatched rewards instead of passing them to reflection.

    This integration test verifies the new hard-validation policy for semantic
    mismatches between reward object lookups and the task description. It is
    needed because mismatch diagnostics are no longer advisory-only, and it
    differs from older feedback-shape tests by asserting reflection is skipped
    and the primary GEPA metric returns a failure score immediately.
    """
    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 1\n"
        "  eval_seed: 2\n",
        encoding="utf-8",
    )

    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"
    captured: dict[str, object] = {}

    class _MismatchEnvRewardGenerator(_DummyRewardGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_env_description = (
                'Task objects: "blue_square" and "green_pyramid".'
            )

    def _fake_run_training_with_reward(_reward_generator, output_dir=None, **_kwargs):
        output_path = Path(str(output_dir))
        output_path.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_path / "eval_trajectory.json"
        trajectory_path.write_text(
            json.dumps({"actions": [3, 4, 3, 4, 0, 1, 2, 0]}),
            encoding="utf-8",
        )
        config = SimpleNamespace(
            env_id="XLand-MiniGrid-R1-9x9",
            benchmark_id="trivial-1m",
        )
        train_info = {
            "loss_info": {"eval/ground_truth_returns_mean": [0.0, 0.1]},
            "component_logs": {"progress": [0.0, 0.2], "penalty": [0.0, -0.1]},
        }
        return SimpleNamespace(
            config=config,
            train_info=train_info,
            artifacts={"eval_trajectory": str(trajectory_path)},
            final_metrics={"solve_rate": 0.0, "eval_successes": 0},
            emitted_reward_code=(
                "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
                "    object_positions = ctx.get('object_positions', {})\n"
                "    _ = object_positions.get('red_key', jnp.array([-1, -1], dtype=jnp.int32))\n"
                "    reward_components = {'progress': jnp.asarray(0.0, dtype=jnp.float32)}\n"
                "    return reward_components['progress'], reward_components\n"
            ),
            ground_truth_eval={"returns": [0.0], "successes": 0},
        )

    def _fake_sparse_baseline(jobs, **_kwargs):
        return ({job.name: {"solve_rate": 0.0} for job in jobs}, 0.0)

    class _CapturingGEPA(_DummyGEPA):
        def compile(self, program, trainset=None, **_kwargs):
            for example in trainset or []:
                prediction = dspy.Prediction(prompt_text="dummy-prompt")
                captured["primary_metric"] = self.metric(
                    example,
                    prediction,
                    trace=[("predictor", {}, {})],
                    pred_name=None,
                    pred_trace=None,
                )
                self.metric(
                    example,
                    prediction,
                    trace=[("predictor", {}, {})],
                    pred_name="prompt_generator",
                    pred_trace=[("predictor", {}, {})],
                )
            class _DummyPromptGenerator:
                def __call__(self, **_kwargs):
                    return "dummy-prompt"

                def dump_state(self):
                    return {}

            program.prompt_generator = _DummyPromptGenerator()
            return program

    monkeypatch.setattr(
        run_reward_batch, "RewardGenerator", _MismatchEnvRewardGenerator
    )
    monkeypatch.setattr(
        run_reward_batch, "run_training_with_reward", _fake_run_training_with_reward
    )
    monkeypatch.setattr(
        run_reward_batch, "ensure_sparse_baseline", _fake_sparse_baseline
    )
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_holdout_sparse_baselines",
        lambda holdout_jobs, **_kwargs: (
            {job.name: {"solve_rate": 0.0} for job in holdout_jobs},
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        lambda jobs, **_kwargs: (
            {
                job.name: {
                    "solve_rate_mean": 0.0,
                    "solve_rate_std": 0.0,
                    "solve_rates": [0.0],
                }
                for job in jobs
            },
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "build_reward_reflection",
        lambda *_args, **_kwargs: pytest.fail("reflection should not run"),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _CapturingGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    metric_result = captured.get("primary_metric")
    assert metric_result is not None
    assert metric_result["score"] == 0.0
    assert "Reward/task mismatch detected after evaluation" in metric_result["feedback"]


def test_gepa_metric_reports_no_missing_object_keys_when_reward_matches_task(
    tmp_path, monkeypatch
):
    """Verify alignment diagnostics stay empty when reward/task keys match.

    This integration test ensures the GEPA metric path still emits diagnostics
    payloads when key alignment is correct, with `missing_from_task` left empty.
    It is needed to distinguish true mismatch warnings from healthy runs.
    """
    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 1\n"
        "  eval_seed: 2\n",
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"
    captured: dict[str, object] = {}

    class _AlignedEnvRewardGenerator(_DummyRewardGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_env_description = 'Task objects: "blue_square" and "green_pyramid".'

    def _fake_run_training_with_reward(_reward_generator, output_dir=None, **_kwargs):
        config = SimpleNamespace(
            env_id="XLand-MiniGrid-R1-9x9",
            benchmark_id="trivial-1m",
        )
        train_info = {
            "loss_info": {
                "eval/ground_truth_returns_mean": [0.0, 0.1],
                "eval/returns_mean": [0.1, 0.2],
            },
            "component_logs": {"progress": [0.0, 0.1]},
        }
        return SimpleNamespace(
            config=config,
            train_info=train_info,
            artifacts={},
            final_metrics={"solve_rate": 0.0, "eval_successes": 0},
            emitted_reward_code=(
                "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
                "    object_positions = ctx.get('object_positions', {})\n"
                "    _ = object_positions.get('blue_square', jnp.array([-1, -1], dtype=jnp.int32))\n"
                "    reward_components = {'progress': jnp.asarray(0.0, dtype=jnp.float32)}\n"
                "    return reward_components['progress'], reward_components\n"
            ),
            ground_truth_eval={"returns": [0.0], "successes": 0},
        )

    def _fake_sparse_baseline(jobs, **_kwargs):
        return ({job.name: {"solve_rate": 0.0} for job in jobs}, 0.0)

    def _capture_reflection(row, **_kwargs):
        captured["row"] = row
        return "reflection"

    monkeypatch.setattr(
        run_reward_batch, "RewardGenerator", _AlignedEnvRewardGenerator
    )
    monkeypatch.setattr(
        run_reward_batch, "run_training_with_reward", _fake_run_training_with_reward
    )
    monkeypatch.setattr(
        run_reward_batch, "ensure_sparse_baseline", _fake_sparse_baseline
    )
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_holdout_sparse_baselines",
        lambda holdout_jobs, **_kwargs: (
            {job.name: {"solve_rate": 0.0} for job in holdout_jobs},
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        lambda jobs, **_kwargs: (
            {
                job.name: {
                    "solve_rate_mean": 0.0,
                    "solve_rate_std": 0.0,
                    "solve_rates": [0.0],
                }
                for job in jobs
            },
            0.0,
        ),
    )
    monkeypatch.setattr(run_reward_batch, "build_reward_reflection", _capture_reflection)
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _DummyGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    row = captured.get("row")
    assert isinstance(row, dict)
    diagnostics = row.get("reward_object_key_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("missing_from_task") == []


def test_gepa_metric_hard_fails_dead_dense_reward_signal(
    tmp_path, monkeypatch
):
    """Reject dense rewards that emit no usable runtime signal.

    This integration test verifies the RL-only dead-signal gate by constructing
    a run result whose dense returns and all component logs remain identically
    zero. It is needed because task-aligned reward code can still be inert at
    runtime, and it differs from mismatch tests by validating the behavioral
    gate rather than static object-key checks.
    """

    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 1\n"
        "  eval_seed: 2\n",
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"
    captured: dict[str, object] = {}

    class _AlignedEnvRewardGenerator(_DummyRewardGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_env_description = (
                'Your task is to move next to the "blue_square". '
                'Success when the agent is adjacent to the "blue_square".'
            )

    class _CapturingGEPA(_DummyGEPA):
        def compile(self, program, trainset=None, **_kwargs):
            for example in trainset or []:
                prediction = dspy.Prediction(prompt_text="dummy-prompt")
                captured["primary_metric"] = self.metric(
                    example,
                    prediction,
                    trace=[("predictor", {}, {})],
                    pred_name=None,
                    pred_trace=None,
                )
                self.metric(
                    example,
                    prediction,
                    trace=[("predictor", {}, {})],
                    pred_name="prompt_generator",
                    pred_trace=[("predictor", {}, {})],
                )
            class _DummyPromptGenerator:
                def __call__(self, **_kwargs):
                    return "dummy-prompt"

                def dump_state(self):
                    return {}

            program.prompt_generator = _DummyPromptGenerator()
            return program

    def _fake_run_training_with_reward(_reward_generator, output_dir=None, **_kwargs):
        config = SimpleNamespace(
            env_id="XLand-MiniGrid-R1-9x9",
            benchmark_id="trivial-1m",
        )
        train_info = {
            "loss_info": {
                "eval/ground_truth_returns_mean": [0.0, 0.0],
                "eval/returns_mean": [0.0, 0.0],
            },
            "component_logs": {"progress": [0.0, 0.0], "penalty": [0.0, 0.0]},
        }
        return SimpleNamespace(
            config=config,
            train_info=train_info,
            artifacts={},
            final_metrics={"solve_rate": 0.0, "eval_successes": 0},
            emitted_reward_code=(
                "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
                "    object_positions = ctx.get('object_positions', {})\n"
                "    blue_square = object_positions.get('blue_square', jnp.array([-1, -1], dtype=jnp.int32))\n"
                "    progress = blue_square[0].astype(jnp.float32) * jnp.asarray(0.0, dtype=jnp.float32)\n"
                "    reward_components = {'progress': progress, 'penalty': progress}\n"
                "    return progress, reward_components\n"
            ),
            ground_truth_eval={"returns": [0.0], "successes": 0},
        )

    def _fake_sparse_baseline(jobs, **_kwargs):
        return ({job.name: {"solve_rate": 0.0} for job in jobs}, 0.0)

    monkeypatch.setattr(
        run_reward_batch, "RewardGenerator", _AlignedEnvRewardGenerator
    )
    monkeypatch.setattr(
        run_reward_batch, "run_training_with_reward", _fake_run_training_with_reward
    )
    monkeypatch.setattr(
        run_reward_batch, "ensure_sparse_baseline", _fake_sparse_baseline
    )
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_holdout_sparse_baselines",
        lambda holdout_jobs, **_kwargs: (
            {job.name: {"solve_rate": 0.0} for job in holdout_jobs},
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        lambda jobs, **_kwargs: (
            {
                job.name: {
                    "solve_rate_mean": 0.0,
                    "solve_rate_std": 0.0,
                    "solve_rates": [0.0],
                }
                for job in jobs
            },
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "build_reward_reflection",
        lambda *_args, **_kwargs: pytest.fail("reflection should not run"),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _CapturingGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    metric_result = captured.get("primary_metric")
    assert metric_result is not None
    assert metric_result["score"] == 0.0
    assert "Dead dense reward signal detected" in metric_result["feedback"]


def test_gepa_metric_reuses_one_ruleset_but_keeps_yaml_training_and_eval_seeds(
    tmp_path, monkeypatch
):
    """Verify GEPA candidates lock task semantics but keep YAML rollout seeds.

    This integration test captures the exact `config_override` passed from the
    on-policy GEPA metric into `run_training_with_reward`. It is needed because
    the repository's intended contract is subtle: one candidate prompt should
    generate a reward from a single task description and then train/evaluate on
    that same task instance. The current implementation also logs derived
    prompt-dependent seeds inside the metric, but the actual training call keeps
    the YAML `train_seed` and `eval_seed` because those fields are already
    present in the job config. It differs from broader pipeline smoke tests by
    asserting the precise deterministic-ruleset and seed-plumbing fields that
    govern the behavior that actually occurs today.
    """

    run_reward_batch = _load_run_reward_batch()

    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "- name: job-1\n"
        "  env_id: XLand-MiniGrid-R1-9x9\n"
        "  benchmark_id: trivial-1m\n"
        "  total_timesteps: 10\n"
        "  train_seed: 123\n"
        "  eval_seed: 456\n",
        encoding="utf-8",
    )

    state_root = tmp_path / "state"
    os.environ["WANDB_DISABLED"] = "1"
    captured: dict[str, object] = {"config_overrides": []}

    def _fake_run_training_with_reward(_reward_generator, output_dir=None, **kwargs):
        config_overrides = captured["config_overrides"]
        assert isinstance(config_overrides, list)
        config_overrides.append(dict(kwargs.get("config_override", {})))
        return _dummy_training_result("XLand-MiniGrid-R1-9x9", "trivial-1m")

    def _fake_sparse_baseline(jobs, **_kwargs):
        return ({job.name: {"solve_rate": 0.0} for job in jobs}, 0.0)

    monkeypatch.setattr(run_reward_batch, "RewardGenerator", _DummyRewardGenerator)
    monkeypatch.setattr(
        run_reward_batch, "run_training_with_reward", _fake_run_training_with_reward
    )
    monkeypatch.setattr(
        run_reward_batch, "ensure_sparse_baseline", _fake_sparse_baseline
    )
    monkeypatch.setattr(
        run_reward_batch,
        "ensure_holdout_sparse_baselines",
        lambda holdout_jobs, **_kwargs: (
            {job.name: {"solve_rate": 0.0} for job in holdout_jobs},
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "evaluate_dense_on_jobs",
        lambda jobs, **_kwargs: (
            {
                job.name: {
                    "solve_rate_mean": 0.0,
                    "solve_rate_std": 0.0,
                    "solve_rates": [0.0],
                }
                for job in jobs
            },
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_reward_batch,
        "build_reward_reflection",
        lambda *_args, **_kwargs: "reflection",
    )
    monkeypatch.setattr(
        run_reward_batch,
        "create_reward_reflection_module",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(run_reward_batch.dspy, "GEPA", _DummyGEPA)

    argv = [
        "run_reward_batch.py",
        "--state-root",
        str(state_root),
        "--env-grid",
        str(env_grid),
        "--max-gepa-iterations",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_reward_batch.run_batch()

    config_overrides = captured.get("config_overrides")
    assert isinstance(config_overrides, list)
    assert config_overrides
    config_override = config_overrides[0]
    assert isinstance(config_override, dict)
    assert config_override["env_id"] == "XLand-MiniGrid-R1-9x9"
    assert config_override["benchmark_id"] == "trivial-1m"
    assert config_override["deterministic_rulesets"] is True
    assert isinstance(config_override.get("fixed_ruleset_seed"), int)
    assert config_override["train_seed"] == 123
    assert config_override["eval_seed"] == 456
