from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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

    def _fake_run_training_with_reward(_reward_generator, _output_dir, **_kwargs):
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
