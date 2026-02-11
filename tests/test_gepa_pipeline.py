from __future__ import annotations

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


def test_job_from_example_requires_job_config():
    run_reward_batch = _load_run_reward_batch()
    example = dspy.Example(env_description="env")
    with pytest.raises(ValueError):
        run_reward_batch.job_from_example(example)


def test_job_from_example_uses_job_config_and_name():
    run_reward_batch = _load_run_reward_batch()
    example = dspy.Example(env_description="env").with_inputs("env_description")
    example.job_name = "job-7"
    example.job_config = {
        "env_id": "XLand-MiniGrid-DoorKey-5x5",
        "benchmark_id": "trivial-1m",
        "total_timesteps": 123,
        "train_seed": 3,
        "eval_seed": 4,
        "name": "explicit-name",
    }
    job = run_reward_batch.job_from_example(example)
    assert job.name == "explicit-name"
    assert job.env_id == "XLand-MiniGrid-DoorKey-5x5"
    assert job.total_timesteps == 123


def test_format_feedback_prefixes_pred_name():
    run_reward_batch = _load_run_reward_batch()
    text = run_reward_batch.format_feedback("hello", pred_name="prompt_generator", pred_trace=None)
    assert text.startswith("[Predictor feedback: prompt_generator]")
    assert text.endswith("hello")


def test_format_feedback_adds_trace_note():
    run_reward_batch = _load_run_reward_batch()
    text = run_reward_batch.format_feedback("hello", pred_name=None, pred_trace={"trace": True})
    assert text.startswith("[Predictor trace present]")


def test_prompt_only_program_returns_prompt_text():
    run_reward_batch = _load_run_reward_batch()

    program = run_reward_batch.PromptOnlyProgram("base-constraints")

    class _DummyRewriter:
        def __call__(self, **_kwargs):
            return SimpleNamespace(prompt_text="rewritten")

        def dump_state(self):
            return {}

    program.prompt_generator.rewriter = _DummyRewriter()
    prediction = program.forward(env_description="env")
    assert getattr(prediction, "prompt_text") == "rewritten"
    assert not hasattr(prediction, "reward_code")


def test_clamp_job_budget_caps_values():
    run_reward_batch = _load_run_reward_batch()
    cfg = {
        "total_timesteps": run_reward_batch.MAX_TOTAL_TIMESTEPS + 1,
        "num_envs": run_reward_batch.MAX_NUM_ENVS + 2,
        "eval_num_envs": run_reward_batch.MAX_EVAL_ENVS + 3,
        "eval_num_episodes": run_reward_batch.MAX_EVAL_EPISODES + 4,
    }
    capped, notes = run_reward_batch.clamp_job_budget(cfg)
    assert capped["total_timesteps"] == run_reward_batch.MAX_TOTAL_TIMESTEPS
    assert capped["num_envs"] == run_reward_batch.MAX_NUM_ENVS
    assert capped["eval_num_envs"] == run_reward_batch.MAX_EVAL_ENVS
    assert capped["eval_num_episodes"] == run_reward_batch.MAX_EVAL_EPISODES
    assert notes


def test_build_examples_attaches_job_config():
    run_reward_batch = _load_run_reward_batch()
    job = run_reward_batch.EnvJob(
        name="job-1",
        env_id="EnvA",
        benchmark_id="bench",
        total_timesteps=1,
        train_seed=2,
        eval_seed=3,
    )
    examples = run_reward_batch.build_examples([job], constraints_text="constraints")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.job_name == "job-1"
    assert ex.job_config["env_id"] == "EnvA"
    assert "name" in ex.job_config


def test_load_prompt_payload_prefers_active_prompt(tmp_path):
    run_reward_batch = _load_run_reward_batch()
    state_root = tmp_path / "state"
    state_root.mkdir()
    active_path = run_reward_batch.get_active_prompt_path(state_root)
    active_path.write_text(
        '{"constraints_text": "from-active", "prompt_state": {"a": 1}}',
        encoding="utf-8",
    )
    text, prompt_state, meta = run_reward_batch.load_prompt_payload(state_root)
    assert text == "from-active"
    assert prompt_state == {"a": 1}
    assert meta["source"] == "active_prompt"


def test_extract_room_count_parses_valid_env_ids():
    run_reward_batch = _load_run_reward_batch()
    assert run_reward_batch.extract_room_count("XLand-MiniGrid-R1-9x9") == 1
    assert run_reward_batch.extract_room_count("XLand-MiniGrid-R4-13x13") == 4
    assert run_reward_batch.extract_room_count("XLand-MiniGrid-R9-25x25") == 9


def test_extract_room_count_rejects_invalid_env_id():
    run_reward_batch = _load_run_reward_batch()
    with pytest.raises(ValueError, match="Could not parse room count"):
        run_reward_batch.extract_room_count("XLand-MiniGrid-9x9")


def test_filter_jobs_by_room_count_keeps_matching_jobs():
    run_reward_batch = _load_run_reward_batch()
    jobs = [
        run_reward_batch.EnvJob(
            name="job-r1",
            env_id="XLand-MiniGrid-R1-9x9",
            benchmark_id="trivial-1m",
            total_timesteps=1,
            train_seed=1,
            eval_seed=2,
        ),
        run_reward_batch.EnvJob(
            name="job-r4",
            env_id="XLand-MiniGrid-R4-9x9",
            benchmark_id="trivial-1m",
            total_timesteps=1,
            train_seed=3,
            eval_seed=4,
        ),
    ]
    filtered = run_reward_batch.filter_jobs_by_room_count(
        jobs,
        [1],
        section_name="training jobs",
    )
    assert [job.name for job in filtered] == ["job-r1"]
    assert filtered[0].train_seed == 1
    assert filtered[0].eval_seed == 2


def test_filter_jobs_by_room_count_raises_when_empty():
    run_reward_batch = _load_run_reward_batch()
    jobs = [
        run_reward_batch.EnvJob(
            name="job-r2",
            env_id="XLand-MiniGrid-R2-9x9",
            benchmark_id="trivial-1m",
            total_timesteps=1,
            train_seed=1,
            eval_seed=2,
        )
    ]
    with pytest.raises(
        ValueError,
        match=r"--room-count filter removed all jobs from training jobs",
    ):
        run_reward_batch.filter_jobs_by_room_count(
            jobs,
            [1],
            section_name="training jobs",
        )
