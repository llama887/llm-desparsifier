"""Focused tests for the standalone batched PuzzleScript GEPA path."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.compare_puzzlescript_batched_prompts import compare_prompt_outputs
from scripts.run_puzzlescript_batched_gepa import (
    DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS,
    DEFAULT_REFLECTION_FEEDBACK_CHARS,
    DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS,
    DEFAULT_REFLECTION_MAX_RECORDS,
    PuzzleScriptBatchedGEPAAdapter,
    assigned_tasks,
    build_sbatch_array_command,
    build_train_dev_tasks,
    candidate_score,
    context_retry_max_tokens,
    evaluate_search_task_with_wall_timeout,
    select_reflection_traces,
    split_train_dev_jobs,
    strip_outer_markdown_fences,
    validate_heuristic_code,
)


def _stuck_search_task_worker(
    _script_doctor: str,
    _task: dict[str, object],
    _astar_timeout_s: float,
    _result_queue: object,
) -> None:
    time.sleep(10.0)


def test_strip_outer_markdown_fences_accepts_python_fence() -> None:
    code = "```python\ndef heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0\n```"

    assert strip_outer_markdown_fences(code) == (
        "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0"
    )


def test_validate_heuristic_code_accepts_expected_signature() -> None:
    code = """
def heuristic_cost_to_go(ts, env_params, ctx):
    if ctx.get("is_winning"):
        return 0.0
    return float(ctx.get("score", 0.0))
"""

    assert validate_heuristic_code(code) is None


def test_validate_heuristic_code_rejects_imports() -> None:
    code = """
import os

def heuristic_cost_to_go(ts, env_params, ctx):
    return 0.0
"""

    issue = validate_heuristic_code(code)

    assert issue is not None
    assert "imports are not allowed" in issue


def test_assigned_tasks_uses_round_robin_partitioning() -> None:
    tasks = [{"task_id": idx} for idx in range(10)]

    assert [row["task_id"] for row in assigned_tasks(tasks, 0, 3)] == [0, 3, 6, 9]
    assert [row["task_id"] for row in assigned_tasks(tasks, 1, 3)] == [1, 4, 7]
    assert [row["task_id"] for row in assigned_tasks(tasks, 2, 3)] == [2, 5, 8]


def test_assigned_tasks_rejects_invalid_index() -> None:
    with pytest.raises(ValueError):
        assigned_tasks([{"task_id": 0}], 3, 3)


def test_build_sbatch_array_command_exports_manifest_and_count() -> None:
    command = build_sbatch_array_command(
        manifest_path=Path("/tmp/eval/search_manifest.json"),
        array_script=Path("sbatch/evaluate_puzzlescript_search_array.s"),
        array_count=8,
        array_concurrency=3,
        extra_sbatch_args=("--time=01:00:00",),
    )

    assert command[:4] == ["sbatch", "--wait", "--parsable", "--array=0-7%3"]
    assert command[4] == (
        "--export=ALL,EVAL_MANIFEST=/tmp/eval/search_manifest.json,SEARCH_ARRAY_COUNT=8"
    )
    assert "--time=01:00:00" in command
    assert command[-1] == "sbatch/evaluate_puzzlescript_search_array.s"


def test_split_train_dev_jobs_is_deterministic_and_non_overlapping() -> None:
    jobs = [{"name": f"game-{idx:02d}"} for idx in range(10)]

    train_a, dev_a = split_train_dev_jobs(jobs, dev_fraction=0.3, seed=17)
    train_b, dev_b = split_train_dev_jobs(jobs, dev_fraction=0.3, seed=17)

    assert train_a == train_b
    assert dev_a == dev_b
    assert len(train_a) == 7
    assert len(dev_a) == 3
    assert {job["name"] for job in train_a}.isdisjoint({job["name"] for job in dev_a})
    assert sorted(job["name"] for job in train_a + dev_a) == [job["name"] for job in jobs]


def test_build_train_dev_tasks_reassigns_task_ids_after_split() -> None:
    tasks = [
        SimpleNamespace(task_id=idx, game=f"game-{idx}", level=0)
        for idx in range(5)
    ]

    train_tasks, dev_tasks = build_train_dev_tasks(tasks, dev_fraction=0.4, seed=3)

    assert [task.task_id for task in train_tasks] == list(range(len(train_tasks)))
    assert [task.task_id for task in dev_tasks] == list(range(len(dev_tasks)))
    assert {task.game for task in train_tasks}.isdisjoint({task.game for task in dev_tasks})


def test_candidate_score_penalizes_lost_solves_and_errors() -> None:
    outputs = [
        {"score": 0.8, "solved": True, "baseline_solved": True, "error": None},
        {"score": 0.9, "solved": False, "baseline_solved": True, "error": None},
        {"score": 0.5, "solved": True, "baseline_solved": False, "error": None},
        {"score": 1.0, "solved": False, "baseline_solved": False, "error": "exit is not allowed"},
    ]

    score = candidate_score(outputs, lost_solve_penalty=0.25, error_penalty=0.1)

    assert score == pytest.approx(((0.8 + 0.9 + 0.5 + 1.0) / 4) - 0.25 - 0.1)


def test_context_retry_max_tokens_uses_reported_prompt_tokens() -> None:
    message = (
        "This model's maximum context length is 32768 tokens. However, you requested "
        "8192 output tokens and your prompt contains at least 24577 input tokens, "
        "for a total of at least 32769 tokens. (parameter=input_tokens, value=24577)"
    )

    retry_tokens = context_retry_max_tokens(
        message,
        current_max_tokens=8192,
        retry_margin_tokens=64,
        min_retry_tokens=256,
    )

    assert retry_tokens == 8127


def test_context_retry_max_tokens_default_keeps_headroom_for_token_recount() -> None:
    first_message = (
        "This model's maximum context length is 32768 tokens. However, you requested "
        "8192 output tokens and your prompt contains at least 24577 input tokens, "
        "for a total of at least 32769 tokens."
    )
    second_message = (
        "This model's maximum context length is 32768 tokens. However, you requested "
        "7679 output tokens and your prompt contains at least 24642 input tokens, "
        "for a total of at least 32321 tokens."
    )

    assert context_retry_max_tokens(first_message, current_max_tokens=8192) == 7679
    assert context_retry_max_tokens(second_message, current_max_tokens=7679) == 7614


def test_make_reflective_dataset_compacts_large_trace_payloads() -> None:
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=object(),  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    eval_batch = SimpleNamespace(
        trajectories=[
            {
                "task": {
                    "game": "large-game",
                    "level": 0,
                    "budget": 100,
                    "env_description": "e" * (DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS + 100),
                },
                "heuristic_code": "h" * (DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS + 100),
                "synthesis_error": None,
                "result": {
                    "feedback": "f" * (DEFAULT_REFLECTION_FEEDBACK_CHARS + 100),
                    "score": 0.25,
                    "solved": False,
                },
            }
        ]
    )

    dataset = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["heuristic_prompt"],
    )

    record = dataset["heuristic_prompt"][0]
    assert len(record["Inputs"]["env_description"]) < DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS + 80
    assert len(record["Generated Outputs"]["heuristic_code"]) < DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS + 80
    assert len(record["Feedback"]) < DEFAULT_REFLECTION_FEEDBACK_CHARS + 80
    assert "[truncated 100 chars]" in record["Inputs"]["env_description"]
    assert "all levels still contributed to the scalar score" in record["Selection"]


def test_select_reflection_traces_keeps_lowest_scoring_failures() -> None:
    trajectories = [
        {
            "task": {"game": f"game-{idx:02d}", "level": idx},
            "result": {"score": float(idx), "solved": idx % 2 == 0},
        }
        for idx in range(DEFAULT_REFLECTION_MAX_RECORDS + 6)
    ]

    selected = select_reflection_traces(trajectories)

    assert len(selected) == DEFAULT_REFLECTION_MAX_RECORDS
    assert [trace["task"]["level"] for trace in selected[:3]] == [1, 3, 5]
    assert all(not trace["result"]["solved"] for trace in selected[:10])


def test_h100_launcher_defaults_to_extended_vllm_context() -> None:
    launcher = Path("sbatch/train_puzzlescript_batched_gepa_gpu.s").read_text(encoding="utf-8")

    assert 'VLLM_MAX_MODEL_LEN:-65536' in launcher
    assert '--val-split "${VAL_SPLIT:-dev}"' in launcher
    assert '--max-gepa-iterations "${MAX_GEPA_ITERATIONS:-16}"' in launcher
    assert 'RUN_HOLDOUT_COMPARE:-1' in launcher
    assert "scripts/compare_puzzlescript_batched_prompts.py" in launcher


def test_search_array_launcher_skips_locked_setup_when_runtime_exists() -> None:
    launcher = Path("sbatch/evaluate_puzzlescript_search_array.s").read_text(encoding="utf-8")

    assert "runtime_ready()" in launcher
    assert "ensure_runtime()" in launcher
    assert "#SBATCH --mem=8G" in launcher
    assert "if runtime_ready; then" in launcher
    assert "[setup] using existing PuzzleScript runtime" in launcher
    assert "else\n    ensure_runtime\nfi" in launcher


def test_evaluate_search_task_with_wall_timeout_terminates_stuck_worker(tmp_path: Path) -> None:
    task = {
        "task_id": 7,
        "game": "stuck-game",
        "level": 2,
        "heuristic_code_path": str(tmp_path / "heuristic.py"),
    }

    result = evaluate_search_task_with_wall_timeout(
        script_doctor=tmp_path,
        task=task,
        astar_timeout_s=1.0,
        wall_timeout_s=0.1,
        worker=_stuck_search_task_worker,
    )

    assert result["task_id"] == 7
    assert result["score"] == 0.0
    assert "wall timeout" in str(result["error"])


def test_compare_prompt_outputs_reports_holdout_deltas() -> None:
    base_outputs = [
        {"task_id": 0, "game": "a", "level": 0, "score": 0.2, "solved": False, "expanded": 10},
        {"task_id": 1, "game": "a", "level": 1, "score": 0.8, "solved": True, "expanded": 2},
        {"task_id": 2, "game": "b", "level": 0, "score": 0.4, "solved": False, "expanded": 9},
    ]
    optimized_outputs = [
        {"task_id": 0, "game": "a", "level": 0, "score": 0.5, "solved": True, "expanded": 4},
        {"task_id": 1, "game": "a", "level": 1, "score": 0.7, "solved": True, "expanded": 3},
        {"task_id": 2, "game": "b", "level": 0, "score": 0.1, "solved": False, "expanded": 12},
    ]

    aggregate, per_level, per_game = compare_prompt_outputs(
        base_outputs=base_outputs,
        optimized_outputs=optimized_outputs,
    )

    assert aggregate["base"]["solved"] == 1
    assert aggregate["optimized"]["solved"] == 2
    assert aggregate["new_solve_count"] == 1
    assert aggregate["lost_solve_count"] == 0
    assert aggregate["better_score_count"] == 1
    assert aggregate["worse_score_count"] == 2
    assert aggregate["solved_delta"] == 1
    assert per_level[0]["score_delta"] == pytest.approx(0.3)
    assert per_game[0]["game"] == "a"
    assert per_game[0]["solved_delta"] == 1


def test_compare_prompt_outputs_can_write_plot_artifacts(tmp_path: Path) -> None:
    from scripts.compare_puzzlescript_batched_prompts import write_comparison_plots

    per_game = [
        {
            "game": "gain",
            "n": 2,
            "base_score_mean": 0.2,
            "optimized_score_mean": 0.5,
            "score_delta": 0.3,
            "base_solved": 0,
            "optimized_solved": 1,
            "solved_delta": 1,
            "better_score_count": 2,
            "worse_score_count": 0,
            "new_solve_count": 1,
            "lost_solve_count": 0,
        },
        {
            "game": "loss",
            "n": 2,
            "base_score_mean": 0.8,
            "optimized_score_mean": 0.4,
            "score_delta": -0.4,
            "base_solved": 2,
            "optimized_solved": 1,
            "solved_delta": -1,
            "better_score_count": 0,
            "worse_score_count": 2,
            "new_solve_count": 0,
            "lost_solve_count": 1,
        },
    ]

    paths = write_comparison_plots(output_dir=tmp_path, per_game=per_game)

    assert {path.name for path in paths} == {
        "holdout_score_delta_by_game.png",
        "holdout_solve_delta_by_game.png",
        "holdout_score_base_vs_optimized.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
