"""Focused tests for the standalone batched PuzzleScript GEPA path."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_puzzlescript_batched_gepa import (
    DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS,
    DEFAULT_REFLECTION_FEEDBACK_CHARS,
    DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS,
    DEFAULT_REFLECTION_MAX_RECORDS,
    PuzzleScriptBatchedGEPAAdapter,
    assigned_tasks,
    build_sbatch_array_command,
    context_retry_max_tokens,
    evaluate_search_task_with_wall_timeout,
    select_reflection_traces,
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


def test_search_array_launcher_skips_locked_setup_when_runtime_exists() -> None:
    launcher = Path("sbatch/evaluate_puzzlescript_search_array.s").read_text(encoding="utf-8")

    assert "runtime_ready()" in launcher
    assert "ensure_runtime()" in launcher
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
