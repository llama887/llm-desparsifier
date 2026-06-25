"""Focused tests for the standalone batched PuzzleScript GEPA path."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_puzzlescript_batched_gepa import (
    assigned_tasks,
    build_sbatch_array_command,
    context_retry_max_tokens,
    strip_outer_markdown_fences,
    validate_heuristic_code,
)


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
