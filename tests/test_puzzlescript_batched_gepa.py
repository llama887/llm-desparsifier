"""Focused tests for the standalone batched PuzzleScript GEPA path."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.compare_puzzlescript_batched_prompts import compare_prompt_outputs
from scripts.run_puzzlescript_batched_gepa import (
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS,
    DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS,
    DEFAULT_REFLECTION_FEEDBACK_CHARS,
    DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS,
    DEFAULT_REFLECTION_MAX_RECORDS,
    GEPA_ADDENDUM_HEADER,
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    PuzzleScriptBatchedGEPAAdapter,
    PuzzleScriptLevelTask,
    SearchArrayConfig,
    SearchArrayStalledError,
    adjusted_candidate_scores,
    assigned_tasks,
    build_reflection_feedback,
    build_sbatch_array_command,
    build_train_dev_tasks,
    candidate_prompt_issue,
    candidate_score,
    context_retry_max_tokens,
    evaluate_search_task_with_wall_timeout,
    heuristic_code_shape,
    merge_validation_guard_tasks,
    parse_guard_level_selection,
    select_reflection_traces,
    split_train_dev_jobs,
    strip_outer_markdown_fences,
    validate_heuristic_code,
    wait_for_shards,
)


def _stuck_search_task_worker(
    _script_doctor: str,
    _task: dict[str, object],
    _astar_timeout_s: float,
    _result_queue: object,
) -> None:
    time.sleep(10.0)


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


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

    assert command[:3] == ["sbatch", "--parsable", "--array=0-7%3"]
    assert "--wait" not in command
    assert command[3] == (
        "--export=ALL,EVAL_MANIFEST=/tmp/eval/search_manifest.json,SEARCH_ARRAY_COUNT=8"
    )
    assert "--time=01:00:00" in command
    assert command[-1] == "sbatch/evaluate_puzzlescript_search_array.s"


def test_wait_for_shards_raises_with_missing_indices_on_stall(tmp_path: Path) -> None:
    (tmp_path / "task-0000-of-0003.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SearchArrayStalledError) as exc_info:
        wait_for_shards(
            shard_dir=tmp_path,
            array_count=3,
            poll_interval_s=0.01,
            stall_timeout_s=0.01,
        )

    assert exc_info.value.present_count == 1
    assert exc_info.value.missing_indices == [1, 2]


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


def test_candidate_score_applies_eval_wide_gate_for_lost_solves_and_errors() -> None:
    outputs = [
        {
            "game": "a",
            "score": 0.9,
            "solved": True,
            "baseline_score": 0.8,
            "baseline_solved": True,
            "error": None,
        },
        {
            "game": "a",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.7,
            "baseline_solved": True,
            "error": None,
        },
        {
            "game": "b",
            "score": 0.6,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "error": None,
        },
        {
            "game": "b",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "error": "exit is not allowed",
        },
    ]

    scores = adjusted_candidate_scores(
        outputs,
        lost_solve_penalty=4.0,
        new_solve_bonus=1.0,
        error_penalty=2.0,
        score_delta_weight=0.25,
        score_delta_clip=0.5,
    )
    score = candidate_score(
        outputs,
        lost_solve_penalty=4.0,
        new_solve_bonus=1.0,
        error_penalty=2.0,
        score_delta_weight=0.25,
        score_delta_clip=0.5,
    )

    assert scores == pytest.approx(
        [
            0.25 * 0.1 - 4.0,
            -4.0 + 0.25 * -0.5 - 4.0,
            1.0 + 0.25 * 0.5 - 4.0,
            -2.0 - 4.0,
        ]
    )
    assert score == pytest.approx(sum(scores) / len(scores))


def test_candidate_score_can_disable_eval_wide_lost_solve_gate() -> None:
    outputs = [
        {
            "game": "a",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.5,
            "baseline_solved": True,
        },
        {
            "game": "a",
            "score": 0.7,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": False,
        },
    ]

    scores = adjusted_candidate_scores(
        outputs,
        lost_solve_penalty=2.0,
        new_solve_bonus=3.0,
        score_delta_weight=1.0,
        score_delta_clip=0.5,
        global_lost_solve_gate_penalty=0.0,
    )

    assert scores == pytest.approx([-2.5, 3.5])
    assert candidate_score(
        outputs,
        lost_solve_penalty=2.0,
        new_solve_bonus=3.0,
        score_delta_weight=1.0,
        score_delta_clip=0.5,
        global_lost_solve_gate_penalty=0.0,
    ) == pytest.approx(0.5)


def test_candidate_score_gates_net_solve_losses_when_lost_gate_is_disabled() -> None:
    outputs = [
        {
            "game": "a",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.5,
            "baseline_solved": True,
        },
        {
            "game": "b",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.5,
            "baseline_solved": True,
        },
        {
            "game": "c",
            "score": 0.7,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": False,
        },
    ]

    scores = adjusted_candidate_scores(
        outputs,
        lost_solve_penalty=2.0,
        new_solve_bonus=3.0,
        score_delta_weight=1.0,
        score_delta_clip=0.5,
        global_lost_solve_gate_penalty=0.0,
        global_net_solve_loss_gate_penalty=4.0,
    )

    assert scores == pytest.approx([-6.5, -6.5, -0.5])


def test_candidate_scores_macro_weight_games() -> None:
    outputs = [
        {
            "game": "large",
            "score": 0.1,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": True,
        },
        {
            "game": "large",
            "score": 0.1,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": True,
        },
        {
            "game": "small",
            "score": 0.5,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": False,
        },
    ]

    scores = adjusted_candidate_scores(
        outputs,
        lost_solve_penalty=4.0,
        new_solve_bonus=1.0,
        error_penalty=2.0,
        score_delta_weight=1.0,
        score_delta_clip=1.0,
    )

    assert scores == pytest.approx([0.075, 0.075, 2.25])
    assert sum(scores) / len(scores) == pytest.approx((0.1 + 1.5) / 2)


def test_candidate_scores_use_partial_progress_when_both_prompts_fail() -> None:
    outputs = [
        {
            "game": "same",
            "score": 0.0,
            "solved": False,
            "partial_progress_score": 0.65,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "baseline_partial_progress_score": 0.25,
        }
    ]

    scores = adjusted_candidate_scores(
        outputs,
        lost_solve_penalty=4.0,
        new_solve_bonus=1.0,
        error_penalty=2.0,
        score_delta_weight=1.0,
        score_delta_clip=1.0,
        partial_progress_weight=0.05,
    )

    assert scores == pytest.approx([0.02])


def test_parse_guard_level_selection_accepts_exact_game_levels() -> None:
    selection = parse_guard_level_selection(
        "Not_Normal_Crates:5,11,12; Ice_Cubes:5,7 ; Beam_Islands:3"
    )

    assert selection == {
        "Not_Normal_Crates": [5, 11, 12],
        "Ice_Cubes": [5, 7],
        "Beam_Islands": [3],
    }


def test_parse_guard_level_selection_rejects_malformed_entries() -> None:
    with pytest.raises(ValueError, match="game:level"):
        parse_guard_level_selection("Not_Normal_Crates")


def test_merge_validation_guard_tasks_deduplicates_and_reassigns_ids() -> None:
    dev_tasks = [
        PuzzleScriptLevelTask(3, "dev", 0, 10, "dev0", "dev.txt"),
        PuzzleScriptLevelTask(4, "shared", 1, 10, "dev-shared", "shared.txt"),
    ]
    guard_tasks = [
        PuzzleScriptLevelTask(99, "shared", 1, 20, "guard-shared", "shared.txt"),
        PuzzleScriptLevelTask(100, "guard", 5, 20, "guard5", "guard.txt"),
    ]

    merged = merge_validation_guard_tasks(dev_tasks, guard_tasks)

    assert [(task.task_id, task.game, task.level, task.env_description) for task in merged] == [
        (0, "dev", 0, "dev0"),
        (1, "shared", 1, "dev-shared"),
        (2, "guard", 5, "guard5"),
    ]


def test_heuristic_code_shape_flags_generic_fallback_and_mechanics_terms() -> None:
    shape = heuristic_code_shape(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    role_positions = {}\n"
        "    if not role_positions: return (1.0 - ctx.get('score_normalized', 0.0)) * 10.0\n"
        "    return 1000.0 if 'door' else 0.0\n"
    )

    assert shape["uses_generic_role_helpers"] is True
    assert shape["uses_score_fallback"] is True
    assert shape["uses_large_penalty"] is True
    assert shape["mentions_mechanics_terms"] is True


def test_candidate_prompt_issue_rejects_code_but_allows_contract_signature() -> None:
    instruction_prompt = (
        "Output exactly Python code defining:\n"
        "def heuristic_cost_to_go(ts, env_params, ctx) -> float\n"
        "Explain the constraints before writing the function."
    )
    code_prompt = (
        "def heuristic_cost_to_go(ts, env_params, ctx) -> float:\n"
        "    if ctx.get('is_winning'):\n"
        "        return 0.0\n"
        "    return 1.0\n"
    )

    assert candidate_prompt_issue(instruction_prompt) is None
    assert "Python implementation" in str(candidate_prompt_issue(code_prompt))


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


def test_adapter_eval_counter_resumes_after_existing_eval_dirs(tmp_path: Path) -> None:
    candidate_eval_root = tmp_path / "candidate_evals"
    (candidate_eval_root / "eval-00035-existing").mkdir(parents=True)
    (candidate_eval_root / "eval-not-a-counter").mkdir()
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=_FakeLLM(""),
        state_root=tmp_path,
        script_doctor=tmp_path,
        search_config=SearchArrayConfig(
            submit=False,
            array_script=tmp_path / "unused.s",
            array_count=1,
            array_concurrency=1,
            poll_interval_s=0.01,
        ),
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="game",
        level=0,
        budget=1,
        env_description="",
        game_text_path="game.txt",
    )

    eval_dir = adapter._next_eval_dir({"heuristic_prompt": "prompt"}, [task])

    assert eval_dir.name.startswith("eval-00036-")


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
    assert record["Comparison"]["classification"] == "persistent_failure"


def test_make_reflective_dataset_includes_regression_comparison(tmp_path: Path) -> None:
    base_code = tmp_path / "base.py"
    base_code.write_text("def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n")
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
                    "game": "regressed-game",
                    "level": 4,
                    "budget": 100,
                    "env_description": "rules",
                },
                "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 2.0\n",
                "synthesis_error": None,
                "result": {
                    "feedback": "candidate exhausted search",
                    "score": 0.0,
                    "solved": False,
                    "expanded": 100,
                    "baseline_score": 0.75,
                    "baseline_solved": True,
                    "baseline_expanded": 12,
                    "baseline_heuristic_code_path": str(base_code),
                    "baseline_feedback": "base solved quickly",
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
    assert record["Comparison"]["classification"] == "lost_baseline_solve"
    assert "REGRESSION" in record["Feedback"]
    assert "base prompt solved" in record["Feedback"]
    assert "candidate failed" in record["Feedback"]
    assert "base solved quickly" in record["Baseline Output"]["feedback"]
    assert "return 1.0" in record["Baseline Output"]["heuristic_code"]


def test_custom_proposer_requests_short_base_anchored_addendum() -> None:
    llm = _FakeLLM(
        "Addendum: keep mechanics-specific object names primary and use score only as a tie-breaker."
    )
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Feedback": "REGRESSION: base prompt solved but candidate failed.",
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"].startswith(
        PUZZLESCRIPT_HEURISTIC_CONTRACT.rstrip() + "\n\n" + GEPA_ADDENDUM_HEADER
    )
    assert "keep mechanics-specific object names primary" in result["heuristic_prompt"]
    assert "short addendum" in llm.prompts[0]
    assert "Do not rewrite the full base prompt" in llm.prompts[0]
    assert "Do not return the base prompt unchanged" in llm.prompts[0]
    assert "missing goal objects" in llm.prompts[0]
    assert "REGRESSION" in llm.prompts[0]


def test_custom_proposer_extracts_addendum_from_full_prompt_output() -> None:
    llm = _FakeLLM(
        PUZZLESCRIPT_HEURISTIC_CONTRACT
        + "\n\n"
        + GEPA_ADDENDUM_HEADER
        + "\nPrefer exact LEGEND names before any generic role fallback."
    )
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "Prefer exact LEGEND names" in result["heuristic_prompt"]


def test_custom_proposer_uses_feedback_fallback_for_noop_output() -> None:
    llm = _FakeLLM(PUZZLESCRIPT_HEURISTIC_CONTRACT)
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {"classification": "persistent_failure"},
                    "Feedback": "Persistent failure: neither prompt solved this level.",
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "low-weight win-condition distance" in result["heuristic_prompt"]
    assert "missing goal objects" in result["heuristic_prompt"]


def test_custom_proposer_uses_code_contract_fallback_for_candidate_errors() -> None:
    llm = _FakeLLM(PUZZLESCRIPT_HEURISTIC_CONTRACT)
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {"classification": "candidate_error"},
                    "Feedback": "CANDIDATE ERROR: imports are not allowed.",
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "No import statements" in result["heuristic_prompt"]
    assert "decorators" in result["heuristic_prompt"]


def test_custom_proposer_rejects_code_as_revised_prompt() -> None:
    llm = _FakeLLM(
        "def heuristic_cost_to_go(ts, env_params, ctx) -> float:\n"
        "    return 0.0\n"
    )
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": "current instruction prompt"},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] == "current instruction prompt"
    assert "Do not output Python code as the revised prompt" in llm.prompts[0]


def test_custom_proposer_rejects_overlong_addendum_instead_of_truncating() -> None:
    current_prompt = PUZZLESCRIPT_HEURISTIC_CONTRACT
    llm = _FakeLLM("x" * (DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS + 1))
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] == current_prompt


def test_custom_proposer_rejects_dangling_addendum_tail() -> None:
    current_prompt = PUZZLESCRIPT_HEURISTIC_CONTRACT
    llm = _FakeLLM("Use mechanics-specific blockers:\n-")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] == current_prompt


def test_base_prompt_evaluation_reuses_stored_baseline_outputs(tmp_path: Path) -> None:
    llm = _FakeLLM("def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=tmp_path,
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    adapter.set_baseline_outputs(
        [
            {
                "task_id": 99,
                "game": "base-game",
                "level": 3,
                "score": 0.75,
                "solved": True,
                "expanded": 12,
                "generated": 20,
                "solution_length": 4,
                "partial_progress_score": 1.0,
                "feedback": "baseline solved",
                "error": None,
            }
        ]
    )
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="base-game",
        level=3,
        budget=100,
        env_description="Win conditions: all crates on targets",
        game_text_path=str(tmp_path / "game.txt"),
    )

    batch = adapter.evaluate(
        batch=[task],
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        capture_traces=False,
    )

    assert llm.prompts == []
    assert batch.outputs[0]["task_id"] == 0
    assert batch.outputs[0]["score"] == 0.75
    assert batch.outputs[0]["solved"] is True
    assert batch.outputs[0]["baseline_solved"] is True
    assert batch.scores == pytest.approx([0.0])
    eval_dirs = list((tmp_path / "candidate_evals").glob("eval-*"))
    assert len(eval_dirs) == 1
    assert (eval_dirs[0] / "baseline_reuse.json").exists()


def test_build_reflection_feedback_includes_trace_diagnostics_for_solved_regression() -> None:
    feedback = build_reflection_feedback(
        {
            "score": 0.4,
            "solved": True,
            "expanded": 400,
            "baseline_score": 0.8,
            "baseline_solved": True,
            "baseline_expanded": 20,
            "trace_summary": {
                "best_seen_h": 3.0,
                "open_set_size_at_end": 17,
                "root_snapshot": {"score_normalized": 0.2},
                "sampled_states": [
                    {"snapshot": {"score_normalized": 0.5}},
                    {"snapshot": {"score_normalized": 0.4}},
                ],
            },
        },
        "solved_regression",
    )

    assert "Candidate trace diagnostics" in feedback
    assert "progress_range=0.200..0.500" in feedback
    assert "open_set_size_at_end=17" in feedback


def test_select_reflection_traces_prioritizes_regressions_and_new_solves() -> None:
    trajectories = [
        {
            "task": {"game": f"game-{idx:02d}", "level": idx},
            "result": {
                "score": float(idx),
                "solved": idx % 2 == 0,
                "baseline_solved": False,
            },
        }
        for idx in range(DEFAULT_REFLECTION_MAX_RECORDS + 6)
    ]
    trajectories.append(
        {
            "task": {"game": "regression", "level": 99},
            "result": {"score": 0.0, "solved": False, "baseline_solved": True},
        }
    )
    trajectories.append(
        {
            "task": {"game": "new-solve", "level": 100},
            "result": {"score": 0.8, "solved": True, "baseline_solved": False},
        }
    )

    selected = select_reflection_traces(trajectories)

    assert len(selected) == DEFAULT_REFLECTION_MAX_RECORDS
    assert selected[0]["task"]["game"] == "regression"
    assert selected[1]["result"]["solved"] is True
    assert selected[1]["result"]["baseline_solved"] is False
    assert "new-solve" in {trace["task"]["game"] for trace in selected}
    assert any(not trace["result"]["solved"] for trace in selected[:10])


def test_select_reflection_traces_keeps_mechanics_diversity() -> None:
    trajectories = [
        {
            "task": {
                "game": f"generic-{idx:02d}",
                "level": idx,
                "env_description": "Win conditions: all target on crate",
            },
            "result": {"score": float(idx) / 100.0, "solved": False, "baseline_solved": False},
        }
        for idx in range(DEFAULT_REFLECTION_MAX_RECORDS + 10)
    ]
    trajectories.append(
        {
            "task": {
                "game": "portal-alias-case",
                "level": 100,
                "env_description": (
                    "Objects: playeru, playerl, portal, crate\n"
                    "Rules: [ portal | > playeru ] -> [ > playeru | portal ]"
                ),
            },
            "result": {"score": 9.0, "solved": False, "baseline_solved": False},
        }
    )

    selected = select_reflection_traces(trajectories)

    assert len(selected) == DEFAULT_REFLECTION_MAX_RECORDS
    assert "portal-alias-case" in {trace["task"]["game"] for trace in selected}


def test_h100_launcher_defaults_to_extended_vllm_context() -> None:
    launcher = Path("sbatch/train_puzzlescript_batched_gepa_gpu.s").read_text(encoding="utf-8")

    assert "#SBATCH --cpus-per-task=2" in launcher
    assert "#SBATCH --time=07:00:00" in launcher
    assert "#SBATCH --gres=gpu:h100:2" in launcher
    assert 'elif [ -n "${RUN_STATE_ROOT:-}" ]; then' in launcher
    assert 'LOCAL_LLM_MODEL:-openai/gpt-oss-120b' in launcher
    assert 'VLLM_TENSOR_PARALLEL_SIZE="${SLURM_GPUS_ON_NODE:-2}"' in launcher
    assert '--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"' in launcher
    assert 'VLLM_MAX_MODEL_LEN:-65536' in launcher
    assert 'VLLM_PORT_SPACING:-20' in launcher
    assert '--shutdown-timeout "${VLLM_SHUTDOWN_TIMEOUT:-30}"' in launcher
    assert f'--temperature "${{LLM_TEMPERATURE:-{DEFAULT_LLM_TEMPERATURE}}}"' in launcher
    assert '--val-split "${VAL_SPLIT:-dev}"' in launcher
    assert '--max-gepa-iterations "${MAX_GEPA_ITERATIONS:-16}"' in launcher
    assert 'search_array_count=${SEARCH_ARRAY_COUNT:-101} concurrency=${SEARCH_ARRAY_CONCURRENCY:-16}' in launcher
    assert '--search-array-concurrency "${SEARCH_ARRAY_CONCURRENCY:-16}"' in launcher
    assert '--search-array-stall-timeout-s "${SEARCH_ARRAY_STALL_TIMEOUT_S:-300}"' in launcher
    assert '--lost-solve-penalty "${LOST_SOLVE_PENALTY:-8.0}"' in launcher
    assert '--new-solve-bonus "${NEW_SOLVE_BONUS:-1.0}"' in launcher
    assert '--score-delta-weight "${SCORE_DELTA_WEIGHT:-1.0}"' in launcher
    assert '--global-lost-solve-gate-penalty "${GLOBAL_LOST_SOLVE_GATE_PENALTY:-${LOST_SOLVE_PENALTY:-8.0}}"' in launcher
    assert '--global-net-solve-loss-gate-penalty "${GLOBAL_NET_SOLVE_LOSS_GATE_PENALTY:-${LOST_SOLVE_PENALTY:-8.0}}"' in launcher
    assert '--guard-levels "${GUARD_LEVELS:-' in launcher
    assert 'RUN_HOLDOUT_COMPARE:-1' in launcher
    assert "scripts/compare_puzzlescript_batched_prompts.py" in launcher

    holdout_launcher = Path("sbatch/compare_puzzlescript_holdout_gpu.s").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --time=01:15:00" in holdout_launcher
    assert 'VLLM_PORT_SPACING:-20' in holdout_launcher
    assert '--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"' in holdout_launcher
    assert 'search_array_count=${SEARCH_ARRAY_COUNT:-101} concurrency=${SEARCH_ARRAY_CONCURRENCY:-16}' in holdout_launcher
    assert '--search-array-concurrency "${SEARCH_ARRAY_CONCURRENCY:-16}"' in holdout_launcher
    assert '--search-array-stall-timeout-s "${SEARCH_ARRAY_STALL_TIMEOUT_S:-300}"' in holdout_launcher
    assert '--shutdown-timeout "${VLLM_SHUTDOWN_TIMEOUT:-30}"' in holdout_launcher
    assert f'--temperature "${{LLM_TEMPERATURE:-{DEFAULT_LLM_TEMPERATURE}}}"' in holdout_launcher

    smoke_launcher = Path("sbatch/smoke_compare_puzzlescript_model_gpu.s").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --time=01:15:00" in smoke_launcher
    assert 'VLLM_PORT_SPACING:-20' in smoke_launcher
    assert '--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"' in smoke_launcher
    assert '--shutdown-timeout "${VLLM_SHUTDOWN_TIMEOUT:-30}"' in smoke_launcher
    assert f'--temperature "${{LLM_TEMPERATURE:-{DEFAULT_LLM_TEMPERATURE}}}"' in smoke_launcher
    assert '--global-lost-solve-gate-penalty "${GLOBAL_LOST_SOLVE_GATE_PENALTY:-${LOST_SOLVE_PENALTY:-4.0}}"' in smoke_launcher
    assert '--global-net-solve-loss-gate-penalty "${GLOBAL_NET_SOLVE_LOSS_GATE_PENALTY:-${LOST_SOLVE_PENALTY:-4.0}}"' in smoke_launcher


def test_search_array_launcher_skips_locked_setup_when_runtime_exists() -> None:
    launcher = Path("sbatch/evaluate_puzzlescript_search_array.s").read_text(encoding="utf-8")

    assert "#SBATCH --cpus-per-task=1" in launcher
    assert "#SBATCH --time=01:00:00" in launcher
    assert "runtime_ready()" in launcher
    assert "ensure_runtime()" in launcher
    assert "#SBATCH --mem=2G" in launcher
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
