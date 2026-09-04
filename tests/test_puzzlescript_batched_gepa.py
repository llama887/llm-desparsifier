"""Focused tests for the standalone batched PuzzleScript GEPA path."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_desparsifier.search.puzzlescript_adapter import win_condition_progress
from scripts.compare_puzzlescript_batched_prompts import (
    build_synthesis_client,
    compare_prompt_outputs,
)
from scripts.plot_puzzlescript_paper_results import write_paper_plots
from scripts.run_puzzlescript_batched_gepa import (
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS,
    DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS,
    DEFAULT_REFLECTION_FEEDBACK_CHARS,
    DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS,
    DEFAULT_REFLECTION_MAX_RECORDS,
    GEPA_ADDENDUM_HEADER,
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    CodexCLITextClient,
    PuzzleScriptBatchedGEPAAdapter,
    PuzzleScriptLevelTask,
    SearchArrayConfig,
    SearchArrayStalledError,
    _common_solve_code_shape_diagnostic_line,
    _gepa_iteration_limit_reached,
    adjusted_candidate_scores,
    aggregate_replicate_results,
    assigned_tasks,
    build_reflection_feedback,
    build_repair_prompt,
    build_sbatch_array_command,
    build_seed_candidate,
    build_train_dev_tasks,
    candidate_prompt_issue,
    candidate_score,
    collect_git_state,
    context_retry_max_tokens,
    evaluate_manifest_shards_locally,
    evaluate_search_task_with_wall_timeout,
    filter_unlearnable_tasks,
    heuristic_code_shape,
    load_scoring_baseline_outputs,
    load_training_targets,
    local_search_fallback_workers,
    merge_validation_guard_tasks,
    parse_guard_level_selection,
    publish_search_pool_manifest,
    read_initial_gepa_addendum,
    run_training_target_sweep,
    select_generalizing_candidate,
    select_jobs_for_training_levels,
    select_reflection_traces,
    select_training_guard_tasks,
    split_train_dev_jobs,
    strip_outer_markdown_fences,
    trace_classification,
    trace_partial_progress_score,
    training_target_state_root,
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


class _SequenceLLM(_FakeLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("")
        self.responses = iter(responses)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_zero_global_gates_leave_local_pareto_scores_untouched() -> None:
    rows = [
        {
            "game": "g",
            "level": 0,
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.5,
            "baseline_solved": True,
            "expanded": 1000,
            "baseline_expanded": 10,
        },
        {
            "game": "g",
            "level": 1,
            "score": 0.5,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "expanded": 10,
            "baseline_expanded": 1000,
        },
        {
            "game": "g",
            "level": 2,
            "score": 0.5,
            "solved": True,
            "baseline_score": 0.5,
            "baseline_solved": True,
            "expanded": 400,
            "baseline_expanded": 100,
        },
        {
            "game": "g",
            "level": 3,
            "score": 0.5,
            "solved": True,
            "baseline_score": 0.5,
            "baseline_solved": True,
            "expanded": 100,
            "baseline_expanded": 100,
        },
    ]

    scores = adjusted_candidate_scores(
        rows,
        global_lost_solve_gate_penalty=0.0,
        global_net_solve_loss_gate_penalty=0.0,
    )

    assert scores[1] > 0.0
    assert scores[2] < 0.0
    assert scores[3] == pytest.approx(0.0)


def test_replicate_aggregation_tracks_solve_probability_and_solved_efficiency() -> None:
    task = PuzzleScriptLevelTask(
        task_id=7,
        game="game",
        level=2,
        budget=1000,
        env_description="description",
        game_text_path="game.txt",
    )
    rows = [
        {
            "solved": True,
            "score": 0.8,
            "expanded": 100,
            "generated": 150,
            "solution_length": 8,
            "heuristic_code_path": "a.py",
        },
        {
            "solved": False,
            "score": 0.0,
            "expanded": 1000,
            "generated": 1400,
            "solution_length": 0,
            "heuristic_code_path": "b.py",
            "synthesis_error": "bad code",
        },
        {
            "solved": True,
            "score": 0.6,
            "expanded": 300,
            "generated": 450,
            "solution_length": 10,
            "heuristic_code_path": "c.py",
        },
    ]

    result = aggregate_replicate_results(task, rows)

    assert result["replicate_count"] == 3
    assert result["solve_rate"] == pytest.approx(2 / 3)
    assert result["solved_expanded_mean"] == pytest.approx(200.0)
    assert result["expanded"] == pytest.approx(1400 / 3)
    assert result["candidate_error_rate"] == pytest.approx(1 / 3)


def test_codex_cli_text_client_uses_structured_stateless_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["input"] = kwargs["input"]
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"text":"generated proposal"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_puzzlescript_batched_gepa.subprocess.run", fake_run)
    client = CodexCLITextClient(
        model="test-codex-model",
        timeout_s=30.0,
        executable="codex-test",
        reasoning_effort="high",
    )

    assert client.complete("improve the prompt") == "generated proposal"
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:2] == ["codex-test", "exec"]
    assert "--ephemeral" in command
    assert ["--sandbox", "read-only"] == command[
        command.index("--sandbox") : command.index("--sandbox") + 2
    ]
    assert ["--model", "test-codex-model"] == command[
        command.index("--model") : command.index("--model") + 2
    ]
    assert command[-1] == "-"
    assert "Do not inspect files" in str(observed["input"])


def test_codex_cli_text_client_can_inspect_trace_artifacts_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        observed["input"] = kwargs["input"]
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"text":"trace-grounded proposal"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_puzzlescript_batched_gepa.subprocess.run", fake_run)
    client = CodexCLITextClient(
        model="test-codex-model",
        timeout_s=30.0,
        working_directory=tmp_path,
        allow_read_tools=True,
    )

    assert client.complete("inspect /trace/scored_results.json") == "trace-grounded proposal"
    command = observed["command"]
    assert isinstance(command, list)
    assert "shell_tool" not in command
    assert observed["cwd"] == tmp_path
    assert "read-only shell tools" in str(observed["input"])


def test_codex_cli_text_client_agentic_synthesis_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        observed["input"] = kwargs["input"]
        observed["has_timeout"] = "timeout" in kwargs
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"text":"def heuristic_cost_to_go(ts, env_params, ctx): return 0.0"}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_puzzlescript_batched_gepa.subprocess.run", fake_run)
    client = CodexCLITextClient(model="luna", timeout_s=30.0, agentic_workspace=True)

    client.complete("write a heuristic")
    command = observed["command"]
    assert isinstance(command, list)
    assert "shell_tool" not in command
    assert ["--sandbox", "workspace-write"] == command[
        command.index("--sandbox") : command.index("--sandbox") + 2
    ]
    assert Path(str(observed["cwd"])).name.startswith("gepa-codex-")
    assert "candidate.py" in str(observed["input"])
    assert observed["has_timeout"] is False


def test_publish_search_pool_manifest_is_atomic(tmp_path: Path) -> None:
    manifest = tmp_path / "eval" / "search_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}")

    pointer = publish_search_pool_manifest(tmp_path / "pool", manifest)

    assert pointer.read_text().strip() == str(manifest.resolve())
    assert not list(pointer.parent.glob("*.tmp"))


def test_win_condition_progress_is_rule_derived_and_dense() -> None:
    positions = {
        "crate": [(1, 1), (2, 2)],
        "target": [(1, 1), (3, 3)],
        "hazard": [(4, 4)],
    }
    winconditions = [
        {"num": 1, "mask1_names": ["crate"], "mask2_names": ["target"]},
        {"num": -1, "mask1_names": ["crate"], "mask2_names": ["hazard"]},
    ]

    assert win_condition_progress(winconditions, positions) == pytest.approx(0.75)


def test_trace_partial_progress_uses_best_and_late_rule_progress() -> None:
    trace = {
        "root_snapshot": {"progress_score": 0.1},
        "sampled_states": [{"snapshot": {"progress_score": 0.2}}],
        "best_progress_snapshot": {"progress_score": 0.8},
        "late_states": [{"snapshot": {"progress_score": 0.6}}],
    }

    assert trace_partial_progress_score(trace) == pytest.approx(0.8)


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


def test_validate_heuristic_code_allows_internal_non_finite_sentinel() -> None:
    code = """
def heuristic_cost_to_go(ts, env_params, ctx):
    if ctx.get("is_winning"):
        return 0.0
    best = float("inf")
    best = min(best, 1.0)
    return best
"""

    issue = validate_heuristic_code(code)

    assert issue is None


def test_validate_heuristic_code_rejects_direct_non_finite_return() -> None:
    code = """
def heuristic_cost_to_go(ts, env_params, ctx):
    if ctx.get("is_winning"):
        return 0.0
    return float("inf")
"""

    issue = validate_heuristic_code(code)

    assert issue is not None
    assert "non-finite" in issue


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


def test_collect_git_state_reports_commit_branch_and_dirty_status() -> None:
    responses = {
        ("rev-parse", "HEAD"): "abc123",
        ("branch", "--show-current"): "feature",
        ("status", "--short"): " M changed.py\n?? new.py",
    }

    state = collect_git_state(
        Path("/repo"),
        git_runner=lambda args: responses.get(tuple(args), ""),
    )

    assert state == {
        "repo_root": "/repo",
        "commit": "abc123",
        "branch": "feature",
        "status_short": [" M changed.py", "?? new.py"],
        "dirty": True,
    }


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


def test_local_search_fallback_workers_uses_allocated_cpus() -> None:
    assert local_search_fallback_workers(
        missing_indices=[0, 1, 2, 3],
        environ={"SLURM_CPUS_PER_TASK": "8"},
    ) == 4
    assert local_search_fallback_workers(
        missing_indices=[0, 1, 2, 3],
        environ={
            "SLURM_CPUS_PER_TASK": "8",
            "SEARCH_LOCAL_FALLBACK_WORKERS": "2",
        },
    ) == 2
    assert local_search_fallback_workers(
        missing_indices=[0, 1, 2, 3],
        environ={"SLURM_CPUS_PER_TASK": "not-an-int"},
    ) == 1


def test_evaluate_manifest_shards_locally_uses_bounded_parallelism(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_evaluate_manifest_shard(
        *,
        manifest_path: Path,
        array_index: int,
        array_count: int,
    ) -> Path:
        nonlocal active, max_active
        del manifest_path, array_count
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        shard_path = tmp_path / f"task-{array_index:04d}-of-0004.json"
        shard_path.write_text("{}\n", encoding="utf-8")
        with lock:
            active -= 1
        return shard_path

    paths = evaluate_manifest_shards_locally(
        manifest_path=tmp_path / "manifest.json",
        array_count=4,
        missing_indices=[0, 1, 2, 3],
        max_workers=2,
        evaluate_fn=fake_evaluate_manifest_shard,
    )

    assert [path.name for path in paths] == [
        "task-0000-of-0004.json",
        "task-0001-of-0004.json",
        "task-0002-of-0004.json",
        "task-0003-of-0004.json",
    ]
    assert max_active == 2


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


def test_build_train_dev_tasks_balances_dev_by_level_count() -> None:
    tasks = [
        *[
            SimpleNamespace(task_id=idx, game="large-game", level=idx)
            for idx in range(8)
        ],
        *[
            SimpleNamespace(task_id=8 + idx, game=f"small-{idx}", level=0)
            for idx in range(4)
        ],
    ]

    train_tasks, dev_tasks = build_train_dev_tasks(tasks, dev_fraction=0.25, seed=1)

    assert len(dev_tasks) == 3
    assert {task.game for task in train_tasks}.isdisjoint({task.game for task in dev_tasks})
    assert {task.game for task in dev_tasks} <= {"small-0", "small-1", "small-2", "small-3"}
    assert [task.task_id for task in train_tasks] == list(range(len(train_tasks)))
    assert [task.task_id for task in dev_tasks] == list(range(len(dev_tasks)))


def test_build_train_dev_tasks_rejects_invalid_dev_fraction() -> None:
    tasks = [SimpleNamespace(task_id=0, game="game", level=0)]

    with pytest.raises(ValueError, match="dev_fraction"):
        build_train_dev_tasks(tasks, dev_fraction=1.0, seed=0)


def test_select_jobs_for_training_levels_can_promote_eval_game() -> None:
    train_jobs = [{"name": "train-game"}]
    eval_jobs = [{"name": "heldout-game"}, {"name": "unused-heldout-game"}]

    selected = select_jobs_for_training_levels(
        train_jobs,
        eval_jobs,
        {"heldout-game": [3]},
    )

    assert selected == [{"name": "heldout-game"}]


def test_select_jobs_for_training_levels_keeps_default_train_boundary() -> None:
    train_jobs = [{"name": "train-game"}]
    eval_jobs = [{"name": "heldout-game"}]

    selected = select_jobs_for_training_levels(train_jobs, eval_jobs, {})

    assert selected == train_jobs


def test_load_training_targets_accepts_manifest_object(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        """{
  "source": "holdout comparison",
  "targets": [
    {"game": "Beam_Islands", "level": 3, "score_delta": -0.9},
    {"game": "Darkness_Sokoban", "level": 0, "score_delta": -0.1}
  ]
}""",
        encoding="utf-8",
    )

    targets = load_training_targets(manifest)

    assert [(target["game"], target["level"]) for target in targets] == [
        ("Beam_Islands", 3),
        ("Darkness_Sokoban", 0),
    ]
    assert targets[0]["score_delta"] == -0.9


def test_load_training_targets_rejects_duplicate_game_level(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        '[{"game": "Beam_Islands", "level": 3}, '
        '{"game": "Beam_Islands", "level": 3}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_training_targets(manifest)


def test_training_target_state_root_is_stable_and_filesystem_safe(tmp_path: Path) -> None:
    state_root = training_target_state_root(
        tmp_path,
        index=2,
        game="Push / Pull: Test",
        level=7,
    )

    assert state_root == tmp_path / "targets" / "02-push-pull-test-level-07"


def test_run_training_target_sweep_isolates_each_gepa_state(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        '[{"game": "Beam_Islands", "level": 3}, '
        '{"game": "Darkness_Sokoban", "level": 0}]',
        encoding="utf-8",
    )
    seen: list[tuple[Path, str, bool]] = []

    def fake_run_target(args: SimpleNamespace) -> None:
        args.state_root.mkdir(parents=True)
        seen.append(
            (args.state_root, args.training_levels, args.allow_case_specific_overfit)
        )
        (args.state_root / "gepa_result.json").write_text(
            '{"best_idx": 0, "val_aggregate_scores": [0.5], "candidates": [{}]}',
            encoding="utf-8",
        )
        (args.state_root / "best_prompt.txt").write_text("prompt", encoding="utf-8")

    group_root = tmp_path / "sweep"
    run_training_target_sweep(
        SimpleNamespace(
            training_targets_file=manifest,
            state_root=group_root,
            training_levels="",
        ),
        run_target=fake_run_target,
    )

    assert seen == [
        (group_root / "targets" / "00-beam-islands-level-03", "Beam_Islands:3", False),
        (
            group_root / "targets" / "01-darkness-sokoban-level-00",
            "Darkness_Sokoban:0",
            False,
        ),
    ]
    summary = json.loads((group_root / "sweep_summary.json").read_text(encoding="utf-8"))
    assert [target["status"] for target in summary["targets"]] == [
        "completed",
        "completed",
    ]


def test_trace_classification_flags_moderate_solved_efficiency_regression() -> None:
    trace = {
        "result": {
            "solved": True,
            "baseline_solved": True,
            "score": 0.84,
            "baseline_score": 0.86,
            "expanded": 1_600,
            "baseline_expanded": 1_400,
        }
    }

    assert trace_classification(trace) == "solved_regression"


def test_trace_classification_flags_solved_efficiency_gain() -> None:
    trace = {
        "result": {
            "solved": True,
            "baseline_solved": True,
            "score": 0.88,
            "baseline_score": 0.86,
            "expanded": 900,
            "baseline_expanded": 1_400,
        }
    }

    assert trace_classification(trace) == "solved_efficiency_gain"


def test_candidate_score_can_apply_explicit_eval_wide_gate_for_lost_solves_and_errors() -> None:
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
        global_lost_solve_gate_penalty=4.0,
    )
    score = candidate_score(
        outputs,
        lost_solve_penalty=4.0,
        new_solve_bonus=1.0,
        error_penalty=2.0,
        score_delta_weight=0.25,
        score_delta_clip=0.5,
        global_lost_solve_gate_penalty=4.0,
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


def test_candidate_score_defaults_to_nonpositive_net_solve_gate() -> None:
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
    )

    assert scores == pytest.approx([-4.5, 1.5])
    assert candidate_score(
        outputs,
        lost_solve_penalty=2.0,
        new_solve_bonus=3.0,
        score_delta_weight=1.0,
        score_delta_clip=0.5,
    ) == pytest.approx(-1.5)


def test_candidate_score_rewards_large_net_solve_gain_by_default() -> None:
    outputs = [
        {
            "game": "loss",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.5,
            "baseline_solved": True,
        },
        *[
            {
                "game": f"gain-{idx}",
                "score": 0.7,
                "solved": True,
                "baseline_score": 0.0,
                "baseline_solved": False,
            }
            for idx in range(5)
        ],
    ]

    assert candidate_score(outputs) > 0.0


def test_candidate_score_keeps_any_net_solve_gain_positive() -> None:
    outputs = [
        {
            "game": "new",
            "score": 0.9,
            "solved": True,
            "expanded": 100,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "baseline_expanded": 10_000,
        },
        {
            "game": "slow-a",
            "score": 0.9,
            "solved": True,
            "expanded": 10_000,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 10,
        },
        {
            "game": "slow-b",
            "score": 0.9,
            "solved": True,
            "expanded": 10_000,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 10,
        },
    ]

    assert candidate_score(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=2.0,
        common_solve_efficiency_clip=2.0,
        new_solve_bonus=1.0,
    ) > 0.0


def test_candidate_scores_penalize_common_solve_expansion_slowdowns() -> None:
    outputs = [
        {
            "game": "same-solve",
            "score": 0.84,
            "solved": True,
            "expanded": 1_600,
            "baseline_score": 0.86,
            "baseline_solved": True,
            "baseline_expanded": 1_400,
        }
    ]

    scores = adjusted_candidate_scores(
        outputs,
        score_delta_weight=1.0,
        score_delta_clip=1.0,
    )

    assert scores[0] < -0.1


def test_candidate_scores_weight_common_solve_efficiency_by_headroom() -> None:
    outputs = [
        {
            "game": "low-headroom",
            "score": 0.9,
            "solved": True,
            "expanded": 24,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 49,
        },
        {
            "game": "high-headroom",
            "score": 0.9,
            "solved": True,
            "expanded": 2_499,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 4_999,
        },
    ]

    scores = adjusted_candidate_scores(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=1.0,
        common_solve_efficiency_clip=2.0,
    )

    assert scores[1] > scores[0] * 3.0


def test_candidate_scores_upweight_very_high_headroom_common_solves() -> None:
    outputs = [
        {
            "game": "barely-high-headroom",
            "score": 0.9,
            "solved": True,
            "expanded": 250,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 500,
        },
        {
            "game": "very-high-headroom",
            "score": 0.9,
            "solved": True,
            "expanded": 2_500,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 5_000,
        },
    ]

    scores = adjusted_candidate_scores(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=1.0,
        common_solve_efficiency_clip=5.0,
    )

    assert scores[1] > scores[0] * 2.5
    assert scores[1] <= 3.0


def test_candidate_scores_penalize_severe_low_headroom_slowdowns() -> None:
    outputs = [
        {
            "game": "low-headroom-collapse",
            "score": 0.9,
            "solved": True,
            "expanded": 629,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 3,
        }
    ]

    scores = adjusted_candidate_scores(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=1.0,
        common_solve_efficiency_clip=1.0,
    )

    assert scores[0] <= -0.9


def test_candidate_score_uses_efficiency_without_overriding_net_solve_gain() -> None:
    outputs = [
        {
            "game": "new",
            "score": 0.9,
            "solved": True,
            "expanded": 100,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "baseline_expanded": 10_000,
        },
        {
            "game": "slow-a",
            "score": 0.9,
            "solved": True,
            "expanded": 300,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 100,
        },
        {
            "game": "slow-b",
            "score": 0.9,
            "solved": True,
            "expanded": 300,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 100,
        },
    ]

    score = candidate_score(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=2.0,
        common_solve_efficiency_clip=1.0,
        new_solve_bonus=4.0,
    )

    assert 0.0 < score < candidate_score(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=0.0,
        new_solve_bonus=4.0,
    )


def test_candidate_score_uses_headroom_weighted_efficiency_as_tiebreaker() -> None:
    outputs = [
        {
            "game": "new",
            "score": 0.9,
            "solved": True,
            "expanded": 100,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "baseline_expanded": 10_000,
        },
        *[
            {
                "game": f"cheap-fast-{idx}",
                "score": 0.9,
                "solved": True,
                "expanded": 4,
                "baseline_score": 0.9,
                "baseline_solved": True,
                "baseline_expanded": 10,
            }
            for idx in range(5)
        ],
        {
            "game": "expensive-slow",
            "score": 0.9,
            "solved": True,
            "expanded": 10_000,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "baseline_expanded": 5_000,
        },
    ]

    score = candidate_score(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=1.0,
        common_solve_efficiency_clip=2.0,
        new_solve_bonus=4.0,
    )

    assert 0.0 < score < candidate_score(
        outputs,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=0.0,
        new_solve_bonus=4.0,
    )


def test_candidate_score_does_not_reward_equal_new_lost_from_efficiency() -> None:
    outputs = [
        {
            "game": "loss",
            "score": 0.0,
            "solved": False,
            "baseline_score": 0.9,
            "baseline_solved": True,
            "expanded": 10_000,
            "baseline_expanded": 100,
        },
        {
            "game": "gain",
            "score": 0.9,
            "solved": True,
            "baseline_score": 0.0,
            "baseline_solved": False,
            "expanded": 100,
            "baseline_expanded": 10_000,
        },
        *[
            {
                "game": f"fast-{idx}",
                "score": 0.99,
                "solved": True,
                "expanded": 100,
                "baseline_score": 0.95,
                "baseline_solved": True,
                "baseline_expanded": 10_000,
            }
            for idx in range(3)
        ],
    ]

    assert candidate_score(
        outputs,
        lost_solve_penalty=4.0,
        new_solve_bonus=4.0,
        score_delta_weight=0.0,
        common_solve_efficiency_weight=2.0,
        common_solve_efficiency_clip=2.0,
    ) < 0.0


def test_candidate_score_can_disable_any_lost_gate_but_keeps_nonpositive_net_gate() -> None:
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

    assert scores == pytest.approx([-4.5, 1.5])
    assert candidate_score(
        outputs,
        lost_solve_penalty=2.0,
        new_solve_bonus=3.0,
        score_delta_weight=1.0,
        score_delta_clip=0.5,
        global_lost_solve_gate_penalty=0.0,
    ) == pytest.approx(-1.5)


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


def test_select_training_guard_tasks_ignores_eval_only_games() -> None:
    tasks = [
        PuzzleScriptLevelTask(7, "train_game", 2, 10, "train2", "train.txt"),
        PuzzleScriptLevelTask(8, "train_game", 3, 10, "train3", "train.txt"),
    ]
    selection = {
        "train_game": [3],
        "heldout_game": [0],
    }

    selected = select_training_guard_tasks(tasks, selection)

    assert [(task.task_id, task.game, task.level, task.env_description) for task in selected] == [
        (0, "train_game", 3, "train3")
    ]


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
        "    crate_keys = ['ncrate', 'mixcrate', 'matchcrate']\n"
        "    players = ['controlplayer', 'dummy']\n"
        "    weighted_switches = len(yellowswitches)\n"
        "    closed_doors = door_cells - open_doors\n"
        "    while queue: reachable = True\n"
        "    return 1000.0 if 'yellowdoor' else len(crate_positions) + len(targets) + corner_deadlock\n"
    )

    assert shape["uses_generic_role_helpers"] is True
    assert shape["uses_score_fallback"] is True
    assert shape["uses_large_penalty"] is True
    assert shape["mentions_mechanics_terms"] is True
    assert shape["uses_pushable_object_terms"] is True
    assert shape["uses_target_terms"] is True
    assert shape["uses_deadlock_checks"] is True
    assert shape["uses_alias_specific_terms"] is True
    assert shape["uses_weighted_switch_terms"] is True
    assert shape["uses_gate_aware_reachability"] is True


def test_heuristic_code_shape_flags_carried_transformed_object_terms() -> None:
    shape = heuristic_code_shape(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    obj_pos = ctx.get('object_positions', {})\n"
        "    crates = obj_pos.get('crate', [])\n"
        "    carried = obj_pos.get('carry', []) + obj_pos.get('pickedup', [])\n"
        "    all_crates = list(crates) + list(carried)\n"
        "    return float(len(all_crates))\n"
    )

    assert shape["uses_transformed_object_terms"] is True
    assert shape["uses_alias_specific_terms"] is True


def test_heuristic_code_shape_flags_player_interaction_distance() -> None:
    shape = heuristic_code_shape(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    player_pos = ctx.get('object_positions', {}).get('player', [])\n"
        "    marked = ctx.get('object_positions', {}).get('marked', [])\n"
        "    if player_pos and marked:\n"
        "        px, py = player_pos[0]\n"
        "        return min(abs(px - mx) + abs(py - my) for mx, my in marked)\n"
        "    return len(marked)\n"
    )

    assert shape["uses_player_terms"] is True
    assert shape["uses_distance_terms"] is True
    assert shape["uses_interaction_object_terms"] is True
    assert shape["uses_player_interaction_distance"] is True


def test_heuristic_code_shape_flags_assignment_and_transition_terms() -> None:
    shape = heuristic_code_shape(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    remaining_crates = crate_positions[:]\n"
        "    remaining_targets = target_positions[:]\n"
        "    for perm in permutations(remaining_crates):\n"
        "        best_sum = min(best_sum, sum(abs(cx - tx) + abs(cy - ty)))\n"
        "    # Push crates into water, pull boxes, and swap through portals.\n"
        "    return float(best_sum)\n"
    )

    assert shape["uses_assignment_matching"] is True
    assert shape["uses_action_transition_terms"] is True


def test_common_solve_efficiency_diagnostic_explains_reachability_regression() -> None:
    baseline_shape = heuristic_code_shape(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    crates = ctx.get('object_positions', {}).get('crate', [])\n"
        "    targets = ctx.get('object_positions', {}).get('target', [])\n"
        "    player = ctx.get('object_positions', {}).get('player', [])\n"
        "    return min(abs(cx - tx) + abs(cy - ty) for cx, cy in crates for tx, ty in targets)\n"
    )
    generated_shape = heuristic_code_shape(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    def bfs(start):\n"
        "        visited = {start}\n"
        "        while queue:\n"
        "            pass\n"
        "    return bfs(player) + (1.0 - ctx.get('score_normalized', 0.0))\n"
    )

    line = _common_solve_code_shape_diagnostic_line(
        {
            "solved": True,
            "baseline_solved": True,
            "expanded": 2186,
            "baseline_expanded": 313,
        },
        baseline_shape,
        generated_shape,
        "solved_regression",
    )

    assert "common solve got slower" in line
    assert "candidate/base_expansion_ratio=6.98" in line
    assert "candidate added reachability/BFS" in line
    assert "score_normalized" in line


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

    record = next(
        item
        for item in dataset["heuristic_prompt"]
        if item["Comparison"]["classification"] == "persistent_failure"
    )
    assert len(record["Inputs"]["env_description"]) < DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS + 80
    assert len(record["Generated Outputs"]["heuristic_code"]) < DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS + 80
    assert len(record["Feedback"]) < DEFAULT_REFLECTION_FEEDBACK_CHARS + 80
    assert "[truncated 100 chars]" in record["Inputs"]["env_description"]
    assert "all levels still contributed to the scalar score" in record["Selection"]
    assert record["Comparison"]["classification"] == "persistent_failure"


def test_make_reflective_dataset_includes_regression_comparison(tmp_path: Path) -> None:
    base_code = tmp_path / "base.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    queue = [(0, 0)]\n"
        "    visited = {(0, 0)}\n"
        "    while queue:\n"
        "        queue.pop(0)\n"
        "    return 1.0\n",
        encoding="utf-8",
    )
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
                "heuristic_code": (
                    "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                    "    if not ctx.get('object_positions'):\n"
                    "        return float('inf')\n"
                    "    return 1000.0\n"
                ),
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

    record = next(
        item
        for item in dataset["heuristic_prompt"]
        if item["Comparison"]["classification"] == "lost_baseline_solve"
    )
    assert record["Comparison"]["classification"] == "lost_baseline_solve"
    assert "REGRESSION" in record["Feedback"]
    assert "base prompt solved" in record["Feedback"]
    assert "candidate failed" in record["Feedback"]
    assert "general precondition" in record["Feedback"]
    assert "Generalization target" in record["Feedback"]
    assert "prompt structure or categorization" in record["Feedback"]
    assert "observable WINCONDITIONS" in record["Feedback"]
    assert "base solved quickly" in record["Baseline Output"]["feedback"]
    assert "return 1.0" in record["Baseline Output"]["heuristic_code"]
    assert record["Baseline Output"]["code_shape"]["uses_reachability_search"] is True
    assert record["Generated Outputs"]["code_shape"]["uses_nonfinite_return"] is True
    assert record["Generated Outputs"]["code_shape"]["uses_large_penalty"] is True
    assert "Code-shape contrast" in record["Feedback"]
    assert "reachability" in record["Feedback"]
    assert "non-finite or huge penalties" in record["Feedback"]


def test_make_reflective_dataset_reports_dropped_player_interaction_distance(
    tmp_path: Path,
) -> None:
    base_code = tmp_path / "base.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    marked = ctx.get('object_positions', {}).get('marked', [])\n"
        "    players = ctx.get('object_positions', {}).get('player', [])\n"
        "    if players and marked:\n"
        "        px, py = players[0]\n"
        "        return min(abs(px - mx) + abs(py - my) for mx, my in marked)\n"
        "    return len(marked)\n",
        encoding="utf-8",
    )
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
                    "game": "drop-swap-like",
                    "level": 1,
                    "budget": 10_000,
                    "env_description": "Win conditions: no marked objects remain.",
                },
                "heuristic_code": (
                    "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                    "    marked = ctx.get('object_positions', {}).get('marked', [])\n"
                    "    return len(marked) + (1.0 - ctx.get('score_normalized', 0.0))\n"
                ),
                "synthesis_error": None,
                "result": {
                    "feedback": "candidate solved slowly",
                    "score": 0.78,
                    "solved": True,
                    "expanded": 2145,
                    "baseline_score": 0.99,
                    "baseline_solved": True,
                    "baseline_expanded": 110,
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

    record = next(
        item
        for item in dataset["heuristic_prompt"]
        if item["Comparison"]["classification"] == "solved_regression"
    )
    assert record["Baseline Output"]["code_shape"]["uses_player_interaction_distance"] is True
    assert record["Generated Outputs"]["code_shape"]["uses_player_interaction_distance"] is False
    assert "player-to-interaction distance" in record["Feedback"]
    assert "count-only" in record["Feedback"]


def test_make_reflective_dataset_includes_mechanics_signature_as_diagnostic_only(
    tmp_path: Path,
) -> None:
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
                    "game": "portal-alias-case",
                    "level": 3,
                    "budget": 100,
                    "env_description": (
                        "Objects: playeru, playerl, portal, crate\n"
                        "Rules: [ portal | > playeru ] -> [ > playeru | portal ]"
                    ),
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

    record = next(
        item
        for item in dataset["heuristic_prompt"]
        if item["Comparison"]["classification"] == "lost_baseline_solve"
    )
    assert record["Comparison"]["mechanics_signature"] == "player-alias+portal"
    assert "Mechanics evidence for diagnosis only: player-alias+portal" in record["Feedback"]
    assert "rule-derived conditions" in record["Feedback"]
    assert "categories" in record["Feedback"]
    assert "abstract preconditions" in record["Feedback"]
    assert "observable mechanics" in record["Feedback"]


def test_make_reflective_dataset_starts_with_aggregate_outcome_summary() -> None:
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
                    "game": "beam-gain",
                    "level": 0,
                    "budget": 100,
                    "env_description": "Rules mention beam and target crates.",
                },
                "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n",
                "synthesis_error": None,
                "result": {
                    "score": 1.0,
                    "solved": True,
                    "baseline_score": 0.0,
                    "baseline_solved": False,
                    "adjusted_score": 4.0,
            },
        },
        {
            "task": {
                "game": "efficient-stable",
                "level": 1,
                "budget": 100,
                "env_description": "All target crate puzzle with useful ordering.",
            },
            "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.5\n",
            "synthesis_error": None,
            "result": {
                "score": 0.92,
                "solved": True,
                "expanded": 80,
                "baseline_score": 0.88,
                "baseline_solved": True,
                "baseline_expanded": 500,
                "adjusted_score": 0.7,
            },
        },
        {
            "task": {
                "game": "stable-loss",
                "level": 2,
                    "budget": 100,
                    "env_description": "Classic all target crate puzzle.",
                },
                "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 2.0\n",
                "synthesis_error": None,
                "result": {
                    "score": 0.0,
                    "solved": False,
                    "baseline_score": 1.0,
                    "baseline_solved": True,
                    "adjusted_score": -8.0,
                },
            },
        ]
    )

    dataset = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["heuristic_prompt"],
    )

    aggregate = dataset["heuristic_prompt"][0]
    assert aggregate["Comparison"]["classification"] == "aggregate_summary"
    assert aggregate["Comparison"]["new_solve_count"] == 1
    assert aggregate["Comparison"]["lost_baseline_solve_count"] == 1
    assert aggregate["Comparison"]["solved_efficiency_gain_count"] == 1
    assert aggregate["Comparison"]["high_headroom_common_solve_count"] == 1
    assert aggregate["Comparison"]["high_headroom_efficiency_gain_count"] == 1
    assert aggregate["Comparison"]["high_headroom_efficiency_regression_count"] == 0
    assert aggregate["Comparison"]["weighted_mean_common_solve_efficiency_delta"] == pytest.approx(1.0)
    assert "solved_efficiency_gains=1" in aggregate["Feedback"]
    assert "high_headroom_common_solves=1" in aggregate["Feedback"]
    assert "high_headroom_efficiency_gains=1" in aggregate["Feedback"]
    assert "weighted_mean_common_solve_efficiency_delta=+1.000" in aggregate["Feedback"]
    assert "efficient-stable" in aggregate["Feedback"]
    assert "beam-gain" in aggregate["Feedback"]
    assert "stable-loss" in aggregate["Feedback"]
    assert "beam" in aggregate["Feedback"]
    assert "code-side routing" in aggregate["Feedback"]


def test_make_reflective_dataset_aggregate_reports_code_shape_losses(
    tmp_path: Path,
) -> None:
    base_code = tmp_path / "base_assignment.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    remaining_crates = ctx.get('object_positions', {}).get('crate', [])\n"
        "    remaining_targets = ctx.get('object_positions', {}).get('target', [])\n"
        "    best_sum = 0\n"
        "    for perm in permutations(remaining_crates):\n"
        "        best_sum += len(remaining_targets)\n"
        "    # push/pull transition cost remains explicit\n"
        "    return float(best_sum)\n",
        encoding="utf-8",
    )
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
                    "game": "assignment-drop",
                    "level": 0,
                    "budget": 100,
                    "env_description": "All target on crate with pull rules.",
                },
                "heuristic_code": (
                    "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                    "    crates = ctx.get('object_positions', {}).get('crate', [])\n"
                    "    return float(len(crates))\n"
                ),
                "synthesis_error": None,
                "result": {
                    "score": 0.60,
                    "solved": True,
                    "expanded": 500,
                    "baseline_score": 0.90,
                    "baseline_solved": True,
                    "baseline_expanded": 100,
                    "baseline_heuristic_code_path": str(base_code),
                    "adjusted_score": -0.4,
                },
            }
        ]
    )

    dataset = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["heuristic_prompt"],
    )

    aggregate = dataset["heuristic_prompt"][0]
    shape_losses = aggregate["Comparison"]["code_shape_loss_counts"]
    assert shape_losses["uses_assignment_matching"] == 1
    assert shape_losses["uses_action_transition_terms"] == 1
    assert "Code-shape losses:" in aggregate["Feedback"]
    assert "uses_assignment_matching=1" in aggregate["Feedback"]
    assert "uses_action_transition_terms=1" in aggregate["Feedback"]


def test_aggregate_code_shape_losses_ignore_helpful_drops(tmp_path: Path) -> None:
    base_code = tmp_path / "base_assignment.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    remaining_crates = ctx.get('object_positions', {}).get('crate', [])\n"
        "    remaining_targets = ctx.get('object_positions', {}).get('target', [])\n"
        "    best_sum = 0\n"
        "    for perm in permutations(remaining_crates):\n"
        "        best_sum += len(remaining_targets)\n"
        "    return float(best_sum)\n",
        encoding="utf-8",
    )
    generated_code = (
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    crates = ctx.get('object_positions', {}).get('crate', [])\n"
        "    return float(len(crates))\n"
    )
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
                    "game": "helpful-drop",
                    "level": 0,
                    "budget": 100,
                    "env_description": "All target on crate.",
                },
                "heuristic_code": generated_code,
                "synthesis_error": None,
                "result": {
                    "score": 0.95,
                    "solved": True,
                    "expanded": 80,
                    "baseline_score": 0.85,
                    "baseline_solved": True,
                    "baseline_expanded": 500,
                    "baseline_heuristic_code_path": str(base_code),
                    "adjusted_score": 0.6,
                },
            },
            {
                "task": {
                    "game": "harmful-drop",
                    "level": 1,
                    "budget": 100,
                    "env_description": "All target on crate.",
                },
                "heuristic_code": generated_code,
                "synthesis_error": None,
                "result": {
                    "score": 0.60,
                    "solved": True,
                    "expanded": 500,
                    "baseline_score": 0.90,
                    "baseline_solved": True,
                    "baseline_expanded": 100,
                    "baseline_heuristic_code_path": str(base_code),
                    "adjusted_score": -0.4,
                },
            },
        ]
    )

    dataset = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["heuristic_prompt"],
    )

    aggregate = dataset["heuristic_prompt"][0]
    assert aggregate["Comparison"]["code_shape_loss_counts"]["uses_assignment_matching"] == 1
    assert "uses_assignment_matching=1" in aggregate["Feedback"]


def test_hybrid_routes_local_synthesis_and_codex_proposals(tmp_path: Path) -> None:
    synthesis_llm = _FakeLLM(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0"
    )
    proposal_llm = _FakeLLM(
        "Addendum: keep mechanics-specific object names primary and use score only as a tie-breaker."
    )
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=synthesis_llm,  # type: ignore[arg-type]
        proposal_llm=proposal_llm,  # type: ignore[arg-type]
        state_root=tmp_path,
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )

    adapter._synthesize_batch(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        batch=[PuzzleScriptLevelTask(0, "game", 0, 10, "description", "game.txt")],
        eval_dir=tmp_path,
    )
    assert len(synthesis_llm.prompts) == 1
    assert proposal_llm.prompts == []

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
    assert len(synthesis_llm.prompts) == 1
    assert "short addendum" in proposal_llm.prompts[0]
    assert "Do not rewrite the full base prompt" in proposal_llm.prompts[0]
    assert "Do not return the base prompt unchanged" in proposal_llm.prompts[0]
    assert "one human heuristic designer's reusable decision procedure" in proposal_llm.prompts[0]
    assert "train/validation performance" in proposal_llm.prompts[0]
    assert "GEPA may" in proposal_llm.prompts[0]
    assert "self-discover categories" in proposal_llm.prompts[0]
    assert "Prompt-internal if-statements" in proposal_llm.prompts[0]
    assert "runner will not implement buckets in code" in proposal_llm.prompts[0]
    assert "repeated solved-efficiency regressions" in proposal_llm.prompts[0]
    assert "preserve base passability/reachability" in proposal_llm.prompts[0]
    assert "candidate added or over-weighted" in proposal_llm.prompts[0]
    assert "preconditioned reasoning" in proposal_llm.prompts[0]
    assert "missing goal objects" in proposal_llm.prompts[0]
    assert "REGRESSION" in proposal_llm.prompts[0]


def test_custom_proposer_includes_current_addendum_when_revising() -> None:
    current_addendum = (
        "For persistent failures, inspect rules for a mechanics-grounded progress "
        "signal before falling back to score."
    )
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
    llm = _FakeLLM(
        "Keep the mechanics-grounded progress signal, but require rule-grounded "
        "preconditions before adding new terms."
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
        candidate={"heuristic_prompt": current_prompt},
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

    assert "Keep the mechanics-grounded progress signal" in result["heuristic_prompt"]
    assert "Current addendum being revised" in llm.prompts[0]
    assert current_addendum in llm.prompts[0]
    assert "Output a replacement addendum" in llm.prompts[0]
    assert "preserve the useful part" in llm.prompts[0]


def test_custom_proposer_replaces_duplicate_in_general_mode() -> None:
    llm = _FakeLLM("Prefer exact LEGEND aliases before generic roles.")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    dataset = {
        "heuristic_prompt": [
            {
                "Comparison": {"classification": "lost_baseline_solve"},
                "Feedback": "REGRESSION: base prompt solved but candidate failed.",
            }
        ]
    }

    first = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset=dataset,
        components_to_update=["heuristic_prompt"],
    )
    second = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset=dataset,
        components_to_update=["heuristic_prompt"],
    )

    assert second["heuristic_prompt"] != first["heuristic_prompt"]


def test_custom_proposer_receives_recent_candidate_outcomes() -> None:
    llm = _FakeLLM("Prefer exact LEGEND aliases before generic roles.")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    adapter.candidate_outcome_history = [
        "candidate=abc123 tasks=140 score=-0.25 new_solves=4 lost_solves=5"
    ]

    adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert "candidate=abc123 tasks=140 score=-0.25 new_solves=4 lost_solves=5" in llm.prompts[0]


def test_custom_proposer_fallback_preserves_current_addendum_when_repairing_errors() -> None:
    current_addendum = (
        "For persistent failures, test exactly one conservative prompt-level "
        "exploration hypothesis: before falling back to score alone, inspect "
        "WINCONDITIONS, RULES, LEGEND aliases, COLLISIONLAYERS, and object counts "
        "for a mechanics-grounded progress signal such as object-goal matching, "
        "player-to-interaction reachability, blocker or terrain distance, staged "
        "transformation progress, or score_normalized as a small tie-breaker. "
        "Use the signal only when its observable precondition is present, keep it "
        "finite/nonnegative, and otherwise preserve the base prompt behavior."
    )
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: imports are not allowed."
                    ),
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_prompt = result["heuristic_prompt"]
    assert "mechanics-grounded progress signal" in revised_prompt
    assert "No import statements" in revised_prompt
    assert len(revised_prompt.split(GEPA_ADDENDUM_HEADER, 1)[1].strip()) <= (
        DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS
    )


def test_custom_proposer_compacts_code_repair_to_preserve_long_addendum() -> None:
    current_addendum = (
        "Build one compact causal sketch from WINCONDITIONS, RULES, LEGEND "
        "aliases/composites, COLLISIONLAYERS, and the initial state, then choose "
        "a rule-grounded regime inside the same prompt. For stable pushable "
        "objects with monotonic target/exit progress and no creation/respawn "
        "aliases, use direct object-target matching, movement distance, and only "
        "provable deadlocks. If rules can transform, create, destroy, swap, pull, "
        "teleport, respawn, or hide win-required objects behind aliases/composites, "
        "avoid huge missing-object penalties unless no rule or alias can recreate "
        "the object; use finite interaction distance, count progress, and "
        "score_normalized as low-weight fallback. Use generic keyword roles only "
        "when WINCONDITIONS or RULES support that role. Use legal "
        "reachability/frontier distances only when terrain, gates, one-way effects, "
        "beams, gravity, or blockers make Manhattan distance misleading."
    )
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: imports are not allowed."
                    ),
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_addendum = result["heuristic_prompt"].split(GEPA_ADDENDUM_HEADER, 1)[1].strip()
    assert "compact causal sketch" in revised_addendum
    assert "no imports" in revised_addendum
    assert "non-finite" in revised_addendum
    assert len(revised_addendum) <= DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS


def test_custom_proposer_preserves_near_budget_addendum_when_repairing_code_errors() -> None:
    current_addendum = (
        "Preserve smooth ranking signal and reachability retention. "
        + "Keep the current rule-grounded efficiency preconditions. " * 18
    ).strip()
    assert len(current_addendum) < DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: imports are not allowed."
                    ),
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_addendum = result["heuristic_prompt"].split(GEPA_ADDENDUM_HEADER, 1)[1].strip()
    assert "Preserve smooth ranking signal" in revised_addendum
    assert "no imports" in revised_addendum
    assert "non-finite" in revised_addendum
    assert len(revised_addendum) <= DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS


def test_custom_proposer_does_not_duplicate_existing_compact_code_repair() -> None:
    compact_repair = (
        "Preserve current mechanics guidance, but enforce code safety: no "
        "imports/decorators/collections.deque or non-finite returns; use bounded "
        "finite values and local lists/dicts/sets/loops."
    )
    current_addendum = (
        "For common-solve efficiency, preserve smooth local ranking and "
        "mechanics-grounded interaction distances. "
        + "Keep the current rule-grounded efficiency preconditions. " * 12
        + compact_repair
    )
    assert len(current_addendum) < DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS
    assert len(current_addendum + " " + compact_repair) <= (
        DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS
    )
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: imports are not allowed."
                    ),
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_addendum = result["heuristic_prompt"].split(GEPA_ADDENDUM_HEADER, 1)[1].strip()
    assert revised_addendum.count(compact_repair) == 1
    assert len(revised_addendum) <= DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS


def test_custom_proposer_does_not_duplicate_semantic_code_safety_seed() -> None:
    current_addendum = Path(
        "configs/gepa_seed_addenda/interaction_alias_code_safety_probe.txt"
    ).read_text(encoding="utf-8")
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {"classification": "candidate_error"},
                    "Feedback": "CANDIDATE ERROR: imports are not allowed.",
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_addendum = result["heuristic_prompt"].split(GEPA_ADDENDUM_HEADER, 1)[1].strip()
    assert revised_addendum.count("no imports/decorators") == 1
    assert "Preserve current mechanics guidance" not in revised_addendum
    assert "bounded finite values and l " not in revised_addendum
    assert "local lists/dicts/sets/loops" in revised_addendum
    assert len(revised_addendum) <= DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS


def test_custom_proposer_preserves_long_seed_when_merging_noncode_fallback() -> None:
    current_addendum = (
        "For efficiency improvements, preserve local ordering signals that the base "
        "prompt often gets right before adding broader mechanics: when the win/rules "
        "imply the player must reach, clear, activate, enter, pick up, pull, swap, "
        "or move marked/goal/wall/switch/exit/crate/block objects, combine "
        "remaining-object counts with a bounded player-to-interaction distance "
        "instead of count-only or score-only progress. If RULES or LEGEND create "
        "transformed, carried, direction-specific, or collision-specific variants "
        "such as picked-up objects, player-state aliases, special walls, "
        "drop/marked objects, doors, switches, or terrain states, include those "
        "observable variants in the same branch before falling back to generic "
        "substring roles. Keep secondary existence/precondition terms finite and "
        "smooth: do not delete them when they protect solvability, but avoid "
        "1e4/1e6 penalties unless the rules prove an irreversible dead state. "
        "Use reachability/BFS only when collision layers, blockers, gates, terrain, "
        "or one-way effects make Manhattan distance misleading; otherwise prefer "
        "cheap matching plus interaction distance and a small score_normalized "
        "tie-breaker."
    )
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {"classification": "solved_regression"},
                    "Baseline Output": {
                        "code_shape": {"uses_reachability_search": True}
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_reachability_search": False,
                            "uses_count_terms": True,
                            "uses_distance_terms": True,
                        }
                    },
                    "Feedback": (
                        "EFFICIENCY REGRESSION: both prompts solved, but the "
                        "candidate expanded far more states after replacing base "
                        "reachability/BFS-style search with plain Manhattan/count terms."
                    ),
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_addendum = result["heuristic_prompt"].split(GEPA_ADDENDUM_HEADER, 1)[1].strip()
    assert "player-to-interaction distance" in revised_addendum
    assert "observable variants" in revised_addendum
    assert "preserve base reachability" in revised_addendum
    assert len(revised_addendum) <= DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS


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


def test_full_prompt_proposer_can_rewrite_every_instruction() -> None:
    replacement = (
        "Study the supplied game and write one finite non-negative Python heuristic. "
        "Return only the function and return zero for a winning state."
    )
    llm = _FakeLLM(replacement)
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        optimize_full_prompt=True,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] == replacement
    assert "rewrite the entire prompt" in llm.prompts[0].lower()


def test_full_prompt_proposer_retries_noop_and_duplicate_with_distinct_focus() -> None:
    replacement = "Write a compact, mechanics-derived heuristic and return only Python code."
    llm = _SequenceLLM(
        [PUZZLESCRIPT_HEURISTIC_CONTRACT, PUZZLESCRIPT_HEURISTIC_CONTRACT, replacement]
    )
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        optimize_full_prompt=True,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] == replacement
    assert len(llm.prompts) == 3
    assert len(set(llm.prompts)) == 3


def test_custom_proposer_uses_persistent_failure_exploration_for_noop_output() -> None:
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
    assert "persistent failures" in result["heuristic_prompt"]
    assert "mechanics-grounded progress signal" in result["heuristic_prompt"]
    assert "WINCONDITIONS" in result["heuristic_prompt"]


def test_custom_proposer_uses_regression_fallback_for_noop_output() -> None:
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
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Feedback": "REGRESSION: base prompt solved but candidate failed.",
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "REGRESSION" not in result["heuristic_prompt"]
    assert "precondition" in result["heuristic_prompt"]
    assert "WINCONDITIONS" in result["heuristic_prompt"]


def test_custom_proposer_uses_player_distance_fallback_for_solved_regression() -> None:
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
                    "Comparison": {"classification": "solved_regression"},
                    "Baseline Output": {
                        "code_shape": {"uses_player_interaction_distance": True}
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_player_interaction_distance": False,
                            "uses_count_terms": True,
                        }
                    },
                    "Feedback": (
                        "EFFICIENCY REGRESSION: base used player-to-interaction "
                        "distance but candidate collapsed to count-only progress."
                    ),
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "player-to-interaction distance" in result["heuristic_prompt"]
    assert "count-only" in result["heuristic_prompt"]
    assert "score_normalized" in result["heuristic_prompt"]


def test_custom_proposer_preserves_base_reachability_for_solved_regression() -> None:
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
                    "Comparison": {"classification": "solved_regression"},
                    "Baseline Output": {
                        "code_shape": {"uses_reachability_search": True}
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_reachability_search": False,
                            "uses_count_terms": True,
                            "uses_distance_terms": True,
                        }
                    },
                    "Feedback": (
                        "EFFICIENCY REGRESSION: both prompts solved, but the "
                        "candidate expanded 971 states versus 95 for the base. "
                        "Code-shape contrast: the base heuristic used "
                        "reachability/BFS-style search, while the candidate "
                        "used plain Manhattan/count terms."
                    ),
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "preserve base reachability" in result["heuristic_prompt"]
    assert "plain Manhattan" in result["heuristic_prompt"]


def test_custom_proposer_preserves_base_solves_when_combining_new_solve_and_efficiency() -> None:
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
                    "Comparison": {"classification": "new_solve"},
                    "Feedback": "POSITIVE EXAMPLE: candidate solved a new level.",
                },
                {
                    "Comparison": {"classification": "solved_regression"},
                    "Feedback": (
                        "EFFICIENCY REGRESSION: both prompts solved, but the "
                        "candidate expanded more."
                    ),
                },
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "new-solve" in result["heuristic_prompt"]
    assert "preserve base-solved behavior" in result["heuristic_prompt"]
    assert "before simplifying" in result["heuristic_prompt"]


def test_custom_proposer_uses_code_contract_fallback_for_lost_solve_errors() -> None:
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
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: imports are not allowed."
                    ),
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "No import statements" in result["heuristic_prompt"]
    assert "including inside" in result["heuristic_prompt"]
    assert "collections.deque" in result["heuristic_prompt"]
    assert "local lists" in result["heuristic_prompt"]
    assert "base-solved" in result["heuristic_prompt"]


def test_custom_proposer_does_not_treat_none_synthesis_error_as_code_error() -> None:
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
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": False,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate exhausted "
                        "the expansion budget without validation errors."
                    ),
                    "Generated Outputs": {"synthesis_error": None},
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "No import statements" not in result["heuristic_prompt"]
    assert "observable precondition" in result["heuristic_prompt"]


def test_custom_proposer_repairs_clean_lost_solve_without_returning_code_safe_seed() -> None:
    current_addendum = Path(
        "configs/gepa_seed_addenda/smooth_reach_code_safety_probe.txt"
    ).read_text(encoding="utf-8")
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": False,
                    },
                    "Feedback": (
                        "REGRESSION: base solved but candidate exhausted the budget "
                        "after adding approximate push reachability."
                    ),
                    "Baseline Output": {
                        "code_shape": {
                            "uses_reachability_search": False,
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                        }
                    },
                    "Generated Outputs": {
                        "synthesis_error": None,
                        "code_shape": {
                            "uses_reachability_search": True,
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                        },
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_addendum = result["heuristic_prompt"].split(GEPA_ADDENDUM_HEADER, 1)[1].strip()
    assert revised_addendum != current_addendum
    assert "simple base-style distance" in revised_addendum
    assert "No import statements" not in revised_addendum


def test_custom_proposer_combines_remote_motion_loss_and_code_contract_fallback() -> None:
    current_addendum = Path(
        "configs/gepa_seed_addenda/persistent_mechanics_probe.txt"
    ).read_text(encoding="utf-8")
    current_prompt = build_seed_candidate(current_addendum)["heuristic_prompt"]
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Comparison": {
                        "classification": "aggregate_summary",
                        "new_solve_count": 2,
                        "lost_baseline_solve_count": 3,
                    },
                    "Feedback": (
                        "AGGREGATE CANDIDATE SUMMARY: new_solves=2 "
                        "lost_base_solves=3\n"
                        "Mechanics losses: beam net=-2 new=0 lost=2"
                    ),
                },
                {
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: imports are not allowed."
                    ),
                    "Generated Outputs": {"synthesis_error": "imports are not allowed"},
                },
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    revised_prompt = result["heuristic_prompt"]
    assert "remote" in revised_prompt
    assert "carried" in revised_prompt
    assert "object names alone" in revised_prompt
    assert "prompt-internal" in revised_prompt
    assert "runner buckets" in revised_prompt
    assert "no imports" in revised_prompt.lower()
    assert "non-finite" in revised_prompt


def test_custom_proposer_uses_finite_contract_fallback_for_lost_solve_errors() -> None:
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
                    "Comparison": {
                        "classification": "lost_baseline_solve",
                        "candidate_error": True,
                    },
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate failed. "
                        "Heuristic validation failed before search: "
                        "direct non-finite heuristic returns are not allowed."
                    ),
                    "Generated Outputs": {
                        "synthesis_error": "direct non-finite heuristic returns are not allowed"
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "non-finite" in result["heuristic_prompt"]
    assert "float('inf')" in result["heuristic_prompt"]
    assert "bounded finite" in result["heuristic_prompt"]


def test_custom_proposer_uses_blocker_preservation_fallback_for_noop_output() -> None:
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
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Feedback": "REGRESSION: base prompt solved but candidate failed.",
                    "Baseline Output": {
                        "code_shape": {
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                            "uses_deadlock_checks": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_pushable_object_terms": False,
                            "uses_target_terms": False,
                            "uses_deadlock_checks": False,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "movable blockers" in result["heuristic_prompt"]
    assert "player-to-goal" in result["heuristic_prompt"]
    assert "crate-target" in result["heuristic_prompt"]


def test_custom_proposer_uses_alias_gate_preservation_fallback_for_noop_output() -> None:
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
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Feedback": "REGRESSION: base prompt solved but candidate failed.",
                    "Baseline Output": {
                        "code_shape": {
                            "uses_alias_specific_terms": True,
                            "uses_weighted_switch_terms": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_alias_specific_terms": False,
                            "uses_weighted_switch_terms": False,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "LEGEND aliases" in result["heuristic_prompt"]
    assert "weighted switch" in result["heuristic_prompt"]
    assert "prompt-internal" in result["heuristic_prompt"]


def test_custom_proposer_uses_gate_reachability_fallback_for_noop_output() -> None:
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
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Feedback": "REGRESSION: base prompt solved but candidate failed.",
                    "Baseline Output": {
                        "code_shape": {
                            "uses_gate_aware_reachability": True,
                            "uses_weighted_switch_terms": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_gate_aware_reachability": False,
                            "uses_weighted_switch_terms": True,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "gate-aware reachability" in result["heuristic_prompt"]
    assert "open/closed doors" in result["heuristic_prompt"]
    assert "Manhattan" in result["heuristic_prompt"]


def test_custom_proposer_rejects_overstrict_role_precondition_addendum() -> None:
    bad_addendum = (
        "Before adding any role-specific term, require the role name to appear in "
        "WINCONDITIONS and require RULES to have a right-hand side that can produce "
        "that role. Only when both checks succeed may the heuristic use distance or "
        "reachability for the role."
    )
    llm = _FakeLLM(bad_addendum)
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
                    "Baseline Output": {
                        "code_shape": {
                            "uses_alias_specific_terms": True,
                            "uses_weighted_switch_terms": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_alias_specific_terms": False,
                            "uses_weighted_switch_terms": False,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "both checks succeed" not in result["heuristic_prompt"]
    assert "right-hand side" not in result["heuristic_prompt"]
    assert "weighted switch" in result["heuristic_prompt"]


def test_custom_proposer_uses_reachability_overfit_fallback_for_noop_output() -> None:
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
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Feedback": (
                        "REGRESSION: base prompt solved but candidate exhausted search "
                        "after adding approximate push reachability."
                    ),
                    "Baseline Output": {
                        "code_shape": {
                            "uses_reachability_search": False,
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_reachability_search": True,
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "simple base-style distance" in result["heuristic_prompt"]
    assert "reachability" in result["heuristic_prompt"]
    assert "player-only" in result["heuristic_prompt"]


def test_case_specific_overfit_can_drop_a_regressed_seed_addendum() -> None:
    seed = build_seed_candidate("Use an intentionally bad case-wide shortcut.")
    llm = _FakeLLM(seed["heuristic_prompt"])
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        allow_case_specific_overfit=True,
    )

    result = adapter.propose_new_texts(
        candidate=seed,
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Inputs": {"game": "Beam_Islands", "level": 3},
                    "Comparison": {"classification": "lost_baseline_solve"},
                    "Baseline Output": {"heuristic_code": "return baseline_distance"},
                    "Feedback": "The base solved, but the seeded prompt failed.",
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"].strip() == PUZZLESCRIPT_HEURISTIC_CONTRACT.strip()
    assert "single-case overfitting diagnostic" in llm.prompts[0]
    assert "Memorization is allowed" in llm.prompts[0]


def test_case_specific_overfit_carries_baseline_code_after_base_noop() -> None:
    llm = _FakeLLM(PUZZLESCRIPT_HEURISTIC_CONTRACT)
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        allow_case_specific_overfit=True,
    )

    result = adapter.propose_new_texts(
        candidate={"heuristic_prompt": PUZZLESCRIPT_HEURISTIC_CONTRACT},
        reflective_dataset={
            "heuristic_prompt": [
                {
                    "Inputs": {"game": "Beam_Islands", "level": 3},
                    "Comparison": {"classification": "solved_regression"},
                    "Baseline Output": {
                        "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                        "    return 10.0 * len(ctx.get('beam', [])) + "
                        "(1.0 - ctx.get('score_normalized', 0.0)) * 5.0"
                    },
                    "Feedback": "The fresh base expanded more states than its reference.",
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "Beam_Islands level 3" in result["heuristic_prompt"]
    assert "known-successful baseline heuristic" in result["heuristic_prompt"]
    assert "return 10.0 * len" in result["heuristic_prompt"]
    assert "omit the score_normalized penalty" in result["heuristic_prompt"]


def test_case_specific_overfit_replaces_a_repeated_rejected_proposal() -> None:
    repeated_addendum = "Use the same rejected Beam distance formula again."
    llm = _FakeLLM(repeated_addendum)
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=Path("/tmp/gepa-state"),
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        allow_case_specific_overfit=True,
    )
    records = {
        "heuristic_prompt": [
            {
                "Inputs": {"game": "Beam_Islands", "level": 3},
                "Comparison": {"classification": "solved_regression"},
                "Baseline Output": {
                    "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                    "    return 10.0 * len(ctx.get('beam', []))"
                },
            }
        ]
    }

    first = adapter.propose_new_texts(
        candidate=build_seed_candidate("Current accepted parent."),
        reflective_dataset=records,
        components_to_update=["heuristic_prompt"],
    )
    second = adapter.propose_new_texts(
        candidate=build_seed_candidate("A different accepted parent."),
        reflective_dataset=records,
        components_to_update=["heuristic_prompt"],
    )

    assert repeated_addendum in first["heuristic_prompt"]
    assert "known-successful baseline heuristic" in second["heuristic_prompt"]
    assert "return 10.0 * len" in second["heuristic_prompt"]


def test_custom_proposer_uses_reachability_overfit_fallback_for_solved_regression() -> None:
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
                    "Comparison": {"classification": "solved_regression"},
                    "Feedback": (
                        "EFFICIENCY REGRESSION: both prompts solved, but the candidate "
                        "expanded 2186 states versus 313 for the base after adding BFS."
                    ),
                    "Baseline Output": {
                        "code_shape": {
                            "uses_reachability_search": False,
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                            "uses_player_interaction_distance": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_reachability_search": True,
                            "uses_pushable_object_terms": True,
                            "uses_target_terms": True,
                            "uses_player_interaction_distance": True,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "base solved without it" in result["heuristic_prompt"]
    assert "approximate reachability" in result["heuristic_prompt"]
    assert "simple base-style distance" in result["heuristic_prompt"]


def test_custom_proposer_uses_transition_fallback_for_solved_regression() -> None:
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
                    "Comparison": {"classification": "solved_regression"},
                    "Feedback": (
                        "Common-solve efficiency diagnosis: base modeled action "
                        "transitions such as push and water/fill effects that the "
                        "candidate omitted."
                    ),
                    "Baseline Output": {
                        "code_shape": {
                            "uses_action_transition_terms": True,
                            "uses_reachability_search": False,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_action_transition_terms": False,
                            "uses_reachability_search": False,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "action-transition costs" in result["heuristic_prompt"]
    assert "state-changing RULES" in result["heuristic_prompt"]
    assert "player-only distance" in result["heuristic_prompt"]


def test_custom_proposer_uses_assignment_fallback_for_solved_regression() -> None:
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
                    "Comparison": {"classification": "solved_regression"},
                    "Feedback": (
                        "Common-solve efficiency diagnosis: base used explicit "
                        "object-target assignment/matching that the candidate weakened."
                    ),
                    "Baseline Output": {
                        "code_shape": {
                            "uses_assignment_matching": True,
                            "uses_reachability_search": False,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_assignment_matching": False,
                            "uses_reachability_search": False,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "object-target assignment or matching" in result["heuristic_prompt"]
    assert "multiple movable objects and goals" in result["heuristic_prompt"]
    assert "nearest-object" in result["heuristic_prompt"]


def test_custom_proposer_uses_aggregate_assignment_loss_fallback() -> None:
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
                    "Comparison": {
                        "classification": "aggregate_summary",
                        "code_shape_loss_counts": {
                            "uses_assignment_matching": 7,
                            "uses_action_transition_terms": 3,
                        },
                    },
                    "Feedback": (
                        "AGGREGATE CANDIDATE SUMMARY\n"
                        "Code-shape losses: uses_assignment_matching=7; "
                        "uses_action_transition_terms=3"
                    ),
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "object-target assignment or matching" in result["heuristic_prompt"]
    assert "multiple movable objects and goals" in result["heuristic_prompt"]
    assert "nearest-object" in result["heuristic_prompt"]


def test_custom_proposer_uses_transformed_variant_fallback_for_solved_regression() -> None:
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
                    "Comparison": {"classification": "solved_regression"},
                    "Feedback": (
                        "Common-solve efficiency diagnosis: base modeled "
                        "carried/transformed object variants that candidate reduced "
                        "to plain object roles."
                    ),
                    "Baseline Output": {
                        "code_shape": {
                            "uses_transformed_object_terms": True,
                            "uses_alias_specific_terms": True,
                        }
                    },
                    "Generated Outputs": {
                        "code_shape": {
                            "uses_transformed_object_terms": False,
                            "uses_alias_specific_terms": False,
                        }
                    },
                }
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert "carried/transformed object variants" in result["heuristic_prompt"]
    assert "picked up, carried, dropped, or transformed" in result["heuristic_prompt"]
    assert "plain object roles" in result["heuristic_prompt"]


def test_custom_proposer_preserves_new_solve_when_efficiency_feedback_is_mixed() -> None:
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
                    "Comparison": {"classification": "new_solve"},
                    "Feedback": "POSITIVE EXAMPLE: candidate solved while base failed.",
                },
                {
                    "Comparison": {"classification": "solved_regression"},
                    "Feedback": (
                        "EFFICIENCY REGRESSION: both prompts solved this level, "
                        "but candidate was materially worse."
                    ),
                },
            ]
        },
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != PUZZLESCRIPT_HEURISTIC_CONTRACT
    assert "new-solve" in result["heuristic_prompt"]
    assert "expensive tie-breakers" in result["heuristic_prompt"]
    assert "score-only fallback" in result["heuristic_prompt"]


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
    assert "collections.deque" in result["heuristic_prompt"]
    assert "decorators" in result["heuristic_prompt"]


def test_repair_prompt_restates_code_contract_requirements() -> None:
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="FiniteRepair",
        level=0,
        budget=100,
        env_description="WINCONDITIONS\nAll goal on target.",
        game_text_path="/tmp/game.txt",
    )

    prompt = build_repair_prompt(
        PUZZLESCRIPT_HEURISTIC_CONTRACT,
        task,
        "def heuristic_cost_to_go(ts, env_params, ctx):\n    return float('inf')",
        "direct non-finite heuristic returns are not allowed",
    )

    assert "No import statements" in prompt
    assert "float('inf')" in prompt
    assert "non-finite" in prompt
    assert "bounded finite" in prompt
    assert "plain local lists" in prompt


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


def test_custom_proposer_allows_rule_derived_conditional_addendum() -> None:
    current_prompt = PUZZLESCRIPT_HEURISTIC_CONTRACT
    llm = _FakeLLM(
        "Use conditional prompt routing: if WINCONDITIONS and RULES prove stable "
        "pushable objects with monotonic target progress, emphasize box-target "
        "matching; otherwise keep interaction distance and score progress secondary."
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
        candidate={"heuristic_prompt": current_prompt},
        reflective_dataset={"heuristic_prompt": []},
        components_to_update=["heuristic_prompt"],
    )

    assert result["heuristic_prompt"] != current_prompt
    assert "conditional prompt routing" in result["heuristic_prompt"]


def test_initial_addendum_file_preserves_prompt_routing_text(tmp_path: Path) -> None:
    addendum = (
        "Let the heuristic design use whatever prompt-level structure the game text supports. "
        "If categories, conditionals, or routing are useful, define them from observable "
        "WINCONDITIONS, RULES, LEGEND aliases, COLLISIONLAYERS, and state properties."
    )
    addendum_path = tmp_path / "seed_addendum.txt"
    addendum_path.write_text(addendum, encoding="utf-8")

    resolved = read_initial_gepa_addendum("", addendum_path)
    prompt = build_seed_candidate(resolved)["heuristic_prompt"]

    assert resolved == addendum
    assert "If categories, conditionals, or routing are useful" in prompt
    assert "COLLISIONLAYERS, and state properties" in prompt


def test_initial_addendum_file_accepts_full_seed_prompt(tmp_path: Path) -> None:
    prompt_path = tmp_path / "best_prompt.txt"
    prompt_path.write_text(
        PUZZLESCRIPT_HEURISTIC_CONTRACT
        + "\n\n"
        + GEPA_ADDENDUM_HEADER
        + "\nPreserve quantitative win-condition terms.",
        encoding="utf-8",
    )

    resolved = read_initial_gepa_addendum("", prompt_path)

    assert resolved == "Preserve quantitative win-condition terms."


def test_code_contract_seed_file_builds_valid_prompt() -> None:
    addendum = Path("configs/gepa_seed_addenda/code_contract_repair.txt").read_text(
        encoding="utf-8"
    )

    prompt = build_seed_candidate(addendum)["heuristic_prompt"]

    assert "No import statements" in prompt
    assert "collections.deque" in prompt
    assert "non-finite" in prompt
    assert "Additional GEPA guidance" in prompt


def test_smooth_reach_code_safety_seed_file_builds_valid_prompt() -> None:
    addendum = Path("configs/gepa_seed_addenda/smooth_reach_code_safety_probe.txt").read_text(
        encoding="utf-8"
    )

    prompt = build_seed_candidate(addendum)["heuristic_prompt"]

    assert "smooth move-scale ranking" in prompt
    assert "finite local reachability" in prompt
    assert "no imports" in prompt
    assert "non-finite returns" in prompt
    assert "Additional GEPA guidance" in prompt


def test_interaction_alias_code_safety_seed_file_builds_valid_prompt() -> None:
    addendum = Path(
        "configs/gepa_seed_addenda/interaction_alias_code_safety_probe.txt"
    ).read_text(encoding="utf-8")

    prompt = build_seed_candidate(addendum)["heuristic_prompt"]

    assert "player-to-interaction distance" in prompt
    assert "observable variants" in prompt
    assert "no imports" in prompt
    assert "non-finite returns" in prompt
    assert "Additional GEPA guidance" in prompt


def test_adaptive_causal_code_safety_seed_file_builds_valid_prompt() -> None:
    addendum = Path(
        "configs/gepa_seed_addenda/adaptive_causal_code_safety_probe.txt"
    ).read_text(encoding="utf-8")

    prompt = build_seed_candidate(addendum)["heuristic_prompt"]

    assert "compact causal sketch" in prompt
    assert "rule-grounded regime" in prompt
    assert "no imports" in prompt
    assert "non-finite returns" in prompt
    assert "Additional GEPA guidance" in prompt


def test_sharp_interaction_reach_code_safety_seed_file_builds_valid_prompt() -> None:
    addendum = Path(
        "configs/gepa_seed_addenda/sharp_interaction_reach_code_safety_probe.txt"
    ).read_text(encoding="utf-8")

    prompt = build_seed_candidate(addendum)["heuristic_prompt"]

    assert "base-simple ranking primary" in prompt
    assert "player-to-interaction" in prompt
    assert "Add reachability/BFS only when" in prompt
    assert "observable WINCONDITIONS/RULES preconditions" in prompt
    assert "no imports" in prompt
    assert "non-finite returns" in prompt
    assert len(addendum) <= DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS
    assert "Additional GEPA guidance" in prompt


def test_initial_addendum_rejects_inline_and_file(tmp_path: Path) -> None:
    addendum_path = tmp_path / "seed_addendum.txt"
    addendum_path.write_text("Use score_normalized only as a tie-breaker.", encoding="utf-8")

    with pytest.raises(ValueError, match="either inline"):
        read_initial_gepa_addendum("Inline addendum.", addendum_path)


def test_load_scoring_baseline_outputs_orders_current_tasks(tmp_path: Path) -> None:
    baseline_path = tmp_path / "scoring_baseline_outputs.json"
    baseline_path.write_text(
        """[
  {"game": "extra", "level": 0, "score": 0.1},
  {"game": "b", "level": 2, "score": 0.8},
  {"game": "a", "level": 1, "score": 0.7}
]""",
        encoding="utf-8",
    )
    tasks = [
        PuzzleScriptLevelTask(0, "a", 1, 100, "env-a", "a.txt"),
        PuzzleScriptLevelTask(1, "b", 2, 100, "env-b", "b.txt"),
    ]

    rows = load_scoring_baseline_outputs(baseline_path, tasks)

    assert [(row["game"], row["level"]) for row in rows] == [("a", 1), ("b", 2)]
    assert [row["score"] for row in rows] == [0.7, 0.8]


def test_load_scoring_baseline_outputs_rejects_missing_task(tmp_path: Path) -> None:
    baseline_path = tmp_path / "scoring_baseline_outputs.json"
    baseline_path.write_text('[{"game": "a", "level": 1, "score": 0.7}]', encoding="utf-8")
    tasks = [
        PuzzleScriptLevelTask(0, "a", 1, 100, "env-a", "a.txt"),
        PuzzleScriptLevelTask(1, "missing", 2, 100, "env-b", "b.txt"),
    ]

    with pytest.raises(RuntimeError, match="missing:2"):
        load_scoring_baseline_outputs(baseline_path, tasks)


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


def test_seeded_prompt_evaluation_reuses_stored_baseline_outputs(tmp_path: Path) -> None:
    seed_candidate = build_seed_candidate("Use exact LEGEND aliases before substring guesses.")
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
                "game": "seeded-game",
                "level": 4,
                "score": 0.5,
                "solved": False,
                "expanded": 34,
                "generated": 45,
                "solution_length": 0,
                "partial_progress_score": 0.25,
                "feedback": "seeded baseline",
                "error": None,
            }
        ],
        candidate=seed_candidate,
    )
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="seeded-game",
        level=4,
        budget=100,
        env_description="Win conditions: all crates on targets",
        game_text_path=str(tmp_path / "game.txt"),
    )

    batch = adapter.evaluate(
        batch=[task],
        candidate=seed_candidate,
        capture_traces=False,
    )

    assert llm.prompts == []
    assert batch.outputs[0]["task_id"] == 0
    assert batch.outputs[0]["score"] == 0.5
    assert batch.outputs[0]["solved"] is False
    assert batch.outputs[0]["baseline_solved"] is False
    eval_dirs = list((tmp_path / "candidate_evals").glob("eval-*"))
    assert len(eval_dirs) == 1
    assert (eval_dirs[0] / "baseline_reuse.json").exists()


def test_seeded_prompt_reuse_can_score_against_original_base_outputs(tmp_path: Path) -> None:
    seed_candidate = build_seed_candidate("Use exact LEGEND aliases before substring guesses.")
    llm = _FakeLLM("def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,  # type: ignore[arg-type]
        state_root=tmp_path,
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        lost_solve_penalty=5.0,
        score_delta_weight=1.0,
        score_delta_clip=1.0,
        partial_progress_weight=0.1,
    )
    adapter.set_baseline_outputs(
        [
            {
                "task_id": 99,
                "game": "seeded-game",
                "level": 4,
                "score": 0.5,
                "solved": False,
                "expanded": 34,
                "generated": 45,
                "solution_length": 0,
                "partial_progress_score": 0.25,
                "feedback": "seeded baseline",
                "error": None,
            }
        ],
        candidate=seed_candidate,
        scoring_outputs=[
            {
                "task_id": 11,
                "game": "seeded-game",
                "level": 4,
                "score": 0.9,
                "solved": True,
                "expanded": 12,
                "generated": 20,
                "solution_length": 6,
                "partial_progress_score": 1.0,
                "feedback": "original base solved",
                "error": None,
            }
        ],
    )
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="seeded-game",
        level=4,
        budget=100,
        env_description="Win conditions: all crates on targets",
        game_text_path=str(tmp_path / "game.txt"),
    )

    batch = adapter.evaluate(
        batch=[task],
        candidate=seed_candidate,
        capture_traces=False,
    )

    assert llm.prompts == []
    assert batch.outputs[0]["solved"] is False
    assert batch.outputs[0]["baseline_solved"] is True
    assert batch.outputs[0]["baseline_score"] == pytest.approx(0.9)
    assert batch.scores[0] < -5.0


def test_candidate_evaluation_can_preload_scoring_baseline_before_reuse_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_candidate = build_seed_candidate("Use exact LEGEND aliases before substring guesses.")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=_FakeLLM(""),
        state_root=tmp_path,
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
        lost_solve_penalty=5.0,
        score_delta_weight=1.0,
        score_delta_clip=1.0,
    )
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="seeded-game",
        level=4,
        budget=100,
        env_description="Win conditions: all crates on targets",
        game_text_path=str(tmp_path / "game.txt"),
    )
    code_path = tmp_path / "seeded_heuristic.py"
    code_path.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n",
        encoding="utf-8",
    )

    adapter.set_scoring_baseline_outputs(
        [
            {
                "task_id": 11,
                "game": "seeded-game",
                "level": 4,
                "score": 0.9,
                "solved": True,
                "expanded": 12,
                "generated": 20,
                "solution_length": 6,
                "partial_progress_score": 1.0,
                "feedback": "original base solved",
                "error": None,
            }
        ]
    )

    def fake_synthesize_batch(**_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "task_id": 0,
                "game": "seeded-game",
                "level": 4,
                "budget": 100,
                "env_description": task.env_description,
                "game_text_path": task.game_text_path,
                "heuristic_code_path": str(code_path),
                "synthesis_error": None,
            }
        ]

    def fake_run_search(**_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "task_id": 0,
                "game": "seeded-game",
                "level": 4,
                "score": 0.5,
                "solved": False,
                "expanded": 34,
                "generated": 45,
                "solution_length": 0,
                "partial_progress_score": 0.25,
                "feedback": "seeded baseline",
                "error": None,
                "heuristic_code_path": str(code_path),
            }
        ]

    monkeypatch.setattr(adapter, "_synthesize_batch", fake_synthesize_batch)
    monkeypatch.setattr(adapter, "_run_search", fake_run_search)

    batch = adapter.evaluate(batch=[task], candidate=seed_candidate, capture_traces=False)

    assert batch.outputs[0]["baseline_solved"] is True
    assert batch.outputs[0]["baseline_score"] == pytest.approx(0.9)
    assert batch.scores[0] < -5.0
    eval_dirs = list((tmp_path / "candidate_evals").glob("eval-*"))
    assert len(eval_dirs) == 1
    scored = (eval_dirs[0] / "scored_results.json").read_text(encoding="utf-8")
    assert "baseline_solved" in scored


def test_nonbaseline_candidate_evaluation_reuses_exact_candidate_task_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = build_seed_candidate("Use role guards before generic keyword helpers.")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=_FakeLLM(""),
        state_root=tmp_path,
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    task = PuzzleScriptLevelTask(
        task_id=0,
        game="cache-game",
        level=2,
        budget=100,
        env_description="Win conditions: all crates on targets",
        game_text_path=str(tmp_path / "game.txt"),
    )
    code_path = tmp_path / "cached_heuristic.py"
    code_path.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n",
        encoding="utf-8",
    )
    calls = {"synthesize": 0, "search": 0}

    def fake_synthesize_batch(**_kwargs: object) -> list[dict[str, object]]:
        calls["synthesize"] += 1
        return [
            {
                "task_id": 0,
                "game": "cache-game",
                "level": 2,
                "budget": 100,
                "env_description": task.env_description,
                "game_text_path": task.game_text_path,
                "heuristic_code_path": str(code_path),
                "synthesis_error": None,
            }
        ]

    def fake_run_search(**_kwargs: object) -> list[dict[str, object]]:
        calls["search"] += 1
        return [
            {
                "task_id": 0,
                "game": "cache-game",
                "level": 2,
                "score": 0.75,
                "solved": True,
                "expanded": 12,
                "generated": 18,
                "solution_length": 4,
                "partial_progress_score": 1.0,
                "feedback": "candidate solved",
                "error": None,
                "heuristic_code_path": str(code_path),
            }
        ]

    monkeypatch.setattr(adapter, "_synthesize_batch", fake_synthesize_batch)
    monkeypatch.setattr(adapter, "_run_search", fake_run_search)

    first = adapter.evaluate(batch=[task], candidate=candidate, capture_traces=False)
    second = adapter.evaluate(batch=[task], candidate=candidate, capture_traces=True)

    assert calls == {"synthesize": 1, "search": 1}
    assert first.outputs[0]["score"] == 0.75
    assert second.outputs[0]["score"] == 0.75
    assert second.trajectories is not None
    assert "return 1.0" in second.trajectories[0]["heuristic_code"]
    eval_dirs = sorted((tmp_path / "candidate_evals").glob("eval-*"))
    assert len(eval_dirs) == 2
    assert (eval_dirs[1] / "candidate_reuse.json").exists()


def test_candidate_hash_ignores_surrounding_prompt_whitespace() -> None:
    first = {"heuristic_prompt": "same prompt"}
    second = {"heuristic_prompt": "\n  same prompt  \n"}

    assert PuzzleScriptBatchedGEPAAdapter._candidate_hash(first) == (
        PuzzleScriptBatchedGEPAAdapter._candidate_hash(second)
    )


def test_nonbaseline_candidate_evaluation_reuses_overlapping_task_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = build_seed_candidate("Use role guards before generic keyword helpers.")
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=_FakeLLM(""),
        state_root=tmp_path,
        script_doctor=Path("/tmp/script-doctor"),
        search_config=SimpleNamespace(),  # type: ignore[arg-type]
        llm_concurrency=1,
        astar_timeout_s=1.0,
    )
    tasks = [
        PuzzleScriptLevelTask(
            task_id=index,
            game=f"cache-game-{index}",
            level=index,
            budget=100,
            env_description="Win conditions: all crates on targets",
            game_text_path=str(tmp_path / f"game-{index}.txt"),
        )
        for index in range(3)
    ]
    code_paths = []
    for task in tasks:
        code_path = tmp_path / f"cached_heuristic_{task.task_id}.py"
        code_path.write_text(
            f"def heuristic_cost_to_go(ts, env_params, ctx):\n    return {task.task_id}.0\n",
            encoding="utf-8",
        )
        code_paths.append(code_path)
    synthesized_batches: list[list[int]] = []
    searched_batches: list[list[int]] = []

    def fake_synthesize_batch(**kwargs: object) -> list[dict[str, object]]:
        batch = list(kwargs["batch"])  # type: ignore[index]
        synthesized_batches.append([task.task_id for task in batch])
        return [
            {
                "task_id": task.task_id,
                "game": task.game,
                "level": task.level,
                "budget": task.budget,
                "env_description": task.env_description,
                "game_text_path": task.game_text_path,
                "heuristic_code_path": str(code_paths[task.task_id]),
                "synthesis_error": None,
            }
            for task in batch
        ]

    def fake_run_search(**kwargs: object) -> list[dict[str, object]]:
        task_rows = list(kwargs["task_rows"])  # type: ignore[index]
        searched_batches.append([int(row["task_id"]) for row in task_rows])
        return [
            {
                "task_id": int(row["task_id"]),
                "game": str(row["game"]),
                "level": int(row["level"]),
                "score": 0.75 + int(row["task_id"]) / 100.0,
                "solved": True,
                "expanded": 12 + int(row["task_id"]),
                "generated": 18,
                "solution_length": 4,
                "partial_progress_score": 1.0,
                "feedback": "candidate solved",
                "error": None,
                "heuristic_code_path": str(row["heuristic_code_path"]),
            }
            for row in task_rows
        ]

    monkeypatch.setattr(adapter, "_synthesize_batch", fake_synthesize_batch)
    monkeypatch.setattr(adapter, "_run_search", fake_run_search)

    first = adapter.evaluate(batch=tasks[:2], candidate=candidate, capture_traces=False)
    second = adapter.evaluate(batch=tasks[1:], candidate=candidate, capture_traces=True)

    assert synthesized_batches == [[0, 1], [2]]
    assert searched_batches == [[0, 1], [2]]
    assert [row["task_id"] for row in first.outputs] == [0, 1]
    assert [row["task_id"] for row in second.outputs] == [1, 2]
    assert [row["score"] for row in second.outputs] == pytest.approx([0.76, 0.77])
    assert second.trajectories is not None
    assert "return 1.0" in second.trajectories[0]["heuristic_code"]
    eval_dirs = sorted((tmp_path / "candidate_evals").glob("eval-*"))
    assert len(eval_dirs) == 2
    reuse_metadata = (eval_dirs[1] / "candidate_reuse.json").read_text(encoding="utf-8")
    assert "per-task candidate prompt cache" in reuse_metadata


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


def test_reflection_feedback_compares_replicates_and_baseline_trace() -> None:
    feedback = build_reflection_feedback(
        {
            "solved": False,
            "solve_rate": 0.2,
            "replicate_count": 5,
            "expanded": 900,
            "replicate_expansions": [800, 900, 1000],
            "trace_summary": {"terminated_reason": "expansion_budget"},
            "baseline_solved": True,
            "baseline_solve_rate": 0.8,
            "baseline_replicate_count": 5,
            "baseline_expanded": 300,
            "baseline_replicate_expansions": [250, 300, 350],
            "baseline_trace_summary": {"terminated_reason": "solved"},
        },
        "lost_baseline_solve",
    )

    assert "solve_rate=0.800" in feedback
    assert "solve_rate=0.200" in feedback
    assert "baseline_termination=solved" in feedback
    assert "candidate_termination=expansion_budget" in feedback
    assert "baseline_expansions=[250, 300, 350]" in feedback


def test_filter_unlearnable_tasks_drops_only_exhaustive_state_spaces() -> None:
    tasks = [
        PuzzleScriptLevelTask(i, "g", i, 100, "env", "/tmp/game.txt")
        for i in range(3)
    ]
    baseline = [
        {
            "game": "g",
            "level": 0,
            "replicate_results": [
                {"trace_summary": {"terminated_reason": "open_set_exhausted"}},
                {"trace_summary": {"terminated_reason": "open_set_exhausted"}},
            ],
        },
        {
            "game": "g",
            "level": 1,
            "replicate_results": [
                {"trace_summary": {"terminated_reason": "expansion_budget"}}
            ],
        },
        {"game": "g", "level": 2, "error": "compile failed"},
    ]

    kept, dropped = filter_unlearnable_tasks(tasks, baseline)

    assert [task.level for task in kept] == [1, 2]
    assert [task.level for task in dropped] == [0]


def test_generalizing_selection_rejects_one_game_efficiency_outlier() -> None:
    baseline = [
        {
            "game": game,
            "level": 0,
            "solve_rate": 1.0,
            "baseline_solve_rate": 1.0,
            "solved_expanded_mean": 100.0,
            "baseline_solved_expanded_mean": 100.0,
        }
        for game in ("a", "b", "c", "d")
    ]
    outlier = [dict(row) for row in baseline]
    outlier[0]["solved_expanded_mean"] = 1.0
    for row in outlier[1:]:
        row["solved_expanded_mean"] = 120.0

    selected, diagnostics = select_generalizing_candidate([baseline, outlier])

    assert selected == 0
    assert diagnostics[1]["positive_games"] == 1
    assert diagnostics[1]["negative_games"] == 3


def test_build_reflection_feedback_includes_solved_efficiency_gain_guidance() -> None:
    feedback = build_reflection_feedback(
        {
            "score": 0.91,
            "solved": True,
            "expanded": 300,
            "baseline_score": 0.86,
            "baseline_solved": True,
            "baseline_expanded": 900,
        },
        "solved_efficiency_gain",
    )

    assert "EFFICIENCY GAIN" in feedback
    assert "both prompts solved" in feedback
    assert "preserve the structural ordering" in feedback
    assert "high-headroom common-solve" in feedback
    assert "expanded=900" in feedback
    assert "expanded=300" in feedback


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


def test_select_reflection_traces_prefers_strong_efficiency_examples() -> None:
    weak_gain = {
        "task": {"game": "weak-gain", "level": 0},
        "result": {
            "score": 0.10,
            "solved": True,
            "baseline_solved": True,
            "expanded": 500,
            "baseline_expanded": 650,
        },
    }
    strong_gain = {
        "task": {"game": "strong-gain", "level": 0},
        "result": {
            "score": 0.95,
            "solved": True,
            "baseline_solved": True,
            "expanded": 200,
            "baseline_expanded": 4000,
        },
    }

    selected = select_reflection_traces([weak_gain, strong_gain], max_records=1)

    assert selected[0]["task"]["game"] == "strong-gain"


def test_select_reflection_traces_balances_efficiency_gains_and_regressions() -> None:
    """Keep repeated solved-slowdown causes visible when gains are plentiful."""

    trajectories = []
    for idx in range(8):
        trajectories.append(
            {
                "task": {"game": f"gain-{idx}", "level": idx},
                "result": {
                    "score": 0.95,
                    "solved": True,
                    "baseline_solved": True,
                    "expanded": 50 + idx,
                    "baseline_expanded": 4_000 + idx,
                },
            }
        )
    for idx in range(4):
        trajectories.append(
            {
                "task": {"game": f"regression-{idx}", "level": idx},
                "result": {
                    "score": 0.20,
                    "baseline_score": 0.90,
                    "solved": True,
                    "baseline_solved": True,
                    "expanded": 2_000 + idx,
                    "baseline_expanded": 200 + idx,
                },
            }
        )

    selected = select_reflection_traces(trajectories, max_records=6)
    selected_classes = [trace_classification(trace) for trace in selected]

    assert selected_classes.count("solved_regression") >= 3
    assert selected_classes.count("solved_efficiency_gain") >= 2


def test_select_reflection_traces_prefers_dropped_transformed_variant_regression(
    tmp_path: Path,
) -> None:
    base_code = tmp_path / "base_variant.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    obj_pos = ctx.get('object_positions', {})\n"
        "    crates = obj_pos.get('crate', []) + obj_pos.get('pickedup', [])\n"
        "    return float(len(crates))\n",
        encoding="utf-8",
    )
    generic_regression = {
        "task": {"game": "generic-regression", "level": 0},
        "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n",
        "result": {
            "score": 0.20,
            "baseline_score": 0.90,
            "solved": True,
            "baseline_solved": True,
            "expanded": 5_000,
            "baseline_expanded": 500,
        },
    }
    transformed_regression = {
        "task": {"game": "transformed-regression", "level": 0},
        "heuristic_code": (
            "def heuristic_cost_to_go(ts, env_params, ctx):\n"
            "    crates = ctx.get('object_positions', {}).get('crate', [])\n"
            "    return float(len(crates))\n"
        ),
        "result": {
            "score": 0.70,
            "baseline_score": 0.90,
            "solved": True,
            "baseline_solved": True,
            "expanded": 320,
            "baseline_expanded": 160,
            "baseline_heuristic_code_path": str(base_code),
        },
    }

    selected = select_reflection_traces(
        [generic_regression, transformed_regression],
        max_records=1,
    )

    assert selected[0]["task"]["game"] == "transformed-regression"


def test_select_reflection_traces_prefers_dropped_assignment_regression(
    tmp_path: Path,
) -> None:
    base_code = tmp_path / "base_assignment.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    remaining_crates = ctx.get('object_positions', {}).get('crate', [])\n"
        "    remaining_targets = ctx.get('object_positions', {}).get('target', [])\n"
        "    best_sum = 0\n"
        "    for perm in permutations(remaining_crates):\n"
        "        best_sum += len(remaining_targets)\n"
        "    return float(best_sum)\n",
        encoding="utf-8",
    )
    generic_regression = {
        "task": {"game": "generic-regression", "level": 0},
        "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n",
        "result": {
            "score": 0.20,
            "baseline_score": 0.90,
            "solved": True,
            "baseline_solved": True,
            "expanded": 5_000,
            "baseline_expanded": 500,
        },
    }
    assignment_regression = {
        "task": {"game": "assignment-regression", "level": 0},
        "heuristic_code": (
            "def heuristic_cost_to_go(ts, env_params, ctx):\n"
            "    crates = ctx.get('object_positions', {}).get('crate', [])\n"
            "    return float(len(crates))\n"
        ),
        "result": {
            "score": 0.70,
            "baseline_score": 0.90,
            "solved": True,
            "baseline_solved": True,
            "expanded": 320,
            "baseline_expanded": 160,
            "baseline_heuristic_code_path": str(base_code),
        },
    }

    selected = select_reflection_traces(
        [generic_regression, assignment_regression],
        max_records=1,
    )

    assert selected[0]["task"]["game"] == "assignment-regression"


def test_select_reflection_traces_prefers_actionable_lost_solve_shape_loss(
    tmp_path: Path,
) -> None:
    base_code = tmp_path / "base_lost_assignment.py"
    base_code.write_text(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    remaining_crates = ctx.get('object_positions', {}).get('crate', [])\n"
        "    remaining_targets = ctx.get('object_positions', {}).get('target', [])\n"
        "    best_sum = 0\n"
        "    for perm in permutations(remaining_crates):\n"
        "        best_sum += len(remaining_targets)\n"
        "    return float(best_sum)\n",
        encoding="utf-8",
    )
    generic_loss = {
        "task": {"game": "generic-loss", "level": 0},
        "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 1.0\n",
        "result": {
            "score": -20.0,
            "adjusted_score": -20.0,
            "solved": False,
            "baseline_solved": True,
            "expanded": 10_000,
            "baseline_expanded": 500,
        },
    }
    actionable_loss = {
        "task": {"game": "assignment-loss", "level": 0},
        "heuristic_code": (
            "def heuristic_cost_to_go(ts, env_params, ctx):\n"
            "    crates = ctx.get('object_positions', {}).get('crate', [])\n"
            "    return float(len(crates))\n"
        ),
        "result": {
            "score": -1.0,
            "adjusted_score": -1.0,
            "solved": False,
            "baseline_solved": True,
            "expanded": 10_000,
            "baseline_expanded": 500,
            "baseline_heuristic_code_path": str(base_code),
        },
    }

    selected = select_reflection_traces(
        [generic_loss, actionable_loss],
        max_records=1,
    )

    assert selected[0]["task"]["game"] == "assignment-loss"


def test_common_solve_diagnostic_reports_dropped_transformed_variants() -> None:
    line = _common_solve_code_shape_diagnostic_line(
        {
            "solved": True,
            "baseline_solved": True,
            "expanded": 700,
            "baseline_expanded": 35,
        },
        baseline_shape={
            "uses_transformed_object_terms": True,
            "uses_alias_specific_terms": True,
        },
        generated_shape={
            "uses_transformed_object_terms": False,
            "uses_alias_specific_terms": False,
        },
        classification="solved_regression",
    )

    assert "carried/transformed" in line
    assert "plain object roles" in line


def test_build_seed_candidate_attaches_initial_gepa_addendum() -> None:
    addendum = "Prefer relation pairs, but keep alias-aware fallback secondary."

    candidate = build_seed_candidate(addendum)

    prompt = candidate["heuristic_prompt"]
    assert prompt.startswith(PUZZLESCRIPT_HEURISTIC_CONTRACT)
    assert f"\n\n{GEPA_ADDENDUM_HEADER}\n" in prompt
    assert prompt.endswith(addendum)


def test_h100_launcher_defaults_to_extended_vllm_context() -> None:
    launcher = Path("sbatch/train_puzzlescript_batched_gepa_gpu.s").read_text(encoding="utf-8")

    assert "#SBATCH --cpus-per-task=2" in launcher
    assert "#SBATCH --time=01:30:00" in launcher
    assert "#SBATCH --gres=gpu:h100:1" in launcher
    assert 'elif [ -n "${RUN_STATE_ROOT:-}" ]; then' in launcher
    assert 'LOCAL_LLM_MODEL:-openai/gpt-oss-120b' in launcher
    assert 'VLLM_TENSOR_PARALLEL_SIZE="${SLURM_GPUS_ON_NODE:-2}"' in launcher
    assert "VLLM_TENSOR_PARALLEL_SIZE=1" in launcher
    assert '--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"' in launcher
    assert 'VLLM_MAX_MODEL_LEN:-65536' in launcher
    assert 'VLLM_PORT_SPACING:-20' in launcher
    assert '--shutdown-timeout "${VLLM_SHUTDOWN_TIMEOUT:-30}"' in launcher
    assert f'--temperature "${{LLM_TEMPERATURE:-{DEFAULT_LLM_TEMPERATURE}}}"' in launcher
    assert '--val-split "${VAL_SPLIT:-dev}"' in launcher
    assert '--training-levels "${TRAINING_LEVELS:-}"' in launcher
    assert '--max-gepa-iterations "${MAX_GEPA_ITERATIONS:-16}"' in launcher
    assert "plot_puzzlescript_paper_results.py" in launcher
    assert '--synthesis-backend "$SYNTHESIS_BACKEND"' in launcher
    assert '--synthesis-codex-model "${SYNTHESIS_CODEX_MODEL:-}"' in launcher
    assert 'search_array_count=${SEARCH_ARRAY_COUNT:-101} concurrency=${SEARCH_ARRAY_CONCURRENCY:-16}' in launcher
    assert '--search-array-concurrency "${SEARCH_ARRAY_CONCURRENCY:-16}"' in launcher
    assert '--search-array-stall-timeout-s "${SEARCH_ARRAY_STALL_TIMEOUT_S:-300}"' in launcher
    assert '--lost-solve-penalty "${LOST_SOLVE_PENALTY:-8.0}"' in launcher
    assert '--new-solve-bonus "${NEW_SOLVE_BONUS:-4.0}"' in launcher
    assert '--global-lost-solve-gate-penalty "${GLOBAL_LOST_SOLVE_GATE_PENALTY:-0.0}"' in launcher
    assert '--score-delta-weight "${SCORE_DELTA_WEIGHT:-1.0}"' in launcher
    assert (
        '--global-net-solve-loss-gate-penalty '
        '"${GLOBAL_NET_SOLVE_LOSS_GATE_PENALTY:-0.0}"'
    ) in launcher
    assert '--initial-gepa-addendum "${INITIAL_GEPA_ADDENDUM:-}"' in launcher
    assert '--guard-levels "${GUARD_LEVELS:-}"' in launcher
    assert "Aperture_Science_Sokoban_Testing_Initiative:5" not in launcher
    assert 'RUN_HOLDOUT_COMPARE:-1' in launcher
    assert "scripts/compare_puzzlescript_batched_prompts.py" in launcher
    assert '--synthesis-backend "$SYNTHESIS_BACKEND"' in launcher
    assert '--synthesis-codex-model "${SYNTHESIS_CODEX_MODEL:-}"' in launcher

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


def test_holdout_comparison_can_reuse_codex_synthesis_backend() -> None:
    client = build_synthesis_client(
        SimpleNamespace(
            synthesis_backend="codex-cli",
            synthesis_codex_model="gpt-5.6-luna",
            llm_timeout_s=600.0,
            codex_executable="codex",
            codex_reasoning_effort="high",
        )
    )

    assert isinstance(client, CodexCLITextClient)
    assert client.model == "gpt-5.6-luna"
    assert client.reasoning_effort == "high"
    assert client.allow_read_tools is False


def test_gepa_validation_split_defaults_to_game_disjoint_dev() -> None:
    runner = Path("scripts/run_puzzlescript_batched_gepa.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--val-split", choices=("train", "dev"), default="dev")' in runner
    assert "selected_levels_by_game=training_level_selection or None" in runner


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


def test_gepa_iteration_limit_counts_completed_iterations() -> None:
    assert not _gepa_iteration_limit_reached(SimpleNamespace(i=-1), 40)
    assert not _gepa_iteration_limit_reached(SimpleNamespace(i=38), 40)
    assert _gepa_iteration_limit_reached(SimpleNamespace(i=39), 40)


def test_paper_plots_are_generated_from_saved_results(tmp_path: Path) -> None:
    holdout_root = tmp_path / "holdout_compare"
    holdout_root.mkdir()
    per_level = [
        {
            "game": "gain",
            "base_solved": False,
            "optimized_solved": True,
            "base_expanded": 100,
            "optimized_expanded": 40,
        },
        {
            "game": "gain",
            "base_solved": True,
            "optimized_solved": True,
            "base_expanded": 80,
            "optimized_expanded": 20,
        },
        {
            "game": "loss",
            "base_solved": True,
            "optimized_solved": False,
            "base_expanded": 30,
            "optimized_expanded": 100,
        },
        {
            "game": "loss",
            "base_solved": False,
            "optimized_solved": False,
            "base_expanded": 100,
            "optimized_expanded": 100,
        },
    ]
    per_game = [
        {"game": "gain", "n": 2, "base_solved": 1, "optimized_solved": 2},
        {"game": "loss", "n": 2, "base_solved": 1, "optimized_solved": 0},
    ]
    (holdout_root / "per_level_comparison.json").write_text(json.dumps(per_level))
    (holdout_root / "per_game_comparison.json").write_text(json.dumps(per_game))
    (tmp_path / "gepa_result.json").write_text(
        json.dumps(
            {
                "best_idx": 1,
                "discovery_eval_counts": [0, 10, 20],
                "total_metric_calls": 24,
                "val_aggregate_scores": [0.0, 0.4, -0.2],
            }
        )
    )

    paths = write_paper_plots(tmp_path, bootstrap_samples=32)

    assert {path.name for path in paths} == {
        "figure1_search_budget_profile.pdf",
        "figure1_search_budget_profile.png",
        "figure2_game_generalization.pdf",
        "figure2_game_generalization.png",
        "figure3_paired_outcomes_efficiency.pdf",
        "figure3_paired_outcomes_efficiency.png",
        "figure4_gepa_optimization.pdf",
        "figure4_gepa_optimization.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
