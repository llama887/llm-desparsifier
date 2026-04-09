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


def _load_run_heuristic_batch():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "run_heuristic_batch.py"
    spec = importlib.util.spec_from_file_location("run_heuristic_batch", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _attach_job_metadata(example: dspy.Example, job: object) -> dspy.Example:
    """Attach the runner's expected `job_name` and `job_config` attributes.

    The heuristic runner reconstructs `EnvJob` objects from metadata attached to
    each DSPy example. This helper is needed because the tests stub out the
    expensive real example builder, and it differs from inline setup by
    centralizing the exact metadata shape the runner expects.
    """

    example.job_name = getattr(job, "name")
    example.job_config = getattr(job, "to_config")()
    return example


def _write_env_grid(tmp_path: Path, *, num_jobs: int, include_eval_job: bool = False) -> Path:
    """Write a minimal search-native env grid for runner tests.

    The curriculum tests only care about ordered training jobs and optional
    holdout execution. This helper is needed because the real default grid has
    many environments and would make the fake schedules harder to reason about,
    and it differs from hand-written YAML snippets by generating exactly the job
    count each scenario needs.
    """

    lines = ["jobs:"]
    for index in range(1, num_jobs + 1):
        lines.extend(
            [
                f"  - name: job-{index}",
                f"    env_id: XLand-MiniGrid-R1-{8 + index}x{8 + index}",
                "    benchmark_id: trivial-1m",
                "    num_gepa_eval_seeds: 2",
                f"    holdout_seeds: [{100 + index}, {200 + index}]",
                "    deterministic_rulesets: true",
                "    astar_max_nodes: 50",
                "    astar_max_expansions: 40",
            ]
        )
    if include_eval_job:
        lines.extend(
            [
                "",
                "eval_jobs:",
                "  - name: holdout-1",
                "    env_id: XLand-MiniGrid-R1-99x99",
                "    benchmark_id: trivial-1m",
                "    num_gepa_eval_seeds: 2",
                "    holdout_seeds: [900, 901]",
                "    deterministic_rulesets: true",
                "    astar_max_nodes: 50",
                "    astar_max_expansions: 40",
            ]
        )
    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_grid


def _build_examples_for_jobs(jobs: list[Any]) -> list[dspy.Example]:
    """Create lightweight DSPy examples matching the runner contract.

    The tests replace the expensive task-materialization path with cheap static
    examples. This helper is needed because the runner's metric only requires
    the attached job metadata, and it differs from the production builder by
    never touching XLand task generation.
    """

    examples: list[dspy.Example] = []
    for job in jobs:
        example = dspy.Example(
            env_description="env",
            heuristic_contract="contract",
            env_id=job.env_id,
            benchmark_id=job.benchmark_id,
        ).with_inputs("env_description", "heuristic_contract", "env_id", "benchmark_id")
        examples.append(_attach_job_metadata(example, job))
    return examples


def test_load_env_grid_defaults_jobs_to_stochastic_rulesets(tmp_path: Path) -> None:
    """Verify omitted ruleset policy now defaults to stochastic task sampling.

    This regression test protects the heuristic-search curriculum against
    silently collapsing onto one benchmark ruleset when a grid author omits the
    `deterministic_rulesets` field. It is needed because the checked-in grid
    now relies on stochastic benchmark-task sampling to vary underlying goals
    across phases, and it differs from broader runner tests by asserting the
    exact YAML-default semantics on parsed `EnvJob` records.
    """

    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = tmp_path / "envs.yaml"
    env_grid.write_text(
        "\n".join(
            [
                "jobs:",
                "  - name: job-1",
                "    env_id: XLand-MiniGrid-R1-9x9",
                "    benchmark_id: trivial-1m",
                "    num_gepa_eval_seeds: 2",
                "    holdout_seeds: [100, 200]",
                "    astar_max_nodes: 50",
                "    astar_max_expansions: 40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    jobs, eval_jobs = run_heuristic_batch.load_env_grid(
        env_grid,
        default_astar_max_nodes=123,
        default_astar_max_expansions=456,
    )

    assert len(jobs) == 1
    assert jobs[0].deterministic_rulesets is False
    assert eval_jobs == []


class _FakePromptGenerator:
    """Minimal prompt generator used by the curriculum runner tests.

    The real runner stores DSPy predictor state and later reconstructs prompt
    text from that state. This fake is needed because the tests only care about
    which prompt text is carried between phases, and it differs from the real
    predictor by encoding the state as a plain `{"prompt_text": ...}` mapping.
    """

    def __init__(self, prompt_text: str) -> None:
        self.prompt_text = prompt_text

    def __call__(self, **_kwargs: Any) -> str:
        return self.prompt_text

    def dump_state(self) -> dict[str, str]:
        """Return a JSON-friendly prompt-state payload.

        The runner persists prompt state inside `active_prompt.json` after every
        phase iteration. This helper is needed because the fake program uses
        plain text instead of DSPy predictor weights, and it differs from the
        production implementation by storing just the chosen prompt text.
        """

        return {"prompt_text": self.prompt_text}


class _FakePromptOnlyProgram:
    """Cheap stand-in for `PromptOnlyProgram` during runner tests.

    The curriculum logic only needs a program object with a prompt generator and
    a stable rewrite-prompt method. This fake is needed because the real DSPy
    module would invoke model-backed prompt rewriting, and it differs from the
    production class by treating the persisted prompt text as the whole state.
    """

    def __init__(
        self,
        base_prompt_text: str,
        prompt_state: dict[str, Any] | None = None,
    ) -> None:
        prompt_text = base_prompt_text
        if isinstance(prompt_state, dict) and isinstance(prompt_state.get("prompt_text"), str):
            prompt_text = str(prompt_state["prompt_text"])
        self.base_prompt_text = base_prompt_text
        self.prompt_generator = _FakePromptGenerator(prompt_text)

    def _build_rewrite_prompt(self) -> str:
        return self.base_prompt_text


def _install_runner_fakes(
    *,
    monkeypatch: pytest.MonkeyPatch,
    run_heuristic_batch: Any,
    prompt_sequence: list[str],
    solve_rate_by_prompt: dict[str, float],
    job_score_by_prompt: dict[str, float] | None,
    captured_compile_history: list[dict[str, Any]],
    captured_eval_history: list[dict[str, Any]],
    no_heuristic_solve_rate: float = 0.0,
    no_heuristic_job_score: float = 0.0,
    captured_plot_calls: list[dict[str, Any]] | None = None,
    metric_examples_per_compile: list[int] | None = None,
) -> None:
    """Install deterministic fakes for the expensive runner dependencies.

    The curriculum behavior under test is the orchestration around GEPA, not
    XLand evaluation or DSPy internals. This helper is needed because the real
    pipeline would be too slow and nondeterministic for unit tests, and it
    differs from broad monkeypatching in each test by wiring one shared fake
    stack that records compile and evaluation history for assertions. The
    optional `metric_examples_per_compile` sequence lets a test simulate GEPA
    returning after only a partial trainset evaluation, which is important for
    covering resume-only optimizer edge cases.
    """

    class _FakeGEPA:
        def __init__(self, metric, *args, **kwargs):
            del args
            self.metric = metric
            self.stats = {}
            self.log_dir = kwargs.get("log_dir")
            self.max_metric_calls = kwargs.get("max_metric_calls")

        def compile(self, program, trainset=None, **_kwargs):
            compile_index = len(captured_compile_history)
            prompt_text = prompt_sequence[compile_index]
            metric_example_limit = len(trainset or [])
            if metric_examples_per_compile is not None and compile_index < len(
                metric_examples_per_compile
            ):
                metric_example_limit = metric_examples_per_compile[compile_index]
            if isinstance(self.log_dir, str):
                log_dir = Path(self.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "gepa_state.bin").write_bytes(b"fake-state")
            captured_compile_history.append(
                {
                    "compile_index": compile_index + 1,
                    "input_prompt_text": program.prompt_generator.prompt_text,
                    "output_prompt_text": prompt_text,
                    "trainset_size": len(trainset or []),
                    "job_names": [getattr(example, "job_name") for example in trainset or []],
                    "evaluated_job_names": [
                        getattr(example, "job_name")
                        for example in (trainset or [])[:metric_example_limit]
                    ],
                    "log_dir": self.log_dir,
                    "max_metric_calls": self.max_metric_calls,
                    "resume_marker_exists": (
                        isinstance(self.log_dir, str)
                        and (Path(self.log_dir) / "prog_candidates").exists()
                    ),
                }
            )
            for example in (trainset or [])[:metric_example_limit]:
                prediction = dspy.Prediction(prompt_text=prompt_text)
                self.metric(example, prediction)
            self.stats = {
                "compile_index": compile_index + 1,
                "trainset_size": len(trainset or []),
                "prompt_text": prompt_text,
            }
            program.prompt_generator = _FakePromptGenerator(prompt_text)
            return program

    def _fake_evaluate_job(*, job, seeds, prompt_text, lm, output_dir):
        del seeds, lm
        output_dir.mkdir(parents=True, exist_ok=True)
        solve_rate = float(solve_rate_by_prompt[prompt_text])
        job_score = float(job_score_by_prompt.get(prompt_text, solve_rate) if job_score_by_prompt else solve_rate)
        captured_eval_history.append(
            {
                "job_name": job.name,
                "env_id": job.env_id,
                "prompt_text": prompt_text,
                "output_dir": str(output_dir),
            }
        )
        return {
            "job_score": job_score,
            "aggregate_stats": {
                "solve_rate": solve_rate,
                "average_expanded_states": 20.0,
                "average_generated_states": 30.0,
                "average_solution_length": 4.0,
            },
            "feedback": f"feedback-{prompt_text}",
            "heuristic_validation": {
                "admissibility_summary": {"admissibility_pass_rate": 0.75}
            },
            "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0\n",
        }

    def _fake_evaluate_no_heuristic_job(*, job, seeds, output_dir):
        del seeds
        output_dir.mkdir(parents=True, exist_ok=True)
        captured_eval_history.append(
            {
                "job_name": job.name,
                "env_id": job.env_id,
                "prompt_text": "<no-heuristic>",
                "output_dir": str(output_dir),
            }
        )
        return {
            "job_score": float(no_heuristic_job_score),
            "aggregate_stats": {
                "solve_rate": float(no_heuristic_solve_rate),
                "average_expanded_states": 40.0,
                "average_generated_states": 60.0,
                "average_solution_length": 0.0,
            },
            "feedback": "feedback-no-heuristic",
            "heuristic_validation": {
                "admissibility_summary": {"admissibility_pass_rate": 1.0}
            },
            "heuristic_code": "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0\n",
        }

    def _fake_write_holdout_comparison_plots(*, logs_root, comparisons):
        if captured_plot_calls is not None:
            captured_plot_calls.append(
                {
                    "logs_root": str(logs_root),
                    "labels": [comparison.label for comparison in comparisons],
                    "solve_rates": [comparison.solve_rate_mean for comparison in comparisons],
                }
            )
        aggregate_path = logs_root / "holdout_comparison_aggregate.png"
        by_env_path = logs_root / "holdout_comparison_by_env.png"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_bytes(b"fake-aggregate")
        by_env_path.write_bytes(b"fake-by-env")
        return [aggregate_path, by_env_path]

    monkeypatch.setattr(run_heuristic_batch.dspy, "GEPA", _FakeGEPA)
    monkeypatch.setattr(run_heuristic_batch, "PromptOnlyProgram", _FakePromptOnlyProgram)
    monkeypatch.setattr(run_heuristic_batch, "configure_gemini_lm", lambda **_kwargs: object())
    monkeypatch.setattr(run_heuristic_batch, "build_examples", _build_examples_for_jobs)
    monkeypatch.setattr(run_heuristic_batch, "evaluate_job", _fake_evaluate_job)
    monkeypatch.setattr(
        run_heuristic_batch,
        "evaluate_no_heuristic_job",
        _fake_evaluate_no_heuristic_job,
    )
    monkeypatch.setattr(
        run_heuristic_batch,
        "write_holdout_comparison_plots",
        _fake_write_holdout_comparison_plots,
    )


def test_sample_eval_seeds_is_stable_and_unique() -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    seeds_a = run_heuristic_batch.sample_eval_seeds(
        global_experiment_seed=7,
        metric_call_idx=3,
        job_name="job-a",
        num_gepa_eval_seeds=4,
    )
    seeds_b = run_heuristic_batch.sample_eval_seeds(
        global_experiment_seed=7,
        metric_call_idx=3,
        job_name="job-a",
        num_gepa_eval_seeds=4,
    )
    assert seeds_a == seeds_b
    assert len(seeds_a) == len(set(seeds_a))


def test_parse_args_uses_max_phase_iterations_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_heuristic_batch.py", "--state-root", "artifacts/gepa_state"],
    )
    args = run_heuristic_batch.parse_args()
    assert args.max_phase_iterations == 10


def test_curriculum_advances_and_final_phase_runs_until_iteration_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    prompt_sequence = [
        "phase1-best",
        "phase2-best",
        "phase3-best",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
        "phase3-flat",
    ]
    solve_rate_by_prompt = {
        "phase1-best": 0.85,
        "phase2-best": 0.82,
        "phase3-best": 0.82,
        "phase3-flat": 0.82,
    }
    job_score_by_prompt = {
        "phase1-best": 0.55,
        "phase2-best": 0.60,
        "phase3-best": 0.61,
        "phase3-flat": 0.61,
    }
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=prompt_sequence,
        solve_rate_by_prompt=solve_rate_by_prompt,
        job_score_by_prompt=job_score_by_prompt,
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
        ],
    )
    run_heuristic_batch.run_batch()

    active_payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    curriculum = active_payload["curriculum"]
    phase_records = curriculum["phase_records"]
    assert curriculum["phase_job_counts"] == [3, 7, 11]
    assert [entry["trainset_size"] for entry in compile_history] == [3, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
    assert [entry["max_metric_calls"] for entry in compile_history] == [3, 7, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
    assert [Path(entry["log_dir"]).name for entry in compile_history] == [
        "phase-01-gepa",
        "phase-02-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
        "phase-03-gepa",
    ]
    assert compile_history[3]["resume_marker_exists"] is True
    assert curriculum["completed_phases"] == [1, 2]
    assert curriculum["current_phase"] == 3
    assert curriculum["stop_reason"] == "phase_iteration_cap"
    assert phase_records["1"]["advanced"] is True
    assert phase_records["1"]["phase_job_count"] == 3
    assert phase_records["2"]["phase_job_count"] == 7
    assert phase_records["3"]["phase_job_count"] == 11
    assert Path(phase_records["1"]["gepa_log_dir"]).name == "phase-01-gepa"
    assert Path(phase_records["3"]["gepa_log_dir"]).name == "phase-03-gepa"
    assert phase_records["1"]["baseline_solve_rate"] == pytest.approx(0.85)
    assert phase_records["3"]["iteration_count"] == 10
    assert phase_records["3"]["best_solve_rate"] == pytest.approx(0.82)
    assert phase_records["3"]["best_job_score"] == pytest.approx(0.61)
    assert phase_records["3"]["non_improving_streak"] == 9
    assert phase_records["3"]["stop_reason"] == "phase_iteration_cap"
    assert active_payload["prompt_state"]["prompt_text"] == "phase3-best"

    candidate_dirs = sorted((state_root / "heuristic_runs").glob("candidate-*"))
    assert len(candidate_dirs) == 120
    assert candidate_dirs[0].name == "candidate-0001-job-1"
    assert candidate_dirs[-1].name == "candidate-0120-job-11"

    stats_payload = json.loads(
        (state_root / "heuristic_runs" / "gepa_stats.json").read_text(encoding="utf-8")
    )
    assert stats_payload["max_phase_iterations"] == 10
    assert stats_payload["curriculum"]["stop_reason"] == "phase_iteration_cap"
    assert len(stats_payload["curriculum"]["phase_records"]) == 3


def test_non_final_phase_threshold_failure_stops_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    prompt_sequence = ["phase1-best", "phase2-best", "phase2-flat", "phase2-flat", "phase2-flat"]
    solve_rate_by_prompt = {
        "phase1-best": 0.90,
        "phase2-best": 0.70,
        "phase2-flat": 0.70,
    }
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=prompt_sequence,
        solve_rate_by_prompt=solve_rate_by_prompt,
        job_score_by_prompt=None,
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
        ],
    )
    run_heuristic_batch.run_batch()

    active_payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    curriculum = active_payload["curriculum"]
    assert curriculum["current_phase"] == 2
    assert curriculum["stop_reason"] == "threshold_failure_early_stop"
    assert curriculum["phase_records"]["2"]["stop_reason"] == "threshold_failure_early_stop"
    assert [entry["trainset_size"] for entry in compile_history] == [3, 7, 7, 7, 7]
    assert [entry["max_metric_calls"] for entry in compile_history] == [3, 7, 14, 21, 28]
    assert [Path(entry["log_dir"]).name for entry in compile_history] == [
        "phase-01-gepa",
        "phase-02-gepa",
        "phase-02-gepa",
        "phase-02-gepa",
        "phase-02-gepa",
    ]


def test_phase_iteration_cap_stops_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=3)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    prompt_sequence = ["cap-a", "cap-b"]
    solve_rate_by_prompt = {"cap-a": 0.30, "cap-b": 0.31}
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=prompt_sequence,
        solve_rate_by_prompt=solve_rate_by_prompt,
        job_score_by_prompt=None,
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "2",
        ],
    )
    run_heuristic_batch.run_batch()

    active_payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    curriculum = active_payload["curriculum"]
    assert len(compile_history) == 2
    assert [entry["max_metric_calls"] for entry in compile_history] == [3, 6]
    assert curriculum["stop_reason"] == "phase_iteration_cap"
    assert curriculum["phase_records"]["1"]["iteration_count"] == 2
    assert curriculum["phase_job_counts"] == [3]


def test_resume_terminal_checkpoint_skips_gepa_reentry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    preexisting_payload = {
        "base_prompt_text": "seed-base",
        "prompt_state": {"prompt_text": "final-best"},
        "heuristic_contract": "contract",
        "curriculum": {
            "version": 1,
            "current_phase": 3,
            "completed_phases": [1, 2],
            "phase_job_counts": [3, 7, 11],
            "total_phases": 3,
            "phase_records": {
                "1": {
                    "phase": 1,
                    "phase_job_count": 3,
                    "active_job_names": ["job-1", "job-2", "job-3"],
                    "active_env_ids": [
                        "XLand-MiniGrid-R1-9x9",
                        "XLand-MiniGrid-R1-10x10",
                        "XLand-MiniGrid-R1-11x11",
                    ],
                    "threshold": 0.8,
                    "baseline_solve_rate": 0.9,
                    "best_solve_rate": 0.9,
                    "baseline_job_score": 0.5,
                    "best_job_score": 0.5,
                    "best_iteration": 1,
                    "iteration_count": 1,
                    "non_improving_streak": 0,
                    "advanced": True,
                    "completed": True,
                    "stop_reason": "advanced_to_next_phase",
                    "iteration_summaries": [],
                    "compiler_stats": [],
                },
                "2": {
                    "phase": 2,
                    "phase_job_count": 7,
                    "active_job_names": [f"job-{index}" for index in range(1, 8)],
                    "active_env_ids": [
                        f"XLand-MiniGrid-R1-{8 + index}x{8 + index}"
                        for index in range(1, 8)
                    ],
                    "threshold": 0.8,
                    "baseline_solve_rate": 0.82,
                    "best_solve_rate": 0.82,
                    "baseline_job_score": 0.55,
                    "best_job_score": 0.55,
                    "best_iteration": 1,
                    "iteration_count": 1,
                    "non_improving_streak": 0,
                    "advanced": True,
                    "completed": True,
                    "stop_reason": "advanced_to_next_phase",
                    "iteration_summaries": [],
                    "compiler_stats": [],
                },
                "3": {
                    "phase": 3,
                    "phase_job_count": 11,
                    "active_job_names": [f"job-{index}" for index in range(1, 12)],
                    "active_env_ids": [
                        f"XLand-MiniGrid-R1-{8 + index}x{8 + index}"
                        for index in range(1, 12)
                    ],
                    "threshold": 0.8,
                    "baseline_solve_rate": 0.79,
                    "best_solve_rate": 0.79,
                    "baseline_job_score": 0.68,
                    "best_job_score": 0.68,
                    "best_iteration": 1,
                    "iteration_count": 7,
                    "non_improving_streak": 6,
                    "advanced": False,
                    "completed": True,
                    "stop_reason": "full_curriculum_early_stop",
                    "iteration_summaries": [],
                    "compiler_stats": [],
                },
            },
            "global_iteration": 11,
            "metric_call_idx": 207,
            "max_phase_iterations": 10,
            "phase_solve_rate_threshold": 0.8,
            "phase_early_stop_patience": 3,
            "total_training_jobs": 11,
            "training_job_names": [f"job-{index}" for index in range(1, 12)],
            "training_env_ids": [
                f"XLand-MiniGrid-R1-{8 + index}x{8 + index}"
                for index in range(1, 12)
            ],
            "stop_reason": "full_curriculum_early_stop",
            "final_prompt_text": "final-best",
        },
    }
    (state_root / "active_prompt.json").write_text(
        json.dumps(preexisting_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=[],
        solve_rate_by_prompt={},
        job_score_by_prompt={},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
        ],
    )
    run_heuristic_batch.run_batch()

    assert not compile_history
    active_payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    curriculum = active_payload["curriculum"]
    assert curriculum["stop_reason"] == "full_curriculum_early_stop"
    assert curriculum["phase_records"]["3"]["iteration_count"] == 7


def test_resume_checkpoint_reuses_saved_phase_and_prompt_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    preexisting_payload = {
        "base_prompt_text": "seed-base",
        "prompt_state": {"prompt_text": "resume-best"},
        "heuristic_contract": "contract",
        "curriculum": {
            "version": 1,
            "current_phase": 2,
            "completed_phases": [1],
            "phase_job_counts": [3, 7, 11],
            "total_phases": 3,
            "phase_records": {
                "1": {
                    "phase": 1,
                    "phase_job_count": 3,
                    "active_job_names": ["job-1", "job-2", "job-3"],
                    "active_env_ids": [
                        "XLand-MiniGrid-R1-9x9",
                        "XLand-MiniGrid-R1-10x10",
                        "XLand-MiniGrid-R1-11x11",
                    ],
                    "threshold": 0.8,
                    "baseline_solve_rate": 0.9,
                    "best_solve_rate": 0.9,
                    "baseline_job_score": 0.5,
                    "best_job_score": 0.5,
                    "best_iteration": 1,
                    "iteration_count": 1,
                    "non_improving_streak": 0,
                    "advanced": True,
                    "completed": True,
                    "stop_reason": "advanced_to_next_phase",
                    "iteration_summaries": [],
                    "compiler_stats": [],
                }
            },
            "global_iteration": 1,
            "metric_call_idx": 1,
            "max_phase_iterations": 10,
            "phase_solve_rate_threshold": 0.8,
            "phase_early_stop_patience": 3,
            "total_training_jobs": 11,
            "training_job_names": [f"job-{index}" for index in range(1, 12)],
            "training_env_ids": [
                f"XLand-MiniGrid-R1-{8 + index}x{8 + index}"
                for index in range(1, 12)
            ],
            "stop_reason": None,
            "final_prompt_text": "resume-best",
        },
    }
    (state_root / "active_prompt.json").write_text(
        json.dumps(preexisting_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    prompt_sequence = ["resume-best", "resume-flat", "resume-flat", "resume-flat"]
    solve_rate_by_prompt = {"resume-best": 0.70, "resume-flat": 0.70}
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=prompt_sequence,
        solve_rate_by_prompt=solve_rate_by_prompt,
        job_score_by_prompt={"resume-best": 0.51, "resume-flat": 0.50},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
        ],
    )
    run_heuristic_batch.run_batch()

    assert all(entry["trainset_size"] == 7 for entry in compile_history)
    assert [entry["max_metric_calls"] for entry in compile_history] == [7, 14, 21, 28]
    assert all(Path(entry["log_dir"]).name == "phase-02-gepa" for entry in compile_history)
    assert compile_history[0]["resume_marker_exists"] is False
    assert compile_history[1]["resume_marker_exists"] is True
    assert compile_history[0]["input_prompt_text"] == "resume-best"
    active_payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    loaded_text, loaded_state, meta = run_heuristic_batch.load_prompt_payload(state_root)
    assert loaded_text == "seed-base"
    assert loaded_state == {"prompt_text": "resume-best"}
    assert meta["source"] == "active_prompt"
    assert active_payload["curriculum"]["global_iteration"] == 5


def test_run_batch_initializes_wandb_project_and_logs_curriculum_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    prompt_sequence = ["wandb-phase1", "wandb-phase2", "wandb-phase2", "wandb-phase2", "wandb-phase2"]
    solve_rate_by_prompt = {"wandb-phase1": 0.85, "wandb-phase2": 0.70}
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=prompt_sequence,
        solve_rate_by_prompt=solve_rate_by_prompt,
        job_score_by_prompt=None,
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    captured_logs: list[dict[str, Any]] = []
    captured_init: dict[str, Any] = {}

    class _FakeTable:
        def __init__(self, **_kwargs):
            self.rows: list[tuple[Any, ...]] = []

        def add_data(self, *row: Any) -> None:
            self.rows.append(row)

    class _FakeRun:
        _is_finished = False
        finished = False

        def log(self, payload: dict[str, Any], **kwargs: Any) -> None:
            captured_logs.append({"payload": payload, "step": kwargs.get("step")})

        def finish(self, quiet: bool = True) -> None:
            del quiet
            self.finished = True

    class _FakeWandb:
        errors = SimpleNamespace(UsageError=RuntimeError)

        @staticmethod
        def init(**kwargs: Any) -> _FakeRun:
            captured_init.update(kwargs)
            return _FakeRun()

        @staticmethod
        def Table(**kwargs: Any) -> _FakeTable:
            return _FakeTable(**kwargs)

    monkeypatch.setattr(run_heuristic_batch, "wandb", _FakeWandb())
    os.environ.pop("WANDB_DISABLED", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
        ],
    )
    run_heuristic_batch.run_batch()

    assert captured_init["project"] == "llm-astar"
    assert captured_init["config"]["max_phase_iterations"] == 10
    assert any("heuristic/candidate_runs" in entry["payload"] for entry in captured_logs)
    assert any("gepa/phase_best_solve_rate" in entry["payload"] for entry in captured_logs)
    assert any("phase_01/solve_rate" in entry["payload"] for entry in captured_logs)
    assert any("phase_02/solve_rate" in entry["payload"] for entry in captured_logs)
    assert any(
        entry["payload"].get("curriculum/phase_failed_to_converge") is True
        for entry in captured_logs
        if isinstance(entry["payload"], dict)
    )
    assert any(
        entry["payload"].get("curriculum/stop_reason") == "threshold_failure_early_stop"
        for entry in captured_logs
        if isinstance(entry["payload"], dict)
    )
    metric_steps = [
        entry["step"]
        for entry in captured_logs
        if isinstance(entry["payload"], dict) and "gepa/phase" in entry["payload"]
    ]
    assert metric_steps == sorted(metric_steps)


def test_better_job_score_is_adopted_within_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=3)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=["score-a", "score-b"],
        solve_rate_by_prompt={"score-a": 0.90, "score-b": 0.90},
        job_score_by_prompt={"score-a": 0.40, "score-b": 0.60},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "2",
        ],
    )
    run_heuristic_batch.run_batch()

    active_payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    phase_record = active_payload["curriculum"]["phase_records"]["1"]
    assert active_payload["prompt_state"]["prompt_text"] == "score-b"
    assert [entry["max_metric_calls"] for entry in compile_history] == [3, 6]
    assert phase_record["best_job_score"] == pytest.approx(0.60)


def test_prompt_below_solve_threshold_does_not_advance_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=["low-solve"],
        solve_rate_by_prompt={"low-solve": 0.79},
        job_score_by_prompt={"low-solve": 0.95},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "1",
        ],
    )
    run_heuristic_batch.run_batch()

    curriculum = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))[
        "curriculum"
    ]
    assert compile_history[0]["max_metric_calls"] == 3
    assert curriculum["current_phase"] == 1
    assert curriculum["completed_phases"] == []


def test_higher_score_prompt_is_kept_when_no_prompt_reaches_solve_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=["fallback-a", "fallback-b"],
        solve_rate_by_prompt={"fallback-a": 0.70, "fallback-b": 0.70},
        job_score_by_prompt={"fallback-a": 0.30, "fallback-b": 0.50},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "2",
        ],
    )
    run_heuristic_batch.run_batch()

    payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    curriculum = payload["curriculum"]
    assert payload["prompt_state"]["prompt_text"] == "fallback-b"
    assert [entry["max_metric_calls"] for entry in compile_history] == [3, 6]
    assert curriculum["current_phase"] == 1


def test_incomplete_compile_retries_same_phase_with_larger_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=3)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=["partial-pass", "recovered-pass"],
        solve_rate_by_prompt={"partial-pass": 0.30, "recovered-pass": 0.31},
        job_score_by_prompt={"partial-pass": 0.20, "recovered-pass": 0.25},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
        metric_examples_per_compile=[2, 3],
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "1",
        ],
    )
    run_heuristic_batch.run_batch()

    payload = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))
    curriculum = payload["curriculum"]
    phase_record = curriculum["phase_records"]["1"]
    assert [entry["max_metric_calls"] for entry in compile_history] == [3, 6]
    assert compile_history[0]["evaluated_job_names"] == ["job-1", "job-2"]
    assert compile_history[1]["evaluated_job_names"] == ["job-1", "job-2", "job-3"]
    assert curriculum["global_iteration"] == 1
    assert curriculum["stop_reason"] == "phase_iteration_cap"
    assert phase_record["iteration_count"] == 1
    assert phase_record["incomplete_compile_retries"] == 0
    assert phase_record["last_incomplete_compile"] is None
    assert payload["prompt_state"]["prompt_text"] == "recovered-pass"
    candidate_dirs = sorted((state_root / "heuristic_runs").glob("candidate-*"))
    assert [path.name for path in candidate_dirs] == [
        "candidate-0001-job-1",
        "candidate-0002-job-2",
        "candidate-0003-job-1",
        "candidate-0004-job-2",
        "candidate-0005-job-3",
    ]


def test_incompatible_saved_curriculum_resets_to_new_phase_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=11)
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / "active_prompt.json").write_text(
        json.dumps(
            {
                "base_prompt_text": "seed-base",
                "prompt_state": {"prompt_text": "stale-prompt"},
                "heuristic_contract": "contract",
                "curriculum": {
                    "version": 1,
                    "current_phase": 2,
                    "completed_phases": [1],
                    "phase_job_counts": [1, 2, 3],
                    "total_phases": 3,
                    "phase_records": {},
                    "global_iteration": 9,
                    "metric_call_idx": 9,
                    "max_phase_iterations": 10,
                    "phase_solve_rate_threshold": 0.8,
                    "phase_early_stop_patience": 3,
                    "total_training_jobs": 11,
                    "training_job_names": [f"job-{index}" for index in range(1, 12)],
                    "training_env_ids": [
                        f"XLand-MiniGrid-R1-{8 + index}x{8 + index}"
                        for index in range(1, 12)
                    ],
                    "stop_reason": None,
                    "final_prompt_text": "stale-prompt",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=["reset-phase"],
        solve_rate_by_prompt={"reset-phase": 0.70},
        job_score_by_prompt={"reset-phase": 0.40},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "1",
        ],
    )
    run_heuristic_batch.run_batch()

    curriculum = json.loads((state_root / "active_prompt.json").read_text(encoding="utf-8"))[
        "curriculum"
    ]
    assert compile_history[0]["trainset_size"] == 3
    assert compile_history[0]["max_metric_calls"] == 3
    assert curriculum["phase_job_counts"] == [3, 7, 11]


def test_run_batch_writes_holdout_baselines_and_comparison_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_heuristic_batch = _load_run_heuristic_batch()
    env_grid = _write_env_grid(tmp_path, num_jobs=3, include_eval_job=True)
    state_root = tmp_path / "state"
    compile_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    plot_calls: list[dict[str, Any]] = []
    _install_runner_fakes(
        monkeypatch=monkeypatch,
        run_heuristic_batch=run_heuristic_batch,
        prompt_sequence=["phase-cap", "phase-cap"],
        solve_rate_by_prompt={"phase-cap": 0.30, run_heuristic_batch.BASE_HEURISTIC_PROMPT: 0.10},
        job_score_by_prompt={"phase-cap": 0.25, run_heuristic_batch.BASE_HEURISTIC_PROMPT: 0.05},
        captured_compile_history=compile_history,
        captured_eval_history=eval_history,
        no_heuristic_solve_rate=0.0,
        no_heuristic_job_score=0.0,
        captured_plot_calls=plot_calls,
    )

    os.environ["WANDB_DISABLED"] = "1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_heuristic_batch.py",
            "--state-root",
            str(state_root),
            "--env-grid",
            str(env_grid),
            "--max-phase-iterations",
            "2",
        ],
    )
    run_heuristic_batch.run_batch()

    stats_payload = json.loads(
        (state_root / "heuristic_runs" / "gepa_stats.json").read_text(encoding="utf-8")
    )
    comparison_by_label = {
        entry["label"]: entry
        for entry in stats_payload["holdout_comparisons"]
    }
    assert comparison_by_label["Optimized prompt"]["solve_rate_mean"] == pytest.approx(0.30)
    assert comparison_by_label["Base prompt"]["solve_rate_mean"] == pytest.approx(0.10)
    assert comparison_by_label["Blind A*"]["solve_rate_mean"] == pytest.approx(0.0)
    assert comparison_by_label["Blind A*"]["job_score_mean"] == pytest.approx(0.0)
    assert stats_payload["holdout_plot_paths"] == [
        str(state_root / "heuristic_runs" / "holdout_comparison_aggregate.png"),
        str(state_root / "heuristic_runs" / "holdout_comparison_by_env.png"),
    ]
    assert any(entry["prompt_text"] == "<no-heuristic>" for entry in eval_history)
    assert plot_calls[0]["labels"] == ["Optimized prompt", "Base prompt", "Blind A*"]
