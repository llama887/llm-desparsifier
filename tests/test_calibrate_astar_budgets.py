from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from llm_desparsifier.search.jaxtar_backend import JAxtarSearchBackend, SearchConfig, SearchTask
from llm_desparsifier.search.metrics import SearchSeedResult
from llm_desparsifier.search.xland_adapter import XLandTaskInstance


def _load_calibration_script() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "calibrate_astar_budgets.py"
    spec = importlib.util.spec_from_file_location("calibrate_astar_budgets", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeEnv:
    """Minimal deterministic search environment for backend timeout tests.

    The backend timeout coverage only needs a stable action space and a cheap
    transition function, not a full XLand environment. This fake is needed
    because the real environments are too heavy for small unit tests, and it
    differs from the production envs by using one integer state with one action.
    """

    def num_actions(self, _env_params: Any) -> int:
        return 1


class _FakeTimestep:
    """Small timestep object that matches the backend's search expectations.

    `JAxtarSearchBackend` reads `.state`, `.reward`, and `.last()` from each
    timestep. This fake is needed because the timeout tests exercise backend
    control flow directly, and it differs from the real XLand timestep by
    representing state as one array-backed counter.
    """

    def __init__(self, value: int, *, solved: bool = False) -> None:
        self.state = {"value": np.asarray([value], dtype=np.int32)}
        self.reward = 1.0 if solved else 0.0

    def last(self) -> bool:
        return False


def _build_fake_task() -> SearchTask:
    """Construct one lightweight search task for backend timeout tests.

    The timeout path only depends on the backend's internal search loop and does
    not require XLand-specific task materialization. This helper is needed
    because the backend protocol expects a full `SearchTask`, and it differs
    from production task creation by stubbing the replay metadata directly.
    """

    def _step_fn(_env_params: Any, timestep: _FakeTimestep, _action: Any) -> _FakeTimestep:
        next_value = int(timestep.state["value"][0]) + 1
        return _FakeTimestep(next_value, solved=next_value >= 3)

    return SearchTask(
        env=_FakeEnv(),
        env_params=object(),
        step_fn=_step_fn,
        root_timestep=_FakeTimestep(0),
        task_instance=XLandTaskInstance(
            env_id="XLand-MiniGrid-R1-11x11",
            benchmark_id="trivial-1m",
            seed=7,
            ruleset_seed=None,
            ruleset_text="GOAL\nreach target",
            reset_key=[1, 2],
            goal_description="reach target",
        ),
    )


def test_backend_timeout_disabled_preserves_current_behavior() -> None:
    backend = JAxtarSearchBackend()
    backend_module = sys.modules[JAxtarSearchBackend.__module__]
    original_ctx_builder = backend_module.build_heuristic_ctx
    backend_module.build_heuristic_ctx = lambda **_kwargs: {}
    try:
        result = backend.solve_many(
            task_batch=[_build_fake_task()],
            heuristic_fn=lambda _ts, _env_params, _ctx: 0.0,
            search_config=SearchConfig(
                max_nodes=100,
                max_expansions=100,
            ),
        ).seed_results[0]
    finally:
        backend_module.build_heuristic_ctx = original_ctx_builder

    assert result.solved is True
    assert result.termination_reason == "solved"
    assert result.expanded_states > 0
    assert result.generated_states > 0


def test_backend_timeout_returns_wall_clock_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = JAxtarSearchBackend()
    monotonic_values = iter([0.0, 0.0, 2.0])
    backend_module = sys.modules[JAxtarSearchBackend.__module__]
    monkeypatch.setattr(backend_module, "build_heuristic_ctx", lambda **_kwargs: {})
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: next(monotonic_values))

    result = backend.solve_many(
        task_batch=[_build_fake_task()],
        heuristic_fn=lambda _ts, _env_params, _ctx: 0.0,
        search_config=SearchConfig(
            max_nodes=100,
            max_expansions=100,
            wall_clock_timeout_seconds=1.0,
        ),
    ).seed_results[0]

    assert result.solved is False
    assert result.termination_reason == "wall_clock_timeout"
    assert result.expanded_states >= 1
    assert result.generated_states >= 1


def _write_env_grid(tmp_path: Path) -> Path:
    """Create a tiny env-grid YAML file for calibration script tests.

    The calibration script only needs realistic job metadata and budget fields
    to test report generation and YAML rewrites. This helper is needed because
    the repo-default grid is much larger than the unit scenarios, and it differs
    from fixture files by building exactly the two-section shape each test uses.
    """

    payload = {
        "jobs": [
            {
                "name": "job-a",
                "env_id": "XLand-MiniGrid-R1-11x11",
                "benchmark_id": "trivial-1m",
                "num_gepa_eval_seeds": 2,
                "holdout_seeds": [10, 20],
                "deterministic_rulesets": True,
                "astar_max_nodes": 20000,
                "astar_max_expansions": 5000,
            }
        ],
        "eval_jobs": [
            {
                "name": "job-b",
                "env_id": "XLand-MiniGrid-R1-17x17",
                "benchmark_id": "trivial-1m",
                "num_gepa_eval_seeds": 2,
                "holdout_seeds": [30],
                "deterministic_rulesets": True,
                "astar_max_nodes": 20000,
                "astar_max_expansions": 5000,
            }
        ],
    }
    path = tmp_path / "envs.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_calibration_report_uses_shrink_and_worst_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_script = _load_calibration_script()
    env_grid_path = _write_env_grid(tmp_path)

    seed_results = {
        10: SearchSeedResult(
            seed=10,
            solved=True,
            expanded_states=100,
            generated_states=200,
            solution_length=5,
            termination_reason="solved",
            actions=[],
            search_trace={},
            validation={},
            seed_score=1.0,
            candidate_cost=100,
        ),
        20: SearchSeedResult(
            seed=20,
            solved=False,
            expanded_states=120,
            generated_states=240,
            solution_length=None,
            termination_reason="wall_clock_timeout",
            actions=[],
            search_trace={},
            validation={},
            seed_score=0.0,
            candidate_cost=121,
        ),
        30: SearchSeedResult(
            seed=30,
            solved=True,
            expanded_states=20,
            generated_states=40,
            solution_length=4,
            termination_reason="solved",
            actions=[],
            search_trace={},
            validation={},
            seed_score=1.0,
            candidate_cost=20,
        ),
    }

    class _FakeBackend:
        def solve_many(self, *, task_batch, heuristic_fn, search_config):
            del heuristic_fn, search_config
            seed = task_batch[0].task_instance.seed
            return SimpleNamespace(seed_results=[seed_results[seed]])

    def _fake_build_task_instance(**kwargs: Any):
        seed = int(kwargs["seed"])
        task_instance = XLandTaskInstance(
            env_id=str(kwargs["env_id"]),
            benchmark_id=str(kwargs["benchmark_id"]),
            seed=seed,
            ruleset_seed=None,
            ruleset_text="GOAL\nstub",
            reset_key=[seed],
            goal_description="stub",
        )
        return object(), object(), object(), object(), None, task_instance

    monkeypatch.setattr(calibration_script, "JAxtarSearchBackend", lambda: _FakeBackend())
    monkeypatch.setattr(calibration_script, "build_task_instance", _fake_build_task_instance)

    report = calibration_script.calibrate_env_grid(
        env_grid_path=env_grid_path,
        timeout_seconds=300.0,
        shrink_ratio=0.95,
    )

    job_a = report["jobs"][0]
    assert job_a["calibrated_astar_max_nodes"] == 240
    assert job_a["calibrated_astar_max_expansions"] == 120
    assert job_a["per_seed"][0]["calibrated_astar_max_nodes"] == 190
    assert job_a["per_seed"][0]["calibrated_astar_max_expansions"] == 95
    assert job_a["per_seed"][1]["calibrated_astar_max_nodes"] == 240
    assert job_a["per_seed"][1]["calibrated_astar_max_expansions"] == 120

    job_b = report["eval_jobs"][0]
    assert job_b["calibrated_astar_max_nodes"] == 38
    assert job_b["calibrated_astar_max_expansions"] == 19


def test_write_flag_updates_only_budget_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration_script = _load_calibration_script()
    env_grid_path = _write_env_grid(tmp_path)
    report_path = tmp_path / "report.json"

    fake_report = {
        "env_grid_path": str(env_grid_path),
        "timeout_seconds": 300.0,
        "shrink_ratio": 0.95,
        "jobs": [
            {
                "name": "job-a",
                "original_astar_max_nodes": 20000,
                "original_astar_max_expansions": 5000,
                "calibrated_astar_max_nodes": 123,
                "calibrated_astar_max_expansions": 45,
            }
        ],
        "eval_jobs": [
            {
                "name": "job-b",
                "original_astar_max_nodes": 20000,
                "original_astar_max_expansions": 5000,
                "calibrated_astar_max_nodes": 321,
                "calibrated_astar_max_expansions": 54,
            }
        ],
    }

    monkeypatch.setattr(calibration_script, "calibrate_env_grid", lambda **_kwargs: fake_report)

    exit_code = calibration_script.main(
        [
            "--env-grid",
            str(env_grid_path),
            "--report-path",
            str(report_path),
            "--write",
        ]
    )

    assert exit_code == 0
    updated = yaml.safe_load(env_grid_path.read_text(encoding="utf-8"))
    assert updated["jobs"][0]["name"] == "job-a"
    assert updated["jobs"][0]["benchmark_id"] == "trivial-1m"
    assert updated["jobs"][0]["astar_max_nodes"] == 123
    assert updated["jobs"][0]["astar_max_expansions"] == 45
    assert updated["eval_jobs"][0]["name"] == "job-b"
    assert updated["eval_jobs"][0]["astar_max_nodes"] == 321
    assert updated["eval_jobs"][0]["astar_max_expansions"] == 54
    assert json.loads(report_path.read_text(encoding="utf-8")) == fake_report
    assert "[jobs]" in capsys.readouterr().out


def test_dry_run_leaves_yaml_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration_script = _load_calibration_script()
    env_grid_path = _write_env_grid(tmp_path)
    original_text = env_grid_path.read_text(encoding="utf-8")

    fake_report = {
        "env_grid_path": str(env_grid_path),
        "timeout_seconds": 300.0,
        "shrink_ratio": 0.95,
        "jobs": [
            {
                "name": "job-a",
                "original_astar_max_nodes": 20000,
                "original_astar_max_expansions": 5000,
                "calibrated_astar_max_nodes": 1,
                "calibrated_astar_max_expansions": 1,
            }
        ],
        "eval_jobs": [
            {
                "name": "job-b",
                "original_astar_max_nodes": 20000,
                "original_astar_max_expansions": 5000,
                "calibrated_astar_max_nodes": 2,
                "calibrated_astar_max_expansions": 2,
            }
        ],
    }

    monkeypatch.setattr(calibration_script, "calibrate_env_grid", lambda **_kwargs: fake_report)

    exit_code = calibration_script.main(["--env-grid", str(env_grid_path)])

    assert exit_code == 0
    assert env_grid_path.read_text(encoding="utf-8") == original_text
