#!/usr/bin/env python3
"""Calibrate per-job A* budgets from blind-search measurements.

This script derives `astar_max_nodes` and `astar_max_expansions` for each job
in the heuristic env grid by running blind A* on the job's deterministic
holdout seeds. It is needed because the heuristic-search pipeline works best
when budgets sit just outside what weak heuristics can solve, and it differs
from hand-editing `configs/gepa_envs.yaml` by measuring actual search effort
and applying the same solved-versus-timeout policy consistently across jobs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from llm_desparsifier.search import (
    JAxtarSearchBackend,
    SearchConfig,
    SearchTask,
    build_task_instance,
)

DEFAULT_ENV_GRID = Path("configs/gepa_envs.yaml")
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_SHRINK_RATIO = 0.95
MAX_CALIBRATION_BUDGET = 2_147_483_647


def _load_run_heuristic_batch_module() -> Any:
    """Load the heuristic runner module so this script can reuse its grid schema.

    The supported env-grid parser currently lives in `scripts/run_heuristic_batch.py`
    rather than a shared library module. This helper is needed because the
    calibration script must stay aligned with that exact `EnvJob` contract, and
    it differs from a normal import by loading the sibling script directly from
    disk so the CLI works even though `scripts/` is not a Python package.
    """

    module_path = Path(__file__).with_name("run_heuristic_batch.py")
    spec = importlib.util.spec_from_file_location("run_heuristic_batch", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_RUN_HEURISTIC_BATCH = _load_run_heuristic_batch_module()
load_env_grid = _RUN_HEURISTIC_BATCH.load_env_grid


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for blind-A* budget calibration.

    The calibration workflow needs a small CLI so users can preview budgets,
    write the YAML in place, and persist a machine-readable report. This helper
    is needed because that behavior is now a supported repo entrypoint, and it
    differs from the GEPA runner parser by exposing timeout, shrink, and write
    controls instead of prompt-optimization settings.
    """

    parser = argparse.ArgumentParser(
        description="Calibrate astar_max_nodes and astar_max_expansions from blind A* runs.",
    )
    parser.add_argument(
        "--env-grid",
        type=Path,
        default=DEFAULT_ENV_GRID,
        help="YAML file describing heuristic-search jobs (default: configs/gepa_envs.yaml).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Wall-clock timeout per seed for blind A* calibration (default: 300).",
    )
    parser.add_argument(
        "--shrink-ratio",
        type=float,
        default=DEFAULT_SHRINK_RATIO,
        help="Multiplier applied to solved blind-A* counts (default: 0.95).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the env grid in place with calibrated budgets.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON path for the full calibration report.",
    )
    return parser.parse_args(argv)


def _zero_heuristic(_ts: Any, _env_params: Any, _ctx: dict[str, Any]) -> float:
    """Return the blind-search heuristic value used during calibration.

    Budget calibration must measure plain A* without any heuristic guidance so
    the resulting caps reflect how much search effort a weak heuristic would
    need. This helper is needed because `JAxtarSearchBackend` always expects a
    heuristic callback, and it differs from synthesized heuristics by always
    returning exactly `0.0`.
    """

    return 0.0


def _calibrated_count(*, observed_count: int, solved: bool, shrink_ratio: float) -> int:
    """Convert one observed search counter into the calibrated budget value.

    The methods policy intentionally treats solved blind-A* seeds differently
    from timed-out seeds. This helper is needed because both node and expansion
    counts follow the same solved-versus-timeout rule, and it differs from raw
    planner output by applying the tightening step only when blind A* solved
    within the allowed wall-clock budget.
    """

    bounded_count = max(1, int(observed_count))
    if not solved:
        return bounded_count
    return max(1, math.floor(float(bounded_count) * float(shrink_ratio)))


def _build_task(job: Any, *, seed: int) -> SearchTask:
    """Materialize one deterministic search task for budget calibration.

    The calibration path should use the same XLand task construction logic as
    heuristic evaluation so the measured counts match actual training and
    holdout search instances. This helper is needed because the script iterates
    over raw seed integers, and it differs from the runner's batch evaluation by
    producing just one `SearchTask` at a time for blind A*.
    """

    env, env_params, step_fn, root_timestep, _reset_key, task_instance = build_task_instance(
        env_id=job.env_id,
        benchmark_id=job.benchmark_id,
        seed=seed,
        deterministic_rulesets=job.deterministic_rulesets,
        fixed_ruleset_seed=job.fixed_ruleset_seed,
    )
    return SearchTask(
        env=env,
        env_params=env_params,
        step_fn=step_fn,
        root_timestep=root_timestep,
        task_instance=task_instance,
    )


def calibrate_job(
    *,
    job: Any,
    timeout_seconds: float,
    shrink_ratio: float,
    backend: JAxtarSearchBackend,
) -> dict[str, Any]:
    """Measure blind A* on all calibration seeds for one env-grid job.

    Each job in `configs/gepa_envs.yaml` ultimately needs a single node cap and
    a single expansion cap. This helper is needed because calibration happens at
    the per-seed level before collapsing to worst-case job budgets, and it
    differs from the GEPA evaluator by running only blind A* over the explicit
    holdout seeds rather than prompt-generated heuristics over sampled training
    seeds.
    """

    seeds = list(job.holdout_seeds)
    if not seeds:
        raise ValueError(f"Job '{job.name}' is missing holdout_seeds for calibration")

    per_seed: list[dict[str, Any]] = []
    calibrated_nodes: list[int] = []
    calibrated_expansions: list[int] = []

    for seed in seeds:
        seed_result = backend.solve_many(
            task_batch=[_build_task(job, seed=seed)],
            heuristic_fn=_zero_heuristic,
            search_config=SearchConfig(
                max_nodes=MAX_CALIBRATION_BUDGET,
                max_expansions=MAX_CALIBRATION_BUDGET,
                wall_clock_timeout_seconds=timeout_seconds,
            ),
        ).seed_results[0]
        seed_nodes = _calibrated_count(
            observed_count=seed_result.generated_states,
            solved=seed_result.solved,
            shrink_ratio=shrink_ratio,
        )
        seed_expansions = _calibrated_count(
            observed_count=seed_result.expanded_states,
            solved=seed_result.solved,
            shrink_ratio=shrink_ratio,
        )
        calibrated_nodes.append(seed_nodes)
        calibrated_expansions.append(seed_expansions)
        per_seed.append(
            {
                "seed": int(seed),
                "solved": bool(seed_result.solved),
                "termination_reason": seed_result.termination_reason,
                "observed_generated_states": int(seed_result.generated_states),
                "observed_expanded_states": int(seed_result.expanded_states),
                "calibrated_astar_max_nodes": int(seed_nodes),
                "calibrated_astar_max_expansions": int(seed_expansions),
            }
        )

    return {
        "name": job.name,
        "env_id": job.env_id,
        "benchmark_id": job.benchmark_id,
        "calibration_seeds": seeds,
        "original_astar_max_nodes": int(job.astar_max_nodes),
        "original_astar_max_expansions": int(job.astar_max_expansions),
        "calibrated_astar_max_nodes": max(calibrated_nodes),
        "calibrated_astar_max_expansions": max(calibrated_expansions),
        "per_seed": per_seed,
    }


def calibrate_env_grid(
    *,
    env_grid_path: Path,
    timeout_seconds: float,
    shrink_ratio: float,
) -> dict[str, Any]:
    """Calibrate every training and holdout job in one env-grid YAML file.

    The main artifact of this workflow is a full-report object that can be
    printed, saved as JSON, and optionally written back into the YAML. This
    helper is needed because the script calibrates two job sections with the
    same policy, and it differs from lower-level per-job helpers by returning
    the complete cross-section report in one stable structure.
    """

    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be > 0")
    if not 0.0 < shrink_ratio <= 1.0:
        raise ValueError("shrink_ratio must be in the interval (0, 1]")

    jobs, eval_jobs = load_env_grid(
        env_grid_path,
        default_astar_max_nodes=MAX_CALIBRATION_BUDGET,
        default_astar_max_expansions=MAX_CALIBRATION_BUDGET,
    )
    backend = JAxtarSearchBackend()

    def _calibrate_section(section_jobs: list[Any]) -> list[dict[str, Any]]:
        """Calibrate one env-grid section while preserving its job ordering.

        The YAML stores training and holdout jobs separately but both sections
        follow the same blind-A* measurement rule. This helper is needed because
        the surrounding function wants one compact report builder, and it differs
        from `calibrate_job(...)` by iterating over an entire section.
        """

        return [
            calibrate_job(
                job=job,
                timeout_seconds=timeout_seconds,
                shrink_ratio=shrink_ratio,
                backend=backend,
            )
            for job in section_jobs
        ]

    return {
        "env_grid_path": str(env_grid_path),
        "timeout_seconds": float(timeout_seconds),
        "shrink_ratio": float(shrink_ratio),
        "jobs": _calibrate_section(jobs),
        "eval_jobs": _calibrate_section(eval_jobs),
    }


def _update_job_entries(
    *,
    entries: list[dict[str, Any]],
    section_report: Sequence[Mapping[str, Any]],
) -> None:
    """Write calibrated budgets into an already-loaded YAML section in place.

    The script must preserve each job's order and unrelated fields while
    updating only the two search-budget keys. This helper is needed because the
    raw YAML document is still the source of truth for ordering and extra
    fields, and it differs from rebuilding the jobs from scratch by mutating
    just the budget values on the existing mappings.
    """

    report_by_name = {str(job_report["name"]): job_report for job_report in section_report}
    for entry in entries:
        job_report = report_by_name[str(entry["name"])]
        entry["astar_max_nodes"] = int(job_report["calibrated_astar_max_nodes"])
        entry["astar_max_expansions"] = int(job_report["calibrated_astar_max_expansions"])


def write_calibrated_env_grid(env_grid_path: Path, report: Mapping[str, Any]) -> None:
    """Rewrite the env-grid YAML with the calibrated budgets from one report.

    Dry-run calibration should be safe by default, but users also need one
    command that applies the measured caps to the canonical env grid. This
    helper is needed because the script writes only when `--write` is set, and
    it differs from JSON report output by preserving the YAML structure already
    present in the repository.
    """

    data = yaml.safe_load(env_grid_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Environment grid must be a mapping to support in-place updates")
    job_entries = data.get("jobs", [])
    eval_entries = data.get("eval_jobs", [])
    if not isinstance(job_entries, list) or not isinstance(eval_entries, list):
        raise ValueError("Environment grid jobs and eval_jobs must both be lists")

    _update_job_entries(entries=job_entries, section_report=report.get("jobs", []))
    _update_job_entries(entries=eval_entries, section_report=report.get("eval_jobs", []))
    env_grid_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _print_report(report: Mapping[str, Any]) -> None:
    """Print a concise human-readable summary of the calibration results.

    The calibration run can take a long time, so users need a terminal summary
    they can skim without opening JSON artifacts. This helper is needed because
    the script is a CLI-first workflow, and it differs from the full report by
    collapsing each job down to its original and calibrated budget pair.
    """

    for section_name in ("jobs", "eval_jobs"):
        print(f"[{section_name}]")
        for job_report in report.get(section_name, []):
            print(
                f"{job_report['name']}: "
                f"nodes {job_report['original_astar_max_nodes']} -> "
                f"{job_report['calibrated_astar_max_nodes']}, "
                f"expansions {job_report['original_astar_max_expansions']} -> "
                f"{job_report['calibrated_astar_max_expansions']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run blind-A* budget calibration and optionally rewrite the env grid.

    This is the supported CLI entrypoint for the new calibration workflow. It
    is needed because methods now require budget calibration before GEPA, and it
    differs from the main heuristic runner by performing only blind search,
    report generation, and optional YAML rewriting.
    """

    args = parse_args(argv)
    env_grid_path = args.env_grid.expanduser().resolve()
    report = calibrate_env_grid(
        env_grid_path=env_grid_path,
        timeout_seconds=float(args.timeout_seconds),
        shrink_ratio=float(args.shrink_ratio),
    )
    _print_report(report)
    if args.report_path is not None:
        report_path = args.report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.write:
        write_calibrated_env_grid(env_grid_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
