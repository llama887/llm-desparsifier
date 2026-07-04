#!/usr/bin/env python3
"""Monitor active batched PuzzleScript GEPA SLURM jobs and artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.summarize_puzzlescript_gepa_artifacts import summarize_root  # noqa: E402

DEFAULT_ACTIVE_RUNS_MANIFEST = Path("configs/gepa_active_h100_runs_s102_s106.json")


def parse_scontrol_show_job(text: str) -> dict[str, str]:
    """Parse useful ``scontrol show job`` key-value fields."""
    return {
        match.group("key"): match.group("value")
        for match in re.finditer(r"(?P<key>[A-Za-z][A-Za-z0-9]*)=(?P<value>\S+)", text)
    }


def scontrol_job_info(job_id: str) -> dict[str, str]:
    """Return parsed SLURM job status for one job id."""
    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    return parse_scontrol_show_job(completed.stdout)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load active GEPA run manifest entries from JSON."""
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def summarize_active_runs(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    job_info_fn: Callable[[str], Mapping[str, str]] = scontrol_job_info,
) -> list[dict[str, Any]]:
    """Return joined SLURM and artifact status for active batched GEPA runs."""
    rows: list[dict[str, Any]] = []
    for entry in manifest_rows:
        job_id = str(entry.get("job_id", ""))
        state_root = Path(str(entry.get("state_root", ""))).expanduser()
        job_info = dict(job_info_fn(job_id)) if job_id else {}
        artifact_summary = summarize_root(state_root)
        best_eval = artifact_summary.get("best_eval")
        best_metric = None
        best_eval_name = None
        solved = None
        baseline_solved = None
        high_eff = None
        if isinstance(best_eval, Mapping):
            best_metric = best_eval.get("current_metric_score")
            best_eval_name = best_eval.get("eval_name")
            solved = best_eval.get("solved")
            baseline_solved = best_eval.get("baseline_solved")
            high_eff = best_eval.get("high_headroom_mean_log2_base_over_candidate")
        rows.append(
            {
                "job_id": job_id,
                "label": entry.get("label"),
                "seed": entry.get("seed"),
                "seed_addendum_file": entry.get("seed_addendum_file"),
                "state_root": str(state_root),
                "job_state": job_info.get("JobState", "UNKNOWN"),
                "reason": job_info.get("Reason", ""),
                "runtime": job_info.get("RunTime", ""),
                "time_limit": job_info.get("TimeLimit", ""),
                "start_time": job_info.get("StartTime", ""),
                "stdout": job_info.get("StdOut", ""),
                "n_scored_evals": artifact_summary.get("n_scored_evals", 0),
                "best_eval": best_eval_name,
                "best_metric": best_metric,
                "best_solved": solved,
                "best_baseline_solved": baseline_solved,
                "best_high_headroom_efficiency": high_eff,
            }
        )
    return rows


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:+.{digits}f}"


def _shorten(value: Any, width: int) -> str:
    text = str(value) if value is not None else "-"
    return text if len(text) <= width else text[: max(0, width - 1)] + "~"


def print_active_run_table(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print a compact active-run table."""
    headers = ["job", "label", "state", "reason", "runtime", "evals", "metric", "sol/base", "high"]
    table: list[list[str]] = []
    for row in rows:
        solved = row.get("best_solved")
        baseline = row.get("best_baseline_solved")
        sol_base = "-" if solved is None or baseline is None else f"{solved}/{baseline}"
        table.append(
            [
                str(row.get("job_id") or "-"),
                _shorten(row.get("label"), 32),
                str(row.get("job_state") or "-"),
                _shorten(row.get("reason"), 16),
                str(row.get("runtime") or "-"),
                str(row.get("n_scored_evals") or 0),
                _format_float(row.get("best_metric")),
                sol_base,
                _format_float(row.get("best_high_headroom_efficiency")),
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor active batched PuzzleScript GEPA jobs and artifact roots.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ACTIVE_RUNS_MANIFEST,
        help="JSON manifest containing job_id/state_root entries.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional status JSON path.")
    args = parser.parse_args()

    rows = summarize_active_runs(load_manifest(args.manifest))
    print_active_run_table(rows)
    if args.output is not None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest),
            "runs": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote status: {args.output}")


if __name__ == "__main__":
    main()
