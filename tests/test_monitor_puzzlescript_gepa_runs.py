"""Tests for active batched PuzzleScript GEPA run monitoring."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.monitor_puzzlescript_gepa_runs import (
    parse_scontrol_show_job,
    summarize_active_runs,
)


def test_parse_scontrol_show_job_extracts_status_fields() -> None:
    parsed = parse_scontrol_show_job(
        """
        JobId=123 JobName=gepa-eff
           JobState=PENDING Reason=QOSGrpGRES Dependency=(null)
           RunTime=00:00:00 TimeLimit=01:00:00 TimeMin=N/A
           StartTime=2026-07-04T02:31:27 EndTime=2026-07-04T03:31:27
           StdOut=/tmp/job.out
        """
    )

    assert parsed["JobState"] == "PENDING"
    assert parsed["Reason"] == "QOSGrpGRES"
    assert parsed["RunTime"] == "00:00:00"
    assert parsed["TimeLimit"] == "01:00:00"
    assert parsed["StartTime"] == "2026-07-04T02:31:27"
    assert parsed["StdOut"] == "/tmp/job.out"


def test_summarize_active_runs_combines_job_and_artifact_state(tmp_path: Path) -> None:
    eval_dir = tmp_path / "run" / "candidate_evals" / "eval-00001-abc-def"
    eval_dir.mkdir(parents=True)
    (eval_dir / "scored_results.json").write_text(
        json.dumps(
            [
                {
                    "game": "a",
                    "solved": True,
                    "baseline_solved": True,
                    "score": 0.9,
                    "baseline_score": 0.9,
                    "expanded": 10,
                    "baseline_expanded": 20,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = [
        {
            "job_id": "123",
            "label": "demo",
            "state_root": str(tmp_path / "run"),
            "seed": 1,
        }
    ]

    rows = summarize_active_runs(
        manifest,
        job_info_fn=lambda _job_id: {"JobState": "RUNNING", "Reason": "None"},
    )

    assert rows[0]["job_id"] == "123"
    assert rows[0]["label"] == "demo"
    assert rows[0]["job_state"] == "RUNNING"
    assert rows[0]["n_scored_evals"] == 1
    assert rows[0]["best_metric"] > 0.0
