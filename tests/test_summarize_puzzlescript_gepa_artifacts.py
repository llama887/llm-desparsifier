"""Tests for batched PuzzleScript GEPA artifact summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_puzzlescript_gepa_artifacts import (
    summarize_eval_dir,
    summarize_scored_results,
)


def test_summarize_scored_results_reports_solve_and_efficiency_gap() -> None:
    rows = [
        {
            "game": "new",
            "solved": True,
            "baseline_solved": False,
            "score": 0.9,
            "baseline_score": 0.0,
            "expanded": 20,
            "baseline_expanded": 10_000,
        },
        {
            "game": "fast",
            "solved": True,
            "baseline_solved": True,
            "score": 0.9,
            "baseline_score": 0.9,
            "expanded": 250,
            "baseline_expanded": 1_000,
        },
        {
            "game": "slow",
            "solved": True,
            "baseline_solved": True,
            "score": 0.9,
            "baseline_score": 0.9,
            "expanded": 1_000,
            "baseline_expanded": 500,
        },
        {
            "game": "lost",
            "solved": False,
            "baseline_solved": True,
            "score": 0.0,
            "baseline_score": 0.8,
            "expanded": 10_000,
            "baseline_expanded": 200,
        },
    ]

    summary = summarize_scored_results(
        rows,
        common_solve_efficiency_weight=2.0,
        common_solve_efficiency_clip=1.0,
    )

    assert summary["n"] == 4
    assert summary["solved"] == 3
    assert summary["baseline_solved"] == 3
    assert summary["net_solve"] == 0
    assert summary["new_solves"] == 1
    assert summary["lost_solves"] == 1
    assert summary["common_solves"] == 2
    assert summary["common_faster"] == 1
    assert summary["common_slower"] == 1
    assert summary["high_headroom_common_solves"] == 2
    assert summary["mean_common_log2_base_over_candidate"] == pytest.approx(0.5, abs=0.02)
    assert summary["current_metric_score"] < 0.0
    by_game = {row["game"]: row for row in summary["game_summaries"]}
    assert by_game["fast"]["common_faster"] == 1
    assert by_game["fast"]["mean_common_log2_base_over_candidate"] > 0.0
    assert by_game["slow"]["common_slower"] == 1
    assert by_game["slow"]["mean_common_log2_base_over_candidate"] < 0.0
    assert by_game["lost"]["lost_solves"] == 1


def test_summarize_eval_dir_reads_scored_results(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval-00001-abc-def"
    eval_dir.mkdir()
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

    summary = summarize_eval_dir(eval_dir)

    assert summary["eval_dir"] == str(eval_dir)
    assert summary["eval_name"] == "eval-00001-abc-def"
    assert summary["n"] == 1
    assert summary["common_faster"] == 1
