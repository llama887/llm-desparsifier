"""Tests for the quick PuzzleScript model smoke runner."""

from __future__ import annotations

from scripts.smoke_compare_puzzlescript_models import (
    select_smoke_jobs,
    summarize_smoke_outputs,
)


def test_select_smoke_jobs_preserves_requested_order_and_fills_missing() -> None:
    eval_jobs = [
        {"name": "first"},
        {"name": "Crates_and_Portals"},
        {"name": "Gravity_Sokoban"},
        {"name": "fallback"},
    ]

    selected = select_smoke_jobs(
        eval_jobs,
        ["Gravity_Sokoban", "missing", "Crates_and_Portals"],
        max_games=3,
    )

    assert [job["name"] for job in selected] == [
        "Gravity_Sokoban",
        "Crates_and_Portals",
        "first",
    ]


def test_summarize_smoke_outputs_reports_code_shape_and_errors() -> None:
    outputs = [
        {
            "score": 0.5,
            "adjusted_score": 0.5,
            "solved": True,
            "expanded": 100,
            "synthesis_error": None,
            "error": None,
        },
        {
            "score": 0.0,
            "adjusted_score": -2.0,
            "solved": False,
            "expanded": 1000,
            "synthesis_error": "bad code",
            "error": "validation failed",
        },
    ]
    trajectories = [
        {
            "heuristic_code": (
                "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                "    object_positions = ctx.get('object_positions', {})\n"
                "    object_names = ctx.get('object_names', [])\n"
                "    win_conditions = ctx.get('win_conditions_text', '')\n"
                "    return float(ctx.get('score_normalized', 0.0))\n"
            )
        },
        {"heuristic_code": "return crate + target\n"},
    ]

    summary = summarize_smoke_outputs(outputs, trajectories)

    assert summary["n"] == 2
    assert summary["solved"] == 1
    assert summary["candidate_errors"] == 1
    assert summary["synthesis_errors"] == 1
    assert summary["starts_with_contract_def"] == 1
    assert summary["uses_object_positions"] == 1
    assert summary["uses_object_names"] == 1
    assert summary["uses_win_text"] == 1
    assert summary["uses_score_fallback"] == 1
    assert summary["mentions_hardcoded_sokoban_roles"] == 1
