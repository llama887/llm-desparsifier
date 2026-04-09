from __future__ import annotations

import pytest

from llm_desparsifier.heuristics.validation import aggregate_validation_results


def test_aggregate_validation_results_computes_pass_rates() -> None:
    result = aggregate_validation_results(
        [
            {
                "goal_zero_pass": True,
                "nonnegative_pass": True,
                "consistency_pass": True,
                "admissibility_goal_violation_count": 0,
                "consistency_violation_count": 0,
                "path_overestimate_count": 0,
                "max_path_overestimate": 0.0,
                "admissibility_pass": True,
            },
            {
                "goal_zero_pass": False,
                "nonnegative_pass": True,
                "consistency_pass": False,
                "admissibility_goal_violation_count": 1,
                "consistency_violation_count": 3,
                "path_overestimate_count": 2,
                "max_path_overestimate": 4.0,
                "admissibility_pass": False,
            },
        ]
    )
    payload = result.to_dict()
    assert payload["goal_zero_pass"] is False
    assert payload["consistency_pass"] is False
    assert payload["admissibility_summary"]["admissibility_pass_rate"] == pytest.approx(0.5)
    assert payload["admissibility_summary"]["max_path_overestimate"] == pytest.approx(4.0)
