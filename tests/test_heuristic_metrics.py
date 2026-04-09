from __future__ import annotations

import pytest

from llm_desparsifier.search.metrics import (
    SearchSeedResult,
    compute_seed_score,
    mean_job_scores,
    summarize_batch,
)


def test_compute_seed_score_matches_plan_formula_for_solved_seed() -> None:
    score, candidate_cost = compute_seed_score(
        solved=True,
        expanded_states=40,
        solution_length=10,
        astar_max_expansions=100,
    )
    assert candidate_cost == 40
    assert score == pytest.approx(61.0 / 101.0)


def test_compute_seed_score_assigns_zero_to_unsolved_seed() -> None:
    score, candidate_cost = compute_seed_score(
        solved=False,
        expanded_states=55,
        solution_length=None,
        astar_max_expansions=100,
    )
    assert candidate_cost == 101
    assert score == 0.0


def test_compute_seed_score_prefers_fewer_expansions_for_solved_seed() -> None:
    fast_score, _ = compute_seed_score(
        solved=True,
        expanded_states=10,
        solution_length=4,
        astar_max_expansions=100,
    )
    slow_score, _ = compute_seed_score(
        solved=True,
        expanded_states=40,
        solution_length=4,
        astar_max_expansions=100,
    )
    assert fast_score > slow_score


def test_compute_seed_score_is_clipped_to_unit_interval() -> None:
    score, candidate_cost = compute_seed_score(
        solved=True,
        expanded_states=-5,
        solution_length=1,
        astar_max_expansions=100,
    )
    assert candidate_cost == 0
    assert score == pytest.approx(1.0)


def test_summarize_batch_averages_seed_scores() -> None:
    batch = summarize_batch(
        [
            SearchSeedResult(
                seed=1,
                solved=True,
                expanded_states=10,
                generated_states=12,
                solution_length=5,
                termination_reason="solved",
                actions=[0],
                search_trace={},
                validation={},
                seed_score=0.5,
                candidate_cost=10,
            ),
            SearchSeedResult(
                seed=2,
                solved=False,
                expanded_states=100,
                generated_states=120,
                solution_length=None,
                termination_reason="max_expansions_reached",
                actions=[],
                search_trace={},
                validation={},
                seed_score=0.0,
                candidate_cost=100,
            ),
        ]
    )
    assert batch.job_score == pytest.approx(0.25)
    assert batch.aggregate_stats["solve_rate"] == pytest.approx(0.5)
    assert batch.aggregate_stats["solved_count"] == 1


def test_mean_job_scores_returns_zero_for_empty_input() -> None:
    assert mean_job_scores([]) == 0.0
