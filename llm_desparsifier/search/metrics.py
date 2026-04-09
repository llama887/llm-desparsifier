"""Metric helpers for heuristic search evaluation.

This module centralizes the expansion-cap-normalized GEPA score used by the
search-only pipeline. It is needed because runner code, tests, and artifact
writers must agree on the exact scalarization, and it differs from the older
solved-path ratio by mapping solved runs directly onto a bounded ``[0, 1]``
scale based on how much of the configured search budget they avoided using.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping


@dataclass(frozen=True)
class SearchSeedResult:
    """Structured search outcome for one environment seed.

    This record holds the per-seed quantities needed for scoring, feedback, and
    replay artifact selection. It is needed because the heuristic prompt is
    evaluated across many seeds per job, and it differs from raw planner output
    by including the derived `seed_score` alongside search statistics.
    """

    seed: int
    solved: bool
    expanded_states: int
    generated_states: int
    solution_length: int | None
    termination_reason: str
    actions: list[int]
    search_trace: dict[str, Any]
    validation: dict[str, Any]
    seed_score: float
    candidate_cost: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of one seed result.

        This helper is used when persisting aggregated search artifacts. It is
        needed because dataclasses containing lists and nested dicts should be
        serialized consistently, and it differs from manual dict construction by
        keeping the artifact schema aligned with the dataclass fields.
        """

        return asdict(self)


@dataclass(frozen=True)
class SearchBatchResult:
    """Aggregated search results for one synthesized heuristic over many seeds.

    This record packages per-seed outcomes with job-level summary statistics. It
    is needed because GEPA consumes a single scalar score per example while the
    artifact model preserves the seed-level detail, and it differs from
    `SearchSeedResult` by including averages and histograms.
    """

    seed_results: list[SearchSeedResult]
    job_score: float
    aggregate_stats: dict[str, Any]


def compute_seed_score(
    *,
    solved: bool,
    expanded_states: int,
    solution_length: int | None,
    astar_max_expansions: int,
) -> tuple[float, int]:
    """Compute the expansion-cap-normalized GEPA score for one seed.

    This helper implements the current heuristic metric requested for GEPA:
    ``((N + 1) - S) / (N + 1)`` where ``N`` is ``astar_max_expansions`` and
    ``S`` is the actual expanded-state count for solved runs. Unsolved runs are
    treated as if they consumed ``N + 1`` states, which yields an exact score of
    ``0.0``. It is needed because the optimizer expects a stable scalar in
    ``[0, 1]``, and it differs from the earlier solved-path ratio by directly
    rewarding budget savings rather than normalizing by solution length.
    """

    del solution_length
    search_cap = max(0, int(astar_max_expansions))
    if solved:
        candidate_cost = max(0, min(int(expanded_states), search_cap))
    else:
        candidate_cost = search_cap + 1
    denominator = float(search_cap + 1)
    if denominator <= 0.0:
        return 0.0, candidate_cost
    score = (denominator - float(candidate_cost)) / denominator
    return max(0.0, min(1.0, score)), candidate_cost


def summarize_batch(seed_results: list[SearchSeedResult]) -> SearchBatchResult:
    """Aggregate a set of per-seed search outcomes into one job-level summary.

    This helper computes the job score and all feedback-facing averages from the
    per-seed results. It is needed because GEPA optimizes one scalar per job
    while reflection and artifacts still need richer summary statistics, and it
    differs from the runner's final environment averaging by staying within one
    `(env_id, benchmark_id)` job.
    """

    if not seed_results:
        raise ValueError("summarize_batch requires at least one seed result")
    solved_results = [result for result in seed_results if result.solved]
    solution_lengths = [
        float(result.solution_length)
        for result in solved_results
        if result.solution_length is not None
    ]
    aggregate_stats = {
        "num_seeds": len(seed_results),
        "solved_count": sum(1 for result in seed_results if result.solved),
        "solve_rate": mean(1.0 if result.solved else 0.0 for result in seed_results),
        "average_expanded_states": mean(result.expanded_states for result in seed_results),
        "average_generated_states": mean(result.generated_states for result in seed_results),
        "average_solution_length": mean(solution_lengths) if solution_lengths else 0.0,
        "average_solved_seed_efficiency": (
            mean(result.seed_score for result in solved_results) if solved_results else 0.0
        ),
        "termination_reasons": dict(
            Counter(result.termination_reason for result in seed_results)
        ),
    }
    job_score = mean(result.seed_score for result in seed_results)
    aggregate_stats["job_score"] = job_score
    return SearchBatchResult(
        seed_results=seed_results,
        job_score=job_score,
        aggregate_stats=aggregate_stats,
    )


def mean_job_scores(job_results: list[Mapping[str, Any]]) -> float:
    """Average job scores across the full GEPA training set.

    This helper produces the final scalar returned to GEPA for a prompt
    candidate. It is needed because the optimizer expects one higher-is-better
    metric value per candidate prompt, and it differs from `summarize_batch` by
    aggregating over heterogeneous jobs rather than seeds of one job.
    """

    if not job_results:
        return 0.0
    return mean(float(job["job_score"]) for job in job_results)


__all__ = [
    "SearchBatchResult",
    "SearchSeedResult",
    "compute_seed_score",
    "mean_job_scores",
    "summarize_batch",
]
