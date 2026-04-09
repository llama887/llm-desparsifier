"""Validation helpers for synthesized heuristics.

This module aggregates static and runtime heuristic diagnostics into a stable
artifact schema. It is needed because GEPA feedback and saved artifacts should
share one authoritative validation payload, and it differs from the search
backend metrics by focusing on correctness warnings rather than optimization
score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class HeuristicValidationResult:
    """Structured validation payload for one synthesized heuristic.

    This dataclass captures the contract and admissibility summary emitted into
    `heuristic_validation.json`. It is needed because downstream feedback,
    logging, and replay tooling should consume one stable schema, and it differs
    from raw backend counters by clearly separating compile errors, contract
    violations, and admissibility warnings.
    """

    compile_ok: bool
    sanitizer_errors: list[str] = field(default_factory=list)
    sanitizer_warnings: list[str] = field(default_factory=list)
    contract_violations: list[str] = field(default_factory=list)
    goal_zero_pass: bool | None = None
    nonnegative_pass: bool | None = None
    consistency_pass: bool | None = None
    admissibility_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the validation result.

        This helper centralizes serialization for artifact writing. It is needed
        because validation results are persisted to disk and embedded in other
        payloads, and it differs from ad hoc dict construction by preserving the
        exact dataclass field names and defaults.
        """

        return asdict(self)


def aggregate_validation_results(seed_results: Iterable[Mapping[str, Any]]) -> HeuristicValidationResult:
    """Aggregate per-seed validation diagnostics into one job-level result.

    This helper merges runtime admissibility checks across all evaluated seeds.
    It is needed because GEPA scores one synthesized heuristic over many seeds,
    and it differs from per-seed statistics by exposing pass/fail summaries at
    the candidate-job level.
    """

    results = list(seed_results)
    if not results:
        return HeuristicValidationResult(
            compile_ok=True,
            sanitizer_warnings=["no seed-level validation results were recorded"],
        )
    goal_zero_values = [bool(item.get("goal_zero_pass", False)) for item in results]
    nonnegative_values = [bool(item.get("nonnegative_pass", False)) for item in results]
    consistency_values = [bool(item.get("consistency_pass", False)) for item in results]
    admissibility_pass_count = sum(
        1 for item in results if bool(item.get("admissibility_pass", False))
    )
    summary = {
        "num_seed_results": len(results),
        "admissibility_pass_count": admissibility_pass_count,
        "admissibility_pass_rate": admissibility_pass_count / float(len(results)),
        "goal_zero_violation_count": sum(
            int(item.get("admissibility_goal_violation_count", 0)) for item in results
        ),
        "consistency_violation_count": sum(
            int(item.get("consistency_violation_count", 0)) for item in results
        ),
        "path_overestimate_count": sum(
            int(item.get("path_overestimate_count", 0)) for item in results
        ),
        "max_path_overestimate": max(
            float(item.get("max_path_overestimate", 0.0)) for item in results
        ),
    }
    return HeuristicValidationResult(
        compile_ok=True,
        goal_zero_pass=all(goal_zero_values),
        nonnegative_pass=all(nonnegative_values),
        consistency_pass=all(consistency_values),
        admissibility_summary=summary,
    )


__all__ = ["HeuristicValidationResult", "aggregate_validation_results"]
