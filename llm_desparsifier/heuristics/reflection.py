"""Deterministic feedback builders for synthesized heuristics.

This module turns search outcomes and admissibility diagnostics into stable
revision guidance for GEPA. It is needed because GEPA optimizes a scalar score
plus text feedback, and it differs from the old reward reflection stack by
describing search behavior rather than policy-learning curves.
"""

from __future__ import annotations

from typing import Any, Mapping


def _format_float(value: Any) -> str:
    """Return a stable short float representation for feedback text.

    This helper keeps deterministic feedback readable and compact. It is needed
    because feedback mixes integer counts with aggregate rates and ratios, and it
    differs from raw `str(float(...))` by consistently using four decimals.
    """

    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def build_heuristic_feedback(
    *,
    env_summary: str,
    heuristic_code: str,
    aggregate_stats: Mapping[str, Any],
    validation_result: Mapping[str, Any],
) -> str:
    """Build deterministic GEPA feedback for one evaluated heuristic.

    This helper writes the five feedback sections required by the refactor plan.
    It is needed because GEPA should receive stable, debuggable guidance even
    when no reflection LM is used, and it differs from the legacy reward
    feedback path by focusing on search efficiency and admissibility rather than
    training curves or dense-reward components.
    """

    termination_hist = aggregate_stats.get("termination_reasons", {})
    admissibility_summary = validation_result.get("admissibility_summary", {})
    contract_violations = validation_result.get("contract_violations", [])
    diagnosis = "Heuristic appears structurally sound."
    guidance = (
        "Keep the heuristic as a lower bound and strengthen it only with clearly "
        "unavoidable costs."
    )
    if contract_violations:
        diagnosis = "The heuristic violated the runtime contract."
        guidance = (
            "Use only the documented ctx fields, return a finite non-negative float, "
            "and avoid unsupported constructs."
        )
    elif not bool(validation_result.get("goal_zero_pass", True)):
        diagnosis = "The heuristic did not return zero on solved states."
        guidance = "Add an explicit solved-state check that returns 0.0."
    elif not bool(validation_result.get("consistency_pass", True)):
        diagnosis = "The heuristic appears to overestimate or violate consistency."
        guidance = (
            "Remove speculative penalties and prefer distances or mandatory-step "
            "lower bounds that remain valid on every completion of the task."
        )
    elif float(aggregate_stats.get("solve_rate", 0.0)) <= 0.0:
        diagnosis = "The heuristic failed to solve any sampled seeds."
        guidance = (
            "Start from a simpler guaranteed lower bound tied directly to the "
            "goal object or mandatory subgoal."
        )
    elif float(aggregate_stats.get("average_solved_seed_efficiency", 0.0)) < 0.1:
        diagnosis = "The heuristic solves some seeds but remains too weak."
        guidance = (
            "Add stronger lower-bound structure such as mandatory pickup, transport, "
            "or obstacle-aware travel distances while preserving admissibility."
        )
    return "\n".join(
        [
            f"Task summary: {env_summary}",
            (
                "Search outcome summary: "
                f"solved {aggregate_stats.get('solved_count', 0)}/"
                f"{aggregate_stats.get('num_seeds', 0)} seeds; "
                f"average expanded states {aggregate_stats.get('average_expanded_states', 0):.2f}; "
                f"average generated states {aggregate_stats.get('average_generated_states', 0):.2f}; "
                f"average solution length {aggregate_stats.get('average_solution_length', 0):.2f}; "
                f"average solved-seed efficiency {_format_float(aggregate_stats.get('average_solved_seed_efficiency', 0.0))}; "
                f"job score {_format_float(aggregate_stats.get('job_score', 0.0))}."
            ),
            (
                "Admissibility summary: "
                f"goal_zero_pass={validation_result.get('goal_zero_pass')}; "
                f"nonnegative_pass={validation_result.get('nonnegative_pass')}; "
                f"consistency_pass={validation_result.get('consistency_pass')}; "
                f"admissibility_pass_rate={_format_float(admissibility_summary.get('admissibility_pass_rate', 0.0))}; "
                f"max_path_overestimate={_format_float(admissibility_summary.get('max_path_overestimate', 0.0))}."
            ),
            (
                "Failure diagnosis: "
                f"{diagnosis} Termination histogram: {dict(termination_hist)}. "
                f"Contract violations: {list(contract_violations)}."
            ),
            f"Revision guidance: {guidance}",
            "",
            "Heuristic code:",
            heuristic_code.strip(),
        ]
    ).strip()


__all__ = ["build_heuristic_feedback"]
