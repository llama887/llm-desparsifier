#!/usr/bin/env python3
"""Summarize batched PuzzleScript GEPA candidate-evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_puzzlescript_batched_gepa import candidate_score  # noqa: E402

DEFAULT_REPORT_EFFICIENCY_WEIGHT = 2.0
DEFAULT_REPORT_EFFICIENCY_CLIP = 1.0
DEFAULT_HIGH_HEADROOM_EXPANSIONS = 500.0


def _optional_float(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _common_solve_log2_delta(row: Mapping[str, Any]) -> Optional[float]:
    if not bool(row.get("solved")) or not bool(row.get("baseline_solved")):
        return None
    expanded = _optional_float(row, "expanded")
    baseline_expanded = _optional_float(row, "baseline_expanded")
    if expanded is None or baseline_expanded is None:
        return None
    if expanded < 0.0 or baseline_expanded < 0.0:
        return None
    return math.log2((baseline_expanded + 1.0) / (expanded + 1.0))


def _is_candidate_error(row: Mapping[str, Any]) -> bool:
    if row.get("synthesis_error") is not None:
        return True
    error = row.get("error")
    if error is None:
        return False
    text = str(error).lower()
    return "game compilation failed" not in text and "compiling game" not in text


def summarize_scored_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    common_solve_efficiency_weight: float = DEFAULT_REPORT_EFFICIENCY_WEIGHT,
    common_solve_efficiency_clip: float = DEFAULT_REPORT_EFFICIENCY_CLIP,
    high_headroom_expansions: float = DEFAULT_HIGH_HEADROOM_EXPANSIONS,
) -> dict[str, Any]:
    """Return solve-rate and efficiency metrics for one scored candidate.

    The summary uses the current GEPA scoring helper for ``current_metric_score``
    so artifact monitoring stays aligned with the optimization objective, while
    also exposing raw counts and paired-expansion diagnostics for interpretation.
    """
    materialized = [dict(row) for row in rows]
    common_rows = [
        row for row in materialized
        if _common_solve_log2_delta(row) is not None
    ]
    common_deltas = [
        cast_delta
        for row in common_rows
        if (cast_delta := _common_solve_log2_delta(row)) is not None
    ]
    high_rows = [
        row for row in common_rows
        if (_optional_float(row, "baseline_expanded") or 0.0) >= high_headroom_expansions
    ]
    high_deltas = [
        cast_delta
        for row in high_rows
        if (cast_delta := _common_solve_log2_delta(row)) is not None
    ]
    solved = sum(bool(row.get("solved")) for row in materialized)
    baseline_solved = sum(bool(row.get("baseline_solved")) for row in materialized)
    new_solves = sum(
        bool(row.get("solved")) and not bool(row.get("baseline_solved"))
        for row in materialized
    )
    lost_solves = sum(
        bool(row.get("baseline_solved")) and not bool(row.get("solved"))
        for row in materialized
    )
    common_faster = sum(
        (_optional_float(row, "expanded") or 0.0)
        < (_optional_float(row, "baseline_expanded") or 0.0)
        for row in common_rows
    )
    common_slower = sum(
        (_optional_float(row, "expanded") or 0.0)
        > (_optional_float(row, "baseline_expanded") or 0.0)
        for row in common_rows
    )
    common_same = len(common_rows) - common_faster - common_slower
    return {
        "n": len(materialized),
        "solved": solved,
        "baseline_solved": baseline_solved,
        "net_solve": solved - baseline_solved,
        "new_solves": new_solves,
        "lost_solves": lost_solves,
        "candidate_errors": sum(_is_candidate_error(row) for row in materialized),
        "common_solves": len(common_rows),
        "common_faster": common_faster,
        "common_slower": common_slower,
        "common_same": common_same,
        "high_headroom_common_solves": len(high_rows),
        "mean_common_log2_base_over_candidate": (
            sum(common_deltas) / len(common_deltas) if common_deltas else 0.0
        ),
        "high_headroom_mean_log2_base_over_candidate": (
            sum(high_deltas) / len(high_deltas) if high_deltas else 0.0
        ),
        "sum_common_expanded": sum(_optional_float(row, "expanded") or 0.0 for row in common_rows),
        "sum_common_baseline_expanded": sum(
            _optional_float(row, "baseline_expanded") or 0.0 for row in common_rows
        ),
        "current_metric_score": candidate_score(
            materialized,
            common_solve_efficiency_weight=common_solve_efficiency_weight,
            common_solve_efficiency_clip=common_solve_efficiency_clip,
        ),
    }


def summarize_eval_dir(
    eval_dir: Path,
    *,
    common_solve_efficiency_weight: float = DEFAULT_REPORT_EFFICIENCY_WEIGHT,
    common_solve_efficiency_clip: float = DEFAULT_REPORT_EFFICIENCY_CLIP,
) -> dict[str, Any]:
    """Summarize one ``candidate_evals/eval-*`` directory."""
    scored_path = eval_dir / "scored_results.json"
    rows = json.loads(scored_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{scored_path} must contain a JSON list")
    summary = summarize_scored_results(
        [row for row in rows if isinstance(row, Mapping)],
        common_solve_efficiency_weight=common_solve_efficiency_weight,
        common_solve_efficiency_clip=common_solve_efficiency_clip,
    )
    summary.update(
        {
            "eval_dir": str(eval_dir),
            "eval_name": eval_dir.name,
            "scored_results_path": str(scored_path),
        }
    )
    return summary


def summarize_root(
    root: Path,
    *,
    common_solve_efficiency_weight: float = DEFAULT_REPORT_EFFICIENCY_WEIGHT,
    common_solve_efficiency_clip: float = DEFAULT_REPORT_EFFICIENCY_CLIP,
) -> dict[str, Any]:
    """Summarize all scored candidate evaluations under one GEPA state root."""
    eval_summaries = [
        summarize_eval_dir(
            path.parent,
            common_solve_efficiency_weight=common_solve_efficiency_weight,
            common_solve_efficiency_clip=common_solve_efficiency_clip,
        )
        for path in sorted(root.glob("candidate_evals/*/scored_results.json"))
    ]
    eval_summaries.sort(key=lambda row: float(row["current_metric_score"]), reverse=True)
    git_state_path = root / "run_git_state.json"
    git_state: dict[str, Any] | None = None
    if git_state_path.exists():
        loaded = json.loads(git_state_path.read_text(encoding="utf-8"))
        git_state = dict(loaded) if isinstance(loaded, Mapping) else None
    return {
        "root": str(root),
        "root_name": root.name,
        "git_state": git_state,
        "n_scored_evals": len(eval_summaries),
        "best_eval": eval_summaries[0] if eval_summaries else None,
        "evals": eval_summaries,
    }


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:+.{digits}f}"


def print_root_table(summaries: Sequence[Mapping[str, Any]], *, limit: int) -> None:
    """Print a compact table of best scored evals per root."""
    headers = ["root", "eval", "metric", "sol/base", "new/lost", "mean", "high", "f/same/sl"]
    table: list[list[str]] = []
    for root_summary in summaries:
        best = root_summary.get("best_eval")
        if not isinstance(best, Mapping):
            table.append([str(root_summary.get("root_name")), "-", "-", "-", "-", "-", "-", "-"])
            continue
        table.append(
            [
                str(root_summary.get("root_name")),
                str(best.get("eval_name")),
                _format_float(best.get("current_metric_score")),
                f"{best.get('solved')}/{best.get('baseline_solved')}",
                f"{best.get('new_solves')}/{best.get('lost_solves')}",
                _format_float(best.get("mean_common_log2_base_over_candidate")),
                _format_float(best.get("high_headroom_mean_log2_base_over_candidate")),
                f"{best.get('common_faster')}/{best.get('common_same')}/{best.get('common_slower')}",
            ]
        )
    table = table[:limit]
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize batched PuzzleScript GEPA candidate_evals artifacts.",
    )
    parser.add_argument("roots", nargs="+", type=Path, help="GEPA state roots or glob patterns.")
    parser.add_argument("--output", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to print.")
    parser.add_argument(
        "--common-solve-efficiency-weight",
        type=float,
        default=DEFAULT_REPORT_EFFICIENCY_WEIGHT,
        help="Metric weight used for current_metric_score reporting.",
    )
    parser.add_argument(
        "--common-solve-efficiency-clip",
        type=float,
        default=DEFAULT_REPORT_EFFICIENCY_CLIP,
        help="Metric clip used for current_metric_score reporting.",
    )
    args = parser.parse_args()

    roots: list[Path] = []
    for root_arg in args.roots:
        matches = sorted(root_arg.parent.glob(root_arg.name)) if any(ch in str(root_arg) for ch in "*?[") else []
        roots.extend(matches or [root_arg])
    summaries = [
        summarize_root(
            root.expanduser().resolve(),
            common_solve_efficiency_weight=args.common_solve_efficiency_weight,
            common_solve_efficiency_clip=args.common_solve_efficiency_clip,
        )
        for root in roots
    ]
    summaries.sort(
        key=lambda summary: (
            float(cast_best["current_metric_score"])
            if isinstance((cast_best := summary.get("best_eval")), Mapping)
            else float("-inf")
        ),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_roots": len(summaries),
        "roots": summaries,
    }
    print_root_table(summaries, limit=max(1, args.limit))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote summary: {args.output}")


if __name__ == "__main__":
    main()
