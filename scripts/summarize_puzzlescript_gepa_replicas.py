#!/usr/bin/env python3
"""Summarize independent PuzzleScript GEPA replica-array runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _replica_sort_key(root: Path) -> tuple[int, str]:
    match = re.fullmatch(r"replica-(\d+)", root.name)
    if match:
        return int(match.group(1)), root.name
    return 10**9, root.name


def _iter_state_roots(state_root: Path) -> list[Path]:
    roots: list[Path] = []
    if (state_root / "curriculum_state.json").exists():
        roots.append(state_root)
    replica_root = state_root / "gepa_replicas"
    if replica_root.exists():
        roots.extend(
            sorted(
                (path for path in replica_root.glob("replica-*") if (path / "curriculum_state.json").exists()),
                key=_replica_sort_key,
            )
        )
    return roots


def _best_observed_iteration(state: Mapping[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_key: tuple[float, float, int, int] | None = None
    phase_records = state.get("phase_records")
    if not isinstance(phase_records, Mapping):
        return best

    for phase_key, phase_record in phase_records.items():
        if not isinstance(phase_record, Mapping):
            continue
        phase = _as_int(phase_key) or 0
        iteration_results = phase_record.get("iteration_results")
        if not isinstance(iteration_results, list):
            continue
        for row in iteration_results:
            if not isinstance(row, Mapping):
                continue
            if row.get("skipped_final_eval"):
                continue
            solve_rate = _as_float(row.get("solve_rate"))
            mean_score = _as_float(row.get("mean_score"))
            iteration = _as_int(row.get("iteration")) or 0
            if solve_rate is None and mean_score is None:
                continue
            key = (solve_rate if solve_rate is not None else -1.0,
                   mean_score if mean_score is not None else -1.0,
                   phase,
                   iteration)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "phase": phase,
                    "iteration": iteration,
                    "mean_score": mean_score,
                    "solve_rate": solve_rate,
                    "n_solved": _as_int(row.get("n_solved")),
                    "selection_reason": row.get("selection_reason"),
                    "source": "iteration_results",
                }
    return best


def _max_observed_solve_rate(state: Mapping[str, Any]) -> Optional[float]:
    values: list[float] = []
    phase_records = state.get("phase_records")
    if not isinstance(phase_records, Mapping):
        return None
    for phase_record in phase_records.values():
        if not isinstance(phase_record, Mapping):
            continue
        max_observed = _as_float(phase_record.get("max_observed_solve_rate"))
        if max_observed is not None:
            values.append(max_observed)
        iteration_results = phase_record.get("iteration_results")
        if isinstance(iteration_results, list):
            for row in iteration_results:
                if isinstance(row, Mapping):
                    solve_rate = _as_float(row.get("solve_rate"))
                    if solve_rate is not None:
                        values.append(solve_rate)
    return max(values) if values else None


def _best_selection(state: Mapping[str, Any]) -> dict[str, Any]:
    selection = state.get("best_prompt_selection")
    if isinstance(selection, Mapping):
        return {
            "phase": _as_int(selection.get("phase")),
            "iteration": _as_int(selection.get("iteration")),
            "mean_score": _as_float(selection.get("mean_score")),
            "solve_rate": _as_float(selection.get("solve_rate")),
            "selection_reason": selection.get("reason"),
            "source": "best_prompt_selection",
        }
    return _best_observed_iteration(state)


def _cost_summary(state_root: Path, state: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = state.get("llm_cost_summary")
    if isinstance(summary, Mapping):
        return summary
    summary_path = state_root / "llm_cost_summary.json"
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, Mapping):
            return loaded
    return {}


def _replica_id(root: Path) -> str:
    if root.name.startswith("replica-"):
        return root.name.removeprefix("replica-")
    return root.name


def summarize_replica(state_root: Path) -> dict[str, Any]:
    state_path = state_root / "curriculum_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, Mapping):
        raise ValueError(f"{state_path} did not contain a JSON object")

    selection = _best_selection(state)
    cost = _cost_summary(state_root, state)
    current_phase = _as_int(state.get("current_phase"))
    total_phases = _as_int(state.get("total_phases"))
    global_iteration = _as_int(state.get("global_iteration")) or 0
    stop_reason = state.get("stop_reason")
    if stop_reason:
        status = str(stop_reason)
    elif current_phase is not None and total_phases is not None and current_phase > total_phases:
        status = "all_phases_completed"
    elif global_iteration > 0:
        status = "running"
    else:
        status = "starting"

    prompt_path = state_root / "best_prompt.txt"
    return {
        "replica": _replica_id(state_root),
        "state_root": str(state_root),
        "status": status,
        "current_phase": current_phase,
        "total_phases": total_phases,
        "global_iteration": global_iteration,
        "completed_phases": state.get("completed_phases", []),
        "best_phase": selection.get("phase"),
        "best_iteration": selection.get("iteration"),
        "best_mean_score": selection.get("mean_score"),
        "best_solve_rate": selection.get("solve_rate"),
        "max_observed_solve_rate": _max_observed_solve_rate(state),
        "selection_source": selection.get("source"),
        "selection_reason": selection.get("selection_reason"),
        "total_cost_usd": _as_float(cost.get("total_cost_usd")),
        "total_calls": _as_int(cost.get("total_calls")),
        "best_prompt_path": str(prompt_path) if prompt_path.exists() else None,
    }


def _rank_key(row: Mapping[str, Any]) -> tuple[float, float, int, int]:
    return (
        _as_float(row.get("best_solve_rate")) or -1.0,
        _as_float(row.get("best_mean_score")) or -1.0,
        _as_int(row.get("best_phase")) or -1,
        _as_int(row.get("best_iteration")) or -1,
    )


def _format_float(value: Any, digits: int) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}f}"


def _format_int(value: Any) -> str:
    parsed = _as_int(value)
    return "-" if parsed is None else str(parsed)


def _shorten(value: Any, width: int) -> str:
    text = str(value) if value is not None else "-"
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "~"


def print_table(rows: list[Mapping[str, Any]]) -> None:
    table_rows: list[list[str]] = []
    for rank, row in enumerate(rows, start=1):
        phase = "-"
        if row.get("best_phase") is not None:
            phase = f"{row.get('best_phase')}/{row.get('best_iteration') or '-'}"
        table_rows.append([
            str(rank),
            str(row.get("replica")),
            _shorten(row.get("status"), 24),
            _format_int(row.get("global_iteration")),
            phase,
            _format_float(row.get("best_solve_rate"), 3),
            _format_float(row.get("best_mean_score"), 4),
            _format_float(row.get("max_observed_solve_rate"), 3),
            _format_float(row.get("total_cost_usd"), 4),
            "yes" if row.get("best_prompt_path") else "no",
        ])
    headers = [
        "rank",
        "replica",
        "status",
        "iters",
        "best",
        "solve",
        "score",
        "maxsolve",
        "cost",
        "prompt",
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize PuzzleScript GEPA replica-array outputs.",
    )
    parser.add_argument("state_root", type=Path, help="Group STATE_ROOT containing gepa_replicas/.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults to STATE_ROOT/gepa_replica_summary.json.",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy the best replica prompt to STATE_ROOT/best_replica_prompt.txt.",
    )
    args = parser.parse_args()

    state_root = args.state_root.expanduser().resolve()
    replica_roots = _iter_state_roots(state_root)
    if not replica_roots:
        raise SystemExit(f"No curriculum_state.json files found under {state_root}")

    rows = [summarize_replica(root) for root in replica_roots]
    rows.sort(key=_rank_key, reverse=True)

    output_path = args.output or (state_root / "gepa_replica_summary.json")
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_root": str(state_root),
        "n_replicas": len(rows),
        "best_replica": rows[0],
        "replicas": rows,
    }

    if args.promote_best and rows[0].get("best_prompt_path"):
        best_prompt_path = Path(str(rows[0]["best_prompt_path"]))
        target_path = state_root / "best_replica_prompt.txt"
        target_path.write_text(best_prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
        summary["promoted_best_prompt_path"] = str(target_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_table(rows)
    print(f"\nWrote summary: {output_path}")
    if summary.get("promoted_best_prompt_path"):
        print(f"Promoted best prompt: {summary['promoted_best_prompt_path']}")


if __name__ == "__main__":
    main()
