#!/usr/bin/env python3
"""Build PuzzleScript per-game expansion-efficiency tables and summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


CATEGORY: dict[str, str] = {
    # Train
    "sokoban_basic": "sokoban",
    "Broken_Leg_Sokoban": "sokoban",
    "Collapsable_Sokoban": "mostly_sokoban",
    "Pulling_Box_Sokoban": "mostly_sokoban",
    "Swap_Sokoban": "mostly_sokoban",
    "Sokoban_Flipped": "mostly_sokoban",
    "Algorithm-Generated_Sokoban_Levels": "mostly_sokoban",
    "Tractor_Beam_Sokoban9": "mostly_sokoban",
    "Muddy_Sokoban_Level_Set_I": "mostly_sokoban",
    "Ultimate_Sokoban_Supreme": "mostly_sokoban",
    "Beams_and_Flowers": "partly_sokoban",
    "Crate_Rotate": "partly_sokoban",
    "Drop_Swap": "partly_sokoban",
    "IceCrates": "partly_sokoban",
    "Laser": "not_sokoban",
    "Merge_and_Swap": "partly_sokoban",
    "Power_Block": "not_sokoban",
    "SwapBot": "not_sokoban",
    "___Hey_you!_Stop_blocking_the_laser!": "not_sokoban",
    # Holdout
    "sokoban_sanity": "sokoban",
    "No_Right_Turn_Sokoban": "sokoban",
    "Cold_Feet_Sokoban": "mostly_sokoban",
    "Soko-bine": "mostly_sokoban",
    "Remote_Control_Sokoban": "partly_sokoban",
    "Darkness_Sokoban": "mostly_sokoban",
    "1D_Sokoban": "sokoban",
    "Aperture_Science_Sokoban_Testing_Initiative": "partly_sokoban",
    "Beam_Islands": "partly_sokoban",
    "Crates_and_Portals": "partly_sokoban",
    "Gravity_Sokoban": "mostly_sokoban",
    "Ice_Cubes": "partly_sokoban",
    "Inswaption": "partly_sokoban",
    "Memory_Push": "partly_sokoban",
    "Not_Normal_Crates": "partly_sokoban",
    "PrograMaze": "not_sokoban",
    "All_These_Damn_Crates": "mostly_sokoban",
    "Boxes_&_Balloons": "not_sokoban",
    "Lawn-Mowing_Robot": "not_sokoban",
    "Robot_Repairs_1.2": "not_sokoban",
    "where_did_all_this_ice_come_from_": "partly_sokoban",
    "ZigZag_Ice": "partly_sokoban",
}


TRAIN_PHASES: list[tuple[int, int]] = [(1, 5), (2, 10), (3, 15), (4, 19)]


def _phase_for_index(index: int) -> int:
    for phase, count in TRAIN_PHASES:
        if index < count:
            return phase
    return 4


def _load_holdout_by_game(stats_path: Path) -> dict[str, dict[str, Any]]:
    if not stats_path.exists():
        return {}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return stats.get("summary", {}).get("by_game", {})


def _safe(bucket: dict[str, Any], key: str) -> float | None:
    if key not in bucket:
        return None
    return float(bucket[key])


def _row_for_game(
    *,
    game: str,
    split: str,
    phase: int | None,
    holdout_by_game: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policies = holdout_by_game.get(game, {})
    best = policies.get("best_prompt", {})
    base = policies.get("base_prompt", {})
    builtin = policies.get("builtin", {})
    blind = policies.get("blind", {})

    best_score = _safe(best, "mean_score")
    base_score = _safe(base, "mean_score")
    best_expanded = _safe(best, "mean_expanded")
    base_expanded = _safe(base, "mean_expanded")

    if best_score is not None and base_score is not None:
        score_gain = best_score - base_score
        score_gain_pct = (score_gain / base_score) if base_score > 0 else None
    else:
        score_gain = None
        score_gain_pct = None

    if best_expanded is not None and base_expanded is not None and base_expanded > 0:
        expansion_reduction = base_expanded - best_expanded
        expansion_reduction_pct = expansion_reduction / base_expanded
        expansion_pct_diff = ((best_expanded - base_expanded) / base_expanded) * 100.0
        efficiency_gain_pct = expansion_reduction_pct * 100.0
    else:
        expansion_reduction = None
        expansion_reduction_pct = None
        expansion_pct_diff = None
        efficiency_gain_pct = None

    return {
        "game": game,
        "split": split,
        "phase": phase,
        "sokoban_category": CATEGORY.get(game, "unknown"),
        "best_solve_rate": _safe(best, "solve_rate"),
        "base_solve_rate": _safe(base, "solve_rate"),
        "builtin_solve_rate": _safe(builtin, "solve_rate"),
        "blind_solve_rate": _safe(blind, "solve_rate"),
        "best_mean_score": best_score,
        "base_mean_score": base_score,
        "builtin_mean_score": _safe(builtin, "mean_score"),
        "blind_mean_score": _safe(blind, "mean_score"),
        "score_gain_best_minus_base": score_gain,
        "score_gain_pct_vs_base": score_gain_pct,
        "best_mean_expanded": best_expanded,
        "base_mean_expanded": base_expanded,
        "expansion_reduction_best_minus_base": expansion_reduction,
        "expansion_reduction_pct_vs_base": expansion_reduction_pct,
        "expansion_pct_diff_best_vs_base": expansion_pct_diff,
        "efficiency_gain_pct_vs_base": efficiency_gain_pct,
        "n_levels": int(best.get("n") or base.get("n") or 0),
        "comparison_available": bool(best and base),
    }


def _format_val(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)

    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        comparable = [r for r in group_rows if r["comparison_available"]]
        def avg(field: str) -> float | None:
            vals = [r[field] for r in comparable if r.get(field) is not None]
            return mean(vals) if vals else None

        solved_best = sum((r.get("best_solve_rate") or 0) * (r.get("n_levels") or 0) for r in comparable)
        solved_base = sum((r.get("base_solve_rate") or 0) * (r.get("n_levels") or 0) for r in comparable)
        n_levels = sum(r.get("n_levels") or 0 for r in comparable)
        row = {k: v for k, v in zip(keys, group_key)}
        row.update(
            {
                "n_games": len(group_rows),
                "n_comparable_games": len(comparable),
                "n_comparable_levels": n_levels,
                "weighted_best_solve_rate": solved_best / n_levels if n_levels else None,
                "weighted_base_solve_rate": solved_base / n_levels if n_levels else None,
                "mean_best_score": avg("best_mean_score"),
                "mean_base_score": avg("base_mean_score"),
                "mean_score_gain": avg("score_gain_best_minus_base"),
                "mean_score_gain_pct_vs_base": avg("score_gain_pct_vs_base"),
                "mean_expansion_reduction_pct_vs_base": avg("expansion_reduction_pct_vs_base"),
                "mean_expansion_pct_diff_best_vs_base": avg("expansion_pct_diff_best_vs_base"),
                "mean_efficiency_gain_pct_vs_base": avg("efficiency_gain_pct_vs_base"),
            }
        )
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_val(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--env-grid", type=Path, default=Path("configs/gepa_puzzlescript_envs.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.state_root / "efficiency_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.env_grid.read_text(encoding="utf-8"))
    train_jobs = cfg["jobs"]
    eval_jobs = cfg["eval_jobs"]
    matched_holdout_stats = (
        args.state_root
        / "best_vs_base_prompt_eval_holdout_matched_cap"
        / "best_prompt_eval_stats.json"
    )
    holdout_stats = (
        matched_holdout_stats
        if matched_holdout_stats.exists()
        else args.state_root / "best_vs_base_prompt_eval_holdout" / "best_prompt_eval_stats.json"
    )
    holdout_by_game = _load_holdout_by_game(holdout_stats)

    rows: list[dict[str, Any]] = []
    for i, job in enumerate(train_jobs):
        rows.append(
            _row_for_game(
                game=job["name"],
                split="train",
                phase=_phase_for_index(i),
                holdout_by_game={},
            )
        )
    for job in eval_jobs:
        rows.append(
            _row_for_game(
                game=job["name"],
                split="holdout",
                phase=None,
                holdout_by_game=holdout_by_game,
            )
        )

    by_category = _summarize(rows, ["sokoban_category"])
    by_split = _summarize(rows, ["split"])
    by_split_category = _summarize(rows, ["split", "sokoban_category"])
    by_phase_category = _summarize([r for r in rows if r["split"] == "train"], ["phase", "sokoban_category"])

    _write_csv(output_dir / "game_efficiency_table.csv", rows)
    _write_csv(output_dir / "summary_by_category.csv", by_category)
    _write_csv(output_dir / "summary_by_split.csv", by_split)
    _write_csv(output_dir / "summary_by_split_category.csv", by_split_category)
    _write_csv(output_dir / "summary_train_by_phase_category.csv", by_phase_category)

    report = [
        "# PuzzleScript Node-Expansion Efficiency Audit",
        "",
        "Search efficiency is reported as pure percentage difference in mean node expansions per search.",
        "",
        "`efficiency_gain_pct_vs_base = 100 * (base_mean_expanded - best_mean_expanded) / base_mean_expanded`.",
        "",
        "Positive values mean the optimized prompt used fewer node expansions than the base prompt. Negative values mean it used more expansions.",
        "",
        "Important: exact best-vs-base comparisons are currently available for the holdout split only. Train rows are categorized by phase and Sokoban category, but train gain fields are blank until a train split base-prompt comparison is generated.",
        "",
        f"Holdout comparison source: `{holdout_stats}`.",
        "",
        "## Per-Game Table",
        "",
        _markdown_table(
            rows,
            [
                "game",
                "split",
                "phase",
                "sokoban_category",
                "best_mean_expanded",
                "base_mean_expanded",
                "efficiency_gain_pct_vs_base",
                "expansion_pct_diff_best_vs_base",
                "best_solve_rate",
                "base_solve_rate",
                "comparison_available",
            ],
        ),
        "",
        "## Summary By Sokoban Category",
        "",
        _markdown_table(by_category, list(by_category[0].keys()) if by_category else []),
        "",
        "## Summary By Split",
        "",
        _markdown_table(by_split, list(by_split[0].keys()) if by_split else []),
        "",
        "## Summary By Split x Sokoban Category",
        "",
        _markdown_table(by_split_category, list(by_split_category[0].keys()) if by_split_category else []),
        "",
        "## Train Summary By Phase x Sokoban Category",
        "",
        _markdown_table(by_phase_category, list(by_phase_category[0].keys()) if by_phase_category else []),
        "",
    ]
    (output_dir / "search_efficiency_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output_dir / 'search_efficiency_audit.md'}")
    print(f"Wrote {output_dir / 'game_efficiency_table.csv'}")


if __name__ == "__main__":
    main()
