#!/usr/bin/env python3
"""Create publication figures from a completed PuzzleScript GEPA run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def _bootstrap_weights(n_games: int, samples: int, seed: int = 0) -> Any:
    import numpy as np

    return np.random.default_rng(seed).multinomial(
        n_games,
        [1.0 / n_games] * n_games,
        size=max(1, samples),
    )


def write_paper_plots(
    state_root: Path,
    *,
    output_dir: Path | None = None,
    bootstrap_samples: int = 2_000,
) -> list[Path]:
    """Write four paired, game-aware figures from saved run artifacts.

    The function reads existing results only; it never reruns synthesis or
    search. PNG files support quick inspection and matching PDFs preserve
    vectors for paper assembly.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    state_root = state_root.expanduser().resolve()
    holdout_root = (
        state_root / "holdout_compare"
        if (state_root / "holdout_compare").is_dir()
        else state_root
    )
    result_root = state_root if (state_root / "gepa_result.json").exists() else state_root.parent
    per_level = _read_json(holdout_root / "per_level_comparison.json")
    per_game = _read_json(holdout_root / "per_game_comparison.json")
    gepa_result = _read_json(result_root / "gepa_result.json")
    if not per_level or not per_game:
        raise ValueError("Paper plots require non-empty per-level and per-game comparisons.")

    output_dir = output_dir or state_root / "paper_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "font.size": 10,
            "figure.dpi": 150,
        }
    )
    base_color = "#4C78A8"
    optimized_color = "#F58518"
    neutral_color = "#777777"
    paths: list[Path] = []

    games = sorted({str(row["game"]) for row in per_level})
    game_index = {game: index for index, game in enumerate(games)}
    n_by_game = np.zeros(len(games), dtype=int)
    max_expanded = max(
        1,
        max(
            max(int(row.get("base_expanded", 0)), int(row.get("optimized_expanded", 0)))
            for row in per_level
        ),
    )
    thresholds = np.unique(np.geomspace(1, max_expanded, 100).astype(int))
    base_counts = np.zeros((len(games), len(thresholds)), dtype=int)
    optimized_counts = np.zeros_like(base_counts)
    for row in per_level:
        index = game_index[str(row["game"])]
        n_by_game[index] += 1
        if bool(row.get("base_solved", False)):
            base_counts[index] += thresholds >= int(row.get("base_expanded", 0))
        if bool(row.get("optimized_solved", False)):
            optimized_counts[index] += thresholds >= int(row.get("optimized_expanded", 0))
    weights = _bootstrap_weights(len(games), bootstrap_samples)
    denominators = weights @ n_by_game
    base_bootstrap = (weights @ base_counts) / denominators[:, None]
    optimized_bootstrap = (weights @ optimized_counts) / denominators[:, None]
    base_profile = base_counts.sum(axis=0) / n_by_game.sum()
    optimized_profile = optimized_counts.sum(axis=0) / n_by_game.sum()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(
        thresholds,
        *np.percentile(base_bootstrap, [2.5, 97.5], axis=0),
        color=base_color,
        alpha=0.16,
        linewidth=0,
    )
    ax.fill_between(
        thresholds,
        *np.percentile(optimized_bootstrap, [2.5, 97.5], axis=0),
        color=optimized_color,
        alpha=0.16,
        linewidth=0,
    )
    ax.step(thresholds, base_profile, where="post", color=base_color, label="Human prompt")
    ax.step(
        thresholds,
        optimized_profile,
        where="post",
        color=optimized_color,
        label="GEPA prompt",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, max(0.6, float(optimized_profile[-1]) + 0.08))
    ax.set_xlabel("A* expansion budget")
    ax.set_ylabel("Fraction of holdout levels solved")
    ax.set_title("Search-budget performance profile")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.annotate(
        f"{int(base_counts[:, -1].sum())}/{int(n_by_game.sum())}",
        (thresholds[-1], base_profile[-1]),
        xytext=(-42, -14),
        textcoords="offset points",
        color=base_color,
    )
    ax.annotate(
        f"{int(optimized_counts[:, -1].sum())}/{int(n_by_game.sum())}",
        (thresholds[-1], optimized_profile[-1]),
        xytext=(-42, 8),
        textcoords="offset points",
        color=optimized_color,
    )
    paths.extend(_save_figure(fig, output_dir, "figure1_search_budget_profile"))
    plt.close(fig)

    game_rows = sorted(
        per_game,
        key=lambda row: (
            int(row.get("optimized_solved", 0)) - int(row.get("base_solved", 0))
        )
        / max(1, int(row.get("n", 1))),
    )
    game_deltas = np.array(
        [
            100.0
            * (int(row.get("optimized_solved", 0)) - int(row.get("base_solved", 0)))
            / max(1, int(row.get("n", 1)))
            for row in game_rows
        ]
    )
    y_positions = np.arange(len(game_rows))
    colors = [
        optimized_color if delta > 0 else base_color if delta < 0 else neutral_color
        for delta in game_deltas
    ]
    sizes = [24 + 3 * int(row.get("n", 1)) for row in game_rows]
    overall_delta = 100.0 * (optimized_counts[:, -1].sum() - base_counts[:, -1].sum()) / n_by_game.sum()
    solve_bootstrap = (
        100.0
        * ((weights @ optimized_counts[:, -1]) - (weights @ base_counts[:, -1]))
        / denominators
    )
    low, high = np.percentile(solve_bootstrap, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.3 * len(game_rows) + 1.5)))
    ax.hlines(y_positions, 0, game_deltas, color=colors, linewidth=1.6)
    ax.scatter(game_deltas, y_positions, color=colors, s=sizes, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row["game"]).replace("_", " ") for row in game_rows], fontsize=8)
    ax.set_xlabel("Solve-rate change (percentage points)")
    ax.set_title(
        f"Generalization by holdout game\nOverall {overall_delta:+.1f} pp "
        f"(game bootstrap 95% CI {low:+.1f} to {high:+.1f})"
    )
    ax.grid(axis="x", alpha=0.2)
    paths.extend(_save_figure(fig, output_dir, "figure2_game_generalization"))
    plt.close(fig)

    both = sum(bool(row["base_solved"]) and bool(row["optimized_solved"]) for row in per_level)
    optimized_only = sum(
        not bool(row["base_solved"]) and bool(row["optimized_solved"]) for row in per_level
    )
    base_only = sum(
        bool(row["base_solved"]) and not bool(row["optimized_solved"]) for row in per_level
    )
    neither = len(per_level) - both - optimized_only - base_only
    transition = np.array([[neither, optimized_only], [base_only, both]])
    common = [
        row for row in per_level if bool(row["base_solved"]) and bool(row["optimized_solved"])
    ]
    log_ratios = np.log2(
        [
            (int(row.get("optimized_expanded", 0)) + 1)
            / (int(row.get("base_expanded", 0)) + 1)
            for row in common
        ]
    )
    aggregate_reduction = 100.0 * (
        1.0
        - sum(int(row.get("optimized_expanded", 0)) for row in common)
        / max(1, sum(int(row.get("base_expanded", 0)) for row in common))
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    left.imshow(transition, cmap="Blues", alpha=0.8)
    for row_index in range(2):
        for column_index in range(2):
            left.text(
                column_index,
                row_index,
                str(transition[row_index, column_index]),
                ha="center",
                va="center",
                fontsize=16,
                color=(
                    "white"
                    if transition[row_index, column_index] > transition.max() / 2
                    else "black"
                ),
            )
    left.set_xticks([0, 1], ["Not solved", "Solved"])
    left.set_yticks([0, 1], ["Not solved", "Solved"])
    left.set_xlabel("GEPA prompt")
    left.set_ylabel("Human prompt")
    left.set_title("Paired solve outcomes")
    if len(log_ratios):
        right.hist(log_ratios, bins=min(18, max(5, len(log_ratios) // 6)), color=optimized_color, alpha=0.82)
        right.axvline(0, color="#333333", linewidth=1.0)
        right.axvline(float(np.median(log_ratios)), color=base_color, linestyle="--", linewidth=1.2)
    right.set_xlabel("log₂(optimized expansions / baseline expansions)")
    right.set_ylabel("Commonly solved levels")
    right.set_title(
        f"Paired search efficiency\n{aggregate_reduction:.1f}% fewer expansions in aggregate"
    )
    right.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    paths.extend(_save_figure(fig, output_dir, "figure3_paired_outcomes_efficiency"))
    plt.close(fig)

    scores = np.asarray(gepa_result.get("val_aggregate_scores", []), dtype=float)
    calls = np.asarray(gepa_result.get("discovery_eval_counts", []), dtype=float)
    count = min(len(scores), len(calls))
    scores = scores[:count]
    calls = calls[:count]
    if not count:
        raise ValueError("GEPA result does not contain candidate validation scores.")
    running_best = np.maximum.accumulate(scores)
    best_index = int(np.argmax(scores))
    total_calls = int(gepa_result.get("total_metric_calls", calls[-1]))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(calls, scores, color=neutral_color, s=34, label="Candidate dev score", zorder=3)
    ax.step(calls, running_best, where="post", color=optimized_color, linewidth=2.0, label="Running best")
    ax.scatter(
        [calls[best_index]],
        [scores[best_index]],
        color=optimized_color,
        edgecolor="#333333",
        s=80,
        zorder=4,
    )
    ax.axhline(0, color=base_color, linewidth=1.0, linestyle="--", label="Human prompt")
    ax.axvline(total_calls, color="#999999", linewidth=0.8, linestyle=":")
    ax.annotate(
        f"selected: {scores[best_index]:.3f}",
        (calls[best_index], scores[best_index]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.set_xlabel("GEPA metric calls")
    ax.set_ylabel("Full development-set aggregate score")
    ax.set_title("Prompt optimization trajectory")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    paths.extend(_save_figure(fig, output_dir, "figure4_gepa_optimization"))
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse the completed run root and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    """Generate and print all publication-figure paths."""
    args = parse_args()
    for path in write_paper_plots(
        args.state_root,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    ):
        print(path)


if __name__ == "__main__":
    main()
