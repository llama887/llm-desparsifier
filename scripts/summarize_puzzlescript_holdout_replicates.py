#!/usr/bin/env python3
"""Aggregate paired holdout replicates and write confidence-interval plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_BUDGETS = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_ci(
    values: Sequence[float], *, samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": None, "ci95": [None, None], "values": []}
    if len(array) == 1:
        low = high = float(array[0])
    else:
        indices = rng.integers(0, len(array), size=(samples, len(array)))
        boot = array[indices].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(array.mean()),
        "ci95": [float(low), float(high)],
        "values": [float(value) for value in array],
    }


def _save_figure(fig: Any, root: Path, stem: str) -> None:
    fig.savefig(root / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")


def _errorbar(ax: Any, x: int, stats: dict[str, Any], *, color: str, label: str) -> None:
    mean = float(stats["mean"])
    low, high = (float(value) for value in stats["ci95"])
    ax.errorbar(
        [x],
        [mean],
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        color=color,
        capsize=5,
        markersize=7,
        label=label,
    )


def _plot_metrics(root: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    colors = {"base": "#4C78A8", "optimized": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for x, label in enumerate(("base", "optimized")):
        _errorbar(
            axes[0], x, summary["solve_rate"][label], color=colors[label], label=label.title()
        )
        _errorbar(
            axes[1], x, summary["expanded_mean"][label], color=colors[label], label=label.title()
        )
    axes[0].set_ylabel("Holdout solve rate")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Mean A* expansions (all levels)")
    for ax in axes:
        ax.set_xticks([0, 1], ["Base", "Optimized"])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Holdout performance across independent synthesis replicates")
    fig.tight_layout()
    _save_figure(fig, root, "holdout_replicate_metrics")
    plt.close(fig)


def _plot_solve_rate(root: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    base = summary["solve_rate"]["base"]
    optimized = summary["solve_rate"]["optimized"]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for base_value, optimized_value in zip(base["values"], optimized["values"]):
        ax.plot(
            [0, 1], [100 * base_value, 100 * optimized_value],
            color="#999999", alpha=0.35, linewidth=0.8, marker="o", markersize=3,
        )
    for x, label, stats, color in (
        (0, "Human prompt", base, "#4C78A8"),
        (1, "GEPA prompt", optimized, "#E45756"),
    ):
        mean = 100 * float(stats["mean"])
        low, high = 100 * np.asarray(stats["ci95"])
        ax.errorbar(
            [x], [mean], yerr=[[mean - low], [high - mean]], fmt="D",
            color=color, capsize=6, markersize=8, label="Mean and 95% CI",
        )
    ax.set_xticks([0, 1], ["Human prompt", "GEPA prompt"])
    ax.set_ylabel("Holdout solve rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Holdout solve rate across synthesis replicates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, root, "holdout_solve_rate")
    plt.close(fig)


def _plot_efficiency_difference(root: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    stats = summary["common_solved_efficiency"]["paired_reduction"]
    values = 100 * np.asarray(stats["values"])
    mean = 100 * float(stats["mean"])
    low, high = 100 * np.asarray(stats["ci95"])
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(np.zeros(len(values)), values, color="#7A5195", alpha=0.65, s=34)
    ax.errorbar(
        [0], [mean], yerr=[[mean - low], [high - mean]], fmt="D",
        color="#E45756", capsize=6, markersize=8,
    )
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Fewer A* expansions with GEPA prompt (%)")
    ax.set_title("Search efficiency on commonly solved levels\nMean and 95% CI across synthesis replicates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, root, "holdout_efficiency_difference")
    plt.close(fig)


def _plot_deltas(root: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    solve = summary["solve_rate"]["paired_delta"]
    expansion = summary["expanded_mean"]["paired_reduction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, stats, scale, ylabel, title in (
        (axes[0], solve, 100.0, "Optimized - base (percentage points)", "Solve-rate delta"),
        (axes[1], expansion, 100.0, "Expansion reduction (%)", "Search-efficiency delta"),
    ):
        values = np.asarray(stats["values"]) * scale
        mean = float(stats["mean"]) * scale
        low, high = np.asarray(stats["ci95"]) * scale
        ax.scatter(np.ones(len(values)), values, color="#7A5195", alpha=0.65, s=28)
        ax.errorbar(
            [1], [mean], yerr=[[mean - low], [high - mean]], fmt="D", color="#E45756",
            capsize=6, markersize=7, label="Mean and 95% CI",
        )
        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Paired holdout effects across synthesis replicates")
    fig.tight_layout()
    _save_figure(fig, root, "holdout_paired_deltas")
    plt.close(fig)


def _curve_ci(
    curves: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = curves.mean(axis=0)
    if len(curves) == 1:
        return mean, mean, mean
    indices = rng.integers(0, len(curves), size=(samples, len(curves)))
    boot = curves[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def _plot_budget_profile(
    root: Path,
    replica_levels: Sequence[Sequence[dict[str, Any]]],
    *,
    budgets: Sequence[int],
    samples: int,
    rng: np.random.Generator,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, color in (("base", "#4C78A8"), ("optimized", "#E45756")):
        curves = []
        for rows in replica_levels:
            curves.append(
                [
                    np.mean(
                        [
                            bool(row[f"{label}_solved"])
                            and float(row[f"{label}_expanded"]) <= budget
                            for row in rows
                        ]
                    )
                    for budget in budgets
                ]
            )
        mean, low, high = _curve_ci(np.asarray(curves), samples=samples, rng=rng)
        ax.plot(budgets, mean, marker="o", color=color, label=label.title())
        ax.fill_between(budgets, low, high, color=color, alpha=0.18)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("A* expansion budget")
    ax.set_ylabel("Holdout solve rate")
    ax.set_title("Search-budget profile (mean and 95% CI)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, root, "holdout_budget_profile")
    plt.close(fig)


def _plot_game_deltas(
    root: Path,
    replica_levels: Sequence[Sequence[dict[str, Any]]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> None:
    import matplotlib.pyplot as plt

    games = sorted({str(row["game"]) for rows in replica_levels for row in rows})
    game_stats: list[tuple[str, dict[str, Any]]] = []
    for game in games:
        deltas = []
        for rows in replica_levels:
            game_rows = [row for row in rows if str(row["game"]) == game]
            if game_rows:
                deltas.append(
                    np.mean([bool(row["optimized_solved"]) for row in game_rows])
                    - np.mean([bool(row["base_solved"]) for row in game_rows])
                )
        game_stats.append((game, _mean_ci(deltas, samples=samples, rng=rng)))
    game_stats.sort(key=lambda item: float(item[1]["mean"]))
    height = max(4.5, 0.30 * len(game_stats) + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, height))
    for y, (game, stats) in enumerate(game_stats):
        mean = float(stats["mean"]) * 100
        low, high = np.asarray(stats["ci95"]) * 100
        ax.errorbar(
            [mean], [y], xerr=[[mean - low], [high - mean]], fmt="o",
            color="#54A24B" if mean >= 0 else "#E45756", capsize=3,
        )
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_yticks(range(len(game_stats)), [item[0] for item in game_stats], fontsize=8)
    ax.set_xlabel("Optimized - base solve rate (percentage points)")
    ax.set_title("Generalization by game (mean and 95% CI)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, root, "holdout_game_solve_rate_delta")
    plt.close(fig)


def summarize_replicates(
    state_root: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Summarize completed ``replicate-*`` directories beneath ``state_root``."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    replicas = sorted(
        path
        for path in state_root.glob("replicate-*")
        if (path / "comparison_summary.json").is_file()
        and (path / "per_level_comparison.json").is_file()
    )
    if not replicas:
        raise ValueError(f"No completed replicas found under {state_root}")
    summaries = [_load_json(path / "comparison_summary.json") for path in replicas]
    levels = [_load_json(path / "per_level_comparison.json") for path in replicas]
    rng = np.random.default_rng(seed)

    base_solve = [float(row["base"]["solve_rate"]) for row in summaries]
    optimized_solve = [float(row["optimized"]["solve_rate"]) for row in summaries]
    base_expanded = [float(row["base"]["expanded_mean"]) for row in summaries]
    optimized_expanded = [float(row["optimized"]["expanded_mean"]) for row in summaries]
    base_score = [float(row["base"]["score_mean"]) for row in summaries]
    optimized_score = [float(row["optimized"]["score_mean"]) for row in summaries]
    expansion_reduction = [
        1.0 - optimized / base if base else 0.0
        for base, optimized in zip(base_expanded, optimized_expanded)
    ]
    common_rows = [
        [row for row in rows if row["base_solved"] and row["optimized_solved"]]
        for rows in levels
    ]
    common_nonempty = [rows for rows in common_rows if rows]
    common_base_expanded = [
        float(np.mean([row["base_expanded"] for row in rows])) for rows in common_nonempty
    ]
    common_optimized_expanded = [
        float(np.mean([row["optimized_expanded"] for row in rows]))
        for rows in common_nonempty
    ]
    common_reduction = [
        1.0
        - sum(float(row["optimized_expanded"]) for row in rows)
        / max(1.0, sum(float(row["base_expanded"]) for row in rows))
        for rows in common_nonempty
    ]

    summary = {
        "n_replicates": len(replicas),
        "replicate_paths": [str(path) for path in replicas],
        "bootstrap_samples": bootstrap_samples,
        "solve_rate": {
            "base": _mean_ci(base_solve, samples=bootstrap_samples, rng=rng),
            "optimized": _mean_ci(optimized_solve, samples=bootstrap_samples, rng=rng),
            "paired_delta": _mean_ci(
                np.subtract(optimized_solve, base_solve), samples=bootstrap_samples, rng=rng
            ),
        },
        "expanded_mean": {
            "base": _mean_ci(base_expanded, samples=bootstrap_samples, rng=rng),
            "optimized": _mean_ci(optimized_expanded, samples=bootstrap_samples, rng=rng),
            "paired_reduction": _mean_ci(
                expansion_reduction, samples=bootstrap_samples, rng=rng
            ),
        },
        "common_solved_efficiency": {
            "n_common": [len(rows) for rows in common_rows],
            "base_expanded_mean": _mean_ci(
                common_base_expanded, samples=bootstrap_samples, rng=rng
            ),
            "optimized_expanded_mean": _mean_ci(
                common_optimized_expanded, samples=bootstrap_samples, rng=rng
            ),
            "paired_reduction": _mean_ci(
                common_reduction, samples=bootstrap_samples, rng=rng
            ),
        },
        "score_mean": {
            "base": _mean_ci(base_score, samples=bootstrap_samples, rng=rng),
            "optimized": _mean_ci(optimized_score, samples=bootstrap_samples, rng=rng),
            "paired_delta": _mean_ci(
                np.subtract(optimized_score, base_score), samples=bootstrap_samples, rng=rng
            ),
        },
    }
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / "replicate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_metrics(state_root, summary)
    _plot_solve_rate(state_root, summary)
    _plot_efficiency_difference(state_root, summary)
    _plot_deltas(state_root, summary)
    _plot_budget_profile(
        state_root, levels, budgets=budgets, samples=bootstrap_samples, rng=rng
    )
    _plot_game_deltas(state_root, levels, samples=bootstrap_samples, rng=rng)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_root", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_replicates(
        args.state_root, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
