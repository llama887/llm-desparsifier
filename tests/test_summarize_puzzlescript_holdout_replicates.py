import json
from pathlib import Path

from scripts.summarize_puzzlescript_holdout_replicates import summarize_replicates


def _write_replica(root: Path, index: int, base_solved: int, optimized_solved: int) -> None:
    replica = root / f"replicate-{index:02d}"
    replica.mkdir(parents=True)
    n = 4
    base_expanded = [10 + index, 20 + index, 30 + index, 40 + index]
    optimized_expanded = [8 + index, 18 + index, 28 + index, 38 + index]
    summary = {
        "base": {
            "n": n,
            "solved": base_solved,
            "solve_rate": base_solved / n,
            "score_mean": 0.4 + index / 100,
            "expanded_mean": sum(base_expanded) / n,
        },
        "optimized": {
            "n": n,
            "solved": optimized_solved,
            "solve_rate": optimized_solved / n,
            "score_mean": 0.5 + index / 100,
            "expanded_mean": sum(optimized_expanded) / n,
        },
    }
    levels = []
    for task_id in range(n):
        levels.append(
            {
                "task_id": task_id,
                "game": "alpha" if task_id < 2 else "beta",
                "base_solved": task_id < base_solved,
                "optimized_solved": task_id < optimized_solved,
                "base_expanded": base_expanded[task_id],
                "optimized_expanded": optimized_expanded[task_id],
            }
        )
    (replica / "comparison_summary.json").write_text(json.dumps(summary))
    (replica / "per_level_comparison.json").write_text(json.dumps(levels))


def test_summarize_replicates_writes_statistics_and_plots(tmp_path: Path) -> None:
    _write_replica(tmp_path, 1, 2, 3)
    _write_replica(tmp_path, 2, 2, 4)
    _write_replica(tmp_path, 3, 3, 3)

    summary = summarize_replicates(tmp_path, bootstrap_samples=64, seed=7)

    assert summary["n_replicates"] == 3
    assert summary["solve_rate"]["base"]["mean"] == 7 / 12
    assert summary["solve_rate"]["optimized"]["mean"] == 10 / 12
    assert summary["solve_rate"]["paired_delta"]["mean"] == 0.25
    assert summary["common_solved_efficiency"]["n_common"] == [2, 2, 3]
    assert (tmp_path / "replicate_summary.json").is_file()
    expected = {
        "holdout_solve_rate.png",
        "holdout_solve_rate.pdf",
        "holdout_efficiency_difference.png",
        "holdout_efficiency_difference.pdf",
        "holdout_replicate_metrics.png",
        "holdout_replicate_metrics.pdf",
        "holdout_paired_deltas.png",
        "holdout_paired_deltas.pdf",
        "holdout_budget_profile.png",
        "holdout_budget_profile.pdf",
        "holdout_game_solve_rate_delta.png",
        "holdout_game_solve_rate_delta.pdf",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
