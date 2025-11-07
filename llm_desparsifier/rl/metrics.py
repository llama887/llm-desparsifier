"""Utilities for logging ground-truth evaluation metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class GroundTruthLogRow:
    run_id: str
    reward_mode: str
    global_step: int
    episode: int
    gt_return: float
    gt_return_std: float
    wall_time: float
    checkpoint_path: str


def _ensure_metrics_dir(output_dir: str) -> Path:
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def dump_ground_truth_logs(output_dir: str, rows: Sequence[GroundTruthLogRow]) -> Path:
    metrics_dir = _ensure_metrics_dir(output_dir)
    csv_path = metrics_dir / "ground_truth_runs.csv"

    fieldnames = [
        "run_id",
        "reward_mode",
        "global_step",
        "episode",
        "gt_return",
        "gt_return_std",
        "wall_time",
        "checkpoint_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return csv_path


def compute_ground_truth_summary(
    rows: Sequence[GroundTruthLogRow],
    *,
    threshold: Optional[float] = None,
) -> dict:
    if not rows:
        return {
            "max_return": None,
            "argmax_step": None,
            "time_to_threshold": None,
            "auc_ground_truth": None,
            "final_return": None,
        }

    steps = np.array([row.global_step for row in rows], dtype=float)
    returns = np.array([row.gt_return for row in rows], dtype=float)

    max_idx = int(np.argmax(returns))
    max_return = float(returns[max_idx])
    argmax_step = int(steps[max_idx])

    auc_value = float(np.trapz(returns, steps))

    time_to_threshold = None
    if threshold is not None:
        meets = np.where(returns >= threshold)[0]
        if meets.size:
            time_to_threshold = int(steps[meets[0]])

    final_return = float(returns[-1])

    return {
        "max_return": max_return,
        "argmax_step": argmax_step,
        "time_to_threshold": time_to_threshold,
        "auc_ground_truth": auc_value,
        "final_return": final_return,
    }


def dump_ground_truth_summary(output_dir: str, summary: dict) -> Path:
    metrics_dir = _ensure_metrics_dir(output_dir)
    summary_path = metrics_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path
