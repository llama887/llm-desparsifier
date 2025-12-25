from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from llm_desparsifier.rl.pipeline import run_training_with_reward


class JobLike(Protocol):
    name: str
    env_id: str

    def to_config(self) -> Dict[str, Any]:
        ...


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_JSON = REPO_ROOT / "sparse_baseline.json"


def to_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    return np.asarray(value, dtype=float).tolist()


def load_sparse_baseline(path: Path) -> Optional[Tuple[Dict[str, Dict[str, Any]], float]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    baselines = payload.get("sparse_baselines")
    if not isinstance(baselines, dict):
        return None
    mean_raw = payload.get("sparse_baseline_mean", 0.0)
    try:
        mean = float(mean_raw)
    except Exception:
        mean = 0.0
    return baselines, mean


def save_sparse_baseline(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def log_sparse_baseline(
    baselines: Mapping[str, Mapping[str, Any]],
    baseline_mean: float,
    log_wandb: Callable[..., None],
) -> None:
    for example_id, payload in baselines.items():
        solve_rate = payload.get("solve_rate")
        env_id = payload.get("env_id")
        if solve_rate is None or env_id is None:
            continue
        log_wandb(
            {
                "gepa/example_id": example_id,
                "gepa/env_id": env_id,
                "gepa/sparse_baseline_solve_rate": float(solve_rate),
            },
            step=0,
        )
    if baselines:
        log_wandb({"gepa/sparse_baseline_solve_rate_mean": baseline_mean}, step=0)


def run_sparse_baseline(
    jobs: List[JobLike],
    logs_root: Path,
    log_wandb: Callable[..., None],
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    class _NullRewardGenerator:
        def generate(self, *_, **__):
            raise RuntimeError("Sparse baseline should not call reward generator")

    baseline_root = logs_root / "sparse_baseline"
    baseline_root.mkdir(exist_ok=True)
    per_env_baselines: List[float] = []
    sparse_baselines: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        baseline_dir = baseline_root / job.name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sparse-baseline] running {job.name} into {baseline_dir}")
        try:
            baseline_result = run_training_with_reward(
                _NullRewardGenerator(),
                output_dir=str(baseline_dir),
                config_override=job.to_config(),
                reward_mode="sparse",
            )
            solve_rate = float(baseline_result.final_metrics.get("solve_rate", 0.0))
            sparse_curve = to_float_list(
                baseline_result.train_info.get("loss_info", {}).get("eval/ground_truth_returns_mean")
            )
            sparse_baselines[job.name] = {
                "solve_rate": solve_rate,
                "sparse_curve": sparse_curve,
                "artifacts": dict(baseline_result.artifacts),
                "env_id": job.env_id,
            }
            per_env_baselines.append(solve_rate)
            print(f"[sparse-baseline] {job.name} solve_rate={solve_rate:.4f}")
        except Exception as exc:  # pragma: no cover - baseline is best-effort
            print(f"[sparse-baseline] FAILED {job.name}: {exc}")

    baseline_mean = float(np.mean(per_env_baselines)) if per_env_baselines else 0.0
    if sparse_baselines:
        log_sparse_baseline(sparse_baselines, baseline_mean, log_wandb)
    return sparse_baselines, baseline_mean


def ensure_sparse_baseline(
    jobs: List[JobLike],
    *,
    logs_root: Path,
    baseline_json_path: Path,
    log_wandb: Callable[..., None],
    env_grid_path: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    cached = load_sparse_baseline(baseline_json_path)
    if cached is not None:
        baselines, baseline_mean = cached
        log_sparse_baseline(baselines, baseline_mean, log_wandb)
        return baselines, baseline_mean

    baselines, baseline_mean = run_sparse_baseline(jobs, logs_root, log_wandb)
    payload: Dict[str, Any] = {
        "created_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "sparse_baseline_mean": baseline_mean,
        "sparse_baselines": baselines,
    }
    if env_grid_path is not None:
        payload["env_grid"] = str(env_grid_path)
    if state_root is not None:
        payload["state_root"] = str(state_root)
    save_sparse_baseline(baseline_json_path, payload)
    return baselines, baseline_mean
