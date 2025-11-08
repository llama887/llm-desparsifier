#!/usr/bin/env python3
"""Run dense-only training for a batch of environments and log GEPA datasets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import yaml

from llm_desparsifier.rewards import (
    RewardGenerator,
    RewardSynthesizer,
    build_reward_reflection,
    create_reward_reflection_module,
)
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT
from llm_desparsifier.rl.pipeline import TrainingResult, run_training_with_reward
from llm_desparsifier.utils import get_active_prompt_path

DEFAULT_ENV_GRID = Path("configs/gepa_envs.yaml")
BASE_PROMPT_PATH = Path("llm_desparsifier/rewards/prompts/base_reward_prompt.txt")


@dataclass
class EnvJob:
    name: str
    env_id: str
    benchmark_id: str
    total_timesteps: int
    train_seed: int
    eval_seed: int

    @classmethod
    def from_mapping(cls, index: int, payload: Mapping[str, Any]) -> "EnvJob":
        name = payload.get("name") or f"job-{index}"
        return cls(
            name=name,
            env_id=str(payload["env_id"]),
            benchmark_id=str(payload.get("benchmark_id", "trivial-1m")),
            total_timesteps=int(payload.get("total_timesteps", 1_000_000)),
            train_seed=int(payload.get("train_seed", index)),
            eval_seed=int(payload.get("eval_seed", index)),
        )

    def to_config(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "benchmark_id": self.benchmark_id,
            "total_timesteps": self.total_timesteps,
            "train_seed": self.train_seed,
            "eval_seed": self.eval_seed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense reward batch and log GEPA dataset")
    parser.add_argument(
        "--state-root",
        type=Path,
        required=True,
        help="Directory for shared GEPA state (active prompt, iteration folders)",
    )
    parser.add_argument(
        "--env-grid",
        type=Path,
        default=DEFAULT_ENV_GRID,
        help="YAML file describing environment jobs (default: configs/gepa_envs.yaml)",
    )
    return parser.parse_args()


def load_env_jobs(env_grid_path: Path) -> List[EnvJob]:
    data = yaml.safe_load(env_grid_path.read_text())
    if isinstance(data, Mapping) and "jobs" in data:
        entries = data["jobs"]
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError("Environment grid must be a list under 'jobs' or a top-level list")
    jobs = [EnvJob.from_mapping(idx, entry) for idx, entry in enumerate(entries)]
    if not jobs:
        raise ValueError("No environment jobs found in grid")
    return jobs


def load_prompt_payload(state_root: Path) -> tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    prompt_path = get_active_prompt_path(state_root)
    if prompt_path.exists():
        payload = json.loads(prompt_path.read_text())
        text = payload.get("constraints_text")
        if isinstance(text, str) and text.strip():
            synth_state = payload.get("synthesizer_state")
            meta = {"source": "active_prompt", "path": str(prompt_path)}
            return text, synth_state, meta
    if BASE_PROMPT_PATH.exists():
        return BASE_PROMPT_PATH.read_text(), None, {
            "source": "base_prompt_file",
            "path": str(BASE_PROMPT_PATH),
        }
    return CONSTRAINTS_TEXT, None, {"source": "default_constraints"}


def build_reward_generator(constraints_text: str, synthesizer_state: Optional[Dict[str, Any]]) -> RewardGenerator:
    synthesizer = RewardSynthesizer()
    if synthesizer_state:
        synthesizer.gen.load_state(synthesizer_state)
    return RewardGenerator(constraints_text=constraints_text, synthesizer=synthesizer, verbose=False)


def create_iteration_directory(state_root: Path) -> Path:
    timestamp = dt.datetime.utcnow().strftime("iter-%Y%m%d-%H%M%S")
    iteration_dir = state_root / timestamp
    iteration_dir.mkdir(parents=True, exist_ok=False)
    (iteration_dir / "runs").mkdir(exist_ok=True)
    return iteration_dir


def to_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    return np.asarray(value, dtype=float).tolist()


def build_dataset_row(job: EnvJob, result: TrainingResult) -> Dict[str, Any]:
    loss_info = result.train_info.get("loss_info", {}) if isinstance(result.train_info, Mapping) else {}
    sparse_curve = to_float_list(loss_info.get("eval/ground_truth_returns_mean"))
    component_logs = result.train_info.get("component_logs", {}) if isinstance(result.train_info, Mapping) else {}
    component_curves = {
        name: to_float_list(series)
        for name, series in component_logs.items()
    }
    return {
        "job_name": job.name,
        "env_id": result.config.env_id,
        "benchmark_id": result.config.benchmark_id,
        "train_seed": job.train_seed,
        "eval_seed": job.eval_seed,
        "reward_code": result.emitted_reward_code,
        "sparse_return_curve": sparse_curve,
        "component_curves": component_curves,
        "final_metrics": result.final_metrics,
        "artifacts": result.artifacts,
    }


def write_metadata(iteration_dir: Path, metadata: Mapping[str, Any]) -> None:
    metadata_path = iteration_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def run_batch() -> None:
    args = parse_args()
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    env_grid_path = args.env_grid.expanduser().resolve()
    jobs = load_env_jobs(env_grid_path)
    constraints_text, synthesizer_state, prompt_meta = load_prompt_payload(state_root)
    iteration_dir = create_iteration_directory(state_root)
    dataset_path = iteration_dir / "train_dense.jsonl"

    metadata = {
        "iteration_dir": str(iteration_dir),
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "env_grid_path": str(env_grid_path),
        "prompt": prompt_meta,
        "jobs": [job.__dict__ for job in jobs],
    }

    reward_generator = build_reward_generator(constraints_text, synthesizer_state)
    reflection_module = create_reward_reflection_module()

    with dataset_path.open("w", encoding="utf-8") as dataset_file:
        for job in jobs:
            print(f"[run_reward_batch] Starting job {job.name} ({job.env_id})")
            job_output_dir = iteration_dir / "runs" / job.name
            job_output_dir.mkdir(parents=True, exist_ok=True)
            result = run_training_with_reward(
                reward_generator,
                output_dir=str(job_output_dir),
                config_override=job.to_config(),
                reward_mode="dense",
            )
            row = build_dataset_row(job, result)
            row["reflection_text"] = build_reward_reflection(row, reflection_module=reflection_module)
            dataset_file.write(json.dumps(row) + "\n")
            dataset_file.flush()
            print(f"[run_reward_batch] Finished job {job.name}")

    write_metadata(iteration_dir, metadata)
    (iteration_dir / "ready_for_gepa").touch()
    print(f"[run_reward_batch] Wrote dataset to {dataset_path}")


if __name__ == "__main__":
    run_batch()
