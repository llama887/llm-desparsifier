#!/usr/bin/env python3
"""Run integrated GEPA prompt optimization with on-policy RL training.

This replaces the previous two-job pipeline (dataset writer + offline GEPA
optimizer). GEPA now proposes reward prompts, we evaluate them with the
existing RL training loop (same budget as before), and feed live rewards +
Eureka-style reflections back into GEPA. No intermediate JSONL datasets or
marker files are produced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import dspy
import numpy as np
import yaml

from llm_desparsifier.rewards import (
    RewardSynthesizer,
    build_reward_reflection,
    create_reward_reflection_module,
    sanitize_and_compile,
    configure_portkey_lm,
)
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT
from llm_desparsifier.rl.pipeline import TrainingResult, run_training_with_reward
from llm_desparsifier.utils import get_active_prompt_path, write_active_prompt

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
    parser = argparse.ArgumentParser(description="Run GEPA with on-policy RL evaluation")
    parser.add_argument(
        "--state-root",
        type=Path,
        required=True,
        help="Directory for shared GEPA state (active prompt, optimizer logs)",
    )
    parser.add_argument(
        "--env-grid",
        type=Path,
        default=DEFAULT_ENV_GRID,
        help="YAML file describing environment jobs (default: configs/gepa_envs.yaml)",
    )
    parser.add_argument(
        "--gepa-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="DSPy GEPA auto budget (controls mutation/eval counts)",
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


class RewardPromptProgram(dspy.Module):
    """DSPy module that synthesizes reward code from env description + constraints."""

    def __init__(self, constraints_text: str, synthesizer_state: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self.constraints_text = constraints_text
        self.synthesizer = RewardSynthesizer()
        if synthesizer_state:
            self.synthesizer.gen.load_state(synthesizer_state)

    def forward(self, env_description: str, constraints: Optional[str] = None):
        text = constraints or self.constraints_text
        reward_code = self.synthesizer(env_description=env_description, constraints=text)
        return dspy.Prediction(reward_code=reward_code)


class StaticRewardGenerator:
    """Reward generator that uses pre-baked reward code (no LLM calls)."""

    def __init__(self, reward_code: str):
        self.reward_code = reward_code

    def generate(self, env, env_params):
        dense_fn = sanitize_and_compile(self.reward_code)
        return dense_fn, self.reward_code


def build_examples(jobs: List[EnvJob], constraints_text: str) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for job in jobs:
        desc = f"{job.env_id} | benchmark={job.benchmark_id}"
        ex = dspy.Example(env_description=desc, constraints=constraints_text).with_inputs(
            "env_description", "constraints"
        )
        cfg = job.to_config()
        cfg["name"] = job.name
        ex.job_config = cfg
        ex.job_name = job.name
        examples.append(ex)
    return examples


def run_batch() -> None:
    args = parse_args()
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "gepa_runs"
    logs_root.mkdir(exist_ok=True)

    env_grid_path = args.env_grid.expanduser().resolve()
    jobs = load_env_jobs(env_grid_path)

    constraints_text, synthesizer_state, prompt_meta = load_prompt_payload(state_root)
    reflection_module = create_reward_reflection_module()

    # Configure LM once; reuse for program + reflection.
    base_lm = configure_portkey_lm()
    dspy.configure(lm=base_lm)

    program = RewardPromptProgram(constraints_text, synthesizer_state)
    examples = build_examples(jobs, constraints_text)
    trainset = valset = examples  # on-policy: no static holdout

    run_counter = itertools.count(1)

    def on_policy_metric(example: dspy.Example, prediction: dspy.Prediction, *_):
        """Evaluate a GEPA candidate by running the full RL loop (existing budget)."""
        reward_code = getattr(prediction, "reward_code", "")
        failsafe_score = 0.0
        if not reward_code.strip():
            return dspy.Prediction(score=failsafe_score, feedback="Empty reward code")

        candidate_id = next(run_counter)
        run_dir = logs_root / f"candidate-{candidate_id:04d}-{getattr(example, 'job_name', 'env')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            reward_generator = StaticRewardGenerator(reward_code)
            job_cfg = getattr(example, "job_config", {}) or {}
            train_cfg = {k: v for k, v in job_cfg.items() if k != "name"}
            result = run_training_with_reward(
                reward_generator,
                output_dir=str(run_dir),
                config_override=train_cfg,
                reward_mode="dense",
            )
            row = build_dataset_row(
                EnvJob.from_mapping(candidate_id, job_cfg),
                result,
            )
            reflection = build_reward_reflection(row, reflection_module=reflection_module)
            sparse_curve = row.get("sparse_return_curve") or []
            final_return = float(sparse_curve[-1]) if sparse_curve else 0.0
            return dspy.Prediction(score=final_return, feedback=reflection)
        except Exception as exc:
            return dspy.Prediction(score=failsafe_score, feedback=f"Training failed: {exc}")

    compiler = dspy.GEPA(
        metric=on_policy_metric,
        auto=args.gepa_auto,
        reflection_lm=base_lm,
        track_stats=True,
    )

    optimized_program = compiler.compile(program, trainset=trainset, valset=valset)

    prompt_payload = {
        "constraints_text": constraints_text,
        "synthesizer_state": optimized_program.synthesizer.gen.dump_state(),
        "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "source": prompt_meta,
    }
    write_active_prompt(state_root, prompt_payload)

    stats_path = logs_root / "gepa_stats.json"
    stats_path.write_text(json.dumps(getattr(compiler, "stats", {}), indent=2, sort_keys=True), encoding="utf-8")

    print(f"[run_reward_batch] GEPA completed. Active prompt updated at {get_active_prompt_path(state_root)}")
    print(f"[run_reward_batch] GEPA stats written to {stats_path}")


if __name__ == "__main__":
    run_batch()
