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
import hashlib
import itertools
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import dspy
import numpy as np
import yaml
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None

from llm_desparsifier.rewards import (
    RewardSynthesizer,
    RewardGenerator,
    build_reward_reflection,
    create_reward_reflection_module,
    configure_portkey_lm,
)
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT
from llm_desparsifier.rl.pipeline import TrainingResult, run_training_with_reward
from llm_desparsifier.utils import get_active_prompt_path, write_active_prompt

DEFAULT_ENV_GRID = Path("configs/gepa_envs.yaml")
BASE_PROMPT_PATH = Path("llm_desparsifier/rewards/prompts/base_reward_prompt.txt")
DEFAULT_MAX_METRIC_CALLS = 80
DEFAULT_REFLECTION_MINIBATCH = 2
MAX_TOTAL_TIMESTEPS = 20_000_000
MAX_NUM_ENVS = 1_024
MAX_EVAL_ENVS = 128
MAX_EVAL_EPISODES = 20


@dataclass
class MetricCacheEntry:
    score: float
    feedback: str
    run_dir: Path
    created_at: float
    sparse_curve: List[float]


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
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--gepa-auto",
        choices=["light", "medium", "heavy"],
        default=None,
        help="DSPy GEPA auto budget (controls mutation/eval counts).",
    )
    budget_group.add_argument(
        "--max-full-evals",
        type=int,
        default=None,
        help="Cap GEPA by the number of full train+val evals (each equals len(train)+len(val) metric calls).",
    )
    budget_group.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Hard cap on GEPA metric calls (overrides auto/full-evals).",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=DEFAULT_REFLECTION_MINIBATCH,
        help="Examples per reflective minibatch (smaller = cheaper feedback passes).",
    )
    parser.add_argument(
        "--disable-merge",
        action="store_true",
        help="Disable GEPA merge proposer to save budget.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Thread pool size for GEPA metric evaluation (None = DSPy default).",
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


def load_prompt_payload(state_root: Path) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    prompt_path = get_active_prompt_path(state_root)
    if prompt_path.exists():
        payload = json.loads(prompt_path.read_text())
        text = payload.get("constraints_text")
        if isinstance(text, str) and text.strip():
            synth_state = payload.get("synthesizer_state")
            prompt_state = payload.get("prompt_state")
            meta = {"source": "active_prompt", "path": str(prompt_path)}
            return text, synth_state, prompt_state, meta
    if BASE_PROMPT_PATH.exists():
        return BASE_PROMPT_PATH.read_text(), None, None, {
            "source": "base_prompt_file",
            "path": str(BASE_PROMPT_PATH),
        }
    return CONSTRAINTS_TEXT, None, None, {"source": "default_constraints"}


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


def clamp_job_budget(job_cfg: Mapping[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Clamp expensive training knobs to keep per-candidate runtime bounded."""

    cfg = dict(job_cfg)
    notes: List[str] = []

    def _clamp(key: str, cap: int) -> None:
        if key in cfg:
            original = int(cfg[key])
            capped = min(original, cap)
            if capped != original:
                notes.append(f"{key} {original}→{capped}")
            cfg[key] = capped
        else:
            cfg[key] = cap
            notes.append(f"{key} default→{cap}")

    _clamp("total_timesteps", MAX_TOTAL_TIMESTEPS)
    _clamp("num_envs", MAX_NUM_ENVS)
    _clamp("eval_num_envs", MAX_EVAL_ENVS)
    _clamp("eval_num_episodes", MAX_EVAL_EPISODES)
    return cfg, notes


def make_candidate_fingerprint(reward_code: str, job_cfg: Mapping[str, Any]) -> str:
    payload = {
        "reward": reward_code,
        "env_id": job_cfg.get("env_id"),
        "benchmark_id": job_cfg.get("benchmark_id"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def derive_seed(example_id: str, candidate_fp: str) -> int:
    digest = hashlib.blake2b(f"{example_id}:{candidate_fp}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2_147_483_647


def get_example_id(example: dspy.Example) -> str:
    return str(getattr(example, "job_name", getattr(example, "env_description", "example")))


class RewardPromptProgram(dspy.Module):
    """DSPy module that synthesizes reward code, but treats the prompt as the optimizable object."""

    def __init__(
        self,
        constraints_text: str,
        synthesizer_state: Optional[Mapping[str, Any]] = None,
        prompt_state: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.base_constraints = constraints_text

        class PromptSearch(dspy.Signature):
            env_description: str = dspy.InputField()
            base_constraints: str = dspy.InputField()
            prompt_text: str = dspy.OutputField(desc="Rewritten constraints for reward synthesis")

        class PromptGenerator(dspy.Module):
            def __init__(self, state: Optional[Mapping[str, Any]] = None):
                super().__init__()
                self.rewriter = dspy.Predict(PromptSearch)
                if state:
                    self.rewriter.load_state(state)

            def dump_state(self) -> Mapping[str, Any]:
                return self.rewriter.dump_state()

            def forward(self, env_description: str, base_constraints: str) -> str:
                out = self.rewriter(env_description=env_description, base_constraints=base_constraints)
                return out.prompt_text

        self.prompt_generator = PromptGenerator(prompt_state)
        self.synthesizer = RewardSynthesizer()
        if synthesizer_state:
            self.synthesizer.gen.load_state(synthesizer_state)

    def forward(self, env_description: str, constraints: Optional[str] = None):
        # GEPA now optimizes the rewrite of the base constraints; fallback to provided constraints.
        prompt_text = constraints or self.prompt_generator(
            env_description=env_description, base_constraints=self.base_constraints
        )
        reward_code = self.synthesizer(env_description=env_description, constraints=prompt_text)
        return dspy.Prediction(reward_code=reward_code, prompt_text=prompt_text)


def build_examples(jobs: List[EnvJob], constraints_text: str) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for job in jobs:
        desc = f"{job.env_id} | benchmark={job.benchmark_id}"
        ex = dspy.Example(env_description=desc).with_inputs("env_description")
        cfg = job.to_config()
        cfg["name"] = job.name
        ex.job_config = cfg
        ex.job_name = job.name
        examples.append(ex)
    return examples


def run_batch() -> None:
    args = parse_args()
    if args.gepa_auto is None and args.max_full_evals is None and args.max_metric_calls is None:
        args.max_metric_calls = DEFAULT_MAX_METRIC_CALLS
    reflection_minibatch_size = max(1, args.reflection_minibatch_size)
    use_merge = not args.disable_merge

    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "gepa_runs"
    logs_root.mkdir(exist_ok=True)

    env_grid_path = args.env_grid.expanduser().resolve()
    jobs = load_env_jobs(env_grid_path)

    constraints_text, synthesizer_state, prompt_state, prompt_meta = load_prompt_payload(state_root)
    reflection_module = create_reward_reflection_module()

    # Configure LM once; reuse for program + reflection.
    base_lm = configure_portkey_lm()
    dspy.configure(lm=base_lm)

    wandb_run = None
    candidate_table = None
    if wandb is not None and not os.environ.get("WANDB_DISABLED"):
        try:
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "llm-desparsifier"),
                name=f"gepa-{state_root.name}",
                config={
                    "state_root": str(state_root),
                    "env_grid": str(env_grid_path),
                    "gepa_auto": args.gepa_auto,
                    "max_full_evals": args.max_full_evals,
                    "max_metric_calls": args.max_metric_calls,
                    "reflection_minibatch_size": reflection_minibatch_size,
                    "use_merge": use_merge,
                    "num_threads": args.num_threads,
                },
            )
            candidate_table = wandb.Table(
                columns=[
                    "candidate_idx",
                    "example_id",
                    "fingerprint",
                    "candidate_prompt",
                    "reward_code",
                    "final_return",
                    "sparse_curve",
                    "feedback",
                    "run_dir",
                    "elapsed_sec",
                    "budget_notes",
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[wandb] init failed, continuing without logging: {exc}")
            wandb_run = None

    program = RewardPromptProgram(constraints_text, synthesizer_state=synthesizer_state, prompt_state=prompt_state)
    examples = build_examples(jobs, constraints_text)
    trainset = valset = examples  # on-policy: no static holdout

    run_counter = itertools.count(1)
    metric_cache: Dict[tuple[str, str], MetricCacheEntry] = {}
    cache_lock = threading.Lock()

    def on_policy_metric(example: dspy.Example, prediction: dspy.Prediction, *_):
        """Evaluate a GEPA candidate by running the full RL loop (existing budget)."""
        reward_code = getattr(prediction, "reward_code", "")
        failsafe_score = 0.0
        if not reward_code.strip():
            return ScoreWithFeedback(score=failsafe_score, feedback="Empty reward code")

        example_id = get_example_id(example)
        job_cfg_raw = getattr(example, "job_config", {}) or {}
        budgeted_cfg, budget_notes = clamp_job_budget(job_cfg_raw)
        candidate_fp = make_candidate_fingerprint(reward_code, budgeted_cfg)
        cache_key = (example_id, candidate_fp)

        with cache_lock:
            cached = metric_cache.get(cache_key)
        if cached is not None:
            print(f"[on_policy_metric] cache hit {example_id}#{candidate_fp} score={cached.score:.4f}")
            return ScoreWithFeedback(score=cached.score, feedback=cached.feedback)

        seed = derive_seed(example_id, candidate_fp)
        budgeted_cfg.setdefault("train_seed", seed)
        budgeted_cfg.setdefault("eval_seed", seed + 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        train_cfg = {k: v for k, v in budgeted_cfg.items() if k != "name"}
        run_id = next(run_counter)
        run_dir = logs_root / f"candidate-{run_id:04d}-{example_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if budget_notes:
            print(f"[on_policy_metric] budget clamp {example_id}: " + "; ".join(budget_notes))
        print(f"[on_policy_metric] cache miss {example_id}#{candidate_fp} seed={seed} dir={run_dir.name}")

        start = time.time()
        try:
            candidate_prompt = getattr(prediction, "prompt_text", None) or constraints_text
            reward_generator = RewardGenerator(
                constraints_text=candidate_prompt,
                lm=base_lm,
                max_sanitize_attempts=5,
                failure_artifact_dir=run_dir / "sanitize_failures",
                bootstrap_code=reward_code,
                verbose=False,
            )
            result = run_training_with_reward(
                reward_generator,
                output_dir=str(run_dir),
                config_override=train_cfg,
                reward_mode="dense",
            )
            row = build_dataset_row(
                EnvJob.from_mapping(run_id, {**train_cfg, "name": job_cfg_raw.get("name", example_id)}),
                result,
            )
            row["env_description"] = getattr(example, "env_description", None)
            row["job_name"] = example_id
            reflection = build_reward_reflection(row, reflection_module=reflection_module)
            sparse_curve = row.get("sparse_return_curve") or []
            final_return = float(sparse_curve[-1]) if sparse_curve else 0.0
            elapsed = time.time() - start
            cache_entry = MetricCacheEntry(
                score=final_return,
                feedback=reflection,
                run_dir=run_dir,
                created_at=time.time(),
                sparse_curve=sparse_curve,
            )
            with cache_lock:
                metric_cache[cache_key] = cache_entry
                if candidate_table is not None:
                    candidate_table.add_data(
                        run_id,
                        example_id,
                        candidate_fp,
                        candidate_prompt,
                        result.emitted_reward_code,
                        final_return,
                        sparse_curve,
                        reflection,
                        str(run_dir),
                        elapsed,
                        "; ".join(budget_notes),
                    )
            if wandb_run is not None:
                enqueue_log(
                    {
                        "gepa/candidate_return": final_return,
                        "gepa/candidate_idx": run_id,
                        "gepa/example_id": example_id,
                    },
                    step=next(log_step_counter),
                )
            print(
                f"[on_policy_metric] completed {example_id}#{candidate_fp} in {elapsed / 60:.2f}m "
                f"score={final_return:.4f}"
            )
            return ScoreWithFeedback(score=final_return, feedback=reflection)
        except Exception as exc:
            elapsed = time.time() - start
            print(f"[on_policy_metric] failure {example_id}#{candidate_fp} after {elapsed / 60:.2f}m: {exc}")
            failure_feedback = f"Training failed: {exc}"
            # Surface sanitizer guidance (if available) so GEPA can mutate prompts effectively.
            if "reward_generator" in locals():
                attempts = getattr(reward_generator, "last_attempt_history", None)
                if attempts:
                    try:
                        extra = reward_generator._build_feedback_block(attempts)  # type: ignore[attr-defined]
                        failure_feedback = f"{failure_feedback}\n\nSanitizer feedback:\n{extra}"
                    except Exception:
                        # Do not let feedback formatting mask the original failure.
                        pass
            cache_entry = MetricCacheEntry(
                score=failsafe_score,
                feedback=failure_feedback,
                run_dir=run_dir,
                created_at=time.time(),
                sparse_curve=[],
            )
            with cache_lock:
                metric_cache[cache_key] = cache_entry
                if candidate_table is not None:
                    candidate_table.add_data(
                        run_id,
                        example_id,
                        candidate_fp,
                        candidate_prompt,
                        reward_code,
                        failsafe_score,
                        [],
                        failure_feedback,
                        str(run_dir),
                        elapsed,
                        "; ".join(budget_notes),
                    )
            if wandb_run is not None:
                enqueue_log(
                    {
                        "gepa/candidate_return": failsafe_score,
                        "gepa/candidate_idx": run_id,
                        "gepa/example_id": example_id,
                        "gepa/error": str(exc),
                    },
                    step=next(log_step_counter),
                )
            return ScoreWithFeedback(score=failsafe_score, feedback=failure_feedback)

    compiler = dspy.GEPA(
        metric=on_policy_metric,
        auto=args.gepa_auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        reflection_lm=base_lm,
        use_merge=use_merge,
        num_threads=args.num_threads,
        track_stats=True,
    )

    optimized_program = compiler.compile(program, trainset=trainset, valset=valset)

    prompt_payload = {
        "constraints_text": constraints_text,
        "synthesizer_state": optimized_program.synthesizer.gen.dump_state(),
        "prompt_state": optimized_program.prompt_generator.dump_state(),
        "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "source": prompt_meta,
    }
    write_active_prompt(state_root, prompt_payload)

    stats_path = logs_root / "gepa_stats.json"
    stats_path.write_text(json.dumps(getattr(compiler, "stats", {}), indent=2, sort_keys=True), encoding="utf-8")

    print(f"[run_reward_batch] GEPA completed. Active prompt updated at {get_active_prompt_path(state_root)}")
    print(f"[run_reward_batch] GEPA stats written to {stats_path}")

    if wandb_run is not None:
        if candidate_table is not None:
            enqueue_log({"gepa/candidates": candidate_table}, step=next(log_step_counter))
        if log_queue is not None:
            log_queue.put(None)
        if log_thread is not None:
            log_thread.join(timeout=5)
        art = wandb.Artifact(f"{state_root.name}-gepa", type="gepa-state")
        _active = get_active_prompt_path(state_root)
        if _active.exists():
            art.add_file(str(_active), name="active_prompt.json")
        if stats_path.exists():
            art.add_file(str(stats_path), name="gepa_stats.json")
        wandb_run.log_artifact(art)
        wandb_run.finish(quiet=True)


if __name__ == "__main__":
    run_batch()
