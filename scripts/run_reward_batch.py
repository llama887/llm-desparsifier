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
import json
import os
import random
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
    RewardGenerator,
    build_reward_reflection,
    create_reward_reflection_module,
    configure_portkey_lm,
)
from llm_desparsifier.rewards.reflection import EUREKA_GUIDANCE
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT
from llm_desparsifier.rl.pipeline import TrainingResult, run_training_with_reward
from llm_desparsifier.rl.sparse_baseline import DEFAULT_BASELINE_JSON, ensure_sparse_baseline
from llm_desparsifier.utils import (
    get_active_prompt_path,
    write_active_prompt,
)

DEFAULT_ENV_GRID = Path("configs/gepa_envs.yaml")
BASE_PROMPT_PATH = Path("llm_desparsifier/rewards/prompts/base_reward_prompt.txt")
DEFAULT_MAX_METRIC_CALLS = 80
MAX_TOTAL_TIMESTEPS = 20_000_000
MAX_NUM_ENVS = 1_024
MAX_EVAL_ENVS = 128
MAX_EVAL_EPISODES = 20


def safe_wandb_log(wandb_run: Any, payload: Mapping[str, Any], **kwargs: Any) -> None:
    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(wandb_run, "finished", False):
        return
    try:
        wandb_run.log(payload, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive for late-finish errors
        if wandb is not None and isinstance(exc, wandb.errors.UsageError) and "finished" in str(exc):
            return
        raise


def safe_wandb_log_artifact(wandb_run: Any, artifact: Any) -> None:
    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(wandb_run, "finished", False):
        return
    try:
        wandb_run.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - defensive for late-finish errors
        if wandb is not None and isinstance(exc, wandb.errors.UsageError) and "finished" in str(exc):
            return
        raise


def safe_wandb_finish(wandb_run: Any) -> None:
    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(wandb_run, "finished", False):
        return
    try:
        wandb_run.finish(quiet=True)
    except Exception as exc:  # pragma: no cover - defensive for late-finish errors
        if wandb is not None and isinstance(exc, wandb.errors.UsageError) and "finished" in str(exc):
            return
        raise


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
        "--max-metric-calls",
        type=int,
        default=None,
        help="Hard cap on GEPA metric calls (defaults to 80).",
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
            prompt_state = payload.get("prompt_state")
            meta = {"source": "active_prompt", "path": str(prompt_path)}
            return text, prompt_state, meta
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


def build_dataset_row(
    job: EnvJob,
    result: TrainingResult,
    *,
    env_description: Optional[str] = None,
    candidate_prompt: Optional[str] = None,
    sanitizer_feedback: Optional[str] = None,
    budget_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
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
        "env_description": env_description,
        "candidate_prompt": candidate_prompt,
        "sanitizer_feedback": sanitizer_feedback,
        "budget_cfg": dict(budget_cfg) if budget_cfg else None,
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
        "eval_num_episodes": job_cfg.get("eval_num_episodes"),
        "eval_seed": job_cfg.get("eval_seed"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def derive_seed(example_id: str, candidate_fp: str) -> int:
    digest = hashlib.blake2b(f"{example_id}:{candidate_fp}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2_147_483_647


def get_example_id(example: dspy.Example) -> str:
    return str(getattr(example, "job_name", getattr(example, "env_description", "example")))


def format_feedback(feedback_text: str, pred_name: Optional[str], pred_trace: Any) -> str:
    if pred_name:
        header = f"[Predictor feedback: {pred_name}]"
        if pred_trace is not None:
            header = f"{header} (trace present)"
        return f"{header}\n{feedback_text}"
    if pred_trace is not None:
        return f"[Predictor trace present]\n{feedback_text}"
    return feedback_text


def write_training_curve_png(
    path: Path,
    gepa_solve_rates: List[float],
    sparse_baseline_mean: Optional[float],
) -> None:
    if not gepa_solve_rates:
        print("[training-curve] no GEPA solve rates to plot; skipping training_curve.png")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[training-curve] matplotlib unavailable; skipping plot ({exc})")
        return

    xs = list(range(1, len(gepa_solve_rates) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(xs, gepa_solve_rates, marker="o", linewidth=2.0, label="GEPA solve rate")
    if sparse_baseline_mean is not None:
        baseline_series = [float(sparse_baseline_mean)] * len(xs)
        plt.plot(xs, baseline_series, linestyle="--", linewidth=2.0, label="Sparse baseline")
    plt.xlabel("GEPA metric call")
    plt.ylabel("Solve rate")
    plt.title("Training Curve")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[training-curve] wrote {path}")


def job_from_example(example: dspy.Example) -> EnvJob:
    job_cfg_raw = getattr(example, "job_config", None)
    if not isinstance(job_cfg_raw, Mapping):
        raise ValueError("Example is missing job_config; cannot evaluate metric")
    example_id = get_example_id(example)
    job_cfg = dict(job_cfg_raw)
    job_cfg.setdefault("name", example_id)
    return EnvJob.from_mapping(0, job_cfg)


class PromptOnlyProgram(dspy.Module):
    """DSPy module that emits only the prompt rewrite for GEPA to optimize."""

    def __init__(
        self,
        constraints_text: str,
        prompt_state: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.base_constraints = constraints_text
        # Static meta-instructions that tell the LM to rewrite constraints only.
        # We keep them outside the constraint block so the rewrite focuses on the
        # content users care about, and we can later strip/verify if needed.
        self.rewrite_preamble = (
            "You are refining reward-spec constraints for a dense reward generator.\n"
            "- Task: rewrite the constraints text to be clearer/safer.\n"
            "- Output: ONLY the rewritten constraints text (no Python code, no code fences).\n"
            "- Preserve all mandatory rules and headings; you may tighten/clarify them.\n"
            "- Do NOT add or output any Python function bodies or examples.\n"
        )

        class PromptSearch(dspy.Signature):
            base_constraints: str = dspy.InputField()
            prompt_text: str = dspy.OutputField(desc="Evolved constraints for reward synthesis")

        class PromptGenerator(dspy.Module):
            def __init__(self, state: Optional[Mapping[str, Any]] = None):
                super().__init__()
                self.rewriter = dspy.Predict(PromptSearch)
                if state:
                    self.rewriter.load_state(state)

            def dump_state(self) -> Mapping[str, Any]:
                return self.rewriter.dump_state()

            def forward(self, base_constraints: str) -> str:
                out = self.rewriter(base_constraints=base_constraints)
                return out.prompt_text

        self.prompt_generator = PromptGenerator(prompt_state)

    def _build_rewrite_prompt(self) -> str:
        """Wrap the base constraints so the LM is steered to rewrite text, not emit code."""
        return (
            f"{self.rewrite_preamble}"
            "=== CONSTRAINTS TO REWRITE START ===\n"
            f"{self.base_constraints.strip()}\n"
            "=== CONSTRAINTS TO REWRITE END ===\n"
            "Return only the rewritten constraints text."
        )

    def forward(self, env_description: str, constraints: Optional[str] = None):
        # GEPA optimizes the rewrite of the base constraints; fallback to provided constraints.
        rewrite_prompt = self._build_rewrite_prompt()
        prompt_text = constraints or self.prompt_generator(base_constraints=rewrite_prompt)
        return dspy.Prediction(prompt_text=prompt_text)


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
    if args.max_metric_calls is None:
        args.max_metric_calls = DEFAULT_MAX_METRIC_CALLS
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "gepa_runs"
    logs_root.mkdir(exist_ok=True)

    env_grid_path = args.env_grid.expanduser().resolve()
    jobs = load_env_jobs(env_grid_path)
    # Use all jobs from the grid; users can edit the YAML to reduce set.

    constraints_text, prompt_state, prompt_meta = load_prompt_payload(state_root)
    reflection_module = create_reward_reflection_module()

    # Configure LM once; reuse for program + reflection.
    base_lm = configure_portkey_lm()
    dspy.configure(lm=base_lm)
    dspy.settings.configure(provide_traceback=True)
    print(f"[run_reward_batch] provide_traceback={dspy.settings.provide_traceback}")

    wandb_run = None
    candidate_table = None
    io_table = None
    sparse_baselines: Dict[str, Dict[str, Any]] = {}
    sparse_baseline_mean: float = 0.0
    gepa_solve_rates: List[float] = []
    last_wandb_step = -1

    if wandb is not None and not os.environ.get("WANDB_DISABLED"):
        try:
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "llm-desparsifier"),
                name=f"gepa-{state_root.name}",
                config={
                    "state_root": str(state_root),
                    "env_grid": str(env_grid_path),
                    "max_metric_calls": args.max_metric_calls,
                },
            )
            candidate_table = wandb.Table(
                columns=[
                    "step",
                    "env_id",
                    "solve_rate",
                    "sparse_baseline",
                    "reward_code_sha16",
                    "prompt_text",
                    "feedback",
                    "run_dir",
                ],
                log_mode="MUTABLE",
            )
            io_table = wandb.Table(
                columns=[
                    "metric_call",
                    "score",
                    "feedback",
                    "prompt_text",
                ],
                log_mode="MUTABLE",
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[wandb] init failed, continuing without logging: {exc}")
            wandb_run = None
            io_table = None

    def log_wandb(payload: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        nonlocal last_wandb_step
        if wandb_run is None:
            return
        if step is None:
            step = last_wandb_step + 1
        else:
            step = max(step, last_wandb_step + 1)
        last_wandb_step = step
        safe_wandb_log(wandb_run, payload, step=step)

    baseline_budget_signature = {
        "total_timesteps": MAX_TOTAL_TIMESTEPS,
        "num_envs": MAX_NUM_ENVS,
        "eval_num_envs": MAX_EVAL_ENVS,
        "eval_num_episodes": MAX_EVAL_EPISODES,
    }
    sparse_baselines, sparse_baseline_mean = ensure_sparse_baseline(
        jobs,
        logs_root=logs_root,
        baseline_json_path=DEFAULT_BASELINE_JSON,
        log_wandb=log_wandb,
        env_grid_path=env_grid_path,
        state_root=state_root,
        config_clamper=clamp_job_budget,
        budget_signature=baseline_budget_signature,
    )

    program = PromptOnlyProgram(constraints_text, prompt_state=prompt_state)
    trainset = build_examples(jobs, constraints_text)  # on-policy: no static holdout

    run_counter = 1
    metric_call_idx = 0
    metric_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
    score_by_prediction_id: Dict[int, float] = {}
    feedback_by_prediction_id: Dict[int, str] = {}

    def on_policy_metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any = None,
        pred_name: Optional[str] = None,
        pred_trace: Any = None,
    ):
        """Evaluate a GEPA candidate by running the full RL loop (existing budget)."""
        nonlocal run_counter, metric_call_idx
        training_curve_path = logs_root / "training_curve.png"
        failsafe_score = 0.0
        candidate_prompt = getattr(prediction, "prompt_text", None)
        if not isinstance(candidate_prompt, str) or not candidate_prompt.strip():
            candidate_prompt = constraints_text
        prediction_key = id(prediction)

        job = job_from_example(example)
        example_id = job.name
        cache_key = (example_id, candidate_prompt)
        if pred_name is not None:
            try:
                from dspy.teleprompt.bootstrap_trace import FailedPrediction  # type: ignore
            except Exception:  # pragma: no cover - defensive
                FailedPrediction = None  # type: ignore

            def _has_failed_prediction(trace_obj: Any) -> bool:
                if FailedPrediction is None or not trace_obj:
                    return False
                for item in trace_obj:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        if isinstance(item[2], FailedPrediction):
                            return True
                return False

            if _has_failed_prediction(pred_trace) or _has_failed_prediction(trace):
                feedback_text = format_feedback(
                    "Trace contains FailedPrediction; returning failure score.",
                    pred_name,
                    pred_trace,
                )
                return ScoreWithFeedback(score=failsafe_score, feedback=feedback_text)

            cached_score = score_by_prediction_id.get(prediction_key)
            cached_feedback = feedback_by_prediction_id.get(prediction_key)
            if cached_score is None:
                print(
                    "[on_policy_metric] missing cached score for predictor feedback "
                    f"example={example_id} pred_name={pred_name} "
                    f"cache_key={cache_key}"
                )
                raise RuntimeError(
                    "Missing cached score for predictor feedback; "
                    "ensure scores are captured during primary metric evaluation."
                )
            feedback_text = format_feedback(
                cached_feedback or "Missing cached feedback for predictor; returning score only.",
                pred_name,
                pred_trace,
            )
            return ScoreWithFeedback(score=cached_score, feedback=feedback_text)
        metric_call_idx += 1
        baseline_solve_rate = sparse_baselines.get(example_id, {}).get("solve_rate")
        baseline_solve_rate_mean = sparse_baseline_mean if sparse_baseline_mean else None

        job_cfg_raw = job.to_config()
        budgeted_cfg, budget_notes = clamp_job_budget(job_cfg_raw)
        seed = derive_seed(example_id, candidate_prompt)
        budgeted_cfg.setdefault("train_seed", seed)
        budgeted_cfg.setdefault("eval_seed", seed + 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        train_cfg = {k: v for k, v in budgeted_cfg.items() if k != "name"}
        run_id = run_counter
        run_counter += 1
        run_dir = logs_root / f"candidate-{run_id:04d}-{example_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if budget_notes:
            print(f"[on_policy_metric] budget clamp {example_id}: " + "; ".join(budget_notes))
        print(f"[on_policy_metric] evaluating {example_id} seed={seed} dir={run_dir.name}")

        start = time.time()
        try:
            reward_generator = RewardGenerator(
                constraints_text=candidate_prompt,
                lm=base_lm,
                max_sanitize_attempts=5,
            )
            result = run_training_with_reward(
                reward_generator,
                output_dir=str(run_dir),
                config_override=train_cfg,
                reward_mode="dense",
            )

            env_description = getattr(reward_generator, "last_env_description", None) or getattr(
                example, "env_description", None
            )
            sanitizer_feedback = None
            if getattr(reward_generator, "last_attempt_history", None):
                try:
                    sanitizer_feedback = reward_generator._build_feedback_block(  # type: ignore[attr-defined]
                        reward_generator.last_attempt_history
                    )
                except Exception:
                    sanitizer_feedback = None

            row = build_dataset_row(
                EnvJob.from_mapping(run_id, {**train_cfg, "name": example_id}),
                result,
                env_description=env_description,
                candidate_prompt=candidate_prompt,
                sanitizer_feedback=sanitizer_feedback,
                budget_cfg=train_cfg,
            )
            row["job_name"] = example_id

            reflection = build_reward_reflection(
                row,
                reflection_module=reflection_module,
                guidance_text=EUREKA_GUIDANCE,
            )
            sparse_curve = row.get("sparse_return_curve") or []

            gt_eval = result.ground_truth_eval or {}
            returns = gt_eval.get("returns") or []
            successes = gt_eval.get("successes")
            if successes is None:
                successes = sum(1 for r in returns if r > 0.0)
            total_eps = len(returns)
            solve_rate = float(successes) / float(total_eps) if total_eps else float(gt_eval.get("success_rate", 0.0))
            # Fallback to the previous metric if eval returns are missing.
            if total_eps == 0 and not gt_eval:
                solve_rate = float(sparse_curve[-1]) if sparse_curve else 0.0

            emitted_code = result.emitted_reward_code or ""
            elapsed = time.time() - start
        except Exception as exc:
            elapsed = time.time() - start
            print(f"[on_policy_metric] failure {example_id} after {elapsed / 60:.2f}m: {exc}")
            failure_feedback = f"Training failed: {exc}"
            if "reward_generator" in locals():
                attempts = getattr(reward_generator, "last_attempt_history", None)
                if attempts:
                    try:
                        extra = reward_generator._build_feedback_block(attempts)  # type: ignore[attr-defined]
                        failure_feedback = f"{failure_feedback}\n\nSanitizer feedback:\n{extra}"
                    except Exception:
                        pass
            metric_cache[cache_key] = {
                "solve_rate": failsafe_score,
                "reflection": failure_feedback,
            }
            score_by_prediction_id[prediction_key] = failsafe_score
            feedback_by_prediction_id[prediction_key] = failure_feedback
            feedback_text = format_feedback(failure_feedback, pred_name, pred_trace)
            gepa_solve_rates.append(failsafe_score)
            baseline_line = sparse_baseline_mean if sparse_baselines else None
            write_training_curve_png(training_curve_path, gepa_solve_rates, baseline_line)
            return ScoreWithFeedback(score=failsafe_score, feedback=feedback_text)

        reward_hash = hashlib.sha256(emitted_code.encode("utf-8")).hexdigest()[:16]
        feedback_text = format_feedback(reflection, pred_name, pred_trace)
        metric_cache[cache_key] = {
            "solve_rate": solve_rate,
            "reflection": reflection,
        }
        score_by_prediction_id[prediction_key] = solve_rate
        feedback_by_prediction_id[prediction_key] = reflection
        gepa_solve_rates.append(solve_rate)
        baseline_line = sparse_baseline_mean if sparse_baselines else None
        write_training_curve_png(training_curve_path, gepa_solve_rates, baseline_line)

        if candidate_table is not None:
            candidate_table.add_data(
                run_id,
                job.env_id,
                solve_rate,
                baseline_solve_rate,
                reward_hash,
                candidate_prompt,
                feedback_text,
                str(run_dir),
            )
            log_wandb({"gepa/candidates": candidate_table}, step=metric_call_idx)

        payload = {
            "gepa/solve_rate": solve_rate,
            "gepa/solve_rate_mean": solve_rate,
        }
        if baseline_solve_rate_mean is not None:
            payload["gepa/sparse_baseline_solve_rate"] = baseline_solve_rate_mean
        log_wandb(payload, step=metric_call_idx)

        print(
            f"[on_policy_metric] solve_rate={solve_rate:.4f} env={example_id} elapsed={elapsed/60:.2f}m"
        )
        if io_table is not None:
            io_table.add_data(metric_call_idx, solve_rate, feedback_text, candidate_prompt)
            log_wandb({"gepa/io_table": io_table}, step=metric_call_idx)
        return ScoreWithFeedback(score=solve_rate, feedback=feedback_text)

    compiler = dspy.GEPA(
        metric=on_policy_metric,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=base_lm,
        track_stats=True,
        num_threads=1,
    )

    optimized_program = compiler.compile(program, trainset=trainset)

    prompt_payload = {
        "constraints_text": constraints_text,
        "prompt_state": optimized_program.prompt_generator.dump_state(),
        "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": prompt_meta,
    }
    write_active_prompt(state_root, prompt_payload)

    stats_path = logs_root / "gepa_stats.json"
    stats_payload = getattr(compiler, "stats", {}) or {}
    stats_payload["sparse_baselines"] = sparse_baselines
    stats_payload["sparse_baseline_mean"] = sparse_baseline_mean
    stats_payload["gepa_solve_rate_series"] = gepa_solve_rates
    stats_payload["baseline_budget_signature"] = baseline_budget_signature
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[run_reward_batch] GEPA completed. Active prompt updated at {get_active_prompt_path(state_root)}")
    print(f"[run_reward_batch] GEPA stats written to {stats_path}")

    training_curve_path = logs_root / "training_curve.png"
    baseline_line = sparse_baseline_mean if sparse_baselines else None
    write_training_curve_png(training_curve_path, gepa_solve_rates, baseline_line)

    if wandb_run is not None:
        if candidate_table is not None:
            log_wandb({"gepa/candidates": candidate_table})
        if io_table is not None:
            log_wandb({"gepa/io_table": io_table})
        art = wandb.Artifact(f"{state_root.name}-gepa", type="gepa-state")
        _active = get_active_prompt_path(state_root)
        if _active.exists():
            art.add_file(str(_active), name="active_prompt.json")
        if stats_path.exists():
            art.add_file(str(stats_path), name="gepa_stats.json")
        if training_curve_path.exists():
            art.add_file(str(training_curve_path), name="training_curve.png")
        safe_wandb_log_artifact(wandb_run, art)
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    run_batch()
