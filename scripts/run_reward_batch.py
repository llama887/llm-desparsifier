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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
    configure_gemini_lm,
    configure_portkey_lm,
    summarize_trajectory_behavior_from_path,
)
from llm_desparsifier.rewards.llm_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL_ALIAS,
)
from llm_desparsifier.rewards.reflection import EUREKA_GUIDANCE
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT
from llm_desparsifier.rl.pipeline import TrainingResult, run_training_with_reward
from llm_desparsifier.rl.sparse_baseline import (
    DEFAULT_BASELINE_JSON,
    ensure_sparse_baseline,
    load_sparse_baseline_payload,
    log_sparse_baseline,
    run_sparse_baseline,
    save_sparse_baseline,
)
from llm_desparsifier.utils import (
    get_active_prompt_path,
    write_active_prompt,
)

DEFAULT_ENV_GRID = Path("configs/gepa_envs.yaml")
BASE_PROMPT_PATH = Path("llm_desparsifier/rewards/prompts/base_reward_prompt.txt")
DEFAULT_MAX_GEPA_ITERATIONS = 25
MAX_TOTAL_TIMESTEPS = 170_000_000
MAX_NUM_ENVS = 2_048
MAX_EVAL_ENVS = 128
MAX_EVAL_EPISODES = 20
DEFAULT_REWARD_LLM_TEMP = 0.1
DEFAULT_REFLECTION_LLM_TEMP = 0.5
HOLDOUT_REWARD_TRIES = 5
SINGLE_ENV_ID = "XLand-MiniGrid-R1-11x11"
SINGLE_ENV_BENCHMARK = "trivial-1m"
SINGLE_ENV_TOTAL_TIMESTEPS = 170_000_000
SINGLE_ENV_TRAIN_SEED = 1000
SINGLE_ENV_EVAL_SEED = 1000
HOLDOUT_ENVS = [
    "XLand-MiniGrid-R1-9x9",
    "XLand-MiniGrid-R1-17x17",
    "XLand-MiniGrid-R2-9x9",
    "XLand-MiniGrid-R2-17x17",
    "XLand-MiniGrid-R4-9x9",
    "XLand-MiniGrid-R4-17x17",
    "XLand-MiniGrid-R6-13x13",
    "XLand-MiniGrid-R6-19x19",
    "XLand-MiniGrid-R9-16x16",
    "XLand-MiniGrid-R9-25x25",
]


def _emit_log_line(payload: Mapping[str, Any]) -> None:
    """Emit a single-line JSON log for long-running GEPA stages.

    This helper standardizes stdout logging so hangs can be diagnosed from the
    W&B logs tab or raw console output. It is needed because many stages (LLM
    calls, JAX compilation, training, artifact uploads) can block for minutes
    without other progress indicators, and structured logs provide a reliable
    heartbeat. It differs from the standard logging module by always emitting a
    single JSON object per line, coercing non-serializable values to strings,
    and flushing immediately so the log line appears even if the process stalls.
    """
    try:
        line = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        line = json.dumps({str(k): str(v) for k, v in payload.items()}, sort_keys=True)
    print(line, flush=True)


def safe_wandb_log(wandb_run: Any, payload: Mapping[str, Any], **kwargs: Any) -> None:
    """Log a payload to W&B while guarding against finished-run errors.

    This wrapper centralizes W&B logging so callers do not have to reimplement
    finished-run checks and error handling. It is needed because GEPA runs can
    be long-lived and W&B sessions may finish or drop unexpectedly, and it
    differs from calling `wandb.log` directly by short-circuiting for finished
    runs, tolerating usage errors that indicate a closed run, and emitting
    timing logs so slow or stuck uploads are visible in stdout.
    """
    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(
        wandb_run, "finished", False
    ):
        return
    payload_keys = sorted(str(k) for k in payload.keys())
    start = time.time()
    _emit_log_line(
        {
            "event": "wandb_log_start",
            "payload_keys": payload_keys,
            "step": kwargs.get("step"),
        }
    )
    try:
        wandb_run.log(payload, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive for late-finish errors
        _emit_log_line(
            {
                "event": "wandb_log_error",
                "elapsed_sec": round(time.time() - start, 4),
                "error": str(exc),
            }
        )
        if (
            wandb is not None
            and isinstance(exc, wandb.errors.UsageError)
            and "finished" in str(exc)
        ):
            return
        raise
    _emit_log_line(
        {
            "event": "wandb_log_end",
            "elapsed_sec": round(time.time() - start, 4),
        }
    )


def safe_wandb_log_artifact(wandb_run: Any, artifact: Any) -> None:
    """Log a W&B artifact while surfacing timing for potential stalls.

    This helper wraps `wandb.log_artifact` with finished-run checks and
    diagnostic timing logs. It is needed because artifact uploads can block on
    network or file IO, and it differs from direct artifact logging by emitting
    start/end log lines to stdout and ignoring finished-run usage errors.
    """
    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(
        wandb_run, "finished", False
    ):
        return
    start = time.time()
    _emit_log_line(
        {
            "event": "wandb_log_artifact_start",
            "artifact_name": getattr(artifact, "name", None),
        }
    )
    try:
        wandb_run.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - defensive for late-finish errors
        _emit_log_line(
            {
                "event": "wandb_log_artifact_error",
                "elapsed_sec": round(time.time() - start, 4),
                "error": str(exc),
            }
        )
        if (
            wandb is not None
            and isinstance(exc, wandb.errors.UsageError)
            and "finished" in str(exc)
        ):
            return
        raise
    _emit_log_line(
        {
            "event": "wandb_log_artifact_end",
            "elapsed_sec": round(time.time() - start, 4),
        }
    )


def safe_wandb_finish(wandb_run: Any) -> None:
    """Finish a W&B run safely while emitting timing diagnostics.

    This guard ensures `wandb.finish()` is only invoked on active runs and
    records how long the finalize step takes. It is needed because some runs
    finalize slowly (e.g., pending uploads), and it differs from direct calls
    by ignoring finished-run usage errors and by logging start/end timestamps
    for debugging.
    """
    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(
        wandb_run, "finished", False
    ):
        return
    start = time.time()
    _emit_log_line({"event": "wandb_finish_start"})
    try:
        wandb_run.finish(quiet=True)
    except Exception as exc:  # pragma: no cover - defensive for late-finish errors
        _emit_log_line(
            {
                "event": "wandb_finish_error",
                "elapsed_sec": round(time.time() - start, 4),
                "error": str(exc),
            }
        )
        if (
            wandb is not None
            and isinstance(exc, wandb.errors.UsageError)
            and "finished" in str(exc)
        ):
            return
        raise
    _emit_log_line(
        {
            "event": "wandb_finish_end",
            "elapsed_sec": round(time.time() - start, 4),
        }
    )


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
    """Parse CLI arguments that configure the GEPA runner.

    This function consolidates all user-facing switches so the rest of the
    pipeline can assume a validated config object. It is needed to keep the
    runner reproducible across local, SLURM, and test entrypoints, and it
    differs from environment grid parsing by only handling top-level CLI
    concerns rather than per-job settings.
    """
    parser = argparse.ArgumentParser(
        description="Run GEPA with on-policy RL evaluation"
    )
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
        "--max-gepa-iterations",
        type=int,
        default=DEFAULT_MAX_GEPA_ITERATIONS,
        help="Number of GEPA iterations (one prompt per full job sweep).",
    )
    parser.add_argument(
        "--llm",
        default=DEFAULT_MODEL_ALIAS,
        help=(
            "Model identifier to use for GEPA; interpreted as a Portkey alias when "
            "--llm-provider=portkey, or as a Gemini model name when --llm-provider=gemini "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["portkey", "gemini"],
        default="portkey",
        help=(
            "LLM backend to use: 'portkey' routes through the gateway, 'gemini' calls "
            "Gemini directly using GEMINI_API_KEY (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--reward-llm-temp",
        type=float,
        default=DEFAULT_REWARD_LLM_TEMP,
        help="Temperature for the reward-synthesis LLM (default: %(default)s)",
    )
    parser.add_argument(
        "--reflection-llm-temp",
        type=float,
        default=DEFAULT_REFLECTION_LLM_TEMP,
        help="Temperature for the GEPA reflection/prompt LLM (default: %(default)s)",
    )
    parser.add_argument(
        "--test-single-env",
        action="store_true",
        help="Run GEPA on a single fixed environment with locked seeds for overfitting.",
    )
    parser.add_argument(
        "--deterministic-envs",
        action="store_true",
        help="Use a fixed benchmark ruleset instead of sampling new ones.",
    )
    parser.add_argument(
        "--plot-training-curves",
        action="store_true",
        help="Generate dense-vs-sparse and holdout plots (default: off).",
    )
    parser.add_argument(
        "--room-count",
        type=int,
        action="append",
        default=None,
        help=(
            "Restrict environments to specific room counts parsed from env_id "
            "(repeatable, e.g. --room-count 1 --room-count 4)."
        ),
    )
    return parser.parse_args()


def load_env_grid(env_grid_path: Path) -> tuple[List[EnvJob], List[EnvJob]]:
    """Load training and holdout environment jobs from a YAML grid file.

    This helper centralizes environment-grid parsing so GEPA optimization and
    holdout evaluation share the same source of truth for job definitions. It is
    needed because the runner now supports a dedicated `eval_jobs` list for
    holdout evaluation while still accepting the legacy top-level list format
    used for training-only runs. It differs from the earlier `load_env_jobs`
    helper by returning both the training set (`jobs`) and the optional holdout
    set (`eval_jobs`) in one pass, validating each section independently.
    """
    data = yaml.safe_load(env_grid_path.read_text())
    if isinstance(data, Mapping):
        job_entries = data.get("jobs", [])
        eval_entries = data.get("eval_jobs", [])
    elif isinstance(data, list):
        job_entries = data
        eval_entries = []
    else:
        raise ValueError("Environment grid must be a mapping or list")

    if not isinstance(job_entries, list):
        raise ValueError("Environment grid 'jobs' must be a list")
    if not isinstance(eval_entries, list):
        raise ValueError("Environment grid 'eval_jobs' must be a list")

    jobs = [EnvJob.from_mapping(idx, entry) for idx, entry in enumerate(job_entries)]
    eval_jobs = [
        EnvJob.from_mapping(idx, entry) for idx, entry in enumerate(eval_entries)
    ]
    if not jobs:
        raise ValueError("No environment jobs found in grid")
    return jobs, eval_jobs


def build_single_env_job() -> List[EnvJob]:
    return [
        EnvJob(
            name="single-env-test",
            env_id=SINGLE_ENV_ID,
            benchmark_id=SINGLE_ENV_BENCHMARK,
            total_timesteps=SINGLE_ENV_TOTAL_TIMESTEPS,
            train_seed=SINGLE_ENV_TRAIN_SEED,
            eval_seed=SINGLE_ENV_EVAL_SEED,
        )
    ]


def extract_room_count(env_id: str) -> int:
    """Extract the XLand room-count token from an environment identifier.

    This helper parses the canonical `-R<rooms>-` segment from an env id such
    as `XLand-MiniGrid-R4-13x13` and returns the integer room count. It is
    needed so room-based filtering is deterministic and centralized across both
    training and holdout job sets, and it differs from ad-hoc string splitting
    by validating the full token shape and failing loudly when the id does not
    include a parseable room-count segment.
    """
    match = re.search(r"-R(\d+)-", env_id)
    if match is None:
        raise ValueError(
            f"Could not parse room count from env_id '{env_id}'. "
            "Expected token shape '-R<digits>-'."
        )
    return int(match.group(1))


def filter_jobs_by_room_count(
    jobs: List[EnvJob],
    allowed_room_counts: List[int],
    section_name: str,
) -> List[EnvJob]:
    """Filter EnvJob entries by allowed room counts parsed from `env_id`.

    This helper applies a user-selected room-count allowlist to an `EnvJob`
    collection and returns only matching entries. It is needed because GEPA
    training and holdout evaluation should share identical room-count semantics,
    and it differs from editing the YAML grid manually by working at runtime for
    all sources of jobs (grid-defined and default holdout fallback). It prints a
    concise before/after summary and raises a `ValueError` when no jobs remain,
    which enforces fail-fast behavior for both train and holdout sections.
    """
    allowed = set(allowed_room_counts)
    filtered = [job for job in jobs if extract_room_count(job.env_id) in allowed]
    print(
        "[run_reward_batch] room-count filter "
        f"section={section_name} allowed={sorted(allowed)} "
        f"before={len(jobs)} after={len(filtered)}"
    )
    if not filtered:
        raise ValueError(
            f"--room-count filter removed all jobs from {section_name}. "
            f"allowed_room_counts={sorted(allowed)}"
        )
    return filtered


def load_prompt_payload(
    state_root: Path,
) -> tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    prompt_path = get_active_prompt_path(state_root)
    if prompt_path.exists():
        payload = json.loads(prompt_path.read_text())
        text = payload.get("constraints_text")
        if isinstance(text, str) and text.strip():
            prompt_state = payload.get("prompt_state")
            meta = {"source": "active_prompt", "path": str(prompt_path)}
            return text, prompt_state, meta
    if BASE_PROMPT_PATH.exists():
        return (
            BASE_PROMPT_PATH.read_text(),
            None,
            {
                "source": "base_prompt_file",
                "path": str(BASE_PROMPT_PATH),
            },
        )
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
    behavior_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a single run record used by reflection and diagnostics.

    This helper normalizes metrics and curves from `TrainingResult` into a
    stable row shape consumed by the reflection module and optional logging
    sinks. It is needed because GEPA metric evaluation constructs this payload
    in multiple branches (success and error paths), and it differs from direct
    `TrainingResult` usage by flattening nested arrays into JSON-friendly
    Python values while attaching run-specific metadata such as sanitizer and
    trajectory behavior summaries.
    """
    loss_info = (
        result.train_info.get("loss_info", {})
        if isinstance(result.train_info, Mapping)
        else {}
    )
    sparse_curve = to_float_list(loss_info.get("eval/ground_truth_returns_mean"))
    component_logs = (
        result.train_info.get("component_logs", {})
        if isinstance(result.train_info, Mapping)
        else {}
    )
    component_curves = {
        name: to_float_list(series) for name, series in component_logs.items()
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
        "behavior_summary": behavior_summary,
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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def derive_seed(example_id: str, candidate_fp: str) -> int:
    digest = hashlib.blake2b(
        f"{example_id}:{candidate_fp}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % 2_147_483_647


def build_holdout_jobs() -> List[EnvJob]:
    """Construct the default holdout EnvJob list when no eval_jobs are configured.

    This helper preserves the historical hard-coded holdout set so the runner
    can still execute a post-GEPA evaluation even if an environment grid omits
    `eval_jobs`. It is needed as a compatibility fallback for older YAML files
    and differs from the main `load_env_grid` path by providing fixed seeds and
    budgets instead of reading job definitions from configuration.
    """

    jobs: List[EnvJob] = []
    for idx, env_id in enumerate(HOLDOUT_ENVS):
        jobs.append(
            EnvJob(
                name=f"holdout-{idx:02d}",
                env_id=env_id,
                benchmark_id="trivial-1m",
                total_timesteps=MAX_TOTAL_TIMESTEPS,
                train_seed=10_000 + idx * 2,
                eval_seed=10_000 + idx * 2 + 1,
            )
        )
    return jobs


def save_best_prompt_text(state_root: Path, model_alias: str, prompt_text: str) -> Path:
    """Persist the best prompt as a plain .txt for quick reuse."""

    safe_model_alias = (
        re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_alias).strip("-") or "model"
    )
    best_prompt_path = state_root / f"{safe_model_alias}.txt"
    best_prompt_path.write_text(prompt_text.strip() + "\n", encoding="utf-8")
    return best_prompt_path


def get_example_id(example: dspy.Example) -> str:
    return str(
        getattr(example, "job_name", getattr(example, "env_description", "example"))
    )


def format_feedback(
    feedback_text: str, pred_name: Optional[str], pred_trace: Any
) -> str:
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
        print(
            "[training-curve] no GEPA solve rates to plot; skipping training_curve.png"
        )
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
        plt.plot(
            xs, baseline_series, linestyle="--", linewidth=2.0, label="Sparse baseline"
        )
    plt.xlabel("GEPA iteration")
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
            prompt_text: str = dspy.OutputField(
                desc="Evolved constraints for reward synthesis"
            )

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

    def forward(
        self,
        env_description: str,
        constraints: Optional[str] = None,
        return_trace: bool = False,
        **kwargs,
    ):
        # GEPA optimizes the rewrite of the base constraints; fallback to provided constraints.
        rewrite_prompt = self._build_rewrite_prompt()
        prompt_input = constraints or rewrite_prompt

        if return_trace:
            # Ask the underlying predictor for its trace so DSPy feedback flows stay rich.
            pred_obj, lm_trace = self.prompt_generator.rewriter(
                base_constraints=prompt_input, return_trace=True
            )
            prompt_text = pred_obj.prompt_text
            prediction = dspy.Prediction(prompt_text=prompt_text)
            trace = {
                "env_description": env_description,
                "prompt_input": prompt_input,
                "prompt_text": prompt_text,
                "lm_trace": lm_trace,
            }
            # Include a lightweight fingerprint of the rewriter state for debugging/deduping.
            try:
                state = self.prompt_generator.rewriter.dump_state()
                state_bytes = json.dumps(state, sort_keys=True, default=str).encode(
                    "utf-8"
                )
                trace["prompt_state_sha256"] = hashlib.sha256(state_bytes).hexdigest()[
                    :16
                ]
            except Exception:
                trace["prompt_state_sha256"] = None
            return prediction, trace

        prompt_text = (
            prompt_input
            if constraints
            else self.prompt_generator(base_constraints=rewrite_prompt)
        )
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


def ensure_holdout_sparse_baselines(
    holdout_jobs: List[EnvJob],
    *,
    logs_root: Path,
    baseline_json_path: Path,
    log_wandb: Any,
    config_clamper: Optional[
        Callable[[Mapping[str, Any]], Tuple[Dict[str, Any], List[str]]]
    ] = None,
    budget_signature: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Cache sparse baselines for holdouts, merging into the shared JSON payload."""

    payload = load_sparse_baseline_payload(baseline_json_path) or {}
    baselines: Dict[str, Dict[str, Any]] = dict(payload.get("sparse_baselines", {}))
    existing_mean = float(payload.get("sparse_baseline_mean", 0.0) or 0.0)
    signature_matches = (
        budget_signature is None or payload.get("budget_signature") == budget_signature
    )

    missing_jobs = [job for job in holdout_jobs if job.name not in baselines]
    if missing_jobs or not signature_matches:
        new_baselines, _ = run_sparse_baseline(
            missing_jobs if signature_matches else holdout_jobs,
            logs_root,
            log_wandb,
            config_clamper=config_clamper,
        )
        baselines.update(new_baselines)
        values = [
            float(row.get("solve_rate", 0.0)) for row in baselines.values() if row
        ]
        existing_mean = float(np.mean(values)) if values else 0.0
        merged_payload = dict(payload)
        merged_payload["sparse_baselines"] = baselines
        merged_payload["sparse_baseline_mean"] = existing_mean
        if budget_signature is not None:
            merged_payload["budget_signature"] = dict(budget_signature)
        save_sparse_baseline(baseline_json_path, merged_payload)

    log_sparse_baseline(
        {k: v for k, v in baselines.items() if k in {j.name for j in holdout_jobs}},
        existing_mean,
        log_wandb,
    )
    return baselines, existing_mean


def evaluate_dense_on_jobs(
    *,
    jobs: List[EnvJob],
    prompt_text: str,
    base_lm: Any,
    logs_root: Path,
    config_clamper: Callable[[Mapping[str, Any]], Tuple[Dict[str, Any], List[str]]],
    num_tries: int,
    plot_training_curves: bool,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Evaluate the optimized prompt on holdout jobs with multiple reward generations.

    This routine runs the dense reward synthesis and RL training loop `num_tries`
    times per holdout environment, aggregates solve rates per environment, and
    writes a per-try reward trace for downstream analysis. It is needed to measure
    variance across stochastic reward generations when the LLM temperature is
    non-zero, and to provide mean +/- std summaries for the holdout plots. It
    differs from the on-policy GEPA metric (which evaluates a single candidate
    per call) and from `evaluate_single_env_dense` by repeatedly sampling reward
    functions for each holdout job and returning aggregated statistics. When
    `plot_training_curves` is enabled, each dense run also emits a per-run
    dense-vs-sparse plot; when disabled, only numeric summaries are recorded.
    """

    if num_tries < 1:
        raise ValueError("num_tries must be >= 1")

    results: Dict[str, Dict[str, Any]] = {}
    reward_records: List[Dict[str, Any]] = []
    per_env_means: List[float] = []
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    for job in jobs:
        job_cfg_raw = job.to_config()
        budgeted_cfg, budget_notes = config_clamper(job_cfg_raw)
        seed = derive_seed(job.name, prompt_text)
        budgeted_cfg.setdefault("train_seed", seed)
        budgeted_cfg.setdefault("eval_seed", seed + 1)

        if budget_notes:
            print(
                f"[holdout-dense] budget clamp {job.name}: " + "; ".join(budget_notes)
            )

        solve_rates: List[float] = []
        reward_hashes: List[str] = []
        reward_code_paths: List[Optional[str]] = []
        run_dirs: List[str] = []
        elapsed_total = 0.0
        for try_idx in range(1, num_tries + 1):
            run_dir = logs_root / "holdout-dense" / job.name / f"try-{try_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                "[holdout-dense] evaluating "
                f"{job.env_id} seed={seed} try={try_idx}/{num_tries} dir={run_dir}"
            )

            start = time.time()
            solve_rate = 0.0
            reward_hash = ""
            emitted_code = ""
            artifacts: Dict[str, str] = {}
            try:
                reward_generator = RewardGenerator(
                    constraints_text=prompt_text,
                    lm=base_lm,
                    max_sanitize_attempts=5,
                )
                result = run_training_with_reward(
                    reward_generator,
                    output_dir=str(run_dir),
                    config_override={
                        k: v for k, v in budgeted_cfg.items() if k != "name"
                    },
                    plot_training_curves=plot_training_curves,
                    reward_mode="dense",
                )
                gt_eval = result.ground_truth_eval or {}
                emitted_code = result.emitted_reward_code or ""
                returns = gt_eval.get("returns") or []
                successes = gt_eval.get("successes")
                if successes is None:
                    successes = sum(1 for r in returns if r > 0.0)
                total_eps = len(returns)
                solve_rate = (
                    float(successes) / float(total_eps)
                    if total_eps
                    else float(gt_eval.get("success_rate", 0.0) or 0.0)
                )
                if total_eps == 0 and gt_eval.get("success_rate") is None:
                    sparse_curve = to_float_list(
                        result.train_info.get("loss_info", {}).get(
                            "eval/ground_truth_returns_mean"
                        )
                    )
                    solve_rate = float(sparse_curve[-1]) if sparse_curve else 0.0
                reward_hash = hashlib.sha256(emitted_code.encode("utf-8")).hexdigest()[
                    :16
                ]
                artifacts = dict(result.artifacts)
                reward_records.append(
                    {
                        "job_name": job.name,
                        "env_id": job.env_id,
                        "prompt_sha16": prompt_hash,
                        "try_idx": try_idx,
                        "train_seed": budgeted_cfg.get("train_seed"),
                        "eval_seed": budgeted_cfg.get("eval_seed"),
                        "reward_hash": reward_hash,
                        "reward_code": emitted_code,
                        "reward_code_path": artifacts.get("dense_reward_path"),
                        "solve_rate": solve_rate,
                        "config": {
                            k: v for k, v in budgeted_cfg.items() if k != "name"
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover - eval best-effort
                print(f"[holdout-dense] FAILED {job.name} try={try_idx}: {exc}")
                reward_records.append(
                    {
                        "job_name": job.name,
                        "env_id": job.env_id,
                        "prompt_sha16": prompt_hash,
                        "try_idx": try_idx,
                        "train_seed": budgeted_cfg.get("train_seed"),
                        "eval_seed": budgeted_cfg.get("eval_seed"),
                        "solve_rate": solve_rate,
                        "error": str(exc),
                    }
                )
            elapsed = time.time() - start
            elapsed_total += elapsed
            solve_rates.append(solve_rate)
            reward_hashes.append(reward_hash)
            reward_code_paths.append(artifacts.get("dense_reward_path"))
            run_dirs.append(str(run_dir))

        solve_rate_mean = float(np.mean(solve_rates)) if solve_rates else 0.0
        solve_rate_std = (
            float(np.std(solve_rates, ddof=1)) if len(solve_rates) > 1 else 0.0
        )
        results[job.name] = {
            "env_id": job.env_id,
            "solve_rates": solve_rates,
            "solve_rate_mean": solve_rate_mean,
            "solve_rate_std": solve_rate_std,
            "reward_hashes": reward_hashes,
            "reward_code_paths": reward_code_paths,
            "run_dirs": run_dirs,
            "train_seed": budgeted_cfg.get("train_seed"),
            "eval_seed": budgeted_cfg.get("eval_seed"),
            "elapsed_sec": elapsed_total,
            "num_tries": num_tries,
        }
        per_env_means.append(solve_rate_mean)

    mean_solve_rate = float(np.mean(per_env_means)) if per_env_means else 0.0
    if reward_records:
        out_path = logs_root / "holdout_reward_functions.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in reward_records) + "\n",
            encoding="utf-8",
        )
        print(f"[holdout-dense] wrote reward functions to {out_path}")
    return results, mean_solve_rate


def extract_solve_rate(result: TrainingResult) -> float:
    """Compute a solve rate from a training result with robust fallbacks.

    This helper centralizes the logic for extracting solve rates from a
    TrainingResult so single-env comparisons and GEPA metrics stay consistent.
    It is needed because ground-truth evaluation metadata can be partially
    populated depending on evaluator settings, and it differs from per-call
    solve-rate snippets by also falling back to the sparse-return curve when
    no explicit success counters are available.
    """

    gt_eval = result.ground_truth_eval or {}
    returns = gt_eval.get("returns") or []
    successes = gt_eval.get("successes")
    if successes is None:
        successes = sum(1 for r in returns if r > 0.0)
    total_eps = len(returns)
    if total_eps:
        return float(successes) / float(total_eps)
    if gt_eval.get("success_rate") is not None:
        return float(gt_eval.get("success_rate", 0.0))
    sparse_curve = (
        result.train_info.get("loss_info", {}).get("eval/ground_truth_returns_mean")
        if isinstance(result.train_info, Mapping)
        else None
    )
    if sparse_curve is None:
        return 0.0
    return float(sparse_curve[-1]) if sparse_curve else 0.0


def evaluate_single_env_dense(
    *,
    job: EnvJob,
    prompt_text: str,
    base_lm: Any,
    logs_root: Path,
    config_clamper: Callable[[Mapping[str, Any]], Tuple[Dict[str, Any], List[str]]],
    plot_training_curves: bool,
) -> Dict[str, Any]:
    """Evaluate the optimized prompt on the single-env test configuration.

    This helper runs the dense reward generation pipeline using the optimized
    prompt on the locked single-env test job so we can compare it directly
    against the sparse baseline within the same GEPA run. It is needed because
    the main GEPA loop only evaluates candidates during optimization, while the
    final prompt should be tested once with fixed seeds. It differs from
    `evaluate_dense_on_jobs` by targeting exactly one job, keeping the original
    train/eval seeds, and returning a compact payload tailored for logging and
    end-of-run summaries. When `plot_training_curves` is enabled, it emits the
    dense-vs-sparse plot for this single run; otherwise it skips plotting.
    """

    job_cfg_raw = job.to_config()
    budgeted_cfg, budget_notes = config_clamper(job_cfg_raw)
    budgeted_cfg["train_seed"] = job.train_seed
    budgeted_cfg["eval_seed"] = job.eval_seed

    run_stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = logs_root / "compare-single-env" / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if budget_notes:
        print(
            f"[single-env-compare] budget clamp {job.name}: " + "; ".join(budget_notes)
        )
    print(
        "[single-env-compare] running dense eval for "
        f"{job.env_id} train_seed={job.train_seed} eval_seed={job.eval_seed} "
        f"dir={run_dir}"
    )

    reward_generator = RewardGenerator(
        constraints_text=prompt_text,
        lm=base_lm,
        max_sanitize_attempts=5,
    )
    start = time.time()
    result = run_training_with_reward(
        reward_generator,
        output_dir=str(run_dir),
        config_override={k: v for k, v in budgeted_cfg.items() if k != "name"},
        plot_training_curves=plot_training_curves,
        reward_mode="dense",
    )
    elapsed = time.time() - start
    solve_rate = extract_solve_rate(result)
    emitted_code = result.emitted_reward_code or ""
    reward_hash = hashlib.sha256(emitted_code.encode("utf-8")).hexdigest()[:16]
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    reward_path = (
        result.artifacts.get("dense_reward_path") if result.artifacts else None
    )

    return {
        "env_id": job.env_id,
        "benchmark_id": job.benchmark_id,
        "train_seed": job.train_seed,
        "eval_seed": job.eval_seed,
        "prompt_sha16": prompt_hash,
        "reward_sha16": reward_hash,
        "solve_rate": solve_rate,
        "elapsed_sec": elapsed,
        "run_dir": str(run_dir),
        "reward_code_path": reward_path,
    }


def write_holdout_bar_plots(
    *,
    dense_results: Mapping[str, Mapping[str, Any]],
    sparse_baselines: Mapping[str, Mapping[str, Any]],
    logs_root: Path,
    num_tries: int,
) -> List[Path]:
    """Create holdout bar charts with error bars for dense solve-rate variance.

    This plotting helper renders two figures: per-environment dense vs sparse
    solve-rate bars (with dense error bars) and an aggregate bar chart with the
    dense mean and its variability across reward generations. It is needed to
    make the stochasticity in reward synthesis visible at a glance, and it
    differs from the training-curve plot by summarizing holdout performance
    across environments instead of showing GEPA optimization over time.
    """

    if num_tries < 1:
        raise ValueError("num_tries must be >= 1")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"[holdout-plots] matplotlib unavailable; skipping plots ({exc})")
        return []

    envs = sorted(dense_results.keys(), key=lambda k: dense_results[k].get("env_id", k))
    dense_means = [float(dense_results[e].get("solve_rate_mean", 0.0)) for e in envs]
    dense_stds = [float(dense_results[e].get("solve_rate_std", 0.0)) for e in envs]
    sparse_vals = [
        float((sparse_baselines.get(e) or {}).get("solve_rate", 0.0)) for e in envs
    ]

    dense_try_means: List[float] = []
    for try_idx in range(num_tries):
        try_vals: List[float] = []
        for env_id in envs:
            rates = dense_results.get(env_id, {}).get("solve_rates", [])
            if try_idx < len(rates):
                try_vals.append(float(rates[try_idx]))
        if try_vals:
            dense_try_means.append(float(np.mean(try_vals)))

    paths: List[Path] = []

    if envs:
        x = np.arange(len(envs))
        width = 0.35
        dense_label = f"Dense (optimized, n={num_tries})"
        plt.figure(figsize=(12, 5))
        plt.bar(
            x - width / 2,
            dense_means,
            width,
            label=dense_label,
            yerr=dense_stds,
            capsize=4,
        )
        plt.bar(x + width / 2, sparse_vals, width, label="Sparse baseline")
        plt.xticks(x, envs, rotation=45, ha="right")
        plt.ylabel("Solve rate")
        plt.title(
            "Holdout solve rate by environment "
            f"(mean +/- std across reward generations, n={num_tries})"
        )
        plt.ylim(0, 1.05)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        out_path = logs_root / "holdout_solve_rates_by_env.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)

    if dense_means or sparse_vals:
        dense_aggregate_mean = (
            float(np.mean(dense_try_means)) if dense_try_means else 0.0
        )
        dense_aggregate_std = (
            float(np.std(dense_try_means, ddof=1)) if len(dense_try_means) > 1 else 0.0
        )
        labels = [f"Dense (optimized, n={num_tries})", "Sparse baseline"]
        means = [
            dense_aggregate_mean,
            float(np.mean(sparse_vals)) if sparse_vals else 0.0,
        ]
        plt.figure(figsize=(6, 4))
        x = np.arange(len(labels))
        plt.bar(
            x,
            means,
            width=0.6,
            color=["#1f77b4", "#ff7f0e"],
            yerr=[dense_aggregate_std, 0.0],
            capsize=4,
        )
        plt.xticks(x, labels, rotation=15, ha="right")
        plt.ylabel("Solve rate")
        plt.title(
            "Holdout aggregate solve rate "
            f"(mean +/- std across reward generations, n={num_tries})"
        )
        plt.ylim(0, 1.05)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = logs_root / "holdout_solve_rate_aggregate.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)

    for p in paths:
        print(f"[holdout-plots] wrote {p}")
    return paths


def run_batch() -> None:
    """Run the integrated GEPA + on-policy RL loop end to end.

    This is the orchestration entrypoint that wires CLI flags into prompt
    loading, LLM configuration, sparse baseline caching, GEPA optimization,
    room-count based environment filtering, holdout evaluation, and artifact
    persistence. It is needed to keep the experiment flow in one place and
    differs from lower-level training helpers by coordinating multiple runs and
    stateful outputs while enforcing top-level fail-fast validation (including
    empty job sets after optional room filtering).
    """
    args = parse_args()
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "gepa_runs"
    logs_root.mkdir(exist_ok=True)

    model_name = args.llm
    if args.llm_provider == "gemini" and model_name == DEFAULT_MODEL_ALIAS:
        model_name = DEFAULT_GEMINI_MODEL

    env_grid_path = args.env_grid.expanduser().resolve()
    selected_room_counts = (
        sorted(set(args.room_count)) if args.room_count is not None else None
    )
    if args.test_single_env:
        jobs = build_single_env_job()
        eval_jobs: List[EnvJob] = []
        print(
            "[run_reward_batch] single-env test enabled: "
            f"{SINGLE_ENV_ID} train_seed={SINGLE_ENV_TRAIN_SEED} eval_seed={SINGLE_ENV_EVAL_SEED}"
        )
    else:
        jobs, eval_jobs = load_env_grid(env_grid_path)
        # Use all jobs from the grid; users can edit the YAML to reduce set.
    if selected_room_counts is not None:
        jobs = filter_jobs_by_room_count(
            jobs,
            selected_room_counts,
            section_name="training jobs",
        )

    constraints_text, prompt_state, prompt_meta = load_prompt_payload(state_root)
    if args.llm_provider == "portkey":
        reward_lm = configure_portkey_lm(
            model_alias=model_name,
            temperature=args.reward_llm_temp,
        )
        reflection_lm = configure_portkey_lm(
            model_alias=model_name,
            temperature=args.reflection_llm_temp,
        )
    else:
        reward_lm = configure_gemini_lm(
            model_name=model_name,
            temperature=args.reward_llm_temp,
        )
        reflection_lm = configure_gemini_lm(
            model_name=model_name,
            temperature=args.reflection_llm_temp,
        )
    reflection_module = create_reward_reflection_module(lm=reflection_lm)
    dspy.configure(lm=reflection_lm)
    dspy.settings.configure(provide_traceback=True)
    print(f"[run_reward_batch] provide_traceback={dspy.settings.provide_traceback}")

    wandb_run = None
    rl_runs_table = None
    sparse_baselines: Dict[str, Dict[str, Any]] = {}
    sparse_baseline_mean: float = 0.0
    last_wandb_step = -1
    single_env_compare: Optional[Dict[str, Any]] = None
    single_env_console_summary: Optional[str] = None

    if wandb is not None and not os.environ.get("WANDB_DISABLED"):
        # Match W&B project to the selected model alias; sanitize to allowed chars.
        safe_model_alias = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_name).strip("-")
        default_project = safe_model_alias or "llm-desparsifier"
        if args.test_single_env:
            default_project = f"single_env_test-{default_project}"
        wandb_project = os.environ.get("WANDB_PROJECT", default_project)
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                name=f"gepa-{state_root.name}",
                config={
                    "state_root": str(state_root),
                    "env_grid": str(env_grid_path),
                    "max_gepa_iterations": args.max_gepa_iterations,
                    "llm_provider": args.llm_provider,
                    "llm_model": model_name,
                    "test_single_env": bool(args.test_single_env),
                    "single_env_id": SINGLE_ENV_ID if args.test_single_env else None,
                    "single_env_train_seed": SINGLE_ENV_TRAIN_SEED
                    if args.test_single_env
                    else None,
                    "single_env_eval_seed": SINGLE_ENV_EVAL_SEED
                    if args.test_single_env
                    else None,
                    "deterministic_envs": bool(args.deterministic_envs),
                    "reward_llm_temp": args.reward_llm_temp,
                    "reflection_llm_temp": args.reflection_llm_temp,
                    "plot_training_curves": bool(args.plot_training_curves),
                    "room_count": selected_room_counts,
                },
            )
            rl_runs_table = wandb.Table(
                columns=[
                    "rl_run_id",
                    "env_id",
                    "env_text",
                    "prompt_text",
                    "reward_code_sha16",
                    "reward_code",
                    "solve_rate",
                    "sparse_baseline",
                    "feedback",
                    "sanitizer_feedback",
                    "run_dir",
                ],
                log_mode="MUTABLE",
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[wandb] init failed, continuing without logging: {exc}")
            wandb_run = None
            rl_runs_table = None

    def log_wandb(payload: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        """Log a payload to W&B while enforcing monotonic step updates.

        This closure standardizes how GEPA metrics advance the W&B step counter
        so plots stay aligned with GEPA iterations. It is needed because
        different call sites may log at explicit steps or rely on auto-steps,
        and it differs from `safe_wandb_log` by tracking `last_wandb_step` and
        ensuring steps only move forward.
        """
        nonlocal last_wandb_step
        if wandb_run is None:
            return
        if step is None:
            step = last_wandb_step + 1
        else:
            step = max(step, last_wandb_step)
        last_wandb_step = step
        safe_wandb_log(wandb_run, payload, step=step)

    last_stage_path = logs_root / "last_stage.json"

    def write_stage_heartbeat(payload: Mapping[str, Any]) -> None:
        """Persist the most recent stage event to disk as a debugging heartbeat.

        This helper writes a small JSON file that records the latest stage and
        timing metadata so operators can inspect progress even if stdout/W&B
        logs stall or are delayed. It is needed because GEPA runs can freeze
        mid-stage with no further output, and it differs from the console logs
        by keeping only the most recent event in a deterministic location.
        """
        try:
            last_stage_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - best-effort heartbeat
            _emit_log_line(
                {
                    "event": "stage_heartbeat_error",
                    "path": str(last_stage_path),
                    "error": str(exc),
                }
            )

    def log_event(
        *,
        stage: str,
        event: str,
        job: Optional[EnvJob] = None,
        prompt_text: Optional[str] = None,
        run_dir: Optional[Path] = None,
        start_time: Optional[float] = None,
        attempt_idx: Optional[int] = None,
        max_attempts: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
        mirror_wandb: bool = True,
    ) -> None:
        """Emit a structured GEPA stage log and update the heartbeat file.

        This helper builds a compact, JSON-serializable payload containing
        stage metadata (job, prompt hash, seeds, elapsed time) and prints it to
        stdout so it appears in the W&B logs tab. It is needed to pinpoint
        which step of the GEPA loop is blocking, and it differs from ad-hoc
        `print` statements by including consistent fields and writing the same
        payload to `last_stage.json` for out-of-band inspection.
        """
        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "stage": stage,
            "event": event,
            "stage_id": f"{stage}:{event}",
        }
        if start_time is not None:
            payload["elapsed_sec"] = round(time.time() - start_time, 4)
        if job is not None:
            payload["job_name"] = job.name
            payload["env_id"] = job.env_id
        if prompt_text is not None:
            prompt_clean = prompt_text.strip()
            if prompt_clean:
                payload["prompt_len"] = len(prompt_clean)
                payload["prompt_sha16"] = hashlib.sha256(
                    prompt_clean.encode("utf-8")
                ).hexdigest()[:16]
        if run_dir is not None:
            payload["run_dir"] = str(run_dir)
        if attempt_idx is not None:
            payload["attempt_idx"] = attempt_idx
        if max_attempts is not None:
            payload["max_attempts"] = max_attempts
        if extra:
            payload.update(dict(extra))
        _emit_log_line(payload)
        write_stage_heartbeat(payload)
        if mirror_wandb and wandb_run is not None and last_wandb_step >= 0:
            wandb_payload: Dict[str, Any] = {
                "gepa/last_stage": stage,
                "gepa/last_event": event,
            }
            if "elapsed_sec" in payload:
                wandb_payload["gepa/last_stage_elapsed_sec"] = payload["elapsed_sec"]
            safe_wandb_log(wandb_run, wandb_payload, step=last_wandb_step)

    def clamp_with_determinism(
        job_cfg: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], List[str]]:
        """Clamp per-job budgets while optionally forcing deterministic rulesets.

        This helper preserves the existing runtime caps but guarantees that the
        deterministic ruleset flag is propagated consistently to baselines,
        GEPA candidates, and holdout checks when enabled via CLI. It differs
        from `clamp_job_budget` by injecting `deterministic_rulesets` into the
        returned config without altering any other knobs.
        """
        cfg, notes = clamp_job_budget(job_cfg)
        if args.deterministic_envs:
            cfg["deterministic_rulesets"] = True
        return cfg, notes

    baseline_budget_signature = {
        "total_timesteps": MAX_TOTAL_TIMESTEPS,
        "num_envs": MAX_NUM_ENVS,
        "eval_num_envs": MAX_EVAL_ENVS,
        "eval_num_episodes": MAX_EVAL_EPISODES,
        "deterministic_envs": bool(args.deterministic_envs),
    }
    baseline_json_path = DEFAULT_BASELINE_JSON
    if args.test_single_env:
        baseline_json_path = state_root / "sparse_baseline.single_env.json"

    sparse_baselines, sparse_baseline_mean = ensure_sparse_baseline(
        jobs,
        logs_root=logs_root,
        baseline_json_path=baseline_json_path,
        log_wandb=log_wandb,
        env_grid_path=None if args.test_single_env else env_grid_path,
        state_root=state_root,
        config_clamper=clamp_with_determinism,
        budget_signature=baseline_budget_signature,
    )

    program = PromptOnlyProgram(constraints_text, prompt_state=prompt_state)
    trainset = build_examples(jobs, constraints_text)  # on-policy: no static holdout
    num_jobs = len(trainset)

    run_counter = 1
    metric_call_idx = 0
    gepa_iteration_idx = 0
    gepa_iteration_solve_rates: List[float] = []
    iteration_scores_by_prompt: Dict[str, List[float]] = {}
    iteration_step_by_prompt: Dict[str, int] = {}
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
        """Score a GEPA prompt candidate by running the full RL training loop.

        This metric orchestrates reward synthesis, PPO training, sparse
        evaluation, and reflection generation to produce a solve-rate score and
        textual feedback for GEPA. It is needed because GEPA optimizes the
        prompt based on on-policy training outcomes rather than static datasets.
        It differs from other evaluation helpers by caching scores for
        predictor-feedback requests and by emitting structured logs and
        heartbeat updates so long-running stages are observable.
        """
        nonlocal run_counter, metric_call_idx, gepa_iteration_idx
        metric_start = time.time()
        training_curve_path = logs_root / "training_curve.png"
        failsafe_score = 0.0
        candidate_prompt = getattr(prediction, "prompt_text", None)
        if not isinstance(candidate_prompt, str) or not candidate_prompt.strip():
            candidate_prompt = constraints_text
        prediction_key = id(prediction)

        job = job_from_example(example)
        example_id = job.name
        cache_key = (example_id, candidate_prompt)
        log_event(
            stage="gepa_metric",
            event="call_start",
            job=job,
            prompt_text=candidate_prompt,
            extra={
                "pred_name": pred_name,
                "metric_call_idx": metric_call_idx,
                "example_id": example_id,
            },
        )
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
                log_event(
                    stage="gepa_metric_feedback",
                    event="failed_prediction",
                    job=job,
                    prompt_text=candidate_prompt,
                    extra={"pred_name": pred_name},
                    mirror_wandb=False,
                )
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
                log_event(
                    stage="gepa_metric_feedback",
                    event="missing_cache",
                    job=job,
                    prompt_text=candidate_prompt,
                    extra={"pred_name": pred_name},
                    mirror_wandb=False,
                )
                raise RuntimeError(
                    "Missing cached score for predictor feedback; "
                    "ensure scores are captured during primary metric evaluation."
                )
            feedback_text = format_feedback(
                cached_feedback
                or "Missing cached feedback for predictor; returning score only.",
                pred_name,
                pred_trace,
            )
            log_event(
                stage="gepa_metric_feedback",
                event="return",
                job=job,
                prompt_text=candidate_prompt,
                extra={"pred_name": pred_name, "score": cached_score},
                start_time=metric_start,
                mirror_wandb=False,
            )
            return ScoreWithFeedback(score=cached_score, feedback=feedback_text)
        metric_call_idx += 1
        baseline_solve_rate = sparse_baselines.get(example_id, {}).get("solve_rate")
        baseline_solve_rate_mean = (
            sparse_baseline_mean if sparse_baseline_mean else None
        )

        job_cfg_raw = job.to_config()
        budgeted_cfg, budget_notes = clamp_with_determinism(job_cfg_raw)
        if args.test_single_env:
            seed = job.train_seed
            eval_seed = job.eval_seed
        else:
            seed = derive_seed(example_id, candidate_prompt)
            eval_seed = seed + 1
        budgeted_cfg.setdefault("train_seed", seed)
        budgeted_cfg.setdefault("eval_seed", eval_seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        train_cfg = {k: v for k, v in budgeted_cfg.items() if k != "name"}
        run_id = run_counter
        run_counter += 1
        run_dir = logs_root / f"candidate-{run_id:04d}-{example_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log_event(
            stage="gepa_metric",
            event="candidate_start",
            job=job,
            prompt_text=candidate_prompt,
            run_dir=run_dir,
            extra={
                "run_id": run_id,
                "train_seed": seed,
                "eval_seed": eval_seed,
                "budget_notes": budget_notes,
            },
        )

        if budget_notes:
            print(
                f"[on_policy_metric] budget clamp {example_id}: "
                + "; ".join(budget_notes)
            )
        print(
            f"[on_policy_metric] evaluating {example_id} seed={seed} dir={run_dir.name}"
        )

        env_description = getattr(example, "env_description", None)
        sanitizer_feedback = None
        behavior_summary = "Behavior summary unavailable (run did not produce trajectory diagnostics)."
        start = time.time()
        train_start = time.time()
        train_status = "pending"
        train_error: Optional[str] = None
        log_event(
            stage="train_with_reward",
            event="start",
            job=job,
            prompt_text=candidate_prompt,
            run_dir=run_dir,
            extra={
                "train_seed": seed,
                "eval_seed": eval_seed,
                "reward_mode": "dense",
                "config_summary": {
                    "total_timesteps": train_cfg.get("total_timesteps"),
                    "num_envs": train_cfg.get("num_envs"),
                    "eval_num_envs": train_cfg.get("eval_num_envs"),
                    "eval_num_episodes": train_cfg.get("eval_num_episodes"),
                    "deterministic_rulesets": train_cfg.get("deterministic_rulesets"),
                },
            },
        )
        try:
            reward_generator = RewardGenerator(
                constraints_text=candidate_prompt,
                lm=reward_lm,
                max_sanitize_attempts=5,
            )
            result = run_training_with_reward(
                reward_generator,
                output_dir=str(run_dir),
                config_override=train_cfg,
                plot_training_curves=args.plot_training_curves,
                reward_mode="dense",
            )

            env_description = getattr(
                reward_generator, "last_env_description", None
            ) or getattr(example, "env_description", None)
            if getattr(reward_generator, "last_attempt_history", None):
                try:
                    sanitizer_feedback = reward_generator._build_feedback_block(  # type: ignore[attr-defined]
                        reward_generator.last_attempt_history
                    )
                except Exception:
                    sanitizer_feedback = None
            trajectory_path = None
            if isinstance(result.artifacts, Mapping):
                trajectory_path = result.artifacts.get("eval_trajectory")
            behavior_summary = summarize_trajectory_behavior_from_path(trajectory_path)

            row = build_dataset_row(
                EnvJob.from_mapping(run_id, {**train_cfg, "name": example_id}),
                result,
                env_description=env_description,
                candidate_prompt=candidate_prompt,
                sanitizer_feedback=sanitizer_feedback,
                budget_cfg=train_cfg,
                behavior_summary=behavior_summary,
            )
            row["job_name"] = example_id

            reflection_start = time.time()
            log_event(
                stage="reflection",
                event="start",
                job=job,
                prompt_text=candidate_prompt,
                run_dir=run_dir,
            )
            reflection = build_reward_reflection(
                row,
                reflection_module=reflection_module,
                guidance_text=EUREKA_GUIDANCE,
            )
            log_event(
                stage="reflection",
                event="end",
                job=job,
                prompt_text=candidate_prompt,
                run_dir=run_dir,
                start_time=reflection_start,
                extra={"reflection_len": len(reflection or "")},
            )
            sparse_curve = row.get("sparse_return_curve") or []

            gt_eval = result.ground_truth_eval or {}
            returns = gt_eval.get("returns") or []
            successes = gt_eval.get("successes")
            if successes is None:
                successes = sum(1 for r in returns if r > 0.0)
            total_eps = len(returns)
            solve_rate = (
                float(successes) / float(total_eps)
                if total_eps
                else float(gt_eval.get("success_rate", 0.0))
            )
            # Fallback to the previous metric if eval returns are missing.
            if total_eps == 0 and not gt_eval:
                solve_rate = float(sparse_curve[-1]) if sparse_curve else 0.0

            emitted_code = result.emitted_reward_code or ""
            elapsed = time.time() - start
            train_status = "ok"
        except Exception as exc:
            elapsed = time.time() - start
            print(
                f"[on_policy_metric] failure {example_id} after {elapsed / 60:.2f}m: {exc}"
            )
            failure_feedback = f"Training failed: {exc}"
            if "reward_generator" in locals():
                attempts = getattr(reward_generator, "last_attempt_history", None)
                if attempts:
                    try:
                        extra = reward_generator._build_feedback_block(attempts)  # type: ignore[attr-defined]
                        failure_feedback = (
                            f"{failure_feedback}\n\nSanitizer feedback:\n{extra}"
                        )
                        sanitizer_feedback = extra
                    except Exception:
                        pass
            solve_rate = failsafe_score
            reflection = failure_feedback
            emitted_code = ""
            train_status = "error"
            train_error = str(exc)
            log_event(
                stage="train_with_reward",
                event="error",
                job=job,
                prompt_text=candidate_prompt,
                run_dir=run_dir,
                start_time=train_start,
                extra={"error": train_error},
            )

        reward_hash = hashlib.sha256(emitted_code.encode("utf-8")).hexdigest()[:16]
        log_event(
            stage="train_with_reward",
            event="end",
            job=job,
            prompt_text=candidate_prompt,
            run_dir=run_dir,
            start_time=train_start,
            extra={
                "status": train_status,
                "error": train_error,
                "reward_sha16": reward_hash,
                "solve_rate": solve_rate,
            },
        )
        feedback_text = format_feedback(reflection, pred_name, pred_trace)
        metric_cache[cache_key] = {
            "solve_rate": solve_rate,
            "reflection": reflection,
        }
        score_by_prediction_id[prediction_key] = solve_rate
        feedback_by_prediction_id[prediction_key] = reflection

        iteration_scores = iteration_scores_by_prompt.setdefault(candidate_prompt, [])
        iteration_scores.append(solve_rate)
        iteration_step = iteration_step_by_prompt.setdefault(
            candidate_prompt, gepa_iteration_idx + 1
        )

        if rl_runs_table is not None:
            wandb_table_start = time.time()
            log_event(
                stage="wandb_rl_runs",
                event="log_start",
                job=job,
                prompt_text=candidate_prompt,
                run_dir=run_dir,
                mirror_wandb=False,
            )
            rl_runs_table.add_data(
                run_id,
                job.env_id,
                env_description,
                candidate_prompt,
                reward_hash,
                emitted_code,
                solve_rate,
                baseline_solve_rate,
                feedback_text,
                sanitizer_feedback,
                str(run_dir),
            )
            if wandb_run is not None:
                # Route table logging through the monotonic-step wrapper so
                # downstream heartbeat logs never regress to an older step.
                log_wandb({"gepa/rl_runs": rl_runs_table}, step=iteration_step)
            log_event(
                stage="wandb_rl_runs",
                event="log_end",
                job=job,
                prompt_text=candidate_prompt,
                run_dir=run_dir,
                start_time=wandb_table_start,
                mirror_wandb=False,
            )

        if len(iteration_scores) >= num_jobs:
            iteration_mean = (
                float(np.mean(iteration_scores)) if iteration_scores else 0.0
            )
            gepa_iteration_idx = max(gepa_iteration_idx, iteration_step)
            gepa_iteration_solve_rates.append(iteration_mean)
            baseline_line = sparse_baseline_mean if sparse_baselines else None
            if args.plot_training_curves:
                write_training_curve_png(
                    training_curve_path, gepa_iteration_solve_rates, baseline_line
                )
            payload = {
                "gepa/solve_rate": iteration_mean,
                "gepa/iteration": gepa_iteration_idx,
            }
            if baseline_solve_rate_mean is not None:
                payload["gepa/sparse_baseline_solve_rate"] = baseline_solve_rate_mean
            log_wandb(payload, step=gepa_iteration_idx)
            iteration_scores_by_prompt.pop(candidate_prompt, None)
            iteration_step_by_prompt.pop(candidate_prompt, None)

        print(
            f"[on_policy_metric] solve_rate={solve_rate:.4f} env={example_id} elapsed={elapsed / 60:.2f}m"
        )
        log_event(
            stage="gepa_metric",
            event="call_end",
            job=job,
            prompt_text=candidate_prompt,
            run_dir=run_dir,
            start_time=metric_start,
            extra={"solve_rate": solve_rate, "status": train_status},
        )
        return ScoreWithFeedback(score=solve_rate, feedback=feedback_text)

    compiler = dspy.GEPA(
        metric=on_policy_metric,
        max_metric_calls=args.max_gepa_iterations * len(trainset),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=1,
        track_stats=True,
        num_threads=1,
    )

    optimized_program = compiler.compile(program, trainset=trainset)

    # Materialize and persist the best prompt text.
    best_prompt_text = optimized_program.prompt_generator(
        base_constraints=optimized_program._build_rewrite_prompt()
    )
    best_prompt_path = save_best_prompt_text(state_root, model_name, best_prompt_text)

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
    stats_payload["gepa_solve_rate_series"] = gepa_iteration_solve_rates
    stats_payload["baseline_budget_signature"] = baseline_budget_signature
    stats_payload["max_gepa_iterations"] = args.max_gepa_iterations
    stats_payload["test_single_env"] = bool(args.test_single_env)
    stats_payload["deterministic_envs"] = bool(args.deterministic_envs)
    stats_payload["llm_provider"] = args.llm_provider
    stats_payload["llm_model"] = model_name
    stats_payload["reward_llm_temp"] = args.reward_llm_temp
    stats_payload["reflection_llm_temp"] = args.reflection_llm_temp
    stats_payload["plot_training_curves"] = bool(args.plot_training_curves)
    stats_payload["room_count"] = selected_room_counts
    if args.test_single_env:
        stats_payload["single_env_job"] = {
            "env_id": SINGLE_ENV_ID,
            "benchmark_id": SINGLE_ENV_BENCHMARK,
            "total_timesteps": SINGLE_ENV_TOTAL_TIMESTEPS,
            "train_seed": SINGLE_ENV_TRAIN_SEED,
            "eval_seed": SINGLE_ENV_EVAL_SEED,
        }
        stats_payload["baseline_json_path"] = str(baseline_json_path)

    # Holdout evaluation using the optimized prompt.
    holdout_plot_paths: List[Path] = []
    if args.test_single_env:
        stats_payload["holdout_skipped"] = True
        stats_payload["holdout_skip_reason"] = "single-env test mode"
    else:
        holdout_jobs = eval_jobs
        if not holdout_jobs:
            print(
                "[run_reward_batch] no eval_jobs in env grid; using default holdout list"
            )
            holdout_jobs = build_holdout_jobs()
        if selected_room_counts is not None:
            holdout_jobs = filter_jobs_by_room_count(
                holdout_jobs,
                selected_room_counts,
                section_name="holdout jobs",
            )
        holdout_sparse, holdout_sparse_mean = ensure_holdout_sparse_baselines(
            holdout_jobs,
            logs_root=logs_root,
            baseline_json_path=DEFAULT_BASELINE_JSON,
            log_wandb=log_wandb,
            config_clamper=clamp_with_determinism,
            budget_signature=baseline_budget_signature,
        )
        holdout_dense, holdout_dense_mean = evaluate_dense_on_jobs(
            jobs=holdout_jobs,
            prompt_text=best_prompt_text,
            base_lm=reward_lm,
            logs_root=logs_root,
            config_clamper=clamp_with_determinism,
            num_tries=HOLDOUT_REWARD_TRIES,
            plot_training_curves=args.plot_training_curves,
        )

        stats_payload["holdout_sparse_baselines"] = holdout_sparse
        stats_payload["holdout_sparse_baseline_mean"] = holdout_sparse_mean
        stats_payload["holdout_dense_results"] = holdout_dense
        stats_payload["holdout_dense_mean"] = holdout_dense_mean
        stats_payload["holdout_dense_num_tries"] = HOLDOUT_REWARD_TRIES
        stats_payload["holdout_dense_error_bars"] = "std_across_reward_generations"

        if args.plot_training_curves:
            holdout_plot_paths = write_holdout_bar_plots(
                dense_results=holdout_dense,
                sparse_baselines=holdout_sparse,
                logs_root=logs_root,
                num_tries=HOLDOUT_REWARD_TRIES,
            )

    if args.test_single_env:
        job = jobs[0]
        single_env_compare = evaluate_single_env_dense(
            job=job,
            prompt_text=best_prompt_text,
            base_lm=reward_lm,
            logs_root=logs_root,
            config_clamper=clamp_with_determinism,
            plot_training_curves=args.plot_training_curves,
        )
        baseline_entry = sparse_baselines.get(job.name, {}) if sparse_baselines else {}
        baseline_solve_rate = float(baseline_entry.get("solve_rate", 0.0))
        dense_solve_rate = float(single_env_compare.get("solve_rate", 0.0))
        delta = dense_solve_rate - baseline_solve_rate
        relative = delta / baseline_solve_rate if baseline_solve_rate > 0 else None
        single_env_compare.update(
            {
                "baseline_solve_rate": baseline_solve_rate,
                "delta": delta,
                "relative_improvement": relative,
            }
        )
        stats_payload["single_env_compare"] = dict(single_env_compare)
        log_wandb(
            {
                "compare/dense_solve_rate": dense_solve_rate,
                "compare/sparse_baseline_solve_rate": baseline_solve_rate,
                "compare/delta": delta,
                "compare/relative_improvement": relative,
                "compare/prompt_sha16": single_env_compare.get("prompt_sha16"),
                "compare/reward_sha16": single_env_compare.get("reward_sha16"),
                "compare/elapsed_sec": single_env_compare.get("elapsed_sec"),
            }
        )
        relative_str = f"{relative:.4f}" if relative is not None else "n/a"
        single_env_console_summary = (
            "[run_reward_batch] single-env comparison "
            f"dense={dense_solve_rate:.4f} "
            f"sparse={baseline_solve_rate:.4f} "
            f"delta={delta:.4f} "
            f"relative={relative_str}"
        )

    stats_path.write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"[run_reward_batch] GEPA completed. Active prompt updated at {get_active_prompt_path(state_root)}"
    )
    print(f"[run_reward_batch] GEPA stats written to {stats_path}")
    print(f"[run_reward_batch] Best prompt saved to {best_prompt_path}")

    training_curve_path = logs_root / "training_curve.png"
    baseline_line = sparse_baseline_mean if sparse_baselines else None
    if args.plot_training_curves:
        write_training_curve_png(
            training_curve_path, gepa_iteration_solve_rates, baseline_line
        )

    if wandb_run is not None:
        if rl_runs_table is not None:
            log_wandb({"gepa/rl_runs": rl_runs_table})
        art = wandb.Artifact(f"{state_root.name}-gepa", type="gepa-state")
        _active = get_active_prompt_path(state_root)
        if _active.exists():
            art.add_file(str(_active), name="active_prompt.json")
        if stats_path.exists():
            art.add_file(str(stats_path), name="gepa_stats.json")
        if best_prompt_path.exists():
            art.add_file(str(best_prompt_path), name=best_prompt_path.name)
        if training_curve_path.exists():
            art.add_file(str(training_curve_path), name="training_curve.png")
        for p in holdout_plot_paths:
            if p.exists():
                art.add_file(str(p), name=p.name)
        safe_wandb_log_artifact(wandb_run, art)
        safe_wandb_finish(wandb_run)

    if single_env_console_summary:
        print(single_env_console_summary)


if __name__ == "__main__":
    run_batch()
