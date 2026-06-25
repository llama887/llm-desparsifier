#!/usr/bin/env python3
"""Run GEPA prompt optimization for synthesized A* heuristics.

This script is the supported batch entrypoint for the refactored repository. It
optimizes prompt text that causes an LLM to emit admissible-leaning heuristics,
evaluates those heuristics through multi-seed search, and feeds the resulting
scalar score plus deterministic feedback back into GEPA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from dspy_cache_control import configure_dspy_cache, prepare_dspy_import

prepare_dspy_import("run_heuristic_batch")
import dspy
configure_dspy_cache(dspy, "run_heuristic_batch")
import yaml
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from llm_desparsifier.heuristics import (
    BASE_HEURISTIC_PROMPT,
    DEFAULT_DEEPSEEK_MODEL,
    HEURISTIC_CONTRACT_TEXT,
    HeuristicGenerator,
    aggregate_validation_results,
    build_heuristic_feedback,
    configure_deepseek_lm,
)

# Compatibility for tests or external scripts that still patch the old name.
configure_gemini_lm = configure_deepseek_lm
from llm_desparsifier.search import (
    JAxtarSearchBackend,
    SearchConfig,
    SearchTask,
    build_task_instance,
    mean_job_scores,
    write_json,
    write_text,
)
from llm_desparsifier.utils import get_active_prompt_path, write_active_prompt

_wandb: Any
try:
    import wandb as _wandb
except ImportError:  # pragma: no cover - optional dependency
    _wandb = None

wandb: Any = _wandb

DEFAULT_ENV_GRID = Path("configs/gepa_envs.yaml")
DEFAULT_MAX_PHASE_ITERATIONS = 10
DEFAULT_ASTAR_MAX_NODES = 200_000
DEFAULT_ASTAR_MAX_EXPANSIONS = 100_000
DEFAULT_GLOBAL_EXPERIMENT_SEED = 0
DEFAULT_HOLDOUT_TRIES = 1
DEFAULT_WANDB_PROJECT = "llm-astar"
PHASE_SOLVE_RATE_THRESHOLD = 0.80
PHASE_EARLY_STOP_PATIENCE = 3
CURRICULUM_PHASE_TARGET_JOB_COUNTS = (3, 7, 11)


@dataclass(frozen=True)
class EnvJob:
    """Search-native environment job definition loaded from the YAML grid.

    This dataclass describes one `(env_id, benchmark_id)` evaluation unit
    together with its multi-seed training and holdout semantics. It is needed
    because the heuristic-only runner no longer accepts PPO-shaped job configs,
    and it differs from the previous job schema by centering search budgets and
    seed sets rather than training timesteps.
    """

    name: str
    env_id: str
    benchmark_id: str
    num_gepa_eval_seeds: int
    holdout_seeds: tuple[int, ...]
    deterministic_rulesets: bool
    fixed_ruleset_seed: int | None
    astar_max_nodes: int
    astar_max_expansions: int

    @classmethod
    def from_mapping(
        cls,
        index: int,
        payload: Mapping[str, Any],
        *,
        default_astar_max_nodes: int,
        default_astar_max_expansions: int,
    ) -> "EnvJob":
        """Create an `EnvJob` from one YAML mapping.

        This constructor centralizes schema validation for both training and
        holdout job entries. It is needed because the grid file is the main
        source of truth for the heuristic-only pipeline, and every downstream
        component assumes the parsed job already has explicit search budgets,
        seed lists, and ruleset-selection semantics. It differs from the legacy
        parser by requiring multi-seed search fields instead of PPO timesteps,
        and by defaulting `deterministic_rulesets` to `False` so newly added
        heuristic-search jobs explore diverse benchmark tasks unless a caller
        explicitly opts into one fixed canonical ruleset.
        """

        holdout_seeds_raw = payload.get("holdout_seeds", [])
        if not isinstance(holdout_seeds_raw, list):
            raise ValueError("holdout_seeds must be a list of integers")
        return cls(
            name=str(payload.get("name") or f"job-{index}"),
            env_id=str(payload["env_id"]),
            benchmark_id=str(payload.get("benchmark_id", "trivial-1m")),
            num_gepa_eval_seeds=int(payload.get("num_gepa_eval_seeds", 4)),
            holdout_seeds=tuple(int(seed) for seed in holdout_seeds_raw),
            deterministic_rulesets=bool(payload.get("deterministic_rulesets", False)),
            fixed_ruleset_seed=(
                None
                if payload.get("fixed_ruleset_seed") is None
                else int(payload["fixed_ruleset_seed"])
            ),
            astar_max_nodes=int(payload.get("astar_max_nodes", default_astar_max_nodes)),
            astar_max_expansions=int(
                payload.get("astar_max_expansions", default_astar_max_expansions)
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the job into a JSON-friendly config mapping.

        This helper attaches the full job config onto DSPy examples and run
        records. It is needed because the metric reconstructs jobs from examples
        during GEPA evaluation, and it differs from `asdict(...)` by keeping the
        intended public field set explicit.
        """

        return {
            "name": self.name,
            "env_id": self.env_id,
            "benchmark_id": self.benchmark_id,
            "num_gepa_eval_seeds": self.num_gepa_eval_seeds,
            "holdout_seeds": list(self.holdout_seeds),
            "deterministic_rulesets": self.deterministic_rulesets,
            "fixed_ruleset_seed": self.fixed_ruleset_seed,
            "astar_max_nodes": self.astar_max_nodes,
            "astar_max_expansions": self.astar_max_expansions,
        }


@dataclass(frozen=True)
class IterationSummary:
    """Store the aggregate metrics for one fully evaluated prompt candidate.

    The curriculum runner advances, early-stops, and checkpoints based on
    phase-level iteration summaries rather than raw per-job rows. This record is
    needed because GEPA metric calls still happen one job at a time, and it
    differs from the per-job payloads written to W&B by representing the single
    score line that controls curriculum progression and best-prompt tracking.
    """

    prompt_text: str
    prompt_sha16: str
    phase: int
    phase_iteration: int
    global_iteration: int
    trainset_size: int
    mean_job_score: float
    mean_solve_rate: float
    mean_expanded_states: float
    mean_generated_states: float
    mean_admissibility_pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary into a JSON-friendly mapping.

        The runner stores curriculum progress in JSON checkpoints after every
        phase iteration. This helper is needed because `active_prompt.json` and
        `gepa_stats.json` both need the same fields, and it differs from
        `__dict__` by explicitly preserving the stable external schema.
        """

        return {
            "prompt_sha16": self.prompt_sha16,
            "phase": self.phase,
            "phase_iteration": self.phase_iteration,
            "global_iteration": self.global_iteration,
            "trainset_size": self.trainset_size,
            "mean_job_score": self.mean_job_score,
            "mean_solve_rate": self.mean_solve_rate,
            "mean_expanded_states": self.mean_expanded_states,
            "mean_generated_states": self.mean_generated_states,
            "mean_admissibility_pass_rate": self.mean_admissibility_pass_rate,
        }


@dataclass(frozen=True)
class HoldoutComparisonSummary:
    """Store one end-of-run holdout baseline comparison bundle.

    The heuristic runner now reports three different holdout policies after
    training finishes: the best learned prompt, the original base prompt, and
    blind A* with no heuristic. This record is needed because JSON stats,
    plotting, and W&B logging all consume the same aggregate metrics, and it
    differs from raw `evaluate_jobs(...)` output by collapsing one policy down
    to the summary fields that comparisons care about.
    """

    label: str
    dir_name: str
    results: dict[str, Any]
    job_score_mean: float
    solve_rate_mean: float
    admissibility_pass_rate_mean: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the comparison summary into a JSON-friendly mapping.

        This helper preserves the end-of-run holdout comparison schema inside
        `gepa_stats.json`. It is needed because the runner writes these
        summaries both for human inspection and later plotting, and it differs
        from the in-memory dataclass by including the nested per-job results in
        the exact shape expected by downstream consumers.
        """

        return {
            "label": self.label,
            "dir_name": self.dir_name,
            "results": self.results,
            "job_score_mean": self.job_score_mean,
            "solve_rate_mean": self.solve_rate_mean,
            "admissibility_pass_rate_mean": self.admissibility_pass_rate_mean,
        }


def _phase_job_counts(jobs: list[EnvJob]) -> list[int]:
    """Return the explicit cumulative curriculum schedule for the current jobs.

    The heuristic runner no longer uses one phase per environment. This helper
    is needed because the curriculum now advances across coarse job-count
    milestones and must persist those exact boundaries for resume safety, and it
    differs from the old implicit scheme by separating phase identity from the
    number of training jobs already unlocked.
    """

    if not jobs:
        return []
    total_jobs = len(jobs)
    counts: list[int] = []
    for target_count in CURRICULUM_PHASE_TARGET_JOB_COUNTS:
        if target_count < total_jobs:
            counts.append(int(target_count))
    counts.append(total_jobs)
    return sorted(set(counts))


def _phase_schedule(jobs: list[EnvJob]) -> list[list[EnvJob]]:
    """Build the ordered curriculum phases from the active training jobs.

    The curriculum state stores stable phase indices, but runner logic still
    needs the concrete job subset assigned to each phase. This helper is needed
    because phase advancement now follows the fixed ``3/7/11`` milestones, and
    it differs from slicing with ``jobs[:current_phase]`` by materializing the
    exact active-job list for every phase up front.
    """

    return [jobs[:job_count] for job_count in _phase_job_counts(jobs)]


def _print_progress_line(message: str) -> None:
    """Print one concise human-readable progress update.

    The heuristic runner can execute for a long time, so some stdout feedback is
    useful. This helper is needed because the previous structured JSON logging
    was too noisy for interactive monitoring, and it differs from those removed
    logs by emitting only short status lines that highlight curriculum progress.
    """

    print(message, flush=True)


def safe_wandb_log(wandb_run: Any, payload: Mapping[str, Any], **kwargs: Any) -> None:
    """Log a payload to W&B while tolerating finished-run edge cases.

    This helper centralizes defensive W&B logging for the heuristic runner. It
    is needed because the optimization loop can be long-lived and network-backed
    logging occasionally outlives the active run object, and it differs from
    calling `wandb.log` directly by short-circuiting finished runs.
    """

    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(wandb_run, "finished", False):
        return
    try:
        wandb_run.log(payload, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        if wandb is not None and isinstance(exc, wandb.errors.UsageError) and "finished" in str(exc):
            return
        raise


def safe_wandb_finish(wandb_run: Any) -> None:
    """Finish a W&B run safely.

    This helper guards against late-finish usage errors. It is needed because
    the runner should close W&B cleanly even after exceptions, and it differs
    from calling `finish()` directly by treating already-finished runs as a
    no-op.
    """

    if wandb_run is None:
        return
    if getattr(wandb_run, "_is_finished", False) or getattr(wandb_run, "finished", False):
        return
    try:
        wandb_run.finish(quiet=True)
    except Exception as exc:  # pragma: no cover - defensive
        if wandb is not None and isinstance(exc, wandb.errors.UsageError) and "finished" in str(exc):
            return
        raise


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for heuristic-only GEPA runs.

    This parser defines the supported user-facing interface of the new runner.
    It is needed because the repository now has a single search-only experiment
    path, and it differs from the removed reward runner by exposing no RL or
    reward-specific switches.
    """

    parser = argparse.ArgumentParser(
        description="Run GEPA to optimize prompts for A* heuristic synthesis"
    )
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument(
        "--max-phase-iterations",
        type=int,
        default=DEFAULT_MAX_PHASE_ITERATIONS,
    )
    parser.add_argument("--llm", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--astar-max-nodes", type=int, default=DEFAULT_ASTAR_MAX_NODES)
    parser.add_argument(
        "--astar-max-expansions",
        type=int,
        default=DEFAULT_ASTAR_MAX_EXPANSIONS,
    )
    parser.add_argument("--deterministic-envs", action="store_true")
    parser.add_argument("--room-count", type=int, action="append", default=None)
    return parser.parse_args()


def load_prompt_payload(
    state_root: Path,
) -> tuple[str, Optional[dict[str, Any]], dict[str, Any]]:
    """Load the active heuristic prompt payload or fall back to the base prompt.

    This helper keeps prompt state persistence compatible across repeated GEPA
    runs. It is needed because the new runner still reads and writes
    `active_prompt.json`, and it differs from the old reward version by storing
    heuristic prompt text rather than reward-constraint text.
    """

    prompt_path = get_active_prompt_path(state_root)
    if prompt_path.exists():
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        text = payload.get("base_prompt_text") or payload.get("constraints_text")
        if isinstance(text, str) and text.strip():
            return text, payload.get("prompt_state"), {
                "source": "active_prompt",
                "path": str(prompt_path),
            }
    return BASE_HEURISTIC_PROMPT, None, {"source": "default_heuristic_prompt"}


def load_active_prompt_payload(state_root: Path) -> Optional[dict[str, Any]]:
    """Return the raw persisted active-prompt payload when one exists.

    The curriculum refactor stores additional checkpoint metadata alongside the
    existing prompt state. This helper is needed because `load_prompt_payload`
    intentionally returns only the prompt-facing fields, and it differs from
    that narrower helper by exposing the full JSON payload for resume logic.
    """

    prompt_path = get_active_prompt_path(state_root)
    if not prompt_path.exists():
        return None
    return json.loads(prompt_path.read_text(encoding="utf-8"))


def save_best_prompt_text(state_root: Path, model_alias: str, prompt_text: str) -> Path:
    """Persist the best prompt text to a stable `.txt` artifact.

    This helper gives users a quick way to inspect or reuse the final prompt.
    It is needed because `active_prompt.json` also stores optimizer state, and
    it differs from the JSON payload by saving only the human-readable prompt.
    """

    safe_model_alias = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_alias).strip("-") or "model"
    output_path = state_root / f"{safe_model_alias}.txt"
    output_path.write_text(prompt_text.strip() + "\n", encoding="utf-8")
    return output_path


def compute_prompt_text(
    *,
    base_prompt_text: str,
    prompt_state: Optional[Mapping[str, Any]],
) -> str:
    """Materialize the rewritten prompt text for a persisted prompt state.

    The curriculum runner checkpoints the best prompt state after every phase
    iteration and needs the concrete rewritten prompt text for holdout
    evaluation, human-readable artifacts, and summary records. This helper is
    needed because the persisted state is DSPy-internal, and it differs from the
    compile path by reconstructing prompt text without running a new search
    iteration.
    """

    program = PromptOnlyProgram(base_prompt_text, prompt_state=prompt_state)
    prompt_text = program.prompt_generator(base_prompt=program._build_rewrite_prompt())
    return str(prompt_text)


def load_env_grid(
    env_grid_path: Path,
    *,
    default_astar_max_nodes: int,
    default_astar_max_expansions: int,
) -> tuple[list[EnvJob], list[EnvJob]]:
    """Load training and holdout jobs from the heuristic env-grid YAML.

    This helper parses the refactored grid schema into `EnvJob` objects. It is
    needed because both GEPA training and holdout evaluation share the same
    source of truth, and it differs from the previous loader by expecting
    search-native fields rather than PPO budgets.
    """

    data = yaml.safe_load(env_grid_path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        job_entries = data.get("jobs", [])
        eval_entries = data.get("eval_jobs", [])
    elif isinstance(data, list):
        job_entries = data
        eval_entries = []
    else:
        raise ValueError("Environment grid must be a mapping or list")
    jobs = [
        EnvJob.from_mapping(
            idx,
            payload,
            default_astar_max_nodes=default_astar_max_nodes,
            default_astar_max_expansions=default_astar_max_expansions,
        )
        for idx, payload in enumerate(job_entries)
    ]
    eval_jobs = [
        EnvJob.from_mapping(
            idx,
            payload,
            default_astar_max_nodes=default_astar_max_nodes,
            default_astar_max_expansions=default_astar_max_expansions,
        )
        for idx, payload in enumerate(eval_entries)
    ]
    if not jobs:
        raise ValueError("No training jobs found in env grid")
    return jobs, eval_jobs


def extract_room_count(env_id: str) -> int:
    """Extract the room-count token from an XLand environment id.

    This helper supports runtime filtering over the standard `-R<rooms>-`
    segment. It is needed because users may want to train or report on only a
    subset of layouts, and it differs from ad hoc string splitting by validating
    the expected token shape explicitly.
    """

    match = re.search(r"-R(\d+)-", env_id)
    if match is None:
        raise ValueError(f"Could not parse room count from env_id '{env_id}'")
    return int(match.group(1))


def filter_jobs_by_room_count(
    jobs: list[EnvJob],
    allowed_room_counts: list[int],
    section_name: str,
) -> list[EnvJob]:
    """Filter jobs by allowed room counts.

    This helper applies the same room-count filter to both training and holdout
    jobs. It is needed because the heuristic runner should support the same
    coarse environment slicing as the legacy script, and it differs from manual
    YAML edits by working at runtime for all loaded jobs.
    """

    allowed = set(allowed_room_counts)
    filtered = [job for job in jobs if extract_room_count(job.env_id) in allowed]
    if not filtered:
        raise ValueError(
            f"--room-count filter removed all jobs from {section_name}. "
            f"allowed_room_counts={sorted(allowed)}"
        )
    return filtered


def _stable_seed_rng(*parts: object) -> random.Random:
    """Create a deterministic RNG keyed by structured input parts.

    This helper provides reproducible seed sampling across GEPA metric calls. It
    is needed because training seeds must vary across calls while remaining
    stable for a fixed experiment seed, and it differs from global RNG usage by
    deriving a dedicated keyed generator each time.
    """

    digest = hashlib.blake2b(
        "::".join(str(part) for part in parts).encode("utf-8"),
        digest_size=16,
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def sample_eval_seeds(
    *,
    global_experiment_seed: int,
    metric_call_idx: int,
    job_name: str,
    num_gepa_eval_seeds: int,
) -> list[int]:
    """Sample reproducible fresh evaluation seeds for one GEPA metric call.

    This helper implements the seed randomization semantics from the refactor
    plan. It is needed because GEPA should not overfit to a tiny fixed seed set,
    and it differs from legacy deterministic seed derivation by sampling without
    replacement within each metric call.
    """

    if num_gepa_eval_seeds <= 0:
        raise ValueError("num_gepa_eval_seeds must be > 0")
    rng = _stable_seed_rng(global_experiment_seed, metric_call_idx, job_name)
    population = range(0, 2**31 - 1)
    return rng.sample(population, num_gepa_eval_seeds)


def build_example_payload(job: EnvJob) -> dict[str, Any]:
    """Materialize the prompt inputs attached to one DSPy example.

    This helper constructs the full environment description and heuristic
    contract stored on each training example. It is needed because the new
    program under optimization should reason over actual task text rather than a
    short `env_id | benchmark` string, and it differs from the old builder by
    materializing a concrete ruleset description up front.
    """

    sample_seed = job.holdout_seeds[0] if job.holdout_seeds else DEFAULT_GLOBAL_EXPERIMENT_SEED
    _, _, _, _, _, task_instance = build_task_instance(
        env_id=job.env_id,
        benchmark_id=job.benchmark_id,
        seed=sample_seed,
        deterministic_rulesets=job.deterministic_rulesets,
        fixed_ruleset_seed=job.fixed_ruleset_seed,
    )
    return {
        "env_description": task_instance.ruleset_text,
        "heuristic_contract": HEURISTIC_CONTRACT_TEXT,
        "env_id": job.env_id,
        "benchmark_id": job.benchmark_id,
    }


def build_examples(jobs: list[EnvJob]) -> list[dspy.Example]:
    """Convert environment jobs into DSPy examples for GEPA.

    This helper creates the training set consumed by `dspy.GEPA`. It is needed
    because the optimizer works over examples rather than raw job configs, and
    it differs from the old implementation by including full task descriptions
    and heuristic contract text as model inputs.
    """

    examples: list[dspy.Example] = []
    for job in jobs:
        payload = build_example_payload(job)
        example = dspy.Example(**payload).with_inputs(
            "env_description", "heuristic_contract", "env_id", "benchmark_id"
        )
        example.job_name = job.name
        example.job_config = job.to_config()
        examples.append(example)
    return examples


def job_from_example(example: dspy.Example) -> EnvJob:
    """Reconstruct the originating `EnvJob` from a DSPy example.

    This helper lets the GEPA metric recover the full search config attached to
    an example. It is needed because the metric receives examples rather than
    direct job objects, and it differs from the old version by restoring the
    refactored search-native schema.
    """

    job_config = getattr(example, "job_config", None)
    if not isinstance(job_config, Mapping):
        raise ValueError("Example is missing job_config")
    return EnvJob.from_mapping(
        0,
        job_config,
        default_astar_max_nodes=int(job_config.get("astar_max_nodes", DEFAULT_ASTAR_MAX_NODES)),
        default_astar_max_expansions=int(
            job_config.get("astar_max_expansions", DEFAULT_ASTAR_MAX_EXPANSIONS)
        ),
    )


class PromptOnlyProgram(dspy.Module):
    """DSPy module that rewrites the heuristic prompt text optimized by GEPA.

    This module mirrors the old prompt-only GEPA pattern while switching the
    language and defaults to heuristic synthesis. It is needed because GEPA
    should optimize prompt text rather than heuristic code directly, and it
    differs from the removed reward version by describing admissible A*
    heuristics rather than dense rewards.
    """

    def __init__(
        self,
        base_prompt_text: str,
        prompt_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.base_prompt_text = base_prompt_text
        self.rewrite_preamble = (
            "You are refining prompt instructions for an LLM that writes admissible A* heuristics.\n"
            "- Rewrite the prompt text to improve heuristic quality and safety.\n"
            "- Output only the rewritten prompt text.\n"
            "- Preserve the heuristic contract and admissibility emphasis.\n"
        )

        class PromptSearch(dspy.Signature):
            base_prompt: str = dspy.InputField()
            prompt_text: str = dspy.OutputField(
                desc="Rewritten prompt text for heuristic synthesis"
            )

        class PromptGenerator(dspy.Module):
            def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
                super().__init__()
                self.rewriter = dspy.Predict(PromptSearch)
                if state:
                    self.rewriter.load_state(state)

            def dump_state(self) -> Mapping[str, Any]:
                """Serialize the underlying DSPy predictor state.

                This helper keeps prompt optimizer state persistence stable across
                repeated runs. It is needed because `active_prompt.json` stores
                the GEPA-updated rewriter state, and it differs from the top-level
                program by returning only the nested predictor state.
                """

                return self.rewriter.dump_state()

            def forward(self, base_prompt: str) -> str:
                """Rewrite the base heuristic prompt text.

                This method forwards the composed rewrite prompt to DSPy and
                returns only the rewritten text. It is needed because GEPA
                optimizes prompt text itself, and it differs from the main
                program's `forward(...)` by ignoring example-specific inputs.
                """

                out = self.rewriter(base_prompt=base_prompt)
                return out.prompt_text

        self.prompt_generator = PromptGenerator(prompt_state)

    def _build_rewrite_prompt(self) -> str:
        """Construct the instruction block used for prompt rewriting.

        This helper wraps the persisted base prompt in meta-instructions that
        tell the model to rewrite text rather than emit code. It is needed
        because the GEPA program optimizes the prompt-only stage, and it differs
        from the heuristic synthesis prompt by never mentioning the current task
        instance directly.
        """

        return (
            f"{self.rewrite_preamble}"
            "=== BASE PROMPT START ===\n"
            f"{self.base_prompt_text.strip()}\n"
            "=== BASE PROMPT END ===\n"
            "Return only the rewritten prompt text."
        )

    def forward(
        self,
        env_description: str,
        heuristic_contract: str,
        env_id: str,
        benchmark_id: str,
        constraints: Optional[str] = None,
    ) -> dspy.Prediction:
        """Return the heuristic prompt text GEPA should evaluate.

        This method defines the program under optimization for DSPy. It is
        needed because GEPA expects a module returning the artifact it is trying
        to optimize, and it differs from the heuristic generator by emitting the
        prompt text rather than heuristic source code.
        """

        del env_description, heuristic_contract, env_id, benchmark_id
        rewrite_prompt = self._build_rewrite_prompt()
        prompt_text = constraints or self.prompt_generator(base_prompt=rewrite_prompt)
        return dspy.Prediction(prompt_text=prompt_text)


def _select_replay_seed(seed_results: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose the representative seed result used for replay artifacts.

    This helper prefers solved seeds when selecting which plan and task instance
    should be written to replay-oriented artifacts. It is needed because one job
    directory stores one representative `astar_plan.json`, and it differs from
    arbitrary first-seed selection by preferring a solved demonstration when one
    exists.
    """

    for seed_result in seed_results:
        if bool(seed_result.get("solved", False)):
            return seed_result
    return seed_results[0]


def _zero_heuristic(_ts: Any, _env_params: Any, _ctx: Mapping[str, Any]) -> float:
    """Return the blind-search heuristic value used for the no-heuristic baseline.

    The end-of-run baseline comparison needs a true blind A* policy rather than
    a prompt-generated heuristic. This helper is needed because the search
    backend always expects a heuristic callback, and it differs from synthesized
    heuristics by unconditionally returning the exact lower bound `0.0`.
    """

    return 0.0


def _zero_heuristic_code() -> str:
    """Return a stable source snippet describing the blind-A* heuristic policy.

    The blind baseline should emit artifacts that look like the prompt-based
    runs so replay and debugging remain uniform. This helper is needed because
    those artifacts expect a code payload even when no LLM generation occurs,
    and it differs from `BASE_HEURISTIC_PROMPT` by representing executable
    baseline behavior rather than prompt instructions.
    """

    return (
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    return 0.0\n"
    )


def _evaluate_fixed_heuristic(
    *,
    job: EnvJob,
    seeds: list[int],
    heuristic_fn: Callable[[Any, Any, Mapping[str, Any]], float],
    heuristic_code: str,
    heuristic_validation_payload: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one already-constructed heuristic function over a job's seeds.

    This helper centralizes the shared multi-seed A* execution path used by
    prompt-generated heuristics and the blind-A* baseline. It is needed because
    both modes should write the same replay artifacts and aggregate metrics, and
    it differs from `evaluate_job(...)` by assuming heuristic generation has
    already happened before the search batch begins.
    """

    example_payload = build_example_payload(job)
    env_summary = str(example_payload["env_description"])
    output_dir.mkdir(parents=True, exist_ok=True)

    task_batch: list[SearchTask] = []
    task_instances_by_seed: dict[int, Any] = {}
    for seed in seeds:
        env, env_params, step_fn, root_timestep, _reset_key, task_instance = build_task_instance(
            env_id=job.env_id,
            benchmark_id=job.benchmark_id,
            seed=seed,
            deterministic_rulesets=job.deterministic_rulesets,
            fixed_ruleset_seed=job.fixed_ruleset_seed,
        )
        task_instances_by_seed[seed] = task_instance
        task_batch.append(
            SearchTask(
                env=env,
                env_params=env_params,
                step_fn=step_fn,
                root_timestep=root_timestep,
                task_instance=task_instance,
            )
        )
    backend = JAxtarSearchBackend()
    batch_result = backend.solve_many(
        task_batch=task_batch,
        heuristic_fn=heuristic_fn,
        search_config=SearchConfig(
            max_nodes=job.astar_max_nodes,
            max_expansions=job.astar_max_expansions,
        ),
    )
    aggregated_validation = aggregate_validation_results(
        [seed_result.validation for seed_result in batch_result.seed_results]
    ).to_dict()
    merged_validation = {
        **dict(heuristic_validation_payload),
        **aggregated_validation,
        "contract_violations": list(
            {
                violation
                for seed_result in batch_result.seed_results
                for violation in seed_result.validation.get("contract_violations", [])
            }
        ),
    }
    feedback = build_heuristic_feedback(
        env_summary=env_summary,
        heuristic_code=heuristic_code,
        aggregate_stats=batch_result.aggregate_stats,
        validation_result=merged_validation,
    )
    write_text(output_dir / "heuristic_synthesized.py", heuristic_code)
    write_json(output_dir / "heuristic_validation.json", merged_validation)
    representative_seed = _select_replay_seed(
        [seed_result.to_dict() for seed_result in batch_result.seed_results]
    )
    representative_task_instance = task_instances_by_seed[int(representative_seed["seed"])]
    write_json(
        output_dir / "astar_search_stats.json",
        {
            "env_id": job.env_id,
            "benchmark_id": job.benchmark_id,
            "max_nodes": job.astar_max_nodes,
            "max_expansions": job.astar_max_expansions,
            "job_score": batch_result.job_score,
            "aggregate_stats": batch_result.aggregate_stats,
            "per_seed": [seed_result.to_dict() for seed_result in batch_result.seed_results],
        },
    )
    write_json(
        output_dir / "astar_plan.json",
        {
            "seed": representative_seed["seed"],
            "actions": representative_seed["actions"],
            "action_names": list(
                ["move_forward", "turn_right", "turn_left", "pick_up", "put_down", "toggle"]
            ),
            "replay_complete": bool(representative_seed["solved"]),
            "final_state_summary": {
                "termination_reason": representative_seed["termination_reason"],
                "solution_length": representative_seed["solution_length"],
            },
        },
    )
    write_json(output_dir / "astar_trace.json", representative_seed["search_trace"])
    write_json(
        output_dir / "task_instance.json",
        {
            **representative_task_instance.to_dict(),
            "reset_payload": {"reset_key": representative_task_instance.reset_key},
        },
    )
    write_text(output_dir / "feedback.txt", feedback)
    return {
        "job_score": batch_result.job_score,
        "aggregate_stats": batch_result.aggregate_stats,
        "feedback": feedback,
        "heuristic_validation": merged_validation,
        "heuristic_code": heuristic_code,
    }


def evaluate_no_heuristic_job(
    *,
    job: EnvJob,
    seeds: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate blind A* on one holdout job and write matching artifacts.

    The end-of-run report compares the learned heuristic against a genuine
    no-heuristic baseline. This helper is needed because that baseline should
    share the exact same budget, replay, and scoring pipeline as prompt-based
    runs, and it differs from `evaluate_job(...)` by bypassing LLM synthesis
    entirely.
    """

    return _evaluate_fixed_heuristic(
        job=job,
        seeds=seeds,
        heuristic_fn=_zero_heuristic,
        heuristic_code=_zero_heuristic_code(),
        heuristic_validation_payload={
            "compile_ok": True,
            "sanitizer_errors": [],
            "sanitizer_warnings": [],
            "contract_violations": [],
        },
        output_dir=output_dir,
    )


def evaluate_job(
    *,
    job: EnvJob,
    seeds: list[int],
    prompt_text: str,
    lm: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate one heuristic, evaluate it over many seeds, and write artifacts.

    This helper is the central unit of search-based evaluation. It is needed
    because both the GEPA metric and holdout reporting share the same
    heuristic-generation and multi-seed search path, and it differs from the old
    reward evaluator by compiling one heuristic and reusing it across all seeds
    for the job.
    """

    example_payload = build_example_payload(job)
    env_summary = str(example_payload["env_description"])
    generator = HeuristicGenerator(
        constraints_text=prompt_text,
        heuristic_contract=HEURISTIC_CONTRACT_TEXT,
        lm=lm,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    heuristic_validation_payload: dict[str, Any]
    try:
        heuristic_fn, heuristic_code = generator.generate(
            env_description=env_summary,
            heuristic_contract=HEURISTIC_CONTRACT_TEXT,
        )
        heuristic_validation_payload = {
            "compile_ok": True,
            "sanitizer_errors": [],
            "sanitizer_warnings": [],
            "contract_violations": [],
        }
    except Exception as exc:
        heuristic_code = ""
        heuristic_validation_payload = {
            "compile_ok": False,
            "sanitizer_errors": [str(exc)],
            "sanitizer_warnings": [],
            "contract_violations": [],
        }
        feedback = build_heuristic_feedback(
            env_summary=env_summary,
            heuristic_code="",
            aggregate_stats={
                "num_seeds": len(seeds),
                "solved_count": 0,
                "solve_rate": 0.0,
                "average_expanded_states": job.astar_max_expansions,
                "average_generated_states": 0.0,
                "average_solution_length": 0.0,
                "average_solved_seed_efficiency": 0.0,
                "termination_reasons": {"generation_failed": len(seeds)},
                "job_score": 0.0,
            },
            validation_result=heuristic_validation_payload,
        )
        write_text(output_dir / "heuristic_synthesized.py", heuristic_code)
        write_json(output_dir / "heuristic_validation.json", heuristic_validation_payload)
        write_text(output_dir / "feedback.txt", feedback)
        return {
            "job_score": 0.0,
            "aggregate_stats": {
                "num_seeds": len(seeds),
                "solved_count": 0,
                "solve_rate": 0.0,
                "average_expanded_states": float(job.astar_max_expansions),
                "average_generated_states": 0.0,
                "average_solution_length": 0.0,
                "average_solved_seed_efficiency": 0.0,
                "termination_reasons": {"generation_failed": len(seeds)},
                "job_score": 0.0,
            },
            "feedback": feedback,
            "heuristic_validation": heuristic_validation_payload,
            "heuristic_code": heuristic_code,
        }

    return _evaluate_fixed_heuristic(
        job=job,
        seeds=seeds,
        heuristic_fn=heuristic_fn,
        heuristic_code=heuristic_code,
        heuristic_validation_payload=heuristic_validation_payload,
        output_dir=output_dir,
    )


def _admissibility_pass_rate(validation_payload: Mapping[str, Any]) -> float:
    """Extract the aggregated admissibility pass rate from a validation payload.

    This helper keeps W&B logging call sites concise while handling missing
    validation summaries safely. It is needed because some failure paths only
    produce partial validation payloads, and it differs from direct dict access
    by returning a stable `0.0` fallback.
    """

    summary = validation_payload.get("admissibility_summary")
    if not isinstance(summary, Mapping):
        return 0.0
    try:
        return float(summary.get("admissibility_pass_rate", 0.0))
    except Exception:
        return 0.0


def _next_candidate_run_counter(logs_root: Path) -> int:
    """Return the next stable `candidate-####-*` counter for a state root.

    Curriculum phases can span many independent GEPA compile calls and may also
    resume from a prior interrupted run. This helper is needed because candidate
    artifact directories must remain monotonic across phases and resumes, and it
    differs from an in-memory counter by discovering the next safe value from
    on-disk artifacts.
    """

    pattern = re.compile(r"^candidate-(\d+)-")
    max_counter = 0
    if logs_root.exists():
        for child in logs_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match is not None:
                max_counter = max(max_counter, int(match.group(1)))
    return max_counter + 1


def _phase_gepa_run_dir(*, logs_root: Path, phase: int) -> Path:
    """Return the stable GEPA checkpoint directory for one curriculum phase.

    The curriculum runner now keeps one long-lived GEPA run per phase instead of
    starting a fresh optimizer for every outer-loop iteration. This helper is
    needed because phase resumes must reuse the same on-disk optimizer archive,
    and it differs from candidate artifact directories by storing GEPA's own
    internal checkpoint state rather than per-job evaluation outputs.
    """

    return logs_root / f"phase-{phase:02d}-gepa"


def _phase_gepa_max_metric_calls(*, phase_iteration: int, trainset_size: int) -> int:
    """Return the cumulative GEPA budget that should be active for this phase.

    DSPy's GEPA resume flow continues optimizing until the persisted run reaches
    the provided `max_metric_calls` budget. This helper is needed because the
    runner wants each outer-loop iteration to grant one additional trainset's
    worth of budget to the same phase-local GEPA run, and it differs from the
    previous fixed budget by returning the cumulative limit required for resume.
    """

    if phase_iteration <= 0:
        raise ValueError("phase_iteration must be > 0")
    if trainset_size <= 0:
        raise ValueError("trainset_size must be > 0")
    return phase_iteration * trainset_size


def _ensure_gepa_resume_compatibility(run_dir: Path) -> None:
    """Create the marker directory required by the pinned GEPA resume check.

    The installed `gepa` package documents `run_dir` resume support, but the
    current version only reloads state when both `gepa_state.bin` and a
    `prog_candidates/` directory exist. This helper is needed because this
    repository relies on resume for one-long-run-per-phase behavior, and it
    differs from GEPA's own save path by adding the missing compatibility marker
    before resumed compile calls.
    """

    state_path = run_dir / "gepa_state.bin"
    if state_path.exists():
        (run_dir / "prog_candidates").mkdir(parents=True, exist_ok=True)


def _build_iteration_summary(
    *,
    prompt_text: str,
    prompt_sha16: str,
    phase: int,
    phase_iteration: int,
    global_iteration: int,
    trainset_size: int,
    job_payloads: list[dict[str, Any]],
) -> IterationSummary:
    """Aggregate per-job metric rows into one curriculum iteration summary.

    GEPA still evaluates one job at a time, but curriculum control decisions are
    phase-wide. This helper is needed because advancement, baseline tracking,
    and early stopping all operate on the mean active-phase solve rate, and it
    differs from the W&B per-job table by collapsing one fully evaluated prompt
    candidate into the single summary the runner uses for control flow.
    """

    num_payloads = float(len(job_payloads))
    return IterationSummary(
        prompt_text=prompt_text,
        prompt_sha16=prompt_sha16,
        phase=phase,
        phase_iteration=phase_iteration,
        global_iteration=global_iteration,
        trainset_size=trainset_size,
        mean_job_score=sum(item["job_score"] for item in job_payloads) / num_payloads,
        mean_solve_rate=sum(item["solve_rate"] for item in job_payloads) / num_payloads,
        mean_expanded_states=sum(item["average_expanded_states"] for item in job_payloads)
        / num_payloads,
        mean_generated_states=sum(item["average_generated_states"] for item in job_payloads)
        / num_payloads,
        mean_admissibility_pass_rate=sum(
            item["admissibility_pass_rate"] for item in job_payloads
        )
        / num_payloads,
    )


def _choose_iteration_summary(
    *,
    completed_summaries: list[IterationSummary],
    optimized_prompt_text: str,
) -> IterationSummary:
    """Choose the summary that represents one GEPA compile call.

    A single DSPy compile invocation may internally touch more than one prompt
    candidate. The curriculum runner therefore needs a deterministic rule for
    selecting the iteration-level score that should count toward advancement and
    early stopping. This helper is needed because the compile result returns the
    chosen prompt text, and it differs from a naive "latest summary wins" rule
    by first matching the prompt text that GEPA actually kept.
    """

    for summary in completed_summaries:
        if summary.prompt_text == optimized_prompt_text:
            return summary
    if completed_summaries:
        return max(
            completed_summaries,
            key=lambda item: (item.mean_job_score, item.mean_solve_rate),
        )
    raise RuntimeError(
        "GEPA compile completed without producing a full-trainset iteration summary. "
        "Increase the per-phase metric-call budget or inspect the DSPy optimizer behavior."
    )


def _new_phase_record(*, phase: int, jobs: list[EnvJob], phase_job_count: int) -> dict[str, Any]:
    """Create the persisted record for one curriculum phase.

    The runner writes one durable phase record into both checkpoints and final
    stats. This helper is needed because resume behavior depends on stable
    phase-level bookkeeping, and it differs from ad hoc dict literals by making
    the curriculum schema explicit in one place.
    """

    return {
        "phase": phase,
        "phase_job_count": phase_job_count,
        "active_job_names": [job.name for job in jobs],
        "active_env_ids": [job.env_id for job in jobs],
        "threshold": PHASE_SOLVE_RATE_THRESHOLD,
        "baseline_solve_rate": None,
        "best_solve_rate": None,
        "baseline_job_score": None,
        "best_job_score": None,
        "best_iteration": None,
        "iteration_count": 0,
        "non_improving_streak": 0,
        "advanced": False,
        "completed": False,
        "stop_reason": None,
        "incomplete_compile_retries": 0,
        "last_incomplete_compile": None,
        "gepa_log_dir": None,
        "iteration_summaries": [],
        "compiler_stats": [],
    }


def _ensure_phase_record(
    curriculum_state: dict[str, Any],
    phase_schedule: list[list[EnvJob]],
) -> dict[str, Any]:
    """Return the mutable record for the current curriculum phase.

    The refactor stores phase progress inside `active_prompt.json` so runs can
    resume without recomputing earlier phases. This helper is needed because the
    active phase changes over time, and it differs from direct indexing by
    allocating a new schema-valid record when the phase has not been seen yet.
    """

    phase = int(curriculum_state["current_phase"])
    phase_key = str(phase)
    phase_records = curriculum_state.setdefault("phase_records", {})
    if phase_key not in phase_records:
        phase_jobs = phase_schedule[phase - 1]
        phase_records[phase_key] = _new_phase_record(
            phase=phase,
            jobs=phase_jobs,
            phase_job_count=len(phase_jobs),
        )
    return phase_records[phase_key]


def _default_curriculum_state(
    *,
    jobs: list[EnvJob],
    max_phase_iterations: int,
) -> dict[str, Any]:
    """Build the default curriculum checkpoint structure for a new run.

    The runner now persists curriculum metadata alongside the optimized prompt
    state. This helper is needed because fresh runs and incompatible resumes
    should share the same default schema, and it differs from the final stats
    payload by keeping only the mutable state required for continuation.
    """

    phase_job_counts = _phase_job_counts(jobs)
    return {
        "version": 1,
        "current_phase": 1,
        "completed_phases": [],
        "phase_records": {},
        "phase_job_counts": phase_job_counts,
        "total_phases": len(phase_job_counts),
        "global_iteration": 0,
        "metric_call_idx": 0,
        "max_phase_iterations": max_phase_iterations,
        "phase_solve_rate_threshold": PHASE_SOLVE_RATE_THRESHOLD,
        "phase_early_stop_patience": PHASE_EARLY_STOP_PATIENCE,
        "total_training_jobs": len(jobs),
        "training_job_names": [job.name for job in jobs],
        "training_env_ids": [job.env_id for job in jobs],
        "stop_reason": None,
        "final_prompt_text": None,
    }


def _load_curriculum_state(
    *,
    active_payload: Optional[Mapping[str, Any]],
    jobs: list[EnvJob],
    max_phase_iterations: int,
) -> dict[str, Any]:
    """Load persisted curriculum progress when it matches the current run shape.

    Users may rerun the same `state_root` with a resumed checkpoint or with a
    different training-job subset after `--room-count` filtering. This helper is
    needed because resume should be safe and predictable, and it differs from
    blindly trusting persisted JSON by resetting only the curriculum metadata
    when the saved job ordering no longer matches the current run.
    """

    default_state = _default_curriculum_state(
        jobs=jobs,
        max_phase_iterations=max_phase_iterations,
    )
    if not isinstance(active_payload, Mapping):
        return default_state
    persisted = active_payload.get("curriculum")
    if not isinstance(persisted, Mapping):
        return default_state
    if list(persisted.get("training_job_names", [])) != default_state["training_job_names"]:
        return default_state
    if list(persisted.get("training_env_ids", [])) != default_state["training_env_ids"]:
        return default_state
    if list(persisted.get("phase_job_counts", [])) != default_state["phase_job_counts"]:
        return default_state

    phase_records = persisted.get("phase_records", {})
    if not isinstance(phase_records, Mapping):
        phase_records = {}
    completed_phases = persisted.get("completed_phases", [])
    if not isinstance(completed_phases, list):
        completed_phases = []

    loaded_state = {
        **default_state,
        **{key: value for key, value in persisted.items() if key != "phase_records"},
        "phase_records": {str(key): value for key, value in phase_records.items()},
        "completed_phases": [int(value) for value in completed_phases],
        "max_phase_iterations": max_phase_iterations,
        "phase_solve_rate_threshold": PHASE_SOLVE_RATE_THRESHOLD,
        "phase_early_stop_patience": PHASE_EARLY_STOP_PATIENCE,
    }
    loaded_state["current_phase"] = max(
        1,
        min(int(loaded_state.get("current_phase", 1)), len(default_state["phase_job_counts"])),
    )
    loaded_state["global_iteration"] = int(loaded_state.get("global_iteration", 0))
    loaded_state["metric_call_idx"] = int(loaded_state.get("metric_call_idx", 0))
    loaded_state["total_training_jobs"] = len(jobs)
    loaded_state["training_job_names"] = default_state["training_job_names"]
    loaded_state["training_env_ids"] = default_state["training_env_ids"]
    loaded_state["phase_job_counts"] = default_state["phase_job_counts"]
    loaded_state["total_phases"] = len(default_state["phase_job_counts"])
    return loaded_state


def _resume_stop_reason(curriculum_state: Mapping[str, Any]) -> Optional[str]:
    """Return a terminal stop reason that was already persisted on disk.

    The heuristic runner supports resuming long-lived GEPA phases from
    `active_prompt.json`. This helper is needed because a resumed checkpoint may
    already represent a completed run, and it differs from trusting only the
    top-level `stop_reason` field by also recovering older checkpoints whose
    current phase record is marked complete even when the run-level stop reason
    was not normalized yet.
    """

    persisted_stop_reason = curriculum_state.get("stop_reason")
    if isinstance(persisted_stop_reason, str) and persisted_stop_reason.strip():
        return persisted_stop_reason

    total_phases = int(curriculum_state.get("total_phases", 0))
    if total_phases <= 0:
        return None
    current_phase = int(curriculum_state.get("current_phase", 1))
    phase_records = curriculum_state.get("phase_records", {})
    if not isinstance(phase_records, Mapping):
        return None
    phase_record = phase_records.get(str(current_phase))
    if not isinstance(phase_record, Mapping):
        return None
    if not bool(phase_record.get("completed")):
        return None
    phase_stop_reason = phase_record.get("stop_reason")
    if not isinstance(phase_stop_reason, str) or not phase_stop_reason.strip():
        return None
    if current_phase >= total_phases:
        return phase_stop_reason
    if phase_stop_reason != "advanced_to_next_phase":
        return phase_stop_reason
    return None


def _write_curriculum_checkpoint(
    *,
    state_root: Path,
    base_prompt_text: str,
    prompt_state: Optional[Mapping[str, Any]],
    prompt_meta: Mapping[str, Any],
    curriculum_state: Mapping[str, Any],
) -> Path:
    """Persist the latest best prompt together with curriculum progress.

    The curriculum runner should resume from the best prompt reached so far
    instead of from the latest attempted prompt. This helper is needed because
    the checkpoint must remain backward-compatible with existing prompt-loading
    logic while also storing richer curriculum metadata, and it differs from the
    legacy final-only write by running after every completed phase iteration.
    """

    prompt_payload = {
        "base_prompt_text": base_prompt_text,
        "prompt_state": prompt_state,
        "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": prompt_meta,
        "heuristic_contract": HEURISTIC_CONTRACT_TEXT,
        "curriculum": curriculum_state,
    }
    return write_active_prompt(state_root, prompt_payload)


def _record_incomplete_compile(
    *,
    phase_record: dict[str, Any],
    phase: int,
    phase_iteration: int,
    budget_iteration: int,
    global_iteration: int,
    trainset_size: int,
    max_metric_calls: int,
    metric_call_idx: int,
    optimized_prompt_text: str,
) -> None:
    """Persist enough metadata to safely resume a partial GEPA compile.

    GEPA occasionally returns from `compile(...)` before the runner has seen a
    prompt evaluated on every active training job. This helper is needed
    because the curriculum loop must preserve those partial optimizer side
    effects on disk without incorrectly counting them as a completed phase
    iteration, and it differs from the normal iteration-summary path by
    recording retry-oriented bookkeeping instead of advancement metrics.
    """

    retries = int(phase_record.get("incomplete_compile_retries", 0)) + 1
    phase_record["incomplete_compile_retries"] = retries
    phase_record["last_incomplete_compile"] = {
        "phase": phase,
        "phase_iteration": phase_iteration,
        "budget_iteration": budget_iteration,
        "global_iteration": global_iteration,
        "trainset_size": trainset_size,
        "max_metric_calls": max_metric_calls,
        "metric_call_idx": metric_call_idx,
        "optimized_prompt_text": optimized_prompt_text,
        "retry_count": retries,
    }


def _build_stats_payload(
    *,
    args: argparse.Namespace,
    model_name: str,
    jobs: list[EnvJob],
    curriculum_state: Mapping[str, Any],
    compiler_stats_by_iteration: list[dict[str, Any]],
    holdout_results: Optional[Mapping[str, Any]] = None,
    holdout_mean: Optional[float] = None,
    holdout_comparisons: Optional[list[HoldoutComparisonSummary]] = None,
    holdout_plot_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    """Assemble the curriculum-aware `gepa_stats.json` payload.

    The previous runner wrote a flat optimizer summary once at the end. This
    helper is needed because the curriculum refactor introduces phase-level stop
    reasons, baselines, and advancement records, and it differs from the old
    payload by exposing the full progression over the ordered training jobs.
    """

    phase_records = curriculum_state.get("phase_records", {})
    ordered_phase_records = [
        phase_records[str(index)]
        for index in range(1, int(curriculum_state.get("total_phases", 0)) + 1)
        if str(index) in phase_records
    ]
    payload: dict[str, Any] = {
        "llm_model": model_name,
        "max_phase_iterations": args.max_phase_iterations,
        "phase_solve_rate_threshold": PHASE_SOLVE_RATE_THRESHOLD,
        "phase_early_stop_patience": PHASE_EARLY_STOP_PATIENCE,
        "room_count": sorted(set(args.room_count)) if args.room_count is not None else None,
        "astar_max_nodes": args.astar_max_nodes,
        "astar_max_expansions": args.astar_max_expansions,
        "deterministic_envs": bool(args.deterministic_envs),
        "training_jobs": [job.to_config() for job in jobs],
        "curriculum": {
            **curriculum_state,
            "phase_records": ordered_phase_records,
        },
        "compiler_stats_by_iteration": compiler_stats_by_iteration,
    }
    if holdout_results is not None:
        payload["holdout_results"] = holdout_results
        payload["holdout_score_mean"] = holdout_mean
        payload["holdout_num_tries"] = DEFAULT_HOLDOUT_TRIES
    if holdout_comparisons is not None:
        payload["holdout_comparisons"] = [
            comparison.to_dict() for comparison in holdout_comparisons
        ]
    if holdout_plot_paths is not None:
        payload["holdout_plot_paths"] = [str(path) for path in holdout_plot_paths]
    return payload


def _phase_metric_prefix(phase: int) -> str:
    """Return the stable W&B metric prefix for one curriculum phase.

    The curriculum runner now emits both global GEPA metrics and phase-scoped
    metrics so W&B charts can be separated by phase. This helper is needed
    because those metric names must remain consistent across iteration and exit
    logs, and it differs from inline string formatting by centralizing the
    naming contract in one place.
    """

    return f"phase_{phase:02d}"


def evaluate_jobs(
    *,
    jobs: list[EnvJob],
    prompt_text: str,
    lm: Any,
    logs_root: Path,
    seed_resolver: Any,
    dir_name: str,
) -> tuple[dict[str, Any], float]:
    """Evaluate a prompt over a set of jobs and return per-job summaries.

    This helper shares the same evaluation path between GEPA training and fixed
    holdout reporting. It is needed because both modes synthesize heuristics and
    run multi-seed search, and it differs from `evaluate_job(...)` by handling
    output directory layout and aggregate job-score means.
    """

    root = logs_root / dir_name
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for job in jobs:
        seeds = list(seed_resolver(job))
        results[job.name] = evaluate_job(
            job=job,
            seeds=seeds,
            prompt_text=prompt_text,
            lm=lm,
            output_dir=root / job.name,
        )
    return results, mean_job_scores(list(results.values()))


def evaluate_no_heuristic_jobs(
    *,
    jobs: list[EnvJob],
    logs_root: Path,
    seed_resolver: Callable[[EnvJob], list[int] | tuple[int, ...]],
    dir_name: str,
) -> tuple[dict[str, Any], float]:
    """Evaluate blind A* over a job list using the runner's standard scoring path.

    The end-of-run comparison needs a no-heuristic baseline on the same holdout
    jobs and seeds as the learned prompt. This helper is needed because that
    baseline should share the same output layout and aggregate scoring contract
    as prompt-based holdout evaluation, and it differs from `evaluate_jobs(...)`
    by skipping LLM-backed heuristic synthesis entirely.
    """

    root = logs_root / dir_name
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for job in jobs:
        seeds = list(seed_resolver(job))
        results[job.name] = evaluate_no_heuristic_job(
            job=job,
            seeds=seeds,
            output_dir=root / job.name,
        )
    return results, mean_job_scores(list(results.values()))


def _mean_holdout_solve_rate(results: Mapping[str, Any]) -> float:
    """Return the mean per-job solve rate for one holdout evaluation bundle.

    The runner compares several end-of-run policies using the same per-job
    result schema. This helper is needed because W&B logs and plots both need a
    stable aggregate solve-rate calculation, and it differs from
    `mean_job_scores(...)` by reading solve-rate fields instead of GEPA scores.
    """

    if not results:
        return 0.0
    return sum(
        float(result["aggregate_stats"].get("solve_rate", 0.0))
        for result in results.values()
    ) / float(len(results))


def _mean_holdout_admissibility(results: Mapping[str, Any]) -> float:
    """Return the mean admissibility pass rate for one holdout evaluation bundle.

    The learned and base-prompt holdout runs both emit aggregated validation
    payloads, and the blind baseline reuses the same artifact shape. This
    helper is needed because end-of-run reporting should compare safety
    alongside performance, and it differs from `_admissibility_pass_rate(...)`
    by averaging across many jobs.
    """

    if not results:
        return 0.0
    return sum(
        _admissibility_pass_rate(result["heuristic_validation"])
        for result in results.values()
    ) / float(len(results))


def summarize_holdout_comparison(
    *,
    label: str,
    dir_name: str,
    results: Mapping[str, Any],
    job_score_mean: float,
) -> HoldoutComparisonSummary:
    """Build the aggregate summary object for one end-of-run holdout policy.

    The runner now compares multiple evaluation policies after training, and it
    needs a uniform summary object for JSON output, plotting, and W&B logging.
    This helper is needed because those consumers should not each recompute the
    same aggregate metrics, and it differs from raw holdout result dicts by
    attaching a human-facing label and precomputed means.
    """

    return HoldoutComparisonSummary(
        label=label,
        dir_name=dir_name,
        results=dict(results),
        job_score_mean=float(job_score_mean),
        solve_rate_mean=_mean_holdout_solve_rate(results),
        admissibility_pass_rate_mean=_mean_holdout_admissibility(results),
    )


def write_holdout_comparison_plots(
    *,
    logs_root: Path,
    comparisons: list[HoldoutComparisonSummary],
) -> list[Path]:
    """Write aggregate and per-environment holdout comparison bar charts.

    The new end-of-run report should make it easy to compare the learned prompt
    against the original prompt and blind A* at a glance. This helper is needed
    because the numeric JSON payload alone is cumbersome to scan, and it
    differs from the reward runner's holdout plots by comparing three heuristic
    policies on the exact same holdout jobs.
    """

    if not comparisons:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"[holdout-comparison-plots] matplotlib unavailable; skipping plots ({exc})")
        return []

    env_names = sorted(
        {
            str(env_name)
            for comparison in comparisons
            for env_name in comparison.results.keys()
        }
    )
    x_positions = list(range(len(comparisons)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    aggregate_path = logs_root / "holdout_comparison_aggregate.png"
    per_env_path = logs_root / "holdout_comparison_by_env.png"
    labels = [comparison.label for comparison in comparisons]
    aggregate_values = [comparison.solve_rate_mean for comparison in comparisons]

    plt.figure(figsize=(7, 4))
    plt.bar(
        x_positions,
        aggregate_values,
        color=colors[: len(comparisons)],
        width=0.65,
    )
    plt.xticks(x_positions, labels, rotation=15, ha="right")
    plt.ylabel("Solve rate")
    plt.title("Aggregate holdout solve rate comparison")
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(aggregate_path, dpi=150)
    plt.close()

    if env_names:
        plt.figure(figsize=(max(9.0, float(len(env_names)) * 1.3), 5))
        width = 0.22 if len(comparisons) >= 3 else 0.3
        offsets = [
            (index - (len(comparisons) - 1) / 2.0) * width
            for index in range(len(comparisons))
        ]
        base_positions = list(range(len(env_names)))
        for index, comparison in enumerate(comparisons):
            values = [
                float(
                    comparison.results.get(env_name, {})
                    .get("aggregate_stats", {})
                    .get("solve_rate", 0.0)
                )
                for env_name in env_names
            ]
            shifted_positions = [
                position + offsets[index]
                for position in base_positions
            ]
            plt.bar(
                shifted_positions,
                values,
                width=width,
                label=comparison.label,
                color=colors[index % len(colors)],
            )
        plt.xticks(base_positions, env_names, rotation=45, ha="right")
        plt.ylabel("Solve rate")
        plt.title("Holdout solve rate by environment")
        plt.ylim(0.0, 1.05)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(per_env_path, dpi=150)
        plt.close()

    paths = [aggregate_path]
    if env_names:
        paths.append(per_env_path)
    for path in paths:
        print(f"[holdout-comparison-plots] wrote {path}")
    return paths


def run_batch() -> None:
    """Run the full heuristic-only GEPA optimization flow.

    This is the orchestration entrypoint for the refactored repository. It is
    needed because prompt loading, example construction, metric caching, holdout
    evaluation, and artifact persistence must happen in one coordinated place,
    and it differs from the legacy runner by using only heuristic synthesis plus
    multi-seed search.
    """

    args = parse_args()
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "heuristic_runs"
    logs_root.mkdir(parents=True, exist_ok=True)
    if args.astar_max_nodes <= 0 or args.astar_max_expansions <= 0:
        raise ValueError("A* budgets must be > 0")

    model_name = args.llm
    base_lm = configure_deepseek_lm(model_name=model_name)
    reflection_lm = configure_deepseek_lm(model_name=model_name)
    dspy.configure(lm=reflection_lm)
    wandb_run = None
    candidate_runs_table = None
    last_wandb_step = -1
    if wandb is not None and not os.environ.get("WANDB_DISABLED"):
        try:
            wandb_run = wandb.init(
                project=DEFAULT_WANDB_PROJECT,
                name=f"heuristic-{state_root.name}",
                config={
                    "state_root": str(state_root),
                    "env_grid": str(args.env_grid),
                    "max_phase_iterations": args.max_phase_iterations,
                    "llm_model": model_name,
                    "astar_max_nodes": args.astar_max_nodes,
                    "astar_max_expansions": args.astar_max_expansions,
                    "deterministic_envs": bool(args.deterministic_envs),
                    "room_count": args.room_count,
                },
            )
            candidate_runs_table = wandb.Table(
                columns=[
                    "global_iteration",
                    "phase",
                    "phase_iteration",
                    "job_name",
                    "env_id",
                    "benchmark_id",
                    "prompt_sha16",
                    "job_score",
                    "solve_rate",
                    "average_expanded_states",
                    "average_generated_states",
                    "average_solution_length",
                    "admissibility_pass_rate",
                    "run_dir",
                ],
                log_mode="MUTABLE",
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[wandb] init failed, continuing without logging: {exc}")
            wandb_run = None
            candidate_runs_table = None

    def log_wandb(payload: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        """Log a payload to W&B while keeping step values monotonic.

        This closure standardizes the step handling for heuristic GEPA logs. It
        is needed because the metric logs both per-job and per-iteration values,
        and it differs from direct `safe_wandb_log(...)` usage by enforcing a
        non-decreasing step counter across the whole run.
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

    env_grid_path = args.env_grid.expanduser().resolve()
    jobs, eval_jobs = load_env_grid(
        env_grid_path,
        default_astar_max_nodes=args.astar_max_nodes,
        default_astar_max_expansions=args.astar_max_expansions,
    )
    selected_room_counts = sorted(set(args.room_count)) if args.room_count is not None else None
    if selected_room_counts is not None:
        jobs = filter_jobs_by_room_count(jobs, selected_room_counts, "training jobs")
        if eval_jobs:
            eval_jobs = filter_jobs_by_room_count(eval_jobs, selected_room_counts, "holdout jobs")
    if args.deterministic_envs:
        jobs = [
            EnvJob(
                name=job.name,
                env_id=job.env_id,
                benchmark_id=job.benchmark_id,
                num_gepa_eval_seeds=job.num_gepa_eval_seeds,
                holdout_seeds=job.holdout_seeds,
                deterministic_rulesets=True,
                fixed_ruleset_seed=job.fixed_ruleset_seed,
                astar_max_nodes=job.astar_max_nodes,
                astar_max_expansions=job.astar_max_expansions,
            )
            for job in jobs
        ]
        eval_jobs = [
            EnvJob(
                name=job.name,
                env_id=job.env_id,
                benchmark_id=job.benchmark_id,
                num_gepa_eval_seeds=job.num_gepa_eval_seeds,
                holdout_seeds=job.holdout_seeds,
                deterministic_rulesets=True,
                fixed_ruleset_seed=job.fixed_ruleset_seed,
                astar_max_nodes=job.astar_max_nodes,
                astar_max_expansions=job.astar_max_expansions,
            )
            for job in eval_jobs
        ]

    if args.max_phase_iterations <= 0:
        raise ValueError("--max-phase-iterations must be > 0")

    active_prompt_payload = load_active_prompt_payload(state_root)
    base_prompt_text, prompt_state, prompt_meta = load_prompt_payload(state_root)
    curriculum_state = _load_curriculum_state(
        active_payload=active_prompt_payload,
        jobs=jobs,
        max_phase_iterations=args.max_phase_iterations,
    )
    phase_schedule = _phase_schedule(jobs)
    if not phase_schedule:
        raise ValueError("Environment grid must include at least one training job")
    best_prompt_state = prompt_state
    best_prompt_text = compute_prompt_text(
        base_prompt_text=base_prompt_text,
        prompt_state=best_prompt_state,
    )
    save_best_prompt_text(state_root, model_name, best_prompt_text)
    _write_curriculum_checkpoint(
        state_root=state_root,
        base_prompt_text=base_prompt_text,
        prompt_state=best_prompt_state,
        prompt_meta=prompt_meta,
        curriculum_state=curriculum_state,
    )

    metric_call_idx = int(curriculum_state["metric_call_idx"])
    run_counter = _next_candidate_run_counter(logs_root)
    compiler_stats_by_iteration: list[dict[str, Any]] = []
    stop_reason = _resume_stop_reason(curriculum_state)
    if stop_reason is not None:
        curriculum_state["stop_reason"] = stop_reason

    while stop_reason is None:
        current_phase = int(curriculum_state["current_phase"])
        total_phases = int(curriculum_state["total_phases"])
        active_jobs = phase_schedule[current_phase - 1]
        phase_record = _ensure_phase_record(curriculum_state, phase_schedule)
        phase_iteration = int(phase_record["iteration_count"]) + 1
        incomplete_compile_retries = int(phase_record.get("incomplete_compile_retries", 0))
        budget_iteration = phase_iteration + incomplete_compile_retries
        global_iteration = int(curriculum_state["global_iteration"]) + 1
        _print_progress_line(
            (
                f"[gepa] phase={current_phase}/{total_phases} "
                f"phase_iteration={phase_iteration}/{args.max_phase_iterations} "
                f"global_iteration={global_iteration}"
            )
        )
        trainset = build_examples(active_jobs)
        phase_gepa_run_dir = _phase_gepa_run_dir(logs_root=logs_root, phase=current_phase)
        phase_gepa_run_dir.mkdir(parents=True, exist_ok=True)
        _ensure_gepa_resume_compatibility(phase_gepa_run_dir)
        phase_record["gepa_log_dir"] = str(phase_gepa_run_dir)
        gepa_max_metric_calls = _phase_gepa_max_metric_calls(
            phase_iteration=budget_iteration,
            trainset_size=len(trainset),
        )
        score_by_prediction_id: dict[int, float] = {}
        feedback_by_prediction_id: dict[int, str] = {}
        iteration_payloads_by_prompt: dict[str, list[dict[str, Any]]] = {}
        completed_summaries: list[IterationSummary] = []

        def metric(
            example: dspy.Example,
            prediction: dspy.Prediction,
            trace: Any = None,
            pred_name: Optional[str] = None,
            pred_trace: Any = None,
        ) -> float | ScoreWithFeedback:
            """Score one prompt candidate on one active curriculum job.

            This closure is recreated for every runner-owned phase iteration so
            the surrounding state can stay phase- and iteration-specific. It is
            needed because GEPA still drives job-level metric calls internally,
            and it differs from the pre-curriculum version by aggregating each
            fully evaluated prompt into an explicit `IterationSummary`.
            """

            nonlocal metric_call_idx, run_counter
            del trace, pred_trace
            prediction_id = id(prediction)
            if pred_name is not None:
                return ScoreWithFeedback(
                    score=score_by_prediction_id.get(prediction_id, 0.0),
                    feedback=feedback_by_prediction_id.get(
                        prediction_id,
                        "No cached feedback.",
                    ),
                )

            prompt_text = getattr(prediction, "prompt_text", None)
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                prompt_text = base_prompt_text
            job = job_from_example(example)
            seeds = sample_eval_seeds(
                global_experiment_seed=DEFAULT_GLOBAL_EXPERIMENT_SEED,
                metric_call_idx=metric_call_idx,
                job_name=job.name,
                num_gepa_eval_seeds=job.num_gepa_eval_seeds,
            )
            run_dir = logs_root / f"candidate-{run_counter:04d}-{job.name}"
            run_counter += 1
            metric_call_idx += 1
            result = evaluate_job(
                job=job,
                seeds=seeds,
                prompt_text=prompt_text,
                lm=base_lm,
                output_dir=run_dir,
            )
            prompt_sha16 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
            aggregate_stats = result["aggregate_stats"]
            validation_payload = result["heuristic_validation"]
            per_job_payload = {
                "job_name": job.name,
                "env_id": job.env_id,
                "benchmark_id": job.benchmark_id,
                "job_score": float(result["job_score"]),
                "solve_rate": float(aggregate_stats.get("solve_rate", 0.0)),
                "average_expanded_states": float(
                    aggregate_stats.get("average_expanded_states", 0.0)
                ),
                "average_generated_states": float(
                    aggregate_stats.get("average_generated_states", 0.0)
                ),
                "average_solution_length": float(
                    aggregate_stats.get("average_solution_length", 0.0)
                ),
                "admissibility_pass_rate": _admissibility_pass_rate(validation_payload),
                "run_dir": str(run_dir),
                "prompt_sha16": prompt_sha16,
            }
            iteration_payloads_by_prompt.setdefault(prompt_text, []).append(per_job_payload)
            if candidate_runs_table is not None:
                candidate_runs_table.add_data(
                    global_iteration,
                    current_phase,
                    phase_iteration,
                    job.name,
                    job.env_id,
                    job.benchmark_id,
                    prompt_sha16,
                    per_job_payload["job_score"],
                    per_job_payload["solve_rate"],
                    per_job_payload["average_expanded_states"],
                    per_job_payload["average_generated_states"],
                    per_job_payload["average_solution_length"],
                    per_job_payload["admissibility_pass_rate"],
                    per_job_payload["run_dir"],
                )
                log_wandb({"heuristic/candidate_runs": candidate_runs_table}, step=global_iteration)
            score_by_prediction_id[prediction_id] = float(result["job_score"])
            feedback_by_prediction_id[prediction_id] = str(result["feedback"])
            prompt_iteration_payloads = iteration_payloads_by_prompt[prompt_text]
            if len(prompt_iteration_payloads) >= len(trainset):
                summary = _build_iteration_summary(
                    prompt_text=prompt_text,
                    prompt_sha16=prompt_sha16,
                    phase=current_phase,
                    phase_iteration=phase_iteration,
                    global_iteration=global_iteration,
                    trainset_size=len(trainset),
                    job_payloads=prompt_iteration_payloads,
                )
                phase_metric_prefix = _phase_metric_prefix(current_phase)
                completed_summaries.append(summary)
                log_wandb(
                    {
                        "gepa/phase": current_phase,
                        "gepa/phase_iteration": phase_iteration,
                        "gepa/active_train_jobs": len(trainset),
                        "gepa/job_score": summary.mean_job_score,
                        "gepa/solve_rate": summary.mean_solve_rate,
                        "gepa/average_expanded_states": summary.mean_expanded_states,
                        "gepa/average_generated_states": summary.mean_generated_states,
                        "gepa/admissibility_pass_rate": summary.mean_admissibility_pass_rate,
                        f"{phase_metric_prefix}/solve_rate": summary.mean_solve_rate,
                        f"{phase_metric_prefix}/job_score": summary.mean_job_score,
                        f"{phase_metric_prefix}/admissibility_pass_rate": (
                            summary.mean_admissibility_pass_rate
                        ),
                        f"{phase_metric_prefix}/active_train_jobs": len(trainset),
                        f"{phase_metric_prefix}/phase_iteration": phase_iteration,
                    },
                    step=global_iteration,
                )
                iteration_payloads_by_prompt.pop(prompt_text, None)
            return ScoreWithFeedback(score=float(result["job_score"]), feedback=str(result["feedback"]))

        program = PromptOnlyProgram(base_prompt_text, prompt_state=best_prompt_state)
        compiler = dspy.GEPA(
            metric=metric,
            max_metric_calls=gepa_max_metric_calls,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=1,
            track_stats=True,
            num_threads=1,
            log_dir=str(phase_gepa_run_dir),
        )
        optimized_program = compiler.compile(program, trainset=trainset)
        optimized_prompt_state = optimized_program.prompt_generator.dump_state()
        optimized_prompt_text = optimized_program.prompt_generator(
            base_prompt=optimized_program._build_rewrite_prompt()
        )
        if not completed_summaries:
            _record_incomplete_compile(
                phase_record=phase_record,
                phase=current_phase,
                phase_iteration=phase_iteration,
                budget_iteration=budget_iteration,
                global_iteration=global_iteration,
                trainset_size=len(trainset),
                max_metric_calls=gepa_max_metric_calls,
                metric_call_idx=metric_call_idx,
                optimized_prompt_text=str(optimized_prompt_text),
            )
            curriculum_state["metric_call_idx"] = metric_call_idx
            curriculum_state["final_prompt_text"] = best_prompt_text
            curriculum_state["stop_reason"] = stop_reason
            save_best_prompt_text(state_root, model_name, best_prompt_text)
            _write_curriculum_checkpoint(
                state_root=state_root,
                base_prompt_text=base_prompt_text,
                prompt_state=best_prompt_state,
                prompt_meta=prompt_meta,
                curriculum_state=curriculum_state,
            )
            _print_progress_line(
                (
                    f"[gepa] phase={current_phase} "
                    f"phase_iteration={phase_iteration} "
                    f"incomplete_compile retry={int(phase_record['incomplete_compile_retries'])} "
                    f"budget_metric_calls={gepa_max_metric_calls}"
                )
            )
            continue
        iteration_summary = _choose_iteration_summary(
            completed_summaries=completed_summaries,
            optimized_prompt_text=str(optimized_prompt_text),
        )
        compiler_stats = getattr(compiler, "stats", {}) or {}
        compiler_stats_by_iteration.append(
            {
                "phase": current_phase,
                "phase_iteration": phase_iteration,
                "global_iteration": global_iteration,
                "max_metric_calls": gepa_max_metric_calls,
                "log_dir": str(phase_gepa_run_dir),
                "stats": compiler_stats,
            }
        )

        phase_record["iteration_count"] = phase_iteration
        phase_record["incomplete_compile_retries"] = 0
        phase_record["last_incomplete_compile"] = None
        phase_record["compiler_stats"].append(
            {
                "phase_iteration": phase_iteration,
                "global_iteration": global_iteration,
                "max_metric_calls": gepa_max_metric_calls,
                "log_dir": str(phase_gepa_run_dir),
                "stats": compiler_stats,
            }
        )
        phase_record["iteration_summaries"].append(iteration_summary.to_dict())
        current_solve_rate = iteration_summary.mean_solve_rate
        current_job_score = iteration_summary.mean_job_score
        improved = False
        if phase_record["baseline_solve_rate"] is None:
            phase_record["baseline_solve_rate"] = current_solve_rate
        if phase_record["baseline_job_score"] is None:
            phase_record["baseline_job_score"] = current_job_score
        best_solve_rate = phase_record["best_solve_rate"]
        best_job_score = phase_record["best_job_score"]
        if best_solve_rate is None or current_solve_rate > float(best_solve_rate):
            phase_record["best_solve_rate"] = current_solve_rate
        if best_job_score is None or current_job_score > float(best_job_score):
            improved = True
            phase_record["best_job_score"] = current_job_score
            phase_record["best_iteration"] = phase_iteration
            phase_record["non_improving_streak"] = 0
            best_prompt_state = optimized_prompt_state
            best_prompt_text = str(optimized_prompt_text)
        else:
            phase_record["non_improving_streak"] = int(phase_record["non_improving_streak"]) + 1

        curriculum_state["global_iteration"] = global_iteration
        curriculum_state["metric_call_idx"] = metric_call_idx
        curriculum_state["final_prompt_text"] = best_prompt_text

        phase_advanced = False
        is_final_phase = current_phase >= total_phases
        if not is_final_phase and current_solve_rate >= PHASE_SOLVE_RATE_THRESHOLD:
            phase_advanced = True
            phase_record["advanced"] = True
            phase_record["completed"] = True
            phase_record["stop_reason"] = "advanced_to_next_phase"
            if not improved:
                best_prompt_state = optimized_prompt_state
                best_prompt_text = str(optimized_prompt_text)
                curriculum_state["final_prompt_text"] = best_prompt_text
            if current_phase not in curriculum_state["completed_phases"]:
                curriculum_state["completed_phases"].append(current_phase)
            curriculum_state["current_phase"] = current_phase + 1
        elif is_final_phase:
            if phase_iteration >= args.max_phase_iterations:
                phase_record["completed"] = True
                phase_record["stop_reason"] = "phase_iteration_cap"
                stop_reason = "phase_iteration_cap"
        else:
            if int(phase_record["non_improving_streak"]) >= PHASE_EARLY_STOP_PATIENCE:
                phase_record["completed"] = True
                phase_record["stop_reason"] = "threshold_failure_early_stop"
                stop_reason = "threshold_failure_early_stop"
            elif phase_iteration >= args.max_phase_iterations:
                phase_record["completed"] = True
                phase_record["stop_reason"] = "phase_iteration_cap"
                stop_reason = "phase_iteration_cap"

        curriculum_state["stop_reason"] = stop_reason
        save_best_prompt_text(state_root, model_name, best_prompt_text)
        _write_curriculum_checkpoint(
            state_root=state_root,
            base_prompt_text=base_prompt_text,
            prompt_state=best_prompt_state,
            prompt_meta=prompt_meta,
            curriculum_state=curriculum_state,
        )
        phase_metric_prefix = _phase_metric_prefix(current_phase)
        log_wandb(
            {
                "gepa/phase": current_phase,
                "gepa/phase_iteration": phase_iteration,
                "gepa/active_train_jobs": len(trainset),
                "gepa/phase_baseline_solve_rate": float(phase_record["baseline_solve_rate"]),
                "gepa/phase_best_solve_rate": float(phase_record["best_solve_rate"]),
                "gepa/phase_baseline_job_score": float(phase_record["baseline_job_score"]),
                "gepa/phase_best_job_score": float(phase_record["best_job_score"]),
                "gepa/phase_non_improving_streak": int(phase_record["non_improving_streak"]),
                "gepa/phase_advanced": phase_advanced,
                "gepa/iteration_improved": improved,
                f"{phase_metric_prefix}/baseline_solve_rate": float(
                    phase_record["baseline_solve_rate"]
                ),
                f"{phase_metric_prefix}/best_solve_rate": float(phase_record["best_solve_rate"]),
                f"{phase_metric_prefix}/baseline_job_score": float(
                    phase_record["baseline_job_score"]
                ),
                f"{phase_metric_prefix}/best_job_score": float(phase_record["best_job_score"]),
                f"{phase_metric_prefix}/non_improving_streak": int(
                    phase_record["non_improving_streak"]
                ),
                f"{phase_metric_prefix}/advanced": phase_advanced,
            },
            step=global_iteration,
        )
        _print_progress_line(
            (
                f"[gepa] phase={current_phase} "
                f"phase_iteration={phase_iteration} "
                f"solve_rate={current_solve_rate:.3f} "
                f"job_score={current_job_score:.4f} "
                f"best_score={float(phase_record['best_job_score']):.4f} "
                f"streak={int(phase_record['non_improving_streak'])}"
            )
        )
        if phase_record["completed"]:
            phase_failed_to_converge = phase_record["stop_reason"] == "threshold_failure_early_stop"
            _print_progress_line(
                (
                    f"[gepa] phase={current_phase} completed "
                    f"reason={phase_record['stop_reason']} "
                    f"best_score={float(phase_record['best_job_score']):.4f}"
                )
            )
            log_wandb(
                {
                    "curriculum/phase": current_phase,
                    "curriculum/phase_exit_reason": str(phase_record["stop_reason"]),
                    "curriculum/phase_failed_to_converge": phase_failed_to_converge,
                    "curriculum/phase_reached_threshold": (
                        float(phase_record["best_solve_rate"]) >= PHASE_SOLVE_RATE_THRESHOLD
                    ),
                    "curriculum/phase_baseline_solve_rate": float(
                        phase_record["baseline_solve_rate"]
                    ),
                    "curriculum/phase_best_solve_rate": float(phase_record["best_solve_rate"]),
                    "curriculum/phase_baseline_job_score": float(
                        phase_record["baseline_job_score"]
                    ),
                    "curriculum/phase_best_job_score": float(phase_record["best_job_score"]),
                    "curriculum/phase_iterations_used": int(phase_record["iteration_count"]),
                    f"{phase_metric_prefix}/exit_reason": str(phase_record["stop_reason"]),
                    f"{phase_metric_prefix}/failed_to_converge": phase_failed_to_converge,
                    f"{phase_metric_prefix}/completed": True,
                },
                step=global_iteration,
            )
        if stop_reason is not None:
            run_failed_to_converge = stop_reason == "threshold_failure_early_stop"
            _print_progress_line(
                (
                    f"[gepa] run complete reason={stop_reason} "
                    f"phase={current_phase} "
                    f"failed_to_converge={run_failed_to_converge}"
                )
            )
            log_wandb(
                {
                    "curriculum/stop_reason": stop_reason,
                    "curriculum/failed_to_converge": run_failed_to_converge,
                    "curriculum/final_phase": current_phase,
                    "curriculum/total_global_iterations": global_iteration,
                },
                step=global_iteration,
            )

    stats_payload = _build_stats_payload(
        args=args,
        model_name=model_name,
        jobs=jobs,
        curriculum_state=curriculum_state,
        compiler_stats_by_iteration=compiler_stats_by_iteration,
    )
    if eval_jobs:
        holdout_results, holdout_mean = evaluate_jobs(
            jobs=eval_jobs,
            prompt_text=best_prompt_text,
            lm=base_lm,
            logs_root=logs_root,
            seed_resolver=lambda job: job.holdout_seeds,
            dir_name="holdout-heuristic",
        )
        base_prompt_holdout_results, base_prompt_holdout_mean = evaluate_jobs(
            jobs=eval_jobs,
            prompt_text=base_prompt_text,
            lm=base_lm,
            logs_root=logs_root,
            seed_resolver=lambda job: job.holdout_seeds,
            dir_name="holdout-base-prompt",
        )
        blind_holdout_results, blind_holdout_mean = evaluate_no_heuristic_jobs(
            jobs=eval_jobs,
            logs_root=logs_root,
            seed_resolver=lambda job: job.holdout_seeds,
            dir_name="holdout-no-heuristic",
        )
        holdout_comparisons = [
            summarize_holdout_comparison(
                label="Optimized prompt",
                dir_name="holdout-heuristic",
                results=holdout_results,
                job_score_mean=holdout_mean,
            ),
            summarize_holdout_comparison(
                label="Base prompt",
                dir_name="holdout-base-prompt",
                results=base_prompt_holdout_results,
                job_score_mean=base_prompt_holdout_mean,
            ),
            summarize_holdout_comparison(
                label="Blind A*",
                dir_name="holdout-no-heuristic",
                results=blind_holdout_results,
                job_score_mean=blind_holdout_mean,
            ),
        ]
        holdout_plot_paths = write_holdout_comparison_plots(
            logs_root=logs_root,
            comparisons=holdout_comparisons,
        )
        stats_payload = _build_stats_payload(
            args=args,
            model_name=model_name,
            jobs=jobs,
            curriculum_state=curriculum_state,
            compiler_stats_by_iteration=compiler_stats_by_iteration,
            holdout_results=holdout_results,
            holdout_mean=holdout_mean,
            holdout_comparisons=holdout_comparisons,
            holdout_plot_paths=holdout_plot_paths,
        )
        if holdout_results:
            holdout_mean_solve_rate = _mean_holdout_solve_rate(holdout_results)
            holdout_mean_admissibility = _mean_holdout_admissibility(holdout_results)
            log_wandb(
                {
                    "holdout/job_score_mean": holdout_mean,
                    "holdout/solve_rate_mean": holdout_mean_solve_rate,
                    "holdout/admissibility_pass_rate_mean": holdout_mean_admissibility,
                    "holdout/base_prompt_job_score_mean": base_prompt_holdout_mean,
                    "holdout/base_prompt_solve_rate_mean": _mean_holdout_solve_rate(
                        base_prompt_holdout_results
                    ),
                    "holdout/base_prompt_admissibility_pass_rate_mean": (
                        _mean_holdout_admissibility(base_prompt_holdout_results)
                    ),
                    "holdout/no_heuristic_job_score_mean": blind_holdout_mean,
                    "holdout/no_heuristic_solve_rate_mean": _mean_holdout_solve_rate(
                        blind_holdout_results
                    ),
                    "holdout/no_heuristic_admissibility_pass_rate_mean": (
                        _mean_holdout_admissibility(blind_holdout_results)
                    ),
                },
                step=max(last_wandb_step, int(curriculum_state["global_iteration"]), 0),
            )
    stats_path = logs_root / "gepa_stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True), encoding="utf-8")
    safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    run_batch()
