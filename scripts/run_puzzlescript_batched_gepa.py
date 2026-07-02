#!/usr/bin/env python3
"""Run standalone GEPA with batched GPU synthesis and CPU-array search.

This is the artifact-oriented path for PuzzleScript heuristic optimization:

1. The GPU controller uses a local OpenAI-compatible model endpoint to generate
   one heuristic per active game level for the current GEPA candidate prompt.
2. The controller writes an evaluation manifest and waits while CPU Slurm array
   tasks evaluate each heuristic with deterministic A*.
3. The merged scores and feedback are returned to standalone `gepa.optimize`.

The existing DSPy runner remains available. This script avoids DSPy in the new
optimization path and uses GEPA's adapter interface directly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any, Mapping, Optional, Protocol, Sequence, cast

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SEARCH_ROOT = _PROJECT_ROOT / "llm_desparsifier" / "search"
if str(_SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEARCH_ROOT))

from puzzle_evaluator import PuzzleScriptEvaluator  # noqa: E402
from puzzlescript_adapter import build_env_description, build_puzzlescript_ctx  # noqa: E402
from puzzlescript_astar import puzzlescript_astar  # noqa: E402
from puzzlescript_sanitizer import sanitize_and_compile_puzzlescript_heuristic  # noqa: E402

DEFAULT_ENV_GRID = Path("configs/gepa_puzzlescript_envs.yaml")
DEFAULT_STATE_ROOT = Path("artifacts/gepa_puzzlescript_batched_state")
DEFAULT_SCRIPT_DOCTOR = _PROJECT_ROOT.parent / "script-doctor"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MAX_MODEL_TOKENS = 8192
DEFAULT_MAX_EXPANSIONS = 50_000
DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL = 10_000
DEFAULT_ASTAR_TIMEOUT_S = 30.0
DEFAULT_CONTEXT_RETRY_MARGIN_TOKENS = 512
DEFAULT_CONTEXT_RETRY_ATTEMPTS = 8
DEFAULT_MIN_RETRY_OUTPUT_TOKENS = 256
DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS = 1200
DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS = 1800
DEFAULT_REFLECTION_FEEDBACK_CHARS = 1600
DEFAULT_REFLECTION_MAX_RECORDS = 24
DEFAULT_SEARCH_TASK_WALL_TIMEOUT_S = 120.0
DEFAULT_DEV_FRACTION = 0.25
DEFAULT_MAX_GEPA_ITERATIONS = 16
DEFAULT_GUARD_LEVELS = (
    "Aperture_Science_Sokoban_Testing_Initiative:5;"
    "Beam_Islands:3;"
    "Gravity_Sokoban:6,9;"
    "Ice_Cubes:5,7,14,26;"
    "Inswaption:7,18;"
    "Memory_Push:0,2;"
    "Not_Normal_Crates:5,11,12;"
    "Sokoban_Dungeon:1;"
    "Sokocross:0,5;"
    "where_did_all_this_ice_come_from_:3,6"
)
DEFAULT_LOST_SOLVE_PENALTY = 8.0
DEFAULT_NEW_SOLVE_BONUS = 1.0
DEFAULT_CANDIDATE_ERROR_PENALTY = 2.0
DEFAULT_SCORE_DELTA_WEIGHT = 1.0
DEFAULT_SCORE_DELTA_CLIP = 0.5
DEFAULT_PARTIAL_PROGRESS_WEIGHT = 0.05
DEFAULT_PROPOSED_PROMPT_MAX_CHARS = 4200
DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS = 900
DEFAULT_REFLECTION_DATASET_CHARS = 24000
DEFAULT_SEARCH_ARRAY_STALL_TIMEOUT_S = 120.0
DEFAULT_LLM_TEMPERATURE = 0.0

HEURISTIC_COMPONENT = "heuristic_prompt"
GEPA_ADDENDUM_HEADER = "Additional GEPA guidance:"
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?"
    r"prompt contains at least\s+(?P<input>\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)

PUZZLESCRIPT_HEURISTIC_CONTRACT = """You are writing a heuristic function for A* search on one PuzzleScript grid puzzle.

Output exactly Python code defining:
def heuristic_cost_to_go(ts, env_params, ctx) -> float

Do not output markdown fences, backticks, prose, imports, print, exec, eval, open,
or file/network access. For PuzzleScript games, ts and env_params are None; the
function must use only ctx plus constants you derive from the prompt-time game
source. The full source, LEGEND, COLLISIONLAYERS, RULES, WINCONDITIONS, and
initial level state may be present in the prompt for analysis, but they are not
runtime inputs except through ctx.

Runtime ctx keys include:
  ctx.get('game_title'): title string
  ctx.get('object_positions'): dict mapping object name -> list of (x,y) tuples
  ctx.get('grid_width'), ctx.get('grid_height'): grid dimensions
  ctx.get('win_conditions_text'): human-readable win conditions
  ctx.get('ascii_state'): text grid of current state
  ctx.get('score'): engine score, lower is closer to solved
  ctx.get('score_normalized'): engine progress in [0,1], higher is closer
  ctx.get('is_winning'): True if state is already won
  ctx.get('object_names'): list of all object type names
  ctx.get('action_names'): action id/name mapping

Read the actual PuzzleScript mechanics before choosing features. Do not assume
the objective is crate-on-target or even Sokoban-like just because object names
look familiar. Derive the main progress terms from WINCONDITIONS, then use RULES
and COLLISIONLAYERS to decide whether distances, reachability, ordering,
alignment, terrain consumption, swapping, pulling, sliding, teleportation,
gravity, beams, or one-way effects matter.

Heuristic requirements:
- Return a non-negative float; lower means closer to a win.
- Return 0.0 when ctx.get('is_winning') is True.
- Produce varied values across plausible successor states; constant heuristics
  turn A* into blind search.
- Prefer conservative, finite penalties. Only use very large deadlock penalties
  for conditions that are provably impossible under this game's rules.
- Use ctx['score_normalized'] or ctx['score'] as a small fallback or tie-breaker
  when the game-specific features are uncertain; do not make them the only
  signal unless there is no reliable object-level signal.
- If unsure, build a finite fallback from win_conditions_text, object names,
  object counts, player-to-interaction distance, and score_normalized instead
  of hard-coding a Sokoban template.
- Do not assume exact object keys such as "player", "crate", or "target".
  Inspect ctx['object_names'], ctx['object_positions'], WINCONDITIONS, and
  LEGEND aliases to collect player variants, crate-like objects, target-like
  objects, exits, portals, terrain, and other mechanic-specific roles.
- Prefer robust helper logic that derives object roles from names and win text
  over a single hard-coded Sokoban key. If aliases exist, use all matching
  variants instead of treating missing "player" or "crate" as an invalid state.
"""


@dataclass(frozen=True)
class PuzzleScriptLevelTask:
    task_id: int
    game: str
    level: int
    budget: int
    env_description: str
    game_text_path: str


@dataclass(frozen=True)
class SearchArrayConfig:
    submit: bool
    array_script: Path
    array_count: int
    array_concurrency: int
    poll_interval_s: float
    stall_timeout_s: float = DEFAULT_SEARCH_ARRAY_STALL_TIMEOUT_S
    extra_sbatch_args: tuple[str, ...] = ()


class TextCompletionClient(Protocol):
    """Minimal text-generation interface used by GEPA synthesis and reflection."""

    def complete(self, prompt: str) -> str:
        """Return one model completion for the supplied prompt."""


@dataclass
class OpenAITextClient:
    model: str
    base_url: str
    api_key: str
    max_tokens: int
    temperature: float
    top_p: float
    timeout_s: float
    context_retry_margin_tokens: int = DEFAULT_CONTEXT_RETRY_MARGIN_TOKENS
    context_retry_attempts: int = DEFAULT_CONTEXT_RETRY_ATTEMPTS
    min_retry_output_tokens: int = DEFAULT_MIN_RETRY_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
        )

    def _complete_with_max_tokens(self, prompt: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return str(response.choices[0].message.content or "").strip()

    def complete(self, prompt: str) -> str:
        from openai import BadRequestError

        max_tokens = self.max_tokens
        for retry_index in range(max(0, self.context_retry_attempts) + 1):
            try:
                return self._complete_with_max_tokens(prompt, max_tokens)
            except BadRequestError as exc:
                retry_max_tokens = context_retry_max_tokens(
                    str(exc),
                    current_max_tokens=max_tokens,
                    retry_margin_tokens=self.context_retry_margin_tokens,
                    min_retry_tokens=self.min_retry_output_tokens,
                )
                if retry_max_tokens is None or retry_index >= self.context_retry_attempts:
                    raise
                print(
                    "[llm] retrying context-limited request with "
                    f"max_tokens={retry_max_tokens} after rejection at max_tokens={max_tokens}",
                    flush=True,
                )
                max_tokens = retry_max_tokens
        raise RuntimeError("unreachable context retry loop exit")



def context_retry_max_tokens(
    error_message: str,
    *,
    current_max_tokens: int,
    retry_margin_tokens: int = DEFAULT_CONTEXT_RETRY_MARGIN_TOKENS,
    min_retry_tokens: int = DEFAULT_MIN_RETRY_OUTPUT_TOKENS,
) -> Optional[int]:
    """Return a smaller completion budget for vLLM context-limit failures.

    vLLM reports both the model context window and the prompt token count in
    OpenAI-compatible 400 errors. GEPA reflection prompts can land close to the
    context boundary, so retrying with the remaining token budget is preferable
    to dropping an otherwise useful proposal.
    """
    match = _CONTEXT_LENGTH_RE.search(error_message)
    if match is None:
        return None
    context_tokens = int(match.group("context"))
    input_tokens = int(match.group("input"))
    retry_tokens = context_tokens - input_tokens - max(0, retry_margin_tokens)
    if retry_tokens < min_retry_tokens or retry_tokens >= current_max_tokens:
        return None
    return retry_tokens


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "item"


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _trace_level(trace: Mapping[str, Any]) -> int:
    task = trace.get("task", {})
    try:
        return int(task.get("level", 0))
    except (TypeError, ValueError):
        return 0


def trace_classification(trace: Mapping[str, Any]) -> str:
    """Classify a trace by how it changes behavior relative to the base prompt."""
    result = trace.get("result", {})
    solved = bool(result.get("solved", False))
    baseline_solved = bool(result.get("baseline_solved", False))
    raw_score = float(result.get("score", 0.0))
    baseline_score = float(result.get("baseline_score", 0.0))
    if is_candidate_error(result):
        return "candidate_error"
    if baseline_solved and not solved:
        return "lost_baseline_solve"
    if not baseline_solved and solved:
        return "new_solve"
    if baseline_solved and solved and raw_score + 0.05 < baseline_score:
        return "solved_regression"
    if not solved:
        return "persistent_failure"
    return "stable_or_improved"


def reflection_trace_priority(trace: Mapping[str, Any]) -> tuple[int, float, str, int]:
    """Return the ordering key used to choose compact reflection examples.

    Full evaluations still contribute to scalar scores, but sending every trace
    into GEPA's prompt can exceed the local model context. Reflection is most
    useful when it can compare the base prompt to the candidate, so baseline
    regressions and new solves are prioritized before ordinary weak cases.
    """
    task = trace.get("task", {})
    result = trace.get("result", {})
    class_rank = {
        "lost_baseline_solve": 0,
        "new_solve": 1,
        "solved_regression": 2,
        "candidate_error": 3,
        "persistent_failure": 4,
        "stable_or_improved": 5,
    }.get(trace_classification(trace), 6)
    return (
        class_rank,
        float(result.get("adjusted_score", result.get("score", 0.0))),
        str(task.get("game", "")),
        _trace_level(trace),
    )


def trace_mechanics_signature(trace: Mapping[str, Any]) -> str:
    """Return a compact mechanics label used to diversify reflection examples."""
    task = trace.get("task", {})
    game = str(task.get("game", ""))
    env_description = str(task.get("env_description", ""))
    text = f"{game}\n{env_description}".lower()
    labels: list[str] = []
    if re.search(r"\bplayer(?:u|d|l|r|v|h|1|2)\b", text):
        labels.append("player-alias")
    if "portal" in text or "teleport" in text or "wrap" in text:
        labels.append("portal")
    if "gravity" in text:
        labels.append("gravity")
    if "ice" in text or "slide" in text or "slipper" in text:
        labels.append("ice")
    if "beam" in text or "laser" in text:
        labels.append("beam")
    if "pull" in text or "[ <" in text:
        labels.append("pull")
    if "swap" in text or "inswap" in text:
        labels.append("swap")
    if "mud" in text or "footprint" in text:
        labels.append("terrain")
    if "all target" in text and "crate" in text:
        labels.append("target-crate")
    elif "all target" in text and "wall" in text:
        labels.append("target-wall")
    elif "win conditions" in text:
        labels.append("other-win")
    return "+".join(labels) if labels else "generic"


def select_reflection_traces(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    max_records: int = DEFAULT_REFLECTION_MAX_RECORDS,
) -> list[Mapping[str, Any]]:
    """Return the bounded trace subset sent to GEPA's reflection LLM.

    This differs from search evaluation, which remains exhaustive over the
    active batch. The cap only limits how much textual evidence is placed in the
    reflection prompt for proposal generation.
    """
    traces = list(trajectories)
    if max_records <= 0 or len(traces) <= max_records:
        return traces
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for trace in traces:
        grouped.setdefault(trace_classification(trace), []).append(trace)
    for values in grouped.values():
        values.sort(key=reflection_trace_priority)

    selected: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    seen_mechanics: set[str] = set()

    def add_trace(trace: Mapping[str, Any]) -> bool:
        if id(trace) in seen_ids:
            return False
        selected.append(trace)
        seen_ids.add(id(trace))
        seen_mechanics.add(trace_mechanics_signature(trace))
        return True

    priority_groups = (
        "lost_baseline_solve",
        "new_solve",
        "solved_regression",
        "candidate_error",
        "persistent_failure",
        "stable_or_improved",
    )

    # Seed reflection with one example from each high-signal category so the
    # LLM sees both what to preserve and what the candidate gained.
    for group_name in priority_groups:
        group = grouped.get(group_name, [])
        if not group:
            continue
        trace = group[0]
        add_trace(trace)
        if len(selected) >= max_records:
            return selected

    # After outcome seeding, add one representative per mechanics signature so
    # crowded generic failures do not hide player-alias, portal, gravity, beam,
    # or terrain cases from the reflection prompt.
    for trace in sorted(traces, key=reflection_trace_priority):
        if id(trace) in seen_ids:
            continue
        signature = trace_mechanics_signature(trace)
        if signature in seen_mechanics:
            continue
        add_trace(trace)
        if len(selected) >= max_records:
            return selected

    for trace in sorted(traces, key=reflection_trace_priority):
        add_trace(trace)
        if len(selected) >= max_records:
            return selected
    return selected


def strip_outer_markdown_fences(code: str) -> str:
    cleaned = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def candidate_prompt_issue(prompt_text: str) -> Optional[str]:
    """Return a prompt-shape issue when GEPA proposes implementation code.

    The optimized component is a reusable instruction prompt that later asks an
    LLM to write one heuristic per level. GEPA can occasionally return a full
    `heuristic_cost_to_go` implementation as the component itself; accepting
    that turns the global prompt into a brittle code template. This guard allows
    instruction prompts that mention the required signature but rejects text that
    is itself a Python implementation.
    """
    cleaned = strip_outer_markdown_fences(prompt_text)
    stripped = cleaned.strip()
    if not stripped:
        return "candidate prompt is empty"
    first_content_line = next(
        (line.strip() for line in stripped.splitlines() if line.strip()),
        "",
    )
    if re.match(
        r"^(?:async\s+def|def\s+heuristic_cost_to_go\s*\(|import\s+|from\s+\S+\s+import\s+)",
        first_content_line,
    ):
        return "candidate prompt is a Python implementation rather than instruction text"
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return None
    if not tree.body:
        return "candidate prompt is empty"
    first_node = tree.body[0]
    if isinstance(first_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
        return "candidate prompt is a Python implementation rather than instruction text"
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "heuristic_cost_to_go"
        for node in tree.body
    ):
        return "candidate prompt contains a heuristic implementation instead of instructions"
    return None


def heuristic_score(solved: bool, expanded: int, max_expansions: int) -> float:
    n = max_expansions
    s = expanded if solved else n + 1
    return ((n + 1) - s) / (n + 1)


def load_env_grid(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(payload.get("jobs", [])), list(payload.get("eval_jobs", []))


def split_train_dev_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split training jobs into deterministic train and development subsets.

    The split happens at game/job granularity, not level granularity, so a
    validation level cannot leak mechanics from the same game into training.
    At least one training job is kept whenever two or more jobs are available.
    """
    rows = [dict(job) for job in jobs]
    if not rows:
        return [], []
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    if len(rows) == 1:
        return rows, []

    n_dev = int(round(len(rows) * dev_fraction))
    n_dev = min(len(rows) - 1, max(1, n_dev))
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    dev_indices = set(indices[:n_dev])
    train = [row for index, row in enumerate(rows) if index not in dev_indices]
    dev = [row for index, row in enumerate(rows) if index in dev_indices]
    return train, dev


def parse_guard_level_selection(selection: str) -> dict[str, list[int]]:
    """Parse an exact `game:level,level;...` guard-level specification.

    Guard levels are held-out benchmark cases that previously exposed a
    base-prompt regression or a useful optimized-prompt win. Keeping the parser
    exact rather than fuzzy prevents accidental validation leakage across whole
    games while still letting the runner protect known mechanics patterns.
    """
    levels_by_game: dict[str, list[int]] = {}
    text = selection.strip()
    if not text:
        return levels_by_game
    for raw_entry in text.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "Guard level entries must use game:level,level format; "
                f"got {entry!r}"
            )
        game, raw_levels = entry.split(":", 1)
        game = game.strip()
        if not game or not raw_levels.strip():
            raise ValueError(
                "Guard level entries must use game:level,level format; "
                f"got {entry!r}"
            )
        parsed_levels: list[int] = []
        for raw_level in raw_levels.split(","):
            raw_level = raw_level.strip()
            if not raw_level:
                continue
            try:
                level = int(raw_level)
            except ValueError as exc:
                raise ValueError(
                    "Guard level entries must use integer levels; "
                    f"got {raw_level!r} in {entry!r}"
                ) from exc
            if level < 0:
                raise ValueError(f"Guard level must be non-negative; got {level}")
            parsed_levels.append(level)
        if not parsed_levels:
            raise ValueError(
                "Guard level entries must use game:level,level format; "
                f"got {entry!r}"
            )
        levels_by_game[game] = sorted(dict.fromkeys(parsed_levels))
    return levels_by_game


def _task_game(task: Any) -> str:
    if isinstance(task, Mapping):
        return str(task["game"])
    return str(getattr(task, "game"))


def _task_level(task: Any) -> int:
    if isinstance(task, Mapping):
        return int(task["level"])
    return int(getattr(task, "level"))


def task_key(task: Any) -> tuple[str, int]:
    """Return the stable cross-evaluation identity for a PuzzleScript task."""
    return _task_game(task), _task_level(task)


def _with_task_id(task: Any, task_id: int) -> Any:
    if is_dataclass(task) and not isinstance(task, type):
        return replace(cast(PuzzleScriptLevelTask, task), task_id=task_id)
    if isinstance(task, Mapping):
        row = dict(task)
        row["task_id"] = task_id
        return row
    if hasattr(task, "__dict__"):
        clone = type(task)(**vars(task))
        setattr(clone, "task_id", task_id)
        return clone
    raise TypeError(f"Cannot reassign task_id for task type {type(task)!r}")


def reassign_task_ids(tasks: Sequence[Any]) -> list[Any]:
    """Return tasks with dense ids starting at zero while preserving order."""
    return [_with_task_id(task, index) for index, task in enumerate(tasks)]


def build_train_dev_tasks(
    tasks: Sequence[Any],
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Split materialized level tasks into game-disjoint train/dev batches.

    This helper differs from `split_train_dev_jobs` by operating after task
    materialization. It is useful when some configured jobs fail to load; the
    held-out split then reflects only tasks that can actually be evaluated.
    """
    games = list(dict.fromkeys(_task_game(task) for task in tasks))
    train_jobs, dev_jobs = split_train_dev_jobs(
        [{"name": game} for game in games],
        dev_fraction=dev_fraction,
        seed=seed,
    )
    train_games = {str(job["name"]) for job in train_jobs}
    dev_games = {str(job["name"]) for job in dev_jobs}
    train_tasks = [task for task in tasks if _task_game(task) in train_games]
    dev_tasks = [task for task in tasks if _task_game(task) in dev_games]
    return reassign_task_ids(train_tasks), reassign_task_ids(dev_tasks)


def unique_tasks_by_key(tasks: Sequence[PuzzleScriptLevelTask]) -> list[PuzzleScriptLevelTask]:
    """Return the first task for each `(game, level)` identity in order."""
    seen: set[tuple[str, int]] = set()
    unique: list[PuzzleScriptLevelTask] = []
    for task in tasks:
        key = task_key(task)
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)
    return unique


def merge_validation_guard_tasks(
    validation_tasks: Sequence[PuzzleScriptLevelTask],
    guard_tasks: Sequence[PuzzleScriptLevelTask],
) -> list[PuzzleScriptLevelTask]:
    """Return validation tasks plus non-duplicate guard tasks with dense ids.

    The guard set augments, rather than replaces, the normal development split.
    When a configured guard level already appears in validation, the original
    validation task wins so the split remains stable and task ids are reassigned
    only after deduplication.
    """
    return cast(
        list[PuzzleScriptLevelTask],
        reassign_task_ids(unique_tasks_by_key([*validation_tasks, *guard_tasks])),
    )


def _is_game_compile_error(error: object) -> bool:
    if error is None:
        return False
    text = str(error).lower()
    return "game compilation failed" in text or "compiling game" in text


def is_candidate_error(output: Mapping[str, Any]) -> bool:
    """Return whether a failed result should count against the generated code.

    PuzzleScript source compilation failures can be properties of the configured
    game rather than the synthesized heuristic. Validation, synthesis, timeout,
    and runtime search failures are candidate-facing and should be discouraged
    by GEPA's scalar objective.
    """
    if output.get("synthesis_error") is not None:
        return True
    error = output.get("error")
    return error is not None and not _is_game_compile_error(error)


def _clamp(value: float, limit: float) -> float:
    bound = max(0.0, limit)
    return max(-bound, min(bound, value))


def _score_normalized_from_snapshot(snapshot: Mapping[str, Any]) -> Optional[float]:
    value = snapshot.get("score_normalized")
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def trace_partial_progress_score(trace_summary: Mapping[str, Any]) -> float:
    """Return the best normalized engine progress visible in an A* trace.

    Unsolved searches otherwise all receive a raw search score of zero. This
    small auxiliary signal gives GEPA a bounded ordering among failed attempts
    without letting partial progress outweigh solve preservation.
    """
    values: list[float] = []
    root_snapshot = trace_summary.get("root_snapshot")
    if isinstance(root_snapshot, Mapping):
        value = _score_normalized_from_snapshot(root_snapshot)
        if value is not None:
            values.append(value)
    for state in trace_summary.get("sampled_states", []) or []:
        if not isinstance(state, Mapping):
            continue
        snapshot = state.get("snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        value = _score_normalized_from_snapshot(snapshot)
        if value is not None:
            values.append(value)
    return max(values) if values else 0.0


def _row_partial_progress_score(row: Mapping[str, Any], *, prefix: str = "") -> float:
    value = row.get(f"{prefix}partial_progress_score")
    if value is not None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
    trace_summary = row.get(f"{prefix}trace_summary")
    if isinstance(trace_summary, Mapping):
        return trace_partial_progress_score(trace_summary)
    return 0.0


def _macro_game_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return per-row weights whose mean is one and games contribute equally."""
    if not rows:
        return []
    counts: dict[str, int] = {}
    for row in rows:
        game = str(row.get("game", ""))
        counts[game] = counts.get(game, 0) + 1
    n_rows = len(rows)
    n_games = max(1, len(counts))
    return [n_rows / (n_games * counts[str(row.get("game", ""))]) for row in rows]


def _has_baseline(row: Mapping[str, Any]) -> bool:
    return "baseline_score" in row or "baseline_solved" in row


def _base_relative_score(
    row: Mapping[str, Any],
    *,
    lost_solve_penalty: float,
    new_solve_bonus: float,
    error_penalty: float,
    score_delta_weight: float,
    score_delta_clip: float,
    partial_progress_weight: float,
) -> float:
    """Score one result by robust improvement over the stored base prompt.

    GEPA only receives per-instance scores and then sums or averages them. This
    transform therefore encodes the desired global-prompt objective locally:
    preserve base solves first, reward new solves second, and let expansion
    efficiency matter only as a capped tie-breaker.
    """
    raw_score = float(row.get("score", 0.0))
    solved = bool(row.get("solved", False))
    partial_weight = max(0.0, partial_progress_weight)
    if not _has_baseline(row):
        score = raw_score if solved else partial_weight * _row_partial_progress_score(row)
    else:
        baseline_score = float(row.get("baseline_score", 0.0))
        baseline_solved = bool(row.get("baseline_solved", False))
        raw_quality = raw_score if solved else partial_weight * _row_partial_progress_score(row)
        baseline_quality = (
            baseline_score
            if baseline_solved
            else partial_weight * _row_partial_progress_score(row, prefix="baseline_")
        )
        delta = _clamp(raw_quality - baseline_quality, score_delta_clip)
        score = max(0.0, score_delta_weight) * delta
        if baseline_solved and not solved:
            score -= max(0.0, lost_solve_penalty)
        elif not baseline_solved and solved:
            score += max(0.0, new_solve_bonus)
    if is_candidate_error(row):
        score -= max(0.0, error_penalty)
    return score


def _result_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric search-result field with a typed fallback."""
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def candidate_score(
    outputs: Sequence[Mapping[str, Any]],
    *,
    lost_solve_penalty: float = DEFAULT_LOST_SOLVE_PENALTY,
    new_solve_bonus: float = DEFAULT_NEW_SOLVE_BONUS,
    error_penalty: float = DEFAULT_CANDIDATE_ERROR_PENALTY,
    score_delta_weight: float = DEFAULT_SCORE_DELTA_WEIGHT,
    score_delta_clip: float = DEFAULT_SCORE_DELTA_CLIP,
    partial_progress_weight: float = DEFAULT_PARTIAL_PROGRESS_WEIGHT,
) -> float:
    """Return the mean GEPA score for one candidate.

    Scores are base-relative when baseline metadata is available and fall back
    to raw A* quality for the initial base-prompt calibration pass. The returned
    mean matches GEPA's validation aggregate because `adjusted_candidate_scores`
    applies the same per-game weights to each instance.
    """
    rows = list(outputs)
    if not rows:
        return 0.0
    scores = adjusted_candidate_scores(
        rows,
        lost_solve_penalty=lost_solve_penalty,
        new_solve_bonus=new_solve_bonus,
        error_penalty=error_penalty,
        score_delta_weight=score_delta_weight,
        score_delta_clip=score_delta_clip,
        partial_progress_weight=partial_progress_weight,
    )
    return sum(scores) / len(scores)


def adjusted_candidate_scores(
    outputs: Sequence[Mapping[str, Any]],
    *,
    lost_solve_penalty: float = DEFAULT_LOST_SOLVE_PENALTY,
    new_solve_bonus: float = DEFAULT_NEW_SOLVE_BONUS,
    error_penalty: float = DEFAULT_CANDIDATE_ERROR_PENALTY,
    score_delta_weight: float = DEFAULT_SCORE_DELTA_WEIGHT,
    score_delta_clip: float = DEFAULT_SCORE_DELTA_CLIP,
    partial_progress_weight: float = DEFAULT_PARTIAL_PROGRESS_WEIGHT,
) -> list[float]:
    """Return macro-game-weighted per-task scores for GEPA.

    The list is still one score per task, matching GEPA's API, but each value is
    weighted so that the arithmetic mean is a per-game macro-average. That keeps
    broad game generalization visible even when one game contributes many more
    levels than another. If any baseline-solved level regresses, an
    evaluation-wide gate penalty is subtracted from every task score. This keeps
    GEPA's Pareto selection from preserving wins produced by globally unsafe
    prompt variants.
    """
    rows = list(outputs)
    weights = _macro_game_weights(rows)
    gate_penalty = (
        max(0.0, lost_solve_penalty)
        if any(
            _has_baseline(row)
            and bool(row.get("baseline_solved", False))
            and not bool(row.get("solved", False))
            for row in rows
        )
        else 0.0
    )
    scores: list[float] = []
    for row, weight in zip(rows, weights, strict=True):
        score = _base_relative_score(
            row,
            lost_solve_penalty=lost_solve_penalty,
            new_solve_bonus=new_solve_bonus,
            error_penalty=error_penalty,
            score_delta_weight=score_delta_weight,
            score_delta_clip=score_delta_clip,
            partial_progress_weight=partial_progress_weight,
        )
        scores.append(score * weight - gate_penalty)
    return scores


def find_game_text_path(name: str, script_doctor: Path) -> Optional[Path]:
    for subdir in ("data/scraped_games", "custom_games"):
        path = script_doctor / subdir / f"{name}.txt"
        if path.exists():
            return path
    return None


def select_level_indices(n_levels: int, requested_levels: int, levels_per_game: int) -> list[int]:
    if n_levels <= 0:
        return []
    capped = min(n_levels, max(1, requested_levels))
    if levels_per_game > 0:
        capped = min(capped, levels_per_game)
        return list(range(capped))
    if capped <= 1:
        return [0]
    return list(range(capped - 1))


def build_level_env_description(
    base_env_description: str,
    engine: Any,
    compiled: Mapping[str, Any],
    level: int,
) -> str:
    engine.load_level(level)
    ctx = build_puzzlescript_ctx(engine, dict(compiled))
    object_positions = {
        str(name): [[int(x), int(y)] for x, y in positions]
        for name, positions in sorted(ctx.get("object_positions", {}).items())
        if str(name).lower() != "background"
    }
    root_summary = {
        "grid_width": ctx.get("grid_width"),
        "grid_height": ctx.get("grid_height"),
        "win_conditions_text": ctx.get("win_conditions_text"),
        "score": ctx.get("score"),
        "score_normalized": ctx.get("score_normalized"),
        "object_counts": {name: len(pos) for name, pos in object_positions.items()},
        "object_positions": object_positions,
    }
    return (
        base_env_description
        + "\n\n"
        + f"Level of interest: {level}\n"
        + "Initial state for this level:\n"
        + str(ctx.get("ascii_state", ""))
        + "\n\nInitial root ctx summary:\n"
        + json.dumps(root_summary, indent=2, sort_keys=True)
        + "\n\nWrite the heuristic for this specific level while still using the rules above."
    )


def build_level_tasks(
    *,
    evaluator: PuzzleScriptEvaluator,
    jobs: Sequence[Mapping[str, Any]],
    script_doctor: Path,
    levels_per_game: int,
    budget: int,
    selected_levels_by_game: Mapping[str, Sequence[int]] | None = None,
) -> list[PuzzleScriptLevelTask]:
    tasks: list[PuzzleScriptLevelTask] = []
    for entry in jobs:
        name = str(entry["name"])
        if selected_levels_by_game is not None and name not in selected_levels_by_game:
            continue
        game_text_path = find_game_text_path(name, script_doctor)
        if game_text_path is None:
            print(f"[inputs] skipping {name}: no game text found")
            continue
        game_text = game_text_path.read_text(encoding="utf-8")
        try:
            json_str = evaluator.compile_game(game_text)
            compiled = json.loads(json_str)
            engine = evaluator.load_engine(json_str)
            engine.load_level(0)
            n_levels = engine.get_num_levels()
            if selected_levels_by_game is not None:
                selected = [
                    int(level)
                    for level in selected_levels_by_game.get(name, [])
                    if 0 <= int(level) < n_levels
                ]
                if len(selected) < len(selected_levels_by_game.get(name, [])):
                    print(
                        f"[inputs] {name}: skipped guard levels outside [0, {n_levels})",
                        flush=True,
                    )
            else:
                requested_levels = max(1, int(entry.get("levels", n_levels) or n_levels))
                selected = select_level_indices(n_levels, requested_levels, levels_per_game)
            base_desc = build_env_description(compiled, engine.get_id_dict(), game_text)
            for level in selected:
                try:
                    env_description = build_level_env_description(base_desc, engine, compiled, level)
                except Exception as exc:
                    print(f"[inputs] skipping {name} level={level}: {exc}")
                    continue
                tasks.append(
                    PuzzleScriptLevelTask(
                        task_id=len(tasks),
                        game=name,
                        level=int(level),
                        budget=int(budget),
                        env_description=env_description,
                        game_text_path=str(game_text_path),
                    )
                )
        except Exception as exc:
            print(f"[inputs] skipping {name}: {exc}")
    return tasks


def build_synthesis_prompt(prompt_text: str, task: PuzzleScriptLevelTask) -> str:
    return (
        prompt_text.strip()
        + "\n\nPuzzleScript game and level context:\n"
        + task.env_description
        + "\n\nReturn only the Python function. Do not include markdown fences."
    )


def build_repair_prompt(
    prompt_text: str,
    task: PuzzleScriptLevelTask,
    bad_code: str,
    issue: str,
) -> str:
    return (
        build_synthesis_prompt(prompt_text, task)
        + "\n\nThe previous output failed validation. Repair it once.\n"
        + f"Validation issue: {issue}\n"
        + "Previous output:\n"
        + truncate_text(strip_outer_markdown_fences(bad_code), 12_000)
    )


def validate_heuristic_code(code: str) -> Optional[str]:
    try:
        sanitize_and_compile_puzzlescript_heuristic(code)
    except Exception as exc:
        return str(exc)
    return None


def synthesize_heuristic_code(
    *,
    llm: TextCompletionClient,
    prompt_text: str,
    task: PuzzleScriptLevelTask,
) -> tuple[str, Optional[str]]:
    first = llm.complete(build_synthesis_prompt(prompt_text, task))
    code = strip_outer_markdown_fences(first)
    issue = validate_heuristic_code(code)
    if issue is None:
        return code, None

    repaired = llm.complete(build_repair_prompt(prompt_text, task, code, issue))
    repaired_code = strip_outer_markdown_fences(repaired)
    repaired_issue = validate_heuristic_code(repaired_code)
    if repaired_issue is None:
        return repaired_code, None
    return repaired_code, repaired_issue


def assigned_tasks(tasks: Sequence[Mapping[str, Any]], array_index: int, array_count: int) -> list[dict[str, Any]]:
    if array_count <= 0:
        raise ValueError("array_count must be > 0")
    if array_index < 0 or array_index >= array_count:
        raise ValueError(f"array_index {array_index} outside [0, {array_count})")
    return [dict(task) for i, task in enumerate(tasks) if i % array_count == array_index]


def search_failure_result(task: Mapping[str, Any], *, feedback: str, error: str) -> dict[str, Any]:
    """Return a standard failed search result for one manifest task.

    The array merger expects every task result to carry the same identity and
    score fields, regardless of whether the failure happened in validation,
    PuzzleScript compilation, A* search, or the outer wall-clock guard.
    """
    code_path = Path(str(task.get("heuristic_code_path", "")))
    return {
        "task_id": int(task["task_id"]),
        "game": str(task["game"]),
        "level": int(task["level"]),
        "score": 0.0,
        "partial_progress_score": 0.0,
        "solved": False,
        "expanded": 0,
        "generated": 0,
        "solution_length": 0,
        "feedback": feedback,
        "heuristic_code_path": str(code_path),
        "error": error,
    }


def evaluate_search_task(
    *,
    evaluator: PuzzleScriptEvaluator,
    task: Mapping[str, Any],
    astar_timeout_s: float,
) -> dict[str, Any]:
    code_path = Path(str(task["heuristic_code_path"]))
    code = code_path.read_text(encoding="utf-8")
    game = str(task["game"])
    level = int(task["level"])
    budget = int(task["budget"])
    validation_error = validate_heuristic_code(code)
    if validation_error:
        return search_failure_result(
            task,
            feedback=f"Heuristic validation failed before search: {validation_error}",
            error=validation_error,
        )

    try:
        raw_fn = sanitize_and_compile_puzzlescript_heuristic(code)

        def heuristic_fn(ctx: dict[str, Any]) -> float:
            return float(raw_fn(None, None, ctx))

        game_text = Path(str(task["game_text_path"])).read_text(encoding="utf-8")
        json_str = evaluator.compile_game(game_text)
        compiled = json.loads(json_str)
        engine = evaluator.load_engine(json_str)
        engine.load_level(level)
        result = puzzlescript_astar(
            engine=engine,
            compiled_json=compiled,
            heuristic_fn=heuristic_fn,
            max_expansions=budget,
            timeout_s=astar_timeout_s,
        )
        score = heuristic_score(result.solved, result.expanded_states, budget)
        partial_progress = trace_partial_progress_score(result.trace_summary)
        feedback = (
            f"Game={game} level={level} solved={result.solved} "
            f"score={score:.4f} expanded={result.expanded_states}/{budget} "
            f"generated={result.generated_states} solution_length={result.solution_length} "
            f"terminated={result.trace_summary.get('terminated_reason', 'unknown')} "
            f"partial_progress={partial_progress:.3f}"
        )
        if not result.solved:
            feedback += "\nTrace summary:\n" + json.dumps(
                result.trace_summary,
                indent=2,
                sort_keys=True,
            )
        return {
            "task_id": int(task["task_id"]),
            "game": game,
            "level": level,
            "score": score,
            "partial_progress_score": partial_progress,
            "raw_search_score": result.score,
            "solved": result.solved,
            "expanded": result.expanded_states,
            "generated": result.generated_states,
            "solution_length": result.solution_length,
            "time_s": result.time_s,
            "feedback": feedback,
            "trace_summary": result.trace_summary,
            "heuristic_code_path": str(code_path),
        }
    except Exception as exc:
        return search_failure_result(
            task,
            feedback=f"Search evaluation failed: {exc}",
            error=str(exc),
        )


def _search_task_worker(
    script_doctor: str,
    task: dict[str, Any],
    astar_timeout_s: float,
    result_queue: Any,
) -> None:
    """Evaluate one search task in a child process and return its result."""
    try:
        evaluator = PuzzleScriptEvaluator(Path(script_doctor))
        result = evaluate_search_task(
            evaluator=evaluator,
            task=task,
            astar_timeout_s=astar_timeout_s,
        )
    except BaseException as exc:  # pragma: no cover - defensive child-process boundary.
        result = search_failure_result(
            task,
            feedback=f"Search evaluation worker failed: {exc}",
            error=str(exc),
        )
    result_queue.put(result)


def _multiprocessing_context() -> Any:
    if "spawn" in mp.get_all_start_methods():
        return mp.get_context("spawn")
    return mp.get_context("fork")


def evaluate_search_task_with_wall_timeout(
    *,
    script_doctor: Path,
    task: Mapping[str, Any],
    astar_timeout_s: float,
    wall_timeout_s: float,
    worker: Any = _search_task_worker,
) -> dict[str, Any]:
    """Evaluate one search task with a killable wall-clock guard.

    `puzzlescript_astar` has an internal timeout, but compilation, engine calls,
    and generated heuristic execution can still wedge inside native or JS code.
    Running each task in a child process lets the array worker convert those
    cases into ordinary failed results instead of blocking the whole GEPA round.
    """
    task_payload = dict(task)
    timeout_s = max(1.0, wall_timeout_s)
    ctx = _multiprocessing_context()
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=worker,
        args=(str(script_doctor), task_payload, astar_timeout_s, result_queue),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        return search_failure_result(
            task_payload,
            feedback=f"Search evaluation timed out after {timeout_s:.1f}s wall time.",
            error=f"search task wall timeout after {timeout_s:.1f}s",
        )
    if process.exitcode not in (0, None):
        return search_failure_result(
            task_payload,
            feedback=f"Search evaluation worker exited with code {process.exitcode}.",
            error=f"worker exit code {process.exitcode}",
        )
    try:
        result = result_queue.get_nowait()
    except Empty:
        return search_failure_result(
            task_payload,
            feedback="Search evaluation worker exited without returning a result.",
            error="missing worker result",
        )
    return dict(result)


def evaluate_manifest_shard(
    *,
    manifest_path: Path,
    array_index: int,
    array_count: int,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_dir = Path(str(manifest["shard_dir"]))
    shard_dir.mkdir(parents=True, exist_ok=True)
    script_doctor = Path(str(manifest["script_doctor"]))
    astar_timeout_s = float(manifest["astar_timeout_s"])
    wall_timeout_s = float(
        manifest.get(
            "task_wall_timeout_s",
            max(DEFAULT_SEARCH_TASK_WALL_TIMEOUT_S, astar_timeout_s + 90.0),
        )
    )
    tasks = assigned_tasks(manifest["tasks"], array_index, array_count)
    results = [
        evaluate_search_task_with_wall_timeout(
            script_doctor=script_doctor,
            task=task,
            astar_timeout_s=astar_timeout_s,
            wall_timeout_s=wall_timeout_s,
        )
        for task in tasks
    ]
    shard_path = shard_dir / f"task-{array_index:04d}-of-{array_count:04d}.json"
    tmp_path = shard_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "array_index": array_index,
                "array_count": array_count,
                "n_tasks": len(tasks),
                "results": results,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(shard_path)
    return shard_path


def build_sbatch_array_command(
    *,
    manifest_path: Path,
    array_script: Path,
    array_count: int,
    array_concurrency: int,
    extra_sbatch_args: Sequence[str] = (),
    wait: bool = False,
) -> list[str]:
    if array_count <= 0:
        raise ValueError("array_count must be > 0")
    if array_concurrency <= 0:
        raise ValueError("array_concurrency must be > 0")
    wait_args = ["--wait"] if wait else []
    return [
        "sbatch",
        "--parsable",
        *wait_args,
        f"--array=0-{array_count - 1}%{array_concurrency}",
        f"--export=ALL,EVAL_MANIFEST={manifest_path},SEARCH_ARRAY_COUNT={array_count}",
        *extra_sbatch_args,
        str(array_script),
    ]


class SearchArrayStalledError(RuntimeError):
    """Raised when no new search shards arrive before the stall timeout."""

    def __init__(
        self,
        *,
        missing_indices: Sequence[int],
        present_count: int,
        array_count: int,
        stall_timeout_s: float,
    ) -> None:
        self.missing_indices = list(missing_indices)
        self.present_count = present_count
        self.array_count = array_count
        self.stall_timeout_s = stall_timeout_s
        super().__init__(
            "Search array stalled after "
            f"{stall_timeout_s:.1f}s with {present_count}/{array_count} shards present; "
            f"missing={self.missing_indices[:12]}"
        )


def expected_shard_paths(*, shard_dir: Path, array_count: int) -> list[Path]:
    """Return the canonical shard paths for one search manifest."""
    return [
        shard_dir / f"task-{idx:04d}-of-{array_count:04d}.json"
        for idx in range(array_count)
    ]


def missing_shard_indices(*, shard_dir: Path, array_count: int) -> list[int]:
    """Return array indices whose shard files have not been materialized."""
    return [
        index
        for index, path in enumerate(
            expected_shard_paths(shard_dir=shard_dir, array_count=array_count)
        )
        if not path.exists()
    ]


def wait_for_shards(
    *,
    shard_dir: Path,
    array_count: int,
    poll_interval_s: float,
    stall_timeout_s: float = 0.0,
) -> list[Path]:
    expected = expected_shard_paths(shard_dir=shard_dir, array_count=array_count)
    last_present_count = -1
    last_progress_time = time.monotonic()
    while True:
        missing = [path for path in expected if not path.exists()]
        if not missing:
            return expected
        present_count = array_count - len(missing)
        if present_count != last_present_count:
            last_present_count = present_count
            last_progress_time = time.monotonic()
        elif stall_timeout_s > 0.0 and time.monotonic() - last_progress_time >= stall_timeout_s:
            raise SearchArrayStalledError(
                missing_indices=[
                    index for index, path in enumerate(expected) if not path.exists()
                ],
                present_count=present_count,
                array_count=array_count,
                stall_timeout_s=stall_timeout_s,
            )
        print(f"[search-array] waiting for {len(missing)} shard(s)...", flush=True)
        time.sleep(poll_interval_s)


def load_shard_results(shard_paths: Sequence[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.extend(dict(row) for row in payload.get("results", []))
    return sorted(results, key=lambda row: int(row["task_id"]))


def _read_optional_text(path_value: object, *, max_chars: int) -> str:
    if not path_value:
        return ""
    try:
        path = Path(str(path_value))
        if not path.exists():
            return ""
        return truncate_text(path.read_text(encoding="utf-8"), max_chars)
    except OSError:
        return ""


def _result_summary(prefix: str, result: Mapping[str, Any], *, baseline: bool = False) -> str:
    key_prefix = "baseline_" if baseline else ""
    return (
        f"{prefix}: solved={bool(result.get(f'{key_prefix}solved', False))} "
        f"score={float(result.get(f'{key_prefix}score', 0.0)):.4f} "
        f"expanded={result.get(f'{key_prefix}expanded', 'unknown')} "
        f"generated={result.get(f'{key_prefix}generated', 'unknown')} "
        f"solution_length={result.get(f'{key_prefix}solution_length', 'unknown')}"
    )


def _compact_trace_diagnostics(result: Mapping[str, Any]) -> str:
    """Return one short trace summary line for reflection feedback."""
    trace_summary = result.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        return ""
    values: list[float] = []
    root_snapshot = trace_summary.get("root_snapshot")
    if isinstance(root_snapshot, Mapping):
        value = _score_normalized_from_snapshot(root_snapshot)
        if value is not None:
            values.append(value)
    for state in trace_summary.get("sampled_states", []) or []:
        if not isinstance(state, Mapping):
            continue
        snapshot = state.get("snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        value = _score_normalized_from_snapshot(snapshot)
        if value is not None:
            values.append(value)
    parts = ["Candidate trace diagnostics:"]
    if values:
        parts.append(f"progress_range={min(values):.3f}..{max(values):.3f}")
    for key in ("best_seen_h", "best_seen_f", "open_set_size_at_end", "terminated_reason"):
        if key in trace_summary:
            parts.append(f"{key}={trace_summary[key]}")
    return " ".join(parts) if len(parts) > 1 else ""


def build_reflection_feedback(result: Mapping[str, Any], classification: str) -> str:
    """Return targeted feedback for GEPA reflection.

    Raw search traces are useful but insufficient by themselves. The reflection
    LLM needs to know whether the candidate regressed a level solved by the base
    prompt, found a new solve worth preserving, or simply remained stuck.
    """
    candidate_feedback = str(result.get("feedback", ""))
    baseline_feedback = str(result.get("baseline_feedback", ""))
    lines: list[str] = []
    if classification == "lost_baseline_solve":
        lines.extend(
            [
                "REGRESSION: the base prompt solved this level, but the candidate failed.",
                _result_summary("Base result", result, baseline=True),
                _result_summary("Candidate result", result),
                "Reflection target: preserve the base behavior for this mechanics pattern while changing only the general rule that caused the regression.",
            ]
        )
    elif classification == "new_solve":
        lines.extend(
            [
                "POSITIVE EXAMPLE: the candidate solved this level while the base prompt did not.",
                _result_summary("Base result", result, baseline=True),
                _result_summary("Candidate result", result),
                "Reflection target: preserve the general insight that produced this new solve without adding narrow game-specific memorization.",
            ]
        )
    elif classification == "solved_regression":
        lines.extend(
            [
                "EFFICIENCY REGRESSION: both prompts solved this level, but the candidate was materially worse.",
                _result_summary("Base result", result, baseline=True),
                _result_summary("Candidate result", result),
                "Reflection target: keep the solve but avoid the extra search caused by the candidate heuristic.",
            ]
        )
    elif classification == "candidate_error":
        lines.extend(
            [
                "CANDIDATE ERROR: the generated heuristic failed validation or search execution.",
                _result_summary("Candidate result", result),
                "Reflection target: strengthen the global code-generation contract without overfitting to this one level.",
            ]
        )
    elif classification == "persistent_failure":
        lines.extend(
            [
                "PERSISTENT FAILURE: neither the base nor candidate solved this level.",
                _result_summary("Base result", result, baseline=True),
                _result_summary("Candidate result", result),
                "Reflection target: propose only a general heuristic principle if the trace reveals one; otherwise avoid bloating the global prompt.",
            ]
        )
    else:
        lines.extend(
            [
                "STABLE OR IMPROVED CASE: use this as low-priority context for what already works.",
                _result_summary("Base result", result, baseline=True),
                _result_summary("Candidate result", result),
            ]
        )

    if baseline_feedback:
        lines.append("Base raw feedback:\n" + baseline_feedback)
    if candidate_feedback:
        lines.append("Candidate raw feedback:\n" + candidate_feedback)
    diagnostics = _compact_trace_diagnostics(result)
    if diagnostics and classification != "stable_or_improved":
        lines.append(diagnostics)
    return "\n".join(lines)


def build_comparison_payload(result: Mapping[str, Any], classification: str) -> dict[str, Any]:
    raw_score = float(result.get("score", 0.0))
    baseline_score = float(result.get("baseline_score", 0.0))
    return {
        "classification": classification,
        "candidate_solved": bool(result.get("solved", False)),
        "baseline_solved": bool(result.get("baseline_solved", False)),
        "candidate_score": raw_score,
        "baseline_score": baseline_score,
        "score_delta": raw_score - baseline_score,
        "candidate_error": is_candidate_error(result),
    }


def _has_dangling_prompt_tail(text: str) -> bool:
    """Return True when prompt text appears cut at an unfinished list or heading.

    GEPA proposals are later reused as high-level instructions. Accepting a
    syntactically valid but truncated note, such as a final bullet marker or
    heading ending in a colon, gives the synthesis model an incoherent contract
    and has been a recurring source of generic fallback heuristics.
    """
    nonempty_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return True
    tail = nonempty_lines[-1].strip()
    if tail in {"-", "*", "•"}:
        return True
    if re.match(r"^(?:[-*]\s*)?(?:\d+[.)]\s*)?$", tail):
        return True
    if re.match(r"^#{1,6}\s+\S.*$", tail):
        return True
    return tail.endswith(":")


def _strip_gepa_addendum(prompt_text: str) -> str:
    """Return the stable base portion of a prompt before GEPA's generated addendum."""
    marker = "\n\n" + GEPA_ADDENDUM_HEADER
    if marker in prompt_text:
        return prompt_text.split(marker, 1)[0].rstrip()
    if GEPA_ADDENDUM_HEADER in prompt_text:
        return prompt_text.split(GEPA_ADDENDUM_HEADER, 1)[0].rstrip()
    return prompt_text.rstrip()


def _compose_prompt_with_addendum(base_prompt: str, addendum: str) -> str:
    """Attach one short GEPA addendum while keeping the base prompt unchanged."""
    return f"{_strip_gepa_addendum(base_prompt)}\n\n{GEPA_ADDENDUM_HEADER}\n{addendum.strip()}"


def _clean_proposed_prompt(text: str, *, max_chars: int) -> str:
    cleaned = strip_outer_markdown_fences(text).strip()
    cleaned = re.sub(
        r"^(?:revised global prompt|new instruction|prompt)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if candidate_prompt_issue(cleaned) is not None:
        return ""
    required_markers = (
        "WINCONDITIONS",
        "RULES",
        "COLLISIONLAYERS",
        "LEGEND",
        "initial level state",
        "score_normalized",
        "Return 0.0",
        "Do not output",
    )
    if any(marker not in cleaned for marker in required_markers):
        return ""
    if _has_dangling_prompt_tail(cleaned):
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return ""


def _clean_proposed_addendum(text: str, *, max_chars: int) -> str:
    """Clean a proposed GEPA prompt addendum and reject drift-prone shapes."""
    cleaned = strip_outer_markdown_fences(text).strip()
    addendum_labels = list(
        re.finditer(
            r"(?:^|\n)\s*(?:additional gepa guidance|addendum|additional guidance)\s*:\s*",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    if addendum_labels:
        cleaned = cleaned[addendum_labels[-1].end() :].strip()
    else:
        cleaned = re.sub(
            r"^(?:revised global prompt|new instruction|prompt)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
    if candidate_prompt_issue(cleaned) is not None:
        return ""
    if not cleaned:
        return ""
    full_prompt_markers = (
        "You are writing a heuristic function",
        "Output exactly Python code defining",
        "Runtime ctx keys include",
    )
    if any(marker in cleaned for marker in full_prompt_markers):
        return ""
    if len(cleaned) > max_chars:
        return ""
    if _has_dangling_prompt_tail(cleaned):
        return ""
    return cleaned


def _fallback_addendum_from_feedback(records: Sequence[Mapping[str, Any]]) -> str:
    """Return a conservative exploration addendum when the proposer no-ops.

    Local LLMs sometimes answer the addendum prompt by restating the base prompt
    or by producing code. Keeping the current prompt in that case makes GEPA
    spend every iteration on an exact baseline reuse. This fallback creates one
    safe, general candidate from the available feedback while preserving the
    base prompt's mechanics-first contract.
    """
    if not records:
        return ""
    classifications = {
        str(cast(Mapping[str, Any], record.get("Comparison", {})).get("classification", ""))
        for record in records
    }
    if "lost_baseline_solve" in classifications:
        return (
            "When a change risks a base-solved level, keep exact WINCONDITIONS, RULES, "
            "LEGEND names, and collision-layer mechanics primary; use score_normalized "
            "only as a tie-breaker and skip unproven deadlock penalties."
        )
    if "new_solve" in classifications:
        return (
            "Generalize new-solve evidence only through exact rule and win-condition "
            "features: count unsatisfied objects by LEGEND name, prefer legal progress "
            "moves, and keep generic role fallback secondary."
        )
    return (
        "For base-unsolved levels, after reading WINCONDITIONS, RULES, and LEGEND, "
        "add one conservative win-condition distance feature using exact object names "
        "and legal progress moves; keep score_normalized only as a tie-breaker."
    )


def heuristic_code_shape(code: str) -> dict[str, bool]:
    """Return compact flags describing generated heuristic strategy shape.

    Reflection already includes truncated code, but these booleans make it cheap
    for the proposal LLM to notice when a candidate drifted from concrete
    mechanics-specific terms into generic role discovery or score-only fallback.
    """
    lowered = code.lower()
    return {
        "uses_generic_role_helpers": "role_positions" in lowered or "role_keywords" in lowered,
        "uses_score_fallback": "score_normalized" in lowered,
        "uses_win_text": "win_conditions_text" in lowered,
        "uses_ascii_state": "ascii_state" in lowered,
        "uses_large_penalty": any(
            marker in lowered for marker in ("1e6", "1000.0", "5000.0", "1e5")
        ),
        "mentions_mechanics_terms": any(
            term in lowered
            for term in (
                "button",
                "door",
                "portal",
                "laser",
                "beam",
                "ice",
                "gravity",
                "swap",
                "pull",
                "mud",
                "boulder",
                "exit",
            )
        ),
    }


def _initial_eval_counter(candidate_eval_root: Path) -> int:
    """Return the highest existing adapter evaluation counter.

    GEPA resume state lives under `gepa_run`, while the batched adapter writes
    per-candidate synthesis and search artifacts under `candidate_evals`. When a
    killed Slurm job is resumed with the same state root, the adapter must append
    new evaluation directories instead of overwriting previous evidence.
    Nonconforming directory names are ignored so manual notes or partial files do
    not block resuming.
    """

    max_counter = 0
    for path in candidate_eval_root.glob("eval-*"):
        parts = path.name.split("-", 2)
        if len(parts) < 2:
            continue
        try:
            counter = int(parts[1])
        except ValueError:
            continue
        max_counter = max(max_counter, counter)
    return max_counter


class PuzzleScriptBatchedGEPAAdapter:
    def __init__(
        self,
        *,
        llm: TextCompletionClient,
        state_root: Path,
        script_doctor: Path,
        search_config: SearchArrayConfig,
        llm_concurrency: int,
        astar_timeout_s: float,
        lost_solve_penalty: float = DEFAULT_LOST_SOLVE_PENALTY,
        new_solve_bonus: float = DEFAULT_NEW_SOLVE_BONUS,
        candidate_error_penalty: float = DEFAULT_CANDIDATE_ERROR_PENALTY,
        score_delta_weight: float = DEFAULT_SCORE_DELTA_WEIGHT,
        score_delta_clip: float = DEFAULT_SCORE_DELTA_CLIP,
        partial_progress_weight: float = DEFAULT_PARTIAL_PROGRESS_WEIGHT,
    ) -> None:
        self.llm = llm
        self.state_root = state_root
        self.script_doctor = script_doctor
        self.search_config = search_config
        self.llm_concurrency = max(1, llm_concurrency)
        self.astar_timeout_s = astar_timeout_s
        self.lost_solve_penalty = max(0.0, lost_solve_penalty)
        self.new_solve_bonus = max(0.0, new_solve_bonus)
        self.candidate_error_penalty = max(0.0, candidate_error_penalty)
        self.score_delta_weight = max(0.0, score_delta_weight)
        self.score_delta_clip = max(0.0, score_delta_clip)
        self.partial_progress_weight = max(0.0, partial_progress_weight)
        self.baseline_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        self.eval_counter = _initial_eval_counter(self.state_root / "candidate_evals")

    def set_baseline_outputs(self, outputs: Sequence[Mapping[str, Any]]) -> None:
        """Store base-prompt outcomes used to penalize solve regressions."""
        self.baseline_by_key = {
            (str(row["game"]), int(row["level"])): dict(row)
            for row in outputs
        }

    def _attach_baseline_metadata(self, result: dict[str, Any]) -> None:
        baseline = self.baseline_by_key.get((str(result["game"]), int(result["level"])))
        if baseline is None:
            return
        result["baseline_score"] = float(baseline.get("score", 0.0))
        result["baseline_solved"] = bool(baseline.get("solved", False))
        result["baseline_error"] = baseline.get("error")
        result["baseline_expanded"] = baseline.get("expanded")
        result["baseline_generated"] = baseline.get("generated")
        result["baseline_solution_length"] = baseline.get("solution_length")
        result["baseline_partial_progress_score"] = baseline.get("partial_progress_score", 0.0)
        result["baseline_feedback"] = baseline.get("feedback")
        result["baseline_heuristic_code_path"] = baseline.get("heuristic_code_path")

    def _next_eval_dir(self, candidate: Mapping[str, str], batch: Sequence[PuzzleScriptLevelTask]) -> Path:
        self.eval_counter += 1
        candidate_hash = hashlib.sha256(
            json.dumps(candidate, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        task_hash = hashlib.sha256(
            ",".join(f"{task.game}:{task.level}" for task in batch).encode("utf-8")
        ).hexdigest()[:8]
        eval_dir = self.state_root / "candidate_evals" / (
            f"eval-{self.eval_counter:05d}-{candidate_hash}-{task_hash}"
        )
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

    def _synthesize_batch(
        self,
        *,
        candidate: Mapping[str, str],
        batch: Sequence[PuzzleScriptLevelTask],
        eval_dir: Path,
    ) -> list[dict[str, Any]]:
        prompt_text = candidate.get(HEURISTIC_COMPONENT, PUZZLESCRIPT_HEURISTIC_CONTRACT)
        heuristics_dir = eval_dir / "heuristics"
        heuristics_dir.mkdir(parents=True, exist_ok=True)
        rows: list[Optional[dict[str, Any]]] = [None] * len(batch)

        def _one(index: int, task: PuzzleScriptLevelTask) -> tuple[int, dict[str, Any]]:
            code, error = synthesize_heuristic_code(llm=self.llm, prompt_text=prompt_text, task=task)
            code_path = heuristics_dir / f"{task.task_id:04d}-{safe_name(task.game)}-level-{task.level:02d}.py"
            code_path.write_text(code + "\n", encoding="utf-8")
            row = asdict(task)
            row["heuristic_code_path"] = str(code_path)
            row["synthesis_error"] = error
            return index, row

        with ThreadPoolExecutor(max_workers=self.llm_concurrency) as pool:
            futures = [pool.submit(_one, index, task) for index, task in enumerate(batch)]
            for future in as_completed(futures):
                index, row = future.result()
                rows[index] = row

        materialized = [row for row in rows if row is not None]
        (eval_dir / "synthesis_manifest.json").write_text(
            json.dumps(materialized, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return materialized

    def _run_search(self, *, eval_dir: Path, task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        array_count = min(max(1, self.search_config.array_count), max(1, len(task_rows)))
        shard_dir = eval_dir / "search_shards"
        manifest_path = eval_dir / "search_manifest.json"
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script_doctor": str(self.script_doctor),
            "astar_timeout_s": self.astar_timeout_s,
            "task_wall_timeout_s": max(
                DEFAULT_SEARCH_TASK_WALL_TIMEOUT_S,
                self.astar_timeout_s + 90.0,
            ),
            "shard_dir": str(shard_dir),
            "array_count": array_count,
            "tasks": task_rows,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        submitted_job_id: Optional[str] = None
        if self.search_config.submit:
            command = build_sbatch_array_command(
                manifest_path=manifest_path,
                array_script=self.search_config.array_script,
                array_count=array_count,
                array_concurrency=min(self.search_config.array_concurrency, array_count),
                extra_sbatch_args=self.search_config.extra_sbatch_args,
            )
            print("[search-array] submitting: " + " ".join(command), flush=True)
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            submitted_job_id = completed.stdout.strip().splitlines()[-1].strip()
            print(f"[search-array] submitted job_id={submitted_job_id}", flush=True)
        else:
            for array_index in range(array_count):
                evaluate_manifest_shard(
                    manifest_path=manifest_path,
                    array_index=array_index,
                    array_count=array_count,
                )

        try:
            shard_paths = wait_for_shards(
                shard_dir=shard_dir,
                array_count=array_count,
                poll_interval_s=self.search_config.poll_interval_s,
                stall_timeout_s=(
                    self.search_config.stall_timeout_s if self.search_config.submit else 0.0
                ),
            )
        except SearchArrayStalledError as exc:
            print(f"[search-array] {exc}; falling back to local missing shards", flush=True)
            if submitted_job_id:
                subprocess.run(["scancel", submitted_job_id], check=False)
                print(f"[search-array] cancelled stalled job_id={submitted_job_id}", flush=True)
            for array_index in exc.missing_indices:
                evaluate_manifest_shard(
                    manifest_path=manifest_path,
                    array_index=array_index,
                    array_count=array_count,
                )
            shard_paths = wait_for_shards(
                shard_dir=shard_dir,
                array_count=array_count,
                poll_interval_s=self.search_config.poll_interval_s,
                stall_timeout_s=0.0,
            )
        results = load_shard_results(shard_paths)
        (eval_dir / "merged_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return results

    def _reuse_baseline_evaluation(
        self,
        *,
        eval_dir: Path,
        batch: Sequence[PuzzleScriptLevelTask],
        capture_traces: bool,
    ) -> Optional[tuple[list[dict[str, Any]], list[float], Optional[list[dict[str, Any]]]]]:
        """Return stored base-prompt outputs for an exact base candidate.

        The baseline pass already synthesized and searched the base prompt for
        every train and validation task. Reusing those rows for later exact
        base-prompt evaluations removes LLM resampling noise from the regression
        metric while leaving all non-base candidate prompts on the normal
        synthesis/search path.
        """
        if not self.baseline_by_key:
            return None
        outputs: list[dict[str, Any]] = []
        trajectories: list[dict[str, Any]] = []
        for task in batch:
            baseline = self.baseline_by_key.get((task.game, task.level))
            if baseline is None:
                return None
            result = dict(baseline)
            result["task_id"] = task.task_id
            result["game"] = task.game
            result["level"] = task.level
            self._attach_baseline_metadata(result)
            outputs.append(result)
            if capture_traces:
                code_path = result.get("heuristic_code_path") or result.get(
                    "baseline_heuristic_code_path"
                )
                heuristic_code = _read_optional_text(code_path, max_chars=16_000)
                trajectories.append(
                    {
                        "task": asdict(task),
                        "heuristic_code_path": code_path,
                        "heuristic_code": heuristic_code,
                        "synthesis_error": result.get("synthesis_error"),
                        "result": result,
                    }
                )

        scores = adjusted_candidate_scores(
            outputs,
            lost_solve_penalty=self.lost_solve_penalty,
            new_solve_bonus=self.new_solve_bonus,
            error_penalty=self.candidate_error_penalty,
            score_delta_weight=self.score_delta_weight,
            score_delta_clip=self.score_delta_clip,
            partial_progress_weight=self.partial_progress_weight,
        )
        for output, adjusted_score in zip(outputs, scores, strict=True):
            output["adjusted_score"] = adjusted_score
        (eval_dir / "baseline_reuse.json").write_text(
            json.dumps(
                {
                    "reused": True,
                    "n_outputs": len(outputs),
                    "reason": "exact base prompt with stored baseline outputs",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (eval_dir / "synthesis_manifest.json").write_text("[]\n", encoding="utf-8")
        (eval_dir / "merged_results.json").write_text(
            json.dumps(outputs, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (eval_dir / "scored_results.json").write_text(
            json.dumps(outputs, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return outputs, scores, trajectories if capture_traces else None

    def evaluate(
        self,
        batch: list[PuzzleScriptLevelTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ):
        from gepa import EvaluationBatch

        eval_dir = self._next_eval_dir(candidate, batch)
        (eval_dir / "candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[adapter] evaluating {len(batch)} level(s) in {eval_dir.name}; "
            f"capture_traces={capture_traces}",
            flush=True,
        )
        prompt_text = candidate.get(HEURISTIC_COMPONENT, PUZZLESCRIPT_HEURISTIC_CONTRACT)
        prompt_issue = candidate_prompt_issue(prompt_text)
        if prompt_issue is not None:
            print(f"[adapter] rejecting candidate prompt: {prompt_issue}", flush=True)
            violation_outputs: list[dict[str, Any]] = []
            violation_trajectories: list[dict[str, Any]] = []
            for task in batch:
                result = {
                    "task_id": task.task_id,
                    "game": task.game,
                    "level": task.level,
                    "score": 0.0,
                    "partial_progress_score": 0.0,
                    "solved": False,
                    "expanded": 0,
                    "generated": 0,
                    "solution_length": 0,
                    "feedback": (
                        "Prompt contract violation: GEPA proposed implementation code "
                        f"instead of reusable instruction text. Issue: {prompt_issue}"
                    ),
                    "error": f"prompt contract violation: {prompt_issue}",
                }
                self._attach_baseline_metadata(result)
                violation_outputs.append(result)
                if capture_traces:
                    violation_trajectories.append(
                        {
                            "task": asdict(task),
                            "heuristic_code_path": None,
                            "heuristic_code": "",
                            "synthesis_error": result["error"],
                            "result": result,
                        }
                    )
            violation_scores = adjusted_candidate_scores(
                violation_outputs,
                lost_solve_penalty=self.lost_solve_penalty,
                new_solve_bonus=self.new_solve_bonus,
                error_penalty=self.candidate_error_penalty,
                score_delta_weight=self.score_delta_weight,
                score_delta_clip=self.score_delta_clip,
                partial_progress_weight=self.partial_progress_weight,
            )
            for output, adjusted_score in zip(violation_outputs, violation_scores, strict=True):
                output["adjusted_score"] = adjusted_score
            (eval_dir / "synthesis_manifest.json").write_text("[]\n", encoding="utf-8")
            (eval_dir / "merged_results.json").write_text(
                json.dumps(violation_outputs, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            (eval_dir / "scored_results.json").write_text(
                json.dumps(violation_outputs, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            return EvaluationBatch(
                outputs=violation_outputs,
                scores=violation_scores,
                trajectories=violation_trajectories if capture_traces else None,
            )

        if prompt_text.strip() == PUZZLESCRIPT_HEURISTIC_CONTRACT.strip():
            reused = self._reuse_baseline_evaluation(
                eval_dir=eval_dir,
                batch=batch,
                capture_traces=capture_traces,
            )
            if reused is not None:
                reused_outputs, reused_scores, trajectories_or_none = reused
                print(
                    f"[adapter] reused baseline outputs for exact base prompt: "
                    f"{len(reused_outputs)} level(s)",
                    flush=True,
                )
                return EvaluationBatch(
                    outputs=reused_outputs,
                    scores=reused_scores,
                    trajectories=trajectories_or_none,
                )

        task_rows = self._synthesize_batch(candidate=candidate, batch=batch, eval_dir=eval_dir)
        results_by_id = {
            int(row["task_id"]): row
            for row in self._run_search(eval_dir=eval_dir, task_rows=task_rows)
        }

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        for task, task_row in zip(batch, task_rows, strict=True):
            result = results_by_id.get(
                task.task_id,
                {
                    "task_id": task.task_id,
                    "game": task.game,
                    "level": task.level,
                    "score": 0.0,
                    "solved": False,
                    "feedback": "Missing search result shard output.",
                    "error": "missing search result shard output",
                },
            )
            result = dict(result)
            result["synthesis_error"] = task_row.get("synthesis_error")
            self._attach_baseline_metadata(result)
            outputs.append(result)
            if capture_traces:
                trajectories.append(
                    {
                        "task": asdict(task),
                        "heuristic_code_path": task_row.get("heuristic_code_path"),
                        "heuristic_code": Path(str(task_row["heuristic_code_path"])).read_text(
                            encoding="utf-8"
                        ),
                        "synthesis_error": task_row.get("synthesis_error"),
                        "result": result,
                    }
                )

        scores = adjusted_candidate_scores(
            outputs,
            lost_solve_penalty=self.lost_solve_penalty,
            new_solve_bonus=self.new_solve_bonus,
            error_penalty=self.candidate_error_penalty,
            score_delta_weight=self.score_delta_weight,
            score_delta_clip=self.score_delta_clip,
            partial_progress_weight=self.partial_progress_weight,
        )
        for output, adjusted_score in zip(outputs, scores, strict=True):
            output["adjusted_score"] = adjusted_score
        (eval_dir / "scored_results.json").write_text(
            json.dumps(outputs, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        aggregate = sum(scores) / len(scores) if scores else 0.0
        raw_aggregate = (
            sum(float(output.get("score", 0.0)) for output in outputs) / len(outputs)
            if outputs
            else 0.0
        )
        solved = sum(1 for output in outputs if output.get("solved"))
        lost_solves = sum(
            1
            for output in outputs
            if bool(output.get("baseline_solved", False)) and not bool(output.get("solved", False))
        )
        candidate_errors = sum(1 for output in outputs if is_candidate_error(output))
        print(
            f"[adapter] merged adjusted_score={aggregate:.4f} raw_score={raw_aggregate:.4f}, "
            f"solved={solved}/{len(outputs)} lost_solves={lost_solves} "
            f"candidate_errors={candidate_errors}",
            flush=True,
        )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build compact GEPA reflection records from traced search results.

        The GPU path evaluates every active level in a round, so unbounded
        PuzzleScript source, generated code, and per-level feedback can push
        GEPA's reflection prompt past the local model context window. This
        dataset keeps scalar scoring exhaustive while sending GEPA a bounded
        low-scoring trace subset with capped high-volume fields.
        """
        del candidate
        trajectories = list(eval_batch.trajectories or [])
        selected_traces = select_reflection_traces(trajectories)
        if len(selected_traces) < len(trajectories):
            print(
                "[adapter] reflection traces selected="
                f"{len(selected_traces)}/{len(trajectories)} "
                f"max_records={DEFAULT_REFLECTION_MAX_RECORDS}",
                flush=True,
            )
        records: list[dict[str, Any]] = []
        selection_note = (
            f"Selected {len(selected_traces)} comparison-focused traces out of "
            f"{len(trajectories)} evaluated levels, prioritizing lost base solves, "
            "new solves, solved regressions, candidate errors, and persistent failures; "
            "all levels still contributed to the scalar score."
        )
        for trace in selected_traces:
            task = trace["task"]
            result = trace["result"]
            classification = trace_classification(trace)
            baseline_code = _read_optional_text(
                result.get("baseline_heuristic_code_path"),
                max_chars=DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS,
            )
            generated_code = trace.get("heuristic_code", "")
            targeted_feedback = build_reflection_feedback(result, classification)
            records.append(
                {
                    "Inputs": {
                        "game": task["game"],
                        "level": task["level"],
                        "budget": task["budget"],
                        "env_description": truncate_text(
                            task["env_description"],
                            DEFAULT_REFLECTION_ENV_DESCRIPTION_CHARS,
                        ),
                    },
                    "Comparison": build_comparison_payload(result, classification),
                    "Baseline Output": {
                        "heuristic_code": baseline_code,
                        "code_shape": heuristic_code_shape(baseline_code),
                        "feedback": truncate_text(
                            str(result.get("baseline_feedback", "")),
                            DEFAULT_REFLECTION_FEEDBACK_CHARS,
                        ),
                    },
                    "Generated Outputs": {
                        "heuristic_code": truncate_text(
                            generated_code,
                            DEFAULT_REFLECTION_HEURISTIC_CODE_CHARS,
                        ),
                        "code_shape": heuristic_code_shape(generated_code),
                        "synthesis_error": trace.get("synthesis_error"),
                    },
                    "Feedback": truncate_text(
                        targeted_feedback,
                        DEFAULT_REFLECTION_FEEDBACK_CHARS,
                    ),
                    "Selection": selection_note,
                    "score": float(result.get("adjusted_score", result.get("score", 0.0))),
                    "raw_score": float(result.get("score", 0.0)),
                    "solved": bool(result.get("solved", False)),
                }
            )
        return {component: records for component in components_to_update}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose compact global prompt revisions from comparative feedback.

        GEPA's default instruction proposer tends to append broad examples. For
        this research path we keep one global prompt, so the reflection request
        explicitly asks for small, generalizable edits that preserve base-solved
        behavior and only encode lessons supported by regression/new-solve
        evidence.
        """
        new_texts: dict[str, str] = {}
        for component in components_to_update:
            current_prompt = candidate[component]
            base_prompt = _strip_gepa_addendum(current_prompt)
            records = reflective_dataset.get(component, [])
            feedback_payload = truncate_text(
                json.dumps(records, indent=2, sort_keys=True, default=str),
                DEFAULT_REFLECTION_DATASET_CHARS,
            )
            max_chars = DEFAULT_PROPOSED_ADDENDUM_MAX_CHARS
            reflection_prompt = f"""You are revising one global prompt used to generate PuzzleScript A* heuristics.

The research goal is a single general prompt, not per-game routing or memorized examples.
The existing base prompt is already strong. Propose only a short addendum that will be attached after it.

Hard requirements:
- Do not rewrite the full base prompt. Output only the short addendum text.
- Do not return the base prompt unchanged. The addendum must test one specific,
  conservative, generalizable hypothesis from the feedback.
- Preserve the exact function contract and safety constraints by leaving the base prompt unchanged.
- Preserve the base prompt's instruction to read WINCONDITIONS, RULES,
  COLLISIONLAYERS, LEGEND aliases, and the initial level state before choosing
  heuristic features.
- Do not replace game-specific mechanics analysis with a generic role-discovery
  template. Generic substring roles and score-based fallback are last resorts,
  not the main heuristic strategy.
- Do not output Python code as the revised prompt; output instruction text that
  will later ask a separate model call to write Python code.
- The revised prompt must not begin with `def`, `import`, or a
  `heuristic_cost_to_go` implementation.
- Do not append long lists of named games, levels, examples, or mechanics inventories.
- Prioritize preventing lost base solves over small expansion-count improvements.
- Treat a candidate that loses base-solved levels as evidence to restore or
  strengthen the base mechanics-reading rule, not as evidence to add broader
  deadlock or fallback heuristics.
- Preserve general principles that created new solves only when they do not conflict with base-solved cases.
- Keep the addendum under {max_chars} characters.
- Output only the addendum text. Do not include markdown, commentary, labels, or headings.

Base prompt that will remain unchanged:
{base_prompt}

Comparison-focused feedback:
{feedback_payload}
"""
            proposed = self.llm.complete(reflection_prompt)
            cleaned = _clean_proposed_addendum(proposed, max_chars=max_chars)
            revised_prompt = (
                _compose_prompt_with_addendum(base_prompt, cleaned) if cleaned else current_prompt
            )
            if revised_prompt.strip() == current_prompt.strip():
                fallback = _fallback_addendum_from_feedback(records)
                if fallback:
                    print(
                        "[proposer] using conservative fallback addendum after no-op proposal",
                        flush=True,
                    )
                    revised_prompt = _compose_prompt_with_addendum(base_prompt, fallback)
            new_texts[component] = revised_prompt
        return new_texts


def parse_extra_sbatch_args(values: Optional[str]) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(part for part in values.split() if part)


def run_standalone_gepa(args: argparse.Namespace) -> None:
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    gepa_run_dir = state_root / "gepa_run"
    gepa_run_dir.mkdir(parents=True, exist_ok=True)
    # GEPA 0.0.7 checks for this directory when deciding whether to resume.
    (gepa_run_dir / "prog_candidates").mkdir(exist_ok=True)

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    train_jobs, eval_jobs = load_env_grid(args.env_grid)
    all_train_tasks = build_level_tasks(
        evaluator=evaluator,
        jobs=train_jobs,
        script_doctor=args.script_doctor,
        levels_per_game=args.levels_per_game,
        budget=max(1, args.max_gepa_expansions_per_level),
    )
    if args.val_split == "dev":
        split_train_tasks, split_val_tasks = build_train_dev_tasks(
            all_train_tasks,
            dev_fraction=args.dev_fraction,
            seed=args.seed,
        )
        train_tasks = [cast(PuzzleScriptLevelTask, task) for task in split_train_tasks]
        val_tasks = [cast(PuzzleScriptLevelTask, task) for task in split_val_tasks]
    else:
        train_tasks = all_train_tasks
        val_tasks = train_tasks if args.val_split == "train" else build_level_tasks(
            evaluator=evaluator,
            jobs=eval_jobs,
            script_doctor=args.script_doctor,
            levels_per_game=args.levels_per_game,
            budget=max(1, args.max_expansions),
        )
    guard_level_selection = parse_guard_level_selection(args.guard_levels)
    guard_tasks: list[PuzzleScriptLevelTask] = []
    if guard_level_selection:
        guard_tasks = build_level_tasks(
            evaluator=evaluator,
            jobs=eval_jobs,
            script_doctor=args.script_doctor,
            levels_per_game=0,
            budget=max(1, args.max_expansions),
            selected_levels_by_game=guard_level_selection,
        )
        val_tasks = merge_validation_guard_tasks(
            [cast(PuzzleScriptLevelTask, task) for task in val_tasks],
            guard_tasks,
        )
    if not train_tasks:
        raise RuntimeError("No train tasks were loadable.")
    if not val_tasks:
        raise RuntimeError("No validation tasks were loadable.")

    input_split = {
        "val_split": args.val_split,
        "dev_fraction": args.dev_fraction if args.val_split == "dev" else None,
        "seed": args.seed,
        "all_train_games": sorted({task.game for task in all_train_tasks}),
        "train_games": sorted({task.game for task in train_tasks}),
        "val_games": sorted({task.game for task in val_tasks}),
        "guard_levels": guard_level_selection,
        "guard_task_count": len(guard_tasks),
        "guard_games": sorted({task.game for task in guard_tasks}),
        "train_task_count": len(train_tasks),
        "val_task_count": len(val_tasks),
    }
    (state_root / "input_split.json").write_text(
        json.dumps(input_split, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (state_root / "train_tasks.json").write_text(
        json.dumps([asdict(task) for task in train_tasks], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (state_root / "val_tasks.json").write_text(
        json.dumps([asdict(task) for task in val_tasks], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    llm = OpenAITextClient(
        model=args.model,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
        max_tokens=args.max_model_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.llm_timeout_s,
    )
    adapter = PuzzleScriptBatchedGEPAAdapter(
        llm=llm,
        state_root=state_root,
        script_doctor=args.script_doctor,
        search_config=SearchArrayConfig(
            submit=args.submit_search_array,
            array_script=args.search_array_script,
            array_count=args.search_array_count,
            array_concurrency=args.search_array_concurrency,
            poll_interval_s=args.search_poll_interval_s,
            stall_timeout_s=args.search_array_stall_timeout_s,
            extra_sbatch_args=parse_extra_sbatch_args(args.extra_sbatch_args),
        ),
        llm_concurrency=args.llm_concurrency,
        astar_timeout_s=max(1.0, args.astar_timeout_s),
        lost_solve_penalty=args.lost_solve_penalty,
        new_solve_bonus=args.new_solve_bonus,
        candidate_error_penalty=args.candidate_error_penalty,
        score_delta_weight=args.score_delta_weight,
        score_delta_clip=args.score_delta_clip,
        partial_progress_weight=args.partial_progress_weight,
    )

    import gepa

    reflection_minibatch_size = (
        len(train_tasks) if args.reflection_minibatch_size <= 0 else args.reflection_minibatch_size
    )
    max_metric_calls = (
        args.max_metric_calls
        if args.max_metric_calls > 0
        else len(val_tasks) + args.max_gepa_iterations * len(train_tasks) * 2
    )
    seed_candidate = {HEURISTIC_COMPONENT: PUZZLESCRIPT_HEURISTIC_CONTRACT}
    baseline_tasks = [
        cast(PuzzleScriptLevelTask, task)
        for task in reassign_task_ids(unique_tasks_by_key([*train_tasks, *val_tasks]))
    ]
    print(
        "[gepa] evaluating base prompt baseline for regression scoring: "
        f"tasks={len(baseline_tasks)} lost_solve_penalty={args.lost_solve_penalty} "
        f"new_solve_bonus={args.new_solve_bonus} "
        f"candidate_error_penalty={args.candidate_error_penalty} "
        f"score_delta_weight={args.score_delta_weight} score_delta_clip={args.score_delta_clip} "
        f"partial_progress_weight={args.partial_progress_weight}",
        flush=True,
    )
    baseline_batch = adapter.evaluate(
        batch=baseline_tasks,
        candidate=seed_candidate,
        capture_traces=False,
    )
    baseline_outputs = [dict(row) for row in baseline_batch.outputs]
    adapter.set_baseline_outputs(baseline_outputs)
    baseline_score_mean = (
        sum(_result_float(row, "score") for row in baseline_outputs) / len(baseline_outputs)
        if baseline_outputs
        else 0.0
    )
    baseline_adjusted_score_mean = (
        sum(
            _result_float(row, "adjusted_score", _result_float(row, "score"))
            for row in baseline_outputs
        )
        / len(baseline_outputs)
        if baseline_outputs
        else 0.0
    )
    baseline_summary = {
        "n": len(baseline_outputs),
        "score_mean": baseline_score_mean,
        "adjusted_score_mean": baseline_adjusted_score_mean,
        "solved": sum(1 for row in baseline_outputs if bool(row.get("solved", False))),
        "candidate_errors": sum(1 for row in baseline_outputs if is_candidate_error(row)),
    }
    (state_root / "baseline_outputs.json").write_text(
        json.dumps(baseline_outputs, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (state_root / "baseline_summary.json").write_text(
        json.dumps(baseline_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        "[gepa] starting standalone optimization: "
        f"train_tasks={len(train_tasks)} val_tasks={len(val_tasks)} "
        f"guard_tasks={len(guard_tasks)} "
        f"val_split={args.val_split} "
        f"reflection_minibatch_size={reflection_minibatch_size} "
        f"max_metric_calls={max_metric_calls}",
        flush=True,
    )
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_tasks,
        valset=val_tasks,
        adapter=adapter,
        reflection_lm=llm.complete,
        reflection_minibatch_size=reflection_minibatch_size,
        max_metric_calls=max_metric_calls,
        run_dir=str(gepa_run_dir),
        display_progress_bar=False,
        raise_on_exception=False,
        seed=args.seed,
    )
    result_path = state_root / "gepa_result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    best_prompt_path = state_root / "best_prompt.txt"
    best_prompt_path.write_text(
        result.best_candidate.get(HEURISTIC_COMPONENT, "") + "\n",
        encoding="utf-8",
    )
    print(f"[gepa] result: {result_path}")
    print(f"[gepa] best prompt: {best_prompt_path}")
    print(f"[gepa] best_idx={result.best_idx} best_score={result.val_aggregate_scores[result.best_idx]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GEPA with batched local-LLM synthesis and CPU-array search."
    )
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--script-doctor", type=Path, default=DEFAULT_SCRIPT_DOCTOR)
    parser.add_argument("--levels-per-game", type=int, default=0)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument(
        "--max-gepa-expansions-per-level",
        type=int,
        default=DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL,
    )
    parser.add_argument("--astar-timeout-s", type=float, default=DEFAULT_ASTAR_TIMEOUT_S)
    parser.add_argument("--model", type=str, default=os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-model-tokens", type=int, default=DEFAULT_MAX_MODEL_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--llm-timeout-s", type=float, default=600.0)
    parser.add_argument("--llm-concurrency", type=int, default=16)
    parser.add_argument("--submit-search-array", action="store_true")
    parser.add_argument(
        "--search-array-script",
        type=Path,
        default=Path("sbatch/evaluate_puzzlescript_search_array.s"),
    )
    parser.add_argument("--search-array-count", type=int, default=101)
    parser.add_argument("--search-array-concurrency", type=int, default=64)
    parser.add_argument("--search-poll-interval-s", type=float, default=15.0)
    parser.add_argument(
        "--search-array-stall-timeout-s",
        type=float,
        default=DEFAULT_SEARCH_ARRAY_STALL_TIMEOUT_S,
    )
    parser.add_argument(
        "--extra-sbatch-args",
        type=str,
        default="",
        help="Optional whitespace-separated sbatch args appended before the array script.",
    )
    parser.add_argument("--val-split", choices=("train", "dev", "eval"), default="dev")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument(
        "--guard-levels",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated game:level,level exact holdout guard set "
            "added to validation for base-solve regression protection."
        ),
    )
    parser.add_argument("--max-gepa-iterations", type=int, default=DEFAULT_MAX_GEPA_ITERATIONS)
    parser.add_argument("--max-metric-calls", type=int, default=0)
    parser.add_argument("--lost-solve-penalty", type=float, default=DEFAULT_LOST_SOLVE_PENALTY)
    parser.add_argument("--new-solve-bonus", type=float, default=DEFAULT_NEW_SOLVE_BONUS)
    parser.add_argument(
        "--candidate-error-penalty",
        type=float,
        default=DEFAULT_CANDIDATE_ERROR_PENALTY,
    )
    parser.add_argument("--score-delta-weight", type=float, default=DEFAULT_SCORE_DELTA_WEIGHT)
    parser.add_argument("--score-delta-clip", type=float, default=DEFAULT_SCORE_DELTA_CLIP)
    parser.add_argument(
        "--partial-progress-weight",
        type=float,
        default=DEFAULT_PARTIAL_PROGRESS_WEIGHT,
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=0,
        help="Use <=0 to evaluate all active levels per GEPA reflection round.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run_standalone_gepa(parse_args())


if __name__ == "__main__":
    main()
