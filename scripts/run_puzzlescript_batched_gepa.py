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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

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
DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MAX_MODEL_TOKENS = 8192
DEFAULT_MAX_EXPANSIONS = 50_000
DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL = 10_000
DEFAULT_ASTAR_TIMEOUT_S = 30.0

HEURISTIC_COMPONENT = "heuristic_prompt"

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
    extra_sbatch_args: tuple[str, ...] = ()


@dataclass
class OpenAITextClient:
    model: str
    base_url: str
    api_key: str
    max_tokens: int
    temperature: float
    top_p: float
    timeout_s: float

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
        )

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return str(response.choices[0].message.content or "").strip()


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "item"


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def strip_outer_markdown_fences(code: str) -> str:
    cleaned = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def heuristic_score(solved: bool, expanded: int, max_expansions: int) -> float:
    n = max_expansions
    s = expanded if solved else n + 1
    return ((n + 1) - s) / (n + 1)


def load_env_grid(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(payload.get("jobs", [])), list(payload.get("eval_jobs", []))


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
) -> list[PuzzleScriptLevelTask]:
    tasks: list[PuzzleScriptLevelTask] = []
    for entry in jobs:
        name = str(entry["name"])
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
    llm: OpenAITextClient,
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
        return {
            "task_id": int(task["task_id"]),
            "game": game,
            "level": level,
            "score": 0.0,
            "solved": False,
            "expanded": 0,
            "generated": 0,
            "solution_length": 0,
            "feedback": f"Heuristic validation failed before search: {validation_error}",
            "heuristic_code_path": str(code_path),
            "error": validation_error,
        }

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
        feedback = (
            f"Game={game} level={level} solved={result.solved} "
            f"score={score:.4f} expanded={result.expanded_states}/{budget} "
            f"generated={result.generated_states} solution_length={result.solution_length} "
            f"terminated={result.trace_summary.get('terminated_reason', 'unknown')}"
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
        return {
            "task_id": int(task["task_id"]),
            "game": game,
            "level": level,
            "score": 0.0,
            "solved": False,
            "expanded": 0,
            "generated": 0,
            "solution_length": 0,
            "feedback": f"Search evaluation failed: {exc}",
            "heuristic_code_path": str(code_path),
            "error": str(exc),
        }


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
    evaluator = PuzzleScriptEvaluator(script_doctor)
    astar_timeout_s = float(manifest["astar_timeout_s"])
    tasks = assigned_tasks(manifest["tasks"], array_index, array_count)
    results = [
        evaluate_search_task(evaluator=evaluator, task=task, astar_timeout_s=astar_timeout_s)
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
) -> list[str]:
    if array_count <= 0:
        raise ValueError("array_count must be > 0")
    if array_concurrency <= 0:
        raise ValueError("array_concurrency must be > 0")
    return [
        "sbatch",
        "--wait",
        "--parsable",
        f"--array=0-{array_count - 1}%{array_concurrency}",
        f"--export=ALL,EVAL_MANIFEST={manifest_path},SEARCH_ARRAY_COUNT={array_count}",
        *extra_sbatch_args,
        str(array_script),
    ]


def wait_for_shards(
    *,
    shard_dir: Path,
    array_count: int,
    poll_interval_s: float,
) -> list[Path]:
    expected = [
        shard_dir / f"task-{idx:04d}-of-{array_count:04d}.json"
        for idx in range(array_count)
    ]
    while True:
        missing = [path for path in expected if not path.exists()]
        if not missing:
            return expected
        print(f"[search-array] waiting for {len(missing)} shard(s)...", flush=True)
        time.sleep(poll_interval_s)


def load_shard_results(shard_paths: Sequence[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.extend(dict(row) for row in payload.get("results", []))
    return sorted(results, key=lambda row: int(row["task_id"]))


class PuzzleScriptBatchedGEPAAdapter:
    def __init__(
        self,
        *,
        llm: OpenAITextClient,
        state_root: Path,
        script_doctor: Path,
        search_config: SearchArrayConfig,
        llm_concurrency: int,
        astar_timeout_s: float,
    ) -> None:
        self.llm = llm
        self.state_root = state_root
        self.script_doctor = script_doctor
        self.search_config = search_config
        self.llm_concurrency = max(1, llm_concurrency)
        self.astar_timeout_s = astar_timeout_s
        self.eval_counter = 0
        self.propose_new_texts = None

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
            "shard_dir": str(shard_dir),
            "array_count": array_count,
            "tasks": task_rows,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        if self.search_config.submit:
            command = build_sbatch_array_command(
                manifest_path=manifest_path,
                array_script=self.search_config.array_script,
                array_count=array_count,
                array_concurrency=min(self.search_config.array_concurrency, array_count),
                extra_sbatch_args=self.search_config.extra_sbatch_args,
            )
            print("[search-array] submitting: " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
        else:
            for array_index in range(array_count):
                evaluate_manifest_shard(
                    manifest_path=manifest_path,
                    array_index=array_index,
                    array_count=array_count,
                )

        shard_paths = wait_for_shards(
            shard_dir=shard_dir,
            array_count=array_count,
            poll_interval_s=self.search_config.poll_interval_s,
        )
        results = load_shard_results(shard_paths)
        (eval_dir / "merged_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return results

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
                },
            )
            outputs.append(result)
            scores.append(float(result.get("score", 0.0)))
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

        aggregate = sum(scores) / len(scores) if scores else 0.0
        solved = sum(1 for output in outputs if output.get("solved"))
        print(
            f"[adapter] merged score={aggregate:.4f}, solved={solved}/{len(outputs)}",
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
        del candidate
        records: list[dict[str, Any]] = []
        for trace in eval_batch.trajectories or []:
            task = trace["task"]
            result = trace["result"]
            records.append(
                {
                    "Inputs": {
                        "game": task["game"],
                        "level": task["level"],
                        "budget": task["budget"],
                        "env_description": truncate_text(task["env_description"], 6000),
                    },
                    "Generated Outputs": {
                        "heuristic_code": truncate_text(trace.get("heuristic_code", ""), 5000),
                        "synthesis_error": trace.get("synthesis_error"),
                    },
                    "Feedback": truncate_text(str(result.get("feedback", "")), 6000),
                    "score": float(result.get("score", 0.0)),
                    "solved": bool(result.get("solved", False)),
                }
            )
        return {component: records for component in components_to_update}


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
    train_tasks = build_level_tasks(
        evaluator=evaluator,
        jobs=train_jobs,
        script_doctor=args.script_doctor,
        levels_per_game=args.levels_per_game,
        budget=max(1, args.max_gepa_expansions_per_level),
    )
    val_tasks = train_tasks if args.val_split == "train" else build_level_tasks(
        evaluator=evaluator,
        jobs=eval_jobs,
        script_doctor=args.script_doctor,
        levels_per_game=args.levels_per_game,
        budget=max(1, args.max_expansions),
    )
    if not train_tasks:
        raise RuntimeError("No train tasks were loadable.")
    if not val_tasks:
        raise RuntimeError("No validation tasks were loadable.")

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
            extra_sbatch_args=parse_extra_sbatch_args(args.extra_sbatch_args),
        ),
        llm_concurrency=args.llm_concurrency,
        astar_timeout_s=max(1.0, args.astar_timeout_s),
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
    print(
        "[gepa] starting standalone optimization: "
        f"train_tasks={len(train_tasks)} val_tasks={len(val_tasks)} "
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
    parser.add_argument("--temperature", type=float, default=0.2)
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
        "--extra-sbatch-args",
        type=str,
        default="",
        help="Optional whitespace-separated sbatch args appended before the array script.",
    )
    parser.add_argument("--val-split", choices=("train", "eval"), default="train")
    parser.add_argument("--max-gepa-iterations", type=int, default=12)
    parser.add_argument("--max-metric-calls", type=int, default=0)
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
