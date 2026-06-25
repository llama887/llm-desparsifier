#!/usr/bin/env python3
"""Run GEPA heuristic optimization on PuzzleScript Sokoban environments.

Uses DSPy GEPA to optimize a prompt that causes an LLM to emit a Python
heuristic function. The heuristic guides A* search on PuzzleScript games
using the C++ engine for fast state transitions.

Curriculum: 10 games -> 15 games -> full training set.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from dspy_cache_control import configure_dspy_cache, prepare_dspy_import

prepare_dspy_import("run_puzzlescript_batch")
import dspy
configure_dspy_cache(dspy, "run_puzzlescript_batch")
import yaml
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Direct imports to avoid heavy __init__.py chains
sys.path.insert(0, str(_PROJECT_ROOT / "llm_desparsifier" / "search"))
from puzzle_evaluator import PuzzleScriptEvaluator  # noqa: E402
from puzzlescript_adapter import (  # noqa: E402
    build_env_description,
    build_puzzlescript_ctx,
    extract_section_text,
)
from puzzlescript_astar import (  # noqa: E402
    PuzzleScriptSearchResult,
    blind_heuristic,
    builtin_heuristic,
    puzzlescript_astar,
)
from puzzlescript_sanitizer import sanitize_and_compile_puzzlescript_heuristic  # noqa: E402

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional in cluster jobs
    load_dotenv = None


class _HeuristicSynthesisSignature(dspy.Signature):
    """Synthesize a PuzzleScript A* heuristic function."""
    synthesis_prompt: str = dspy.InputField(desc="Instructions for writing the heuristic")
    env_description: str = dspy.InputField(desc="Game description with rules and objects")
    heuristic_code: str = dspy.OutputField(
        desc="Python function heuristic_cost_to_go(ts, env_params, ctx)")

_heuristic_predictor = dspy.Predict(_HeuristicSynthesisSignature)


class _PuzzleScriptFeedbackReflectionSignature(dspy.Signature):
    """Interpret PuzzleScript search behavior and explain heuristic failures."""

    env_description: str = dspy.InputField(desc="Game description and rules text")
    heuristic_code: str = dspy.InputField(desc="Current synthesized heuristic implementation")
    search_summary: str = dspy.InputField(desc="Deterministic summary of search outcomes and trace snippets")
    reflection: str = dspy.OutputField(
        desc=(
            "Mechanism-level feedback explaining what the heuristic appears to miss, "
            "what the search trace suggests, and concrete guidance for the next revision."
        )
    )


_feedback_reflector = dspy.Predict(_PuzzleScriptFeedbackReflectionSignature)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DOCTOR_PATH = _PROJECT_ROOT.parent / "script-doctor"
DEFAULT_ENV_GRID = Path("configs/gepa_puzzlescript_envs.yaml")
DEFAULT_STATE_ROOT = Path("artifacts/gepa_puzzlescript_state")

# The 5-game warmup advanced immediately in practice and spent budget on an
# easy slice. Start with a broader phase, then expand once before the full set.
CURRICULUM_PHASE_GAME_COUNTS = (10, 15)
PHASE_SOLVE_RATE_THRESHOLD = 0.60
PHASE_NEAR_THRESHOLD = 0.55
PHASE_NEAR_THRESHOLD_PATIENCE = 3
PHASE_EARLY_STOP_PATIENCE = 8
DEFAULT_MAX_PHASE_ITERATIONS = 10
DEFAULT_ASTAR_MAX_EXPANSIONS = 50_000
DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL = 10_000
DEFAULT_ASTAR_TIMEOUT_S = 30.0
DEFAULT_LLM = "deepseek/deepseek-v4-pro"
DEFAULT_LLM_MAX_TOKENS = 384_000
DEFAULT_LEVELS_PER_GAME = 0
DEFAULT_GEPA_NUM_THREADS = 1
BASELINE_CACHE_VERSION = 1
GEPA_MAX_METRIC_CALLS_MULTIPLIER = 12
PAIRWISE_SOLVE_BASE_BOTH_SOLVE = 0.85
PAIRWISE_SOLVE_BASE_BOTH_FAIL = 0.02
PAIRWISE_LOST_SOLVE_PENALTY = 1.0
PAIRWISE_EFFICIENCY_WEIGHT = 0.10
PAIRWISE_SPEEDUP_WEIGHT = 0.05
PAIRWISE_REGRESSION_WEIGHT = 0.10
BEST_PROMPT_MIN_SCORE_DELTA = 0.005
BEST_PROMPT_SCORE_BACKOFF_FOR_SOLVE_GAIN = 0.010


def _phase_gepa_max_metric_calls(*, phase_iteration: int, trainset_size: int) -> int:
    """Return the capped cumulative GEPA metric-call cap for one phase run."""

    if phase_iteration <= 0:
        raise ValueError("phase_iteration must be > 0")
    if trainset_size <= 0:
        raise ValueError("trainset_size must be > 0")
    uncapped = phase_iteration * trainset_size * 3
    cap = trainset_size * GEPA_MAX_METRIC_CALLS_MULTIPLIER
    return min(uncapped, cap)


def build_curriculum_phase_schedule(train_jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Build curriculum phases, ending with the complete training split."""

    phase_schedule: list[list[dict[str, Any]]] = []
    for count in CURRICULUM_PHASE_GAME_COUNTS:
        phase_schedule.append(train_jobs[:count])
    if len(train_jobs) > CURRICULUM_PHASE_GAME_COUNTS[-1]:
        phase_schedule.append(train_jobs)
    return phase_schedule


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


def load_local_env() -> None:
    """Load repo-local .env credentials when available."""

    if load_dotenv is not None:
        load_dotenv(_PROJECT_ROOT / ".env")


class LMCostLogger:
    """Append DSPy LM history costs and write cumulative GEPA cost summaries."""

    DEEPSEEK_V4_PRO_PRICES_PER_MILLION = {
        "prompt_cache_hit_tokens": 0.003625,
        "prompt_cache_miss_tokens": 0.435,
        "completion_tokens": 0.87,
    }

    def __init__(self, lm: Any, output_dir: Path) -> None:
        self.lm = lm
        self.output_dir = output_dir
        self.events_path = output_dir / "llm_cost_events.jsonl"
        self.summary_path = output_dir / "llm_cost_summary.json"
        self._seen_history = 0
        self._events: list[dict[str, Any]] = []

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _usage_payload(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, Mapping):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            return dict(usage.model_dump())
        if hasattr(usage, "dict"):
            return dict(usage.dict())
        return {}

    @classmethod
    def _usage_int(cls, usage: Mapping[str, Any], key: str) -> int:
        value = usage.get(key)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _estimate_deepseek_cost(cls, model: Any, response_model: Any, usage: Mapping[str, Any]) -> float:
        model_name = f"{model or ''} {response_model or ''}".lower()
        if "deepseek" not in model_name or "v4-pro" not in model_name:
            return 0.0

        hit_tokens = cls._usage_int(usage, "prompt_cache_hit_tokens")
        miss_tokens = cls._usage_int(usage, "prompt_cache_miss_tokens")
        prompt_tokens = cls._usage_int(usage, "prompt_tokens")
        completion_tokens = cls._usage_int(usage, "completion_tokens")

        if hit_tokens == 0 and miss_tokens == 0 and prompt_tokens > 0:
            miss_tokens = prompt_tokens

        prices = cls.DEEPSEEK_V4_PRO_PRICES_PER_MILLION
        return (
            hit_tokens * prices["prompt_cache_hit_tokens"]
            + miss_tokens * prices["prompt_cache_miss_tokens"]
            + completion_tokens * prices["completion_tokens"]
        ) / 1_000_000

    def sync(self, label: str, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Record new LM calls since the previous sync and return a summary."""

        history = list(getattr(self.lm, "history", []) or [])
        new_entries = history[self._seen_history :]
        self._seen_history = len(history)

        new_events: list[dict[str, Any]] = []
        for entry in new_entries:
            if not isinstance(entry, Mapping):
                continue
            usage = self._usage_payload(entry.get("usage"))
            logged_cost = self._as_float(entry.get("cost"))
            estimated_cost = self._estimate_deepseek_cost(
                entry.get("model"),
                entry.get("response_model"),
                usage,
            )
            event = {
                "label": label,
                "timestamp": entry.get("timestamp"),
                "uuid": entry.get("uuid"),
                "model": entry.get("model"),
                "response_model": entry.get("response_model"),
                "model_type": entry.get("model_type"),
                "cost_usd": logged_cost or estimated_cost,
                "logged_cost_usd": logged_cost,
                "estimated_cost_usd": estimated_cost,
                "usage": usage,
            }
            if extra:
                event.update({f"extra_{key}": value for key, value in extra.items()})
            new_events.append(event)

        if new_events:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as f:
                for event in new_events:
                    f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            self._events.extend(new_events)

        summary = self.summary()
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            f"  [llm-cost] {label}: +${sum(e['cost_usd'] for e in new_events):.6f} "
            f"({len(new_events)} calls), total=${summary['total_cost_usd']:.6f} "
            f"({summary['total_calls']} calls)"
        )
        return summary

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, dict[str, Any]] = {}
        token_totals = {
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for event in self._events:
            model = str(event.get("model") or event.get("response_model") or "unknown")
            row = by_model.setdefault(model, {"calls": 0, "cost_usd": 0.0, "tokens": dict(token_totals)})
            row["calls"] += 1
            row["cost_usd"] += self._as_float(event.get("cost_usd"))
            usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
            for key in token_totals:
                tokens = self._usage_int(usage, key)
                token_totals[key] += tokens
                row["tokens"][key] += tokens
        return {
            "total_calls": len(self._events),
            "total_cost_usd": sum(self._as_float(e.get("cost_usd")) for e in self._events),
            "token_totals": token_totals,
            "by_model": by_model,
            "events_path": str(self.events_path),
            "pricing_note": (
                "DeepSeek V4 Pro estimated from official USD prices per 1M tokens: "
                "cache_hit=$0.003625, cache_miss=$0.435, output=$0.87. "
                "If cache hit/miss fields are absent, prompt tokens are treated as cache misses."
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _process_action_with_again(engine, action: int) -> bool:
    """Apply one action and drain PuzzleScript `again` loops.

    The local feedback probes need to inspect immediate successor states using
    the same transition semantics as the main search loop. This helper mirrors
    the A* runner's action application without importing a private function.
    """

    changed = engine.process_input(action)
    again = 0
    while engine.is_againing() and again < 50:
        changed = engine.process_input(-1) or changed
        again += 1
    return changed


def load_env_grid(grid_path: Path) -> tuple[list[dict], list[dict]]:
    with open(grid_path) as f:
        raw = yaml.safe_load(f)
    return raw.get("jobs", []), raw.get("eval_jobs", [])


def load_game_text(name: str, sd_path: Path) -> Optional[str]:
    for subdir in ("data/scraped_games", "custom_games"):
        path = sd_path / subdir / f"{name}.txt"
        if path.exists():
            return path.read_text()
    return None


def build_level_env_description(
    base_env_description: str,
    engine: Any,
    compiled: dict[str, Any],
    level_i: int,
) -> str:
    """Add the specific level's initial state to a game description."""

    engine.load_level(level_i)
    ctx = build_puzzlescript_ctx(engine, compiled)
    object_positions = {
        str(name): [[int(x), int(y)] for x, y in positions]
        for name, positions in sorted(ctx.get("object_positions", {}).items())
        if str(name).lower() != "background"
    }
    object_counts = {
        name: len(positions)
        for name, positions in object_positions.items()
    }
    root_summary = {
        "grid_width": ctx.get("grid_width"),
        "grid_height": ctx.get("grid_height"),
        "win_conditions_text": ctx.get("win_conditions_text"),
        "score": ctx.get("score"),
        "score_normalized": ctx.get("score_normalized"),
        "object_counts": object_counts,
        "object_positions": object_positions,
    }
    return (
        base_env_description
        + "\n\n"
        + f"Level of interest: {level_i}\n"
        + "Initial state for this level:\n"
        + str(ctx.get("ascii_state", ""))
        + "\n\n"
        + "Initial root ctx summary:\n"
        + json.dumps(root_summary, indent=2, sort_keys=True)
        + "\n\n"
        + "Write the heuristic for this specific level while still using the rules above."
    )


def gepa_score(solved: bool, expanded: int, max_expansions: int) -> float:
    n = max_expansions
    s = expanded if solved else n + 1
    return ((n + 1) - s) / (n + 1)


def select_level_indices(n_levels: int, requested_levels: int, levels_per_game: int) -> list[int]:
    """Select levels for GEPA.

    `levels_per_game <= 0` means use every loadable level except the final level,
    which is often a credits/thanks or very small terminal level in PuzzleScript
    games. If a game only exposes one level, keep that level.
    """

    if n_levels <= 0:
        return []
    capped = min(n_levels, max(1, requested_levels))
    if levels_per_game > 0:
        capped = min(capped, levels_per_game)
        return list(range(capped))
    if capped <= 1:
        return [0]
    return list(range(capped - 1))


def filter_loadable_level_indices(
    engine: Any,
    level_indices: list[int],
    game_name: str,
) -> list[int]:
    """Return only selected level indices that the PuzzleScript engine can load."""

    loadable: list[int] = []
    for level_i in level_indices:
        try:
            engine.load_level(level_i)
        except Exception as e:
            print(f"  [WARN] Skipping {game_name} level={level_i}: {e}")
            continue
        loadable.append(level_i)
    return loadable


def prepare_puzzlescript_inputs(
    evaluator: PuzzleScriptEvaluator,
    train_jobs: list[dict[str, Any]],
    eval_jobs: list[dict[str, Any]],
    sd_path: Path,
    levels_per_game: int,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[int, str]],
    dict[str, list[int]],
]:
    """Compile selected games and build per-level prompt descriptions."""

    all_game_texts: dict[str, str] = {}
    all_env_descs: dict[str, str] = {}
    all_level_env_descs: dict[str, dict[int, str]] = {}
    level_indices_by_game: dict[str, list[int]] = {}
    for entry in train_jobs + eval_jobs:
        name = str(entry["name"])
        if name in all_game_texts:
            continue
        text = load_game_text(name, sd_path)
        if text:
            try:
                json_str = evaluator.compile_game(text)
                compiled = json.loads(json_str)
                engine = evaluator.load_engine(json_str)
                engine.load_level(0)
                n_levels = engine.get_num_levels()
                requested_levels = max(1, int(entry.get("levels", n_levels) or n_levels))
                selected_level_indices = select_level_indices(
                    n_levels=n_levels,
                    requested_levels=requested_levels,
                    levels_per_game=levels_per_game,
                )
                loadable_level_indices = filter_loadable_level_indices(
                    engine=engine,
                    level_indices=selected_level_indices,
                    game_name=name,
                )
                if not loadable_level_indices:
                    print(f"  [WARN] Skipping {name}: no selected levels were loadable")
                    continue
                all_game_texts[name] = text
                level_indices_by_game[name] = loadable_level_indices
                base_desc = build_env_description(compiled, engine.get_id_dict(), text)
                all_env_descs[name] = base_desc
                all_level_env_descs[name] = {
                    level_i: build_level_env_description(base_desc, engine, compiled, level_i)
                    for level_i in level_indices_by_game[name]
                }
            except Exception as e:
                print(f"  [WARN] Could not compile {name}: {e}")

    return all_game_texts, all_env_descs, all_level_env_descs, level_indices_by_game


def build_training_level_examples(
    train_jobs: list[dict[str, Any]],
    all_game_texts: Mapping[str, str],
    level_indices_by_game: Mapping[str, list[int]],
) -> list[dict[str, int | str]]:
    """Return selected train game/level examples in stable grid order."""

    examples: list[dict[str, int | str]] = []
    for entry in train_jobs:
        name = str(entry["name"])
        if name not in all_game_texts:
            continue
        for level_i in level_indices_by_game.get(name, [0]):
            examples.append({"game": name, "level": int(level_i)})
    return examples


def _baseline_cache_path(state_root: Path) -> Path:
    return state_root / "puzzlescript_baselines.json"


def _baseline_shard_dir(state_root: Path) -> Path:
    return state_root / "baseline_shards"


def build_baseline_cache_signature(
    *,
    train_jobs: list[dict[str, Any]],
    level_indices_by_game: Mapping[str, list[int]],
    max_expansions: int,
    max_gepa_expansions_per_level: int,
    astar_timeout_s: float,
    levels_per_game: int,
    llm_name: str,
    llm_max_tokens: int,
) -> dict[str, Any]:
    """Build the compatibility key for baseline caches and array shards."""

    train_levels = {
        str(entry["name"]): [
            int(level_i) for level_i in level_indices_by_game.get(str(entry["name"]), [])
        ]
        for entry in train_jobs
        if str(entry["name"]) in level_indices_by_game
    }
    contract_sha16 = hashlib.sha256(
        PUZZLESCRIPT_HEURISTIC_CONTRACT.encode("utf-8")
    ).hexdigest()[:16]
    return {
        "version": BASELINE_CACHE_VERSION,
        "train_levels": train_levels,
        "max_expansions": int(max_expansions),
        "max_gepa_expansions_per_level": int(max_gepa_expansions_per_level),
        "astar_timeout_s": float(astar_timeout_s),
        "levels_per_game": int(levels_per_game),
        "llm_name": str(llm_name),
        "llm_max_tokens": int(llm_max_tokens),
        "base_prompt_sha16": contract_sha16,
    }


def _level_key_to_int(level_key: Any) -> int:
    text = str(level_key)
    if text.startswith("level-"):
        text = text.split("-", 1)[1]
    return int(text)


def _normalize_baseline_map(raw: Any) -> dict[str, dict[int, dict[str, Any]]]:
    normalized: dict[str, dict[int, dict[str, Any]]] = {}
    if not isinstance(raw, Mapping):
        return normalized
    for game, levels in raw.items():
        if not isinstance(levels, Mapping):
            continue
        game_key = str(game)
        for level_key, payload in levels.items():
            if not isinstance(payload, Mapping):
                continue
            try:
                level_i = _level_key_to_int(level_key)
            except (TypeError, ValueError):
                continue
            normalized.setdefault(game_key, {})[level_i] = dict(payload)
    return normalized


def _normalize_budget_map(raw: Any) -> dict[str, dict[int, int]]:
    normalized: dict[str, dict[int, int]] = {}
    if not isinstance(raw, Mapping):
        return normalized
    for game, levels in raw.items():
        if not isinstance(levels, Mapping):
            continue
        game_key = str(game)
        for level_key, budget in levels.items():
            try:
                level_i = _level_key_to_int(level_key)
                normalized.setdefault(game_key, {})[level_i] = int(budget)
            except (TypeError, ValueError):
                continue
    return normalized


def _jsonable_nested_map(
    mapping: Mapping[str, Mapping[int, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(game): {str(int(level_i)): value for level_i, value in sorted(levels.items())}
        for game, levels in sorted(mapping.items())
    }


def _merge_nested_map(
    target: dict[str, dict[int, Any]],
    incoming: Mapping[str, Mapping[int, Any]],
) -> None:
    for game, levels in incoming.items():
        target.setdefault(str(game), {}).update(
            {int(level_i): value for level_i, value in levels.items()}
        )


def load_cached_puzzlescript_baselines(
    state_root: Path,
    signature: Mapping[str, Any],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, int]],
    list[Path],
]:
    """Load matching canonical baseline cache and any completed array shards."""

    blind_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    builtin_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    base_prompt_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    per_game_budgets: dict[str, dict[int, int]] = {}
    loaded_paths: list[Path] = []
    candidate_paths = [_baseline_cache_path(state_root)]
    shard_dir = _baseline_shard_dir(state_root)
    if shard_dir.exists():
        candidate_paths.extend(sorted(shard_dir.glob("*.json")))

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  [baseline-cache] ignoring unreadable {path}: {exc}")
            continue
        if payload.get("signature") != dict(signature):
            print(f"  [baseline-cache] ignoring stale {path}")
            continue
        _merge_nested_map(
            blind_baselines,
            _normalize_baseline_map(payload.get("blind_baselines")),
        )
        _merge_nested_map(
            builtin_baselines,
            _normalize_baseline_map(payload.get("builtin_baselines")),
        )
        _merge_nested_map(
            base_prompt_baselines,
            _normalize_baseline_map(payload.get("base_prompt_baselines")),
        )
        _merge_nested_map(
            per_game_budgets,
            _normalize_budget_map(payload.get("per_game_budgets")),
        )
        loaded_paths.append(path)
    return (
        blind_baselines,
        builtin_baselines,
        base_prompt_baselines,
        per_game_budgets,
        loaded_paths,
    )


def _baseline_payload(
    *,
    signature: Mapping[str, Any],
    blind_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    builtin_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    base_prompt_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    per_game_budgets: Mapping[str, Mapping[int, int]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "signature": dict(signature),
        "blind_baselines": _jsonable_nested_map(blind_baselines),
        "builtin_baselines": _jsonable_nested_map(builtin_baselines),
        "base_prompt_baselines": _jsonable_nested_map(base_prompt_baselines),
        "per_game_budgets": _jsonable_nested_map(per_game_budgets),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def save_puzzlescript_baseline_cache(
    state_root: Path,
    *,
    signature: Mapping[str, Any],
    blind_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    builtin_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    base_prompt_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    per_game_budgets: Mapping[str, Mapping[int, int]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the merged canonical baseline cache atomically."""

    state_root.mkdir(parents=True, exist_ok=True)
    path = _baseline_cache_path(state_root)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            _baseline_payload(
                signature=signature,
                blind_baselines=blind_baselines,
                builtin_baselines=builtin_baselines,
                base_prompt_baselines=base_prompt_baselines,
                per_game_budgets=per_game_budgets,
                metadata=metadata,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def save_puzzlescript_baseline_shard(
    state_root: Path,
    shard_name: str,
    *,
    signature: Mapping[str, Any],
    blind_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    builtin_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    base_prompt_baselines: Mapping[str, Mapping[int, dict[str, Any]]],
    per_game_budgets: Mapping[str, Mapping[int, int]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write one Slurm array task's baseline shard."""

    shard_dir = _baseline_shard_dir(state_root)
    shard_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", shard_name).strip("_") or "task"
    path = shard_dir / f"{safe_name}.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            _baseline_payload(
                signature=signature,
                blind_baselines=blind_baselines,
                builtin_baselines=builtin_baselines,
                base_prompt_baselines=base_prompt_baselines,
                per_game_budgets=per_game_budgets,
                metadata=metadata,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def missing_puzzlescript_baseline_examples(
    examples: list[Mapping[str, Any]],
    *,
    blind_baselines: Mapping[str, Mapping[int, Any]],
    builtin_baselines: Mapping[str, Mapping[int, Any]],
    base_prompt_baselines: Mapping[str, Mapping[int, Any]],
    per_game_budgets: Mapping[str, Mapping[int, int]],
) -> list[dict[str, int | str]]:
    """Return examples that do not have all three baseline families cached."""

    missing: list[dict[str, int | str]] = []
    for example in examples:
        game = str(example["game"])
        level_i = int(example["level"])
        if (
            level_i not in blind_baselines.get(game, {})
            or level_i not in builtin_baselines.get(game, {})
            or level_i not in base_prompt_baselines.get(game, {})
            or level_i not in per_game_budgets.get(game, {})
        ):
            missing.append({"game": game, "level": level_i})
    return missing


def compute_puzzlescript_baselines_for_examples(
    *,
    evaluator: PuzzleScriptEvaluator,
    examples: list[Mapping[str, Any]],
    all_game_texts: Mapping[str, str],
    all_level_env_descs: Mapping[str, Mapping[int, str]],
    all_env_descs: Mapping[str, str],
    max_expansions: int,
    max_gepa_expansions_per_level: int,
    astar_timeout_s: float,
    lm: Any,
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, int]],
]:
    """Compute blind, built-in, and base-prompt baselines for game/level examples."""

    blind_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    builtin_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    base_prompt_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    per_game_budgets: dict[str, dict[int, int]] = {}

    for example in examples:
        name = str(example["game"])
        level_i = int(example["level"])
        if name not in all_game_texts:
            continue
        print(f"  [baseline] {name} level={level_i}: blind A*")
        blind_result = evaluate_one_game(
            evaluator,
            name,
            all_game_texts[name],
            blind_heuristic,
            max_expansions,
            level_i=level_i,
            astar_timeout_s=astar_timeout_s,
        )
        blind_baselines.setdefault(name, {})[level_i] = blind_result
        if blind_result["solved"] and blind_result["expanded"] > 0:
            calibrated_budget = max(math.floor(0.95 * blind_result["expanded"]), 1)
        else:
            calibrated_budget = max_expansions
        budget = min(calibrated_budget, max_gepa_expansions_per_level)
        per_game_budgets.setdefault(name, {})[level_i] = budget
        print(
            f"    blind solved={blind_result['solved']} expanded={blind_result['expanded']} "
            f"score={blind_result['score']:.4f} -> gepa_budget={budget}"
        )

        print(f"  [baseline] {name} level={level_i}: built-in heuristic")
        builtin_result = evaluate_one_game(
            evaluator,
            name,
            all_game_texts[name],
            builtin_heuristic,
            budget,
            level_i=level_i,
            astar_timeout_s=astar_timeout_s,
        )
        builtin_baselines.setdefault(name, {})[level_i] = builtin_result
        print(
            f"    builtin solved={builtin_result['solved']} "
            f"expanded={builtin_result['expanded']} score={builtin_result['score']:.4f}"
        )

        print(f"  [baseline] {name} level={level_i}: base-prompt LLM heuristic")
        env_desc = all_level_env_descs.get(name, {}).get(level_i, all_env_descs.get(name, name))
        heuristic_fn, code, error = synthesize_heuristic_from_prompt(
            PUZZLESCRIPT_HEURISTIC_CONTRACT,
            env_desc,
            lm,
            preflight_evaluator=evaluator,
            preflight_game_text=all_game_texts[name],
            preflight_level_i=level_i,
        )
        if error:
            print(f"    base-prompt synthesis error: {error[:160]}")
            heuristic_fn = builtin_heuristic
            code = f"# FALLBACK: {error[:200]}"
        base_result = evaluate_one_game(
            evaluator,
            name,
            all_game_texts[name],
            heuristic_fn,
            budget,
            level_i=level_i,
            blind_baseline=blind_result,
            builtin_baseline=builtin_result,
            env_description=env_desc,
            heuristic_code=code,
            astar_timeout_s=astar_timeout_s,
        )
        base_result["heuristic_code"] = code
        base_prompt_baselines.setdefault(name, {})[level_i] = base_result
        print(
            f"    base_prompt solved={base_result['solved']} "
            f"expanded={base_result['expanded']} score={base_result['score']:.4f}"
        )

    return blind_baselines, builtin_baselines, base_prompt_baselines, per_game_budgets


def _pairwise_gepa_metric(
    *,
    candidate: PuzzleScriptSearchResult | Mapping[str, Any],
    base_prompt_baseline: Optional[Mapping[str, Any]],
    max_expansions: int,
) -> dict[str, Any]:
    """Return one scalar metric plus a detailed base-prompt comparison."""

    cand_solved = bool(
        candidate.solved if isinstance(candidate, PuzzleScriptSearchResult)
        else candidate.get("solved", False)
    )
    cand_expanded = int(
        candidate.expanded_states if isinstance(candidate, PuzzleScriptSearchResult)
        else candidate.get("expanded", max_expansions)
    )
    if base_prompt_baseline is None:
        fallback = gepa_score(cand_solved, cand_expanded, max_expansions)
        return {
            "metric": fallback,
            "raw_metric": fallback,
            "outcome_class": "no_base_prompt_baseline",
            "base_solved": None,
            "base_expanded": None,
            "candidate_solved": cand_solved,
            "candidate_expanded": cand_expanded,
            "efficiency_gain_pct": None,
            "solve_component": fallback,
            "efficiency_component": 0.0,
            "speedup_component": 0.0,
            "regression_penalty": 0.0,
        }

    base_solved = bool(base_prompt_baseline.get("solved", False))
    base_expanded = int(base_prompt_baseline.get("expanded", max_expansions))
    if cand_solved and not base_solved:
        solve_component = 1.0
        outcome_class = "new_solve"
    elif cand_solved and base_solved:
        solve_component = PAIRWISE_SOLVE_BASE_BOTH_SOLVE
        outcome_class = "preserved_solve"
    elif not cand_solved and not base_solved:
        solve_component = PAIRWISE_SOLVE_BASE_BOTH_FAIL
        outcome_class = "both_failed"
    else:
        solve_component = -PAIRWISE_LOST_SOLVE_PENALTY
        outcome_class = "lost_solve"

    if cand_solved:
        efficiency_component = max(0.0, 1.0 - cand_expanded / max(max_expansions, 1))
    else:
        efficiency_component = 0.0

    speedup_component = 0.0
    if cand_solved and base_solved:
        denom = max(base_expanded, 1)
        regression_penalty = max(0.0, cand_expanded - base_expanded) / denom
        efficiency_gain_pct = 100.0 * (base_expanded - cand_expanded) / denom
        speedup_component = max(
            -1.0,
            min(1.0, math.log((base_expanded + 1) / max(cand_expanded + 1, 1))),
        )
        if cand_expanded < base_expanded:
            outcome_class = "preserved_solve_faster"
        elif cand_expanded > base_expanded:
            outcome_class = "preserved_solve_slower"
        else:
            outcome_class = "preserved_solve_same_expansions"
    elif base_solved and not cand_solved:
        regression_penalty = 1.0
        efficiency_gain_pct = 100.0 * (base_expanded - cand_expanded) / max(base_expanded, 1)
    elif base_expanded > 0:
        regression_penalty = 0.0
        efficiency_gain_pct = 100.0 * (base_expanded - cand_expanded) / base_expanded
    else:
        regression_penalty = 0.0
        efficiency_gain_pct = None

    raw_metric = (
        solve_component
        + PAIRWISE_EFFICIENCY_WEIGHT * efficiency_component
        + PAIRWISE_SPEEDUP_WEIGHT * speedup_component
        - PAIRWISE_REGRESSION_WEIGHT * regression_penalty
    )
    metric = max(-1.0, min(1.0, raw_metric))
    return {
        "metric": metric,
        "raw_metric": raw_metric,
        "outcome_class": outcome_class,
        "base_solved": base_solved,
        "base_expanded": base_expanded,
        "candidate_solved": cand_solved,
        "candidate_expanded": cand_expanded,
        "efficiency_gain_pct": efficiency_gain_pct,
        "solve_component": solve_component,
        "efficiency_component": efficiency_component,
        "speedup_component": speedup_component,
        "regression_penalty": regression_penalty,
    }


def _object_counts_from_ctx(root_ctx: Optional[Mapping[str, Any]]) -> dict[str, int]:
    if not root_ctx:
        return {}
    counts: dict[str, int] = {}
    for name, positions in (root_ctx.get("object_positions") or {}).items():
        if str(name).lower() == "background":
            continue
        try:
            count = len(positions)
        except TypeError:
            continue
        if count:
            counts[str(name)] = int(count)
    return dict(sorted(counts.items()))


def _relevant_object_names(root_ctx: Optional[Mapping[str, Any]]) -> set[str]:
    if not root_ctx:
        return set()
    names = {str(name) for name in (root_ctx.get("object_positions") or {})}
    win_text = str(root_ctx.get("win_conditions_text", "")).lower()
    object_names = [str(name) for name in root_ctx.get("object_names", [])]
    for name in object_names:
        if name.lower() in win_text:
            names.add(name)
    return {name for name in names if name and name.lower() != "background"}


def _extract_relevant_rule_snippets(
    game_text: str,
    root_ctx: Optional[Mapping[str, Any]],
    *,
    limit: int = 8,
) -> list[str]:
    """Return source-grounded rule lines mentioning current or win-condition objects."""

    rules_text = extract_section_text(game_text, "RULES")
    rule_lines = [line.strip() for line in rules_text.splitlines() if line.strip()]
    if not rule_lines:
        return []

    object_names = sorted(_relevant_object_names(root_ctx), key=len, reverse=True)
    relevant: list[str] = []
    for line in rule_lines:
        lowered = line.lower()
        if any(re.search(rf"\b{re.escape(name.lower())}\b", lowered) for name in object_names):
            relevant.append(line)
        if len(relevant) >= limit:
            break
    if relevant:
        return relevant
    return rule_lines[:limit]


def _source_context_feedback(
    *,
    game_text: str,
    root_ctx: Optional[Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if root_ctx:
        win_text = str(root_ctx.get("win_conditions_text", "unknown"))
        lines.append(f"Runtime win_conditions_text: {win_text}")
        counts = _object_counts_from_ctx(root_ctx)
        if counts:
            lines.append("Initial object counts: " + json.dumps(counts, sort_keys=True))
        score = root_ctx.get("score")
        score_norm = root_ctx.get("score_normalized")
        lines.append(f"Initial engine progress: score={score} score_normalized={score_norm}")

    winconditions_source = extract_section_text(game_text, "WINCONDITIONS")
    if winconditions_source:
        compact_win = " | ".join(winconditions_source.splitlines()[:4])
        lines.append(f"WINCONDITIONS source: {compact_win}")

    snippets = _extract_relevant_rule_snippets(game_text, root_ctx)
    if snippets:
        lines.append("Relevant RULES snippets: " + " | ".join(snippets))
    else:
        lines.append("No RULES snippets were available; rely on win conditions and runtime ctx.")
    return lines


def _sample_local_heuristic_diagnostics(
    engine,
    compiled_json: dict[str, Any],
    heuristic_fn: Callable,
) -> dict[str, Any]:
    """Probe the heuristic on the root state and immediate successors.

    The current GEPA feedback lacks any explanation of the heuristic's local
    shape. This helper adds cheap probes that can catch near-constant outputs,
    oversized penalties, and rankings that disagree with the engine's native
    progress signal.
    """

    initial_backup = engine.backup_level()
    root_ctx = build_puzzlescript_ctx(engine, compiled_json)
    def _safe_heuristic_value(ctx: Mapping[str, Any]) -> float:
        try:
            value = float(heuristic_fn(ctx))
        except Exception:
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return max(0.0, value)

    root_h = _safe_heuristic_value(root_ctx)

    has_action = not engine.has_metadata("noaction") if hasattr(engine, "has_metadata") else True
    n_actions = 5 if has_action else 4

    successors: list[dict[str, Any]] = []
    for action in range(n_actions):
        engine.restore_level(initial_backup)
        if not _process_action_with_again(engine, action):
            continue
        ctx = build_puzzlescript_ctx(engine, compiled_json)
        h_val = _safe_heuristic_value(ctx)
        successors.append(
            {
                "action": action,
                "heuristic": h_val,
                "score_normalized": float(ctx.get("score_normalized", 0.0)),
                "is_winning": bool(ctx.get("is_winning", False)),
            }
        )

    engine.restore_level(initial_backup)

    if not successors:
        return {
            "root_heuristic": root_h,
            "n_successors": 0,
            "heuristic_range": 0.0,
            "heuristic_stddev": 0.0,
            "winning_successor_count": 0,
            "top_progress_action": None,
            "top_heuristic_action": None,
            "ranking_mismatch": False,
        }

    heuristic_values = [float(entry["heuristic"]) for entry in successors]
    rounded_values = {round(value, 3) for value in heuristic_values}
    top_progress = max(successors, key=lambda entry: entry["score_normalized"])
    top_heuristic = min(successors, key=lambda entry: entry["heuristic"])
    return {
        "root_heuristic": root_h,
        "n_successors": len(successors),
        "heuristic_range": max(heuristic_values) - min(heuristic_values),
        "heuristic_stddev": statistics.pstdev(heuristic_values) if len(heuristic_values) > 1 else 0.0,
        "winning_successor_count": sum(1 for entry in successors if entry["is_winning"]),
        "top_progress_action": top_progress["action"],
        "top_progress_score_normalized": top_progress["score_normalized"],
        "top_heuristic_action": top_heuristic["action"],
        "top_heuristic_value": top_heuristic["heuristic"],
        "ranking_mismatch": (
            len(successors) >= 3
            and top_progress["score_normalized"] > 0.0
            and top_progress["action"] != top_heuristic["action"]
        ),
        "constant_like": len(rounded_values) <= 1,
        "penalty_dominated": max(heuristic_values) >= max(min(heuristic_values) + 1000.0, 1000.0),
    }


def _build_feedback_report(
    *,
    game_name: str,
    game_text: str,
    result: PuzzleScriptSearchResult,
    max_expansions: int,
    diagnostics: dict[str, Any],
    root_ctx: Optional[Mapping[str, Any]] = None,
    blind_baseline: Optional[dict[str, Any]] = None,
    builtin_baseline: Optional[dict[str, Any]] = None,
    base_prompt_baseline: Optional[dict[str, Any]] = None,
    metric_breakdown: Optional[dict[str, Any]] = None,
) -> str:
    """Build structured feedback for GEPA reflection.

    The previous feedback was a short scalar summary. This version adds
    actionable observations about local heuristic shape, search efficiency, and
    likely mechanic families that deserve attention.
    """

    metric_breakdown = metric_breakdown or {}
    outcome_lines = [
        f"Game: {game_name}",
        (
            f"Outcome: solved={result.solved} raw_search_score={result.score:.4f} "
            f"metric={float(metric_breakdown.get('metric', result.score)):.4f}"
        ),
        (
            "Search stats: "
            f"expanded={result.expanded_states}/{max_expansions} "
            f"generated={result.generated_states} "
            f"solution_length={result.solution_length}"
        ),
    ]
    if base_prompt_baseline is not None:
        base_expanded = int(base_prompt_baseline.get("expanded", max_expansions))
        base_solved = bool(base_prompt_baseline.get("solved", False))
        gain = metric_breakdown.get("efficiency_gain_pct")
        gain_text = "n/a" if gain is None else f"{float(gain):+.1f}%"
        outcome_lines.append(
            "Base-prompt comparison: "
            f"outcome={metric_breakdown.get('outcome_class', 'unknown')} "
            f"base_solved={base_solved} candidate_solved={result.solved} "
            f"base_expanded={base_expanded} candidate_expanded={result.expanded_states} "
            f"efficiency_gain={gain_text}"
        )
        outcome_lines.append(
            "Metric components: "
            f"solve={float(metric_breakdown.get('solve_component', 0.0)):.3f} "
            f"efficiency={float(metric_breakdown.get('efficiency_component', 0.0)):.3f} "
            f"speedup={float(metric_breakdown.get('speedup_component', 0.0)):.3f} "
            f"regression_penalty={float(metric_breakdown.get('regression_penalty', 0.0)):.3f}"
        )

    if blind_baseline is not None:
        blind_expanded = int(blind_baseline.get("expanded", 0))
        if blind_expanded > 0:
            delta = result.expanded_states - blind_expanded
            outcome_lines.append(
                "Blind comparison: "
                f"expanded={result.expanded_states} vs blind={blind_expanded} "
                f"(delta={delta:+d})"
            )
    if builtin_baseline is not None:
        builtin_expanded = int(builtin_baseline.get("expanded", 0))
        if builtin_expanded > 0:
            delta = result.expanded_states - builtin_expanded
            outcome_lines.append(
                "Builtin comparison: "
                f"expanded={result.expanded_states} vs builtin={builtin_expanded} "
                f"(delta={delta:+d})"
            )

    observed_issues: list[str] = []
    counterexamples: list[str] = []

    expansion_ratio = result.expanded_states / max(max_expansions, 1)
    if not result.solved:
        if expansion_ratio >= 0.95:
            observed_issues.append("Search nearly exhausted the full expansion budget before failing.")
        else:
            observed_issues.append("Search failed before solving, so the heuristic is not guiding search toward a valid plan reliably enough.")
    elif result.solution_length > 0:
        work_ratio = result.expanded_states / max(result.solution_length, 1)
        if work_ratio >= 8.0:
            observed_issues.append(
                f"A* still expanded {work_ratio:.1f} states per solution step, which suggests weak prioritization among near-goal states."
            )

    if diagnostics.get("constant_like"):
        observed_issues.append("Immediate successors receive nearly constant heuristic values, so the heuristic provides little local ranking signal.")
    elif diagnostics.get("heuristic_range", 0.0) < 1.0 and diagnostics.get("n_successors", 0) >= 2:
        observed_issues.append("Immediate successor scores vary only slightly, so the heuristic may be too flat near the root.")

    if diagnostics.get("penalty_dominated"):
        observed_issues.append("Large penalty values dominate local scoring, which can drown out incremental progress signals.")

    if diagnostics.get("ranking_mismatch"):
        observed_issues.append("The locally best-ranked action does not match the action with the strongest engine progress signal.")
        counterexamples.append(
            "At the root, the heuristic prefers action "
            f"{diagnostics.get('top_heuristic_action')}, but the engine's normalized progress score prefers action "
            f"{diagnostics.get('top_progress_action')}."
        )

    root_h = diagnostics.get("root_heuristic")
    if not result.solved and isinstance(root_h, (float, int)) and root_h == 0.0:
        observed_issues.append("The root state heuristic is 0 on a non-winning state, which makes the search behave like blind search near the start.")

    if result.solved and blind_baseline is not None:
        blind_expanded = int(blind_baseline.get("expanded", 0))
        if blind_expanded > 0 and result.expanded_states >= blind_expanded:
            observed_issues.append("This solved run does not outperform blind search on expansion count, so the heuristic is adding little value.")

    outcome_class = metric_breakdown.get("outcome_class")
    if outcome_class == "new_solve":
        observed_issues.append(
            "This is a new solve relative to the base prompt; preserve the mechanic-specific idea that made the level solvable."
        )
        counterexamples.append(
            "Prompt repair: identify which rule family enabled the new solve and make that rule-analysis step explicit in the rewritten prompt."
        )
    elif outcome_class == "preserved_solve_faster":
        observed_issues.append(
            "The candidate preserves the base solve and improves expansion efficiency; keep the ranking features that reduced search."
        )
        counterexamples.append(
            "Prompt repair: strengthen this style of successor ranking and tie-breaking for similar games."
        )
    elif outcome_class == "preserved_solve_slower":
        observed_issues.append(
            "The candidate preserves solvability but is slower than the base prompt, so the prompt likely weakened local ranking or introduced broad plateaus."
        )
        counterexamples.append(
            "Prompt repair: add finer tie-breakers such as player-to-interaction distance, object progress, and small penalties for reversible wandering; avoid broad constant values."
        )
    elif outcome_class == "lost_solve":
        observed_issues.append(
            "The candidate lost a level that the base prompt solved, suggesting an over-specific assumption, wrong object mapping, or unjustified deadlock penalty."
        )
        counterexamples.append(
            "Prompt repair: require a conservative fallback based directly on win_conditions_text and object names before adding hard deadlocks or Sokoban-specific assumptions."
        )
    elif outcome_class == "both_failed":
        observed_issues.append(
            "Both candidate and base failed, so the prompt likely needs a new game-mechanic abstraction rather than small distance tuning."
        )
        counterexamples.append(
            "Prompt repair: derive subgoals from the actual win condition and rules; if the objective is not object-on-target, do not force a crate-target heuristic."
        )

    if not observed_issues:
        observed_issues.append("The run solved the game cleanly; focus on sharper mechanic-specific ranking to cut expansions further.")

    source_context_lines = _source_context_feedback(game_text=game_text, root_ctx=root_ctx)

    profile_line = (
        "Local heuristic profile: "
        f"root={diagnostics.get('root_heuristic', 0.0):.3f} "
        f"successors={diagnostics.get('n_successors', 0)} "
        f"range={diagnostics.get('heuristic_range', 0.0):.3f} "
        f"stddev={diagnostics.get('heuristic_stddev', 0.0):.3f}"
    )

    sections = [
        "\n".join(outcome_lines),
        "Observed issues:\n- " + "\n- ".join(observed_issues),
        profile_line,
        "Source-grounded mechanics context:\n- " + "\n- ".join(source_context_lines),
    ]
    if counterexamples:
        sections.append("Counterexamples:\n- " + "\n- ".join(counterexamples))
    return "\n\n".join(sections)


def _build_trace_prompt_block(result: PuzzleScriptSearchResult) -> str:
    """Serialize the compact A* trace summary for reflection prompts."""

    trace_summary = result.trace_summary or {}
    if not trace_summary:
        return "Search trace summary unavailable."
    return json.dumps(trace_summary, indent=2, sort_keys=True)


def _reflect_with_llm(
    *,
    env_description: Optional[str],
    heuristic_code: Optional[str],
    deterministic_feedback: str,
    result: PuzzleScriptSearchResult,
    reflection_lm: Any = None,
) -> str:
    """Run LLM reflection over deterministic search evidence with fallback."""

    if reflection_lm is None or not env_description or not heuristic_code:
        return deterministic_feedback

    should_reflect = not result.solved
    trace_summary = result.trace_summary or {}
    if not should_reflect:
        if trace_summary.get("terminated_reason") != "solved":
            should_reflect = True
        elif "Observed issues:\n- The run solved the game cleanly;" not in deterministic_feedback:
            should_reflect = True
    if not should_reflect:
        return deterministic_feedback

    prompt_summary = (
        deterministic_feedback
        + "\n\nCompact search trace:\n"
        + _build_trace_prompt_block(result)
        + "\n\nTask:\n"
        + "1. Identify the most likely heuristic failure modes.\n"
        + "2. Explain which PuzzleScript mechanics the heuristic seems to ignore or model incorrectly.\n"
        + "3. Ground the explanation in the observed search behavior.\n"
        + "4. Give concrete guidance for the next heuristic revision.\n"
        + "Keep the response concise and mechanism-specific."
    )

    try:
        with dspy.context(lm=reflection_lm):
            prediction = _feedback_reflector(
                env_description=env_description,
                heuristic_code=heuristic_code,
                search_summary=prompt_summary,
            )
        reflection_text = str(prediction.reflection).strip()
    except Exception as exc:
        return deterministic_feedback + f"\n\nLLM reflection unavailable: {exc}"

    if not reflection_text:
        return deterministic_feedback

    return deterministic_feedback + "\n\nLLM analysis:\n" + reflection_text


# ---------------------------------------------------------------------------
# DSPy Program: the thing GEPA optimizes
# ---------------------------------------------------------------------------
class PuzzleScriptPromptProgram(dspy.Module):
    """DSPy module whose prompt text GEPA optimizes."""

    def __init__(self, base_prompt: str, prompt_state: Optional[Mapping] = None):
        super().__init__()
        self.base_prompt = base_prompt

        class PromptSearch(dspy.Signature):
            base_prompt: str = dspy.InputField()
            prompt_text: str = dspy.OutputField(
                desc="Rewritten prompt for PuzzleScript heuristic synthesis")

        class PromptGenerator(dspy.Module):
            def __init__(self, state=None):
                super().__init__()
                self.rewriter = dspy.Predict(PromptSearch)
                if state:
                    self.rewriter.load_state(state)

            def dump_state(self):
                return self.rewriter.dump_state()

            def forward(self, base_prompt: str) -> str:
                return self.rewriter(base_prompt=base_prompt).prompt_text

        self.prompt_generator = PromptGenerator(prompt_state)

    def _build_rewrite_prompt(self) -> str:
        return (
            "Rewrite the following prompt to produce better PuzzleScript heuristics.\n"
            "Focus on game-specific mechanics, not generic distance metrics.\n\n"
            + self.base_prompt
        )

    def forward(self, env_description: str, heuristic_contract: str, **kwargs):
        prompt_text = self.prompt_generator(self._build_rewrite_prompt())
        return dspy.Prediction(prompt_text=prompt_text)


# ---------------------------------------------------------------------------
# Evaluate one heuristic on one game
# ---------------------------------------------------------------------------
def evaluate_one_game(
    evaluator: PuzzleScriptEvaluator,
    game_name: str,
    game_text: str,
    heuristic_fn: Callable,
    max_expansions: int,
    output_dir: Optional[Path] = None,
    level_i: int = 0,
    blind_baseline: Optional[dict[str, Any]] = None,
    builtin_baseline: Optional[dict[str, Any]] = None,
    base_prompt_baseline: Optional[dict[str, Any]] = None,
    env_description: Optional[str] = None,
    heuristic_code: Optional[str] = None,
    reflection_lm: Any = None,
    astar_timeout_s: float = DEFAULT_ASTAR_TIMEOUT_S,
) -> dict[str, Any]:
    """Compile game, run A* on one level, return result dict with feedback."""
    json_str = evaluator.compile_game(game_text)
    compiled = json.loads(json_str)
    engine = evaluator.load_engine(json_str)
    engine.load_level(level_i)
    root_ctx = build_puzzlescript_ctx(engine, compiled)
    diagnostics = _sample_local_heuristic_diagnostics(engine, compiled, heuristic_fn)
    engine.load_level(level_i)

    result = puzzlescript_astar(
        engine=engine, compiled_json=compiled,
        heuristic_fn=heuristic_fn, max_expansions=max_expansions,
        timeout_s=astar_timeout_s,
    )
    metric_breakdown = _pairwise_gepa_metric(
        candidate=result,
        base_prompt_baseline=base_prompt_baseline,
        max_expansions=max_expansions,
    )

    deterministic_feedback = _build_feedback_report(
        game_name=game_name,
        game_text=game_text,
        result=result,
        max_expansions=max_expansions,
        diagnostics=diagnostics,
        root_ctx=root_ctx,
        blind_baseline=blind_baseline,
        builtin_baseline=builtin_baseline,
        base_prompt_baseline=base_prompt_baseline,
        metric_breakdown=metric_breakdown,
    )
    feedback = _reflect_with_llm(
        env_description=env_description,
        heuristic_code=heuristic_code,
        deterministic_feedback=deterministic_feedback,
        result=result,
        reflection_lm=reflection_lm,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "search_stats.json").write_text(json.dumps({
            "game": game_name, "level": level_i, "solved": result.solved,
            "expanded": result.expanded_states,
            "generated": result.generated_states,
            "solution_length": result.solution_length,
            "score": metric_breakdown["metric"],
            "raw_search_score": result.score,
            "time_s": result.time_s,
            "max_expansions": max_expansions,
            "astar_timeout_s": astar_timeout_s,
            "metric_breakdown": metric_breakdown,
            "feedback_diagnostics": diagnostics,
            "trace_summary": result.trace_summary,
            "deterministic_feedback": deterministic_feedback,
            "feedback": feedback,
        }, indent=2))

    return {
        "score": metric_breakdown["metric"],
        "raw_search_score": result.score,
        "level": level_i,
        "solved": result.solved,
        "expanded": result.expanded_states,
        "generated": result.generated_states,
        "solution_length": result.solution_length,
        "feedback": feedback,
        "deterministic_feedback": deterministic_feedback,
        "feedback_diagnostics": diagnostics,
        "trace_summary": result.trace_summary,
        "metric_breakdown": metric_breakdown,
    }


def _aggregate_level_results(
    *,
    game_name: str,
    level_results: list[dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Aggregate multiple level evaluations into one GEPA metric payload."""

    if not level_results:
        return {
            "score": 0.0,
            "solved": False,
            "expanded": 0,
            "generated": 0,
            "solution_length": 0,
            "feedback": f"No levels evaluated for {game_name}",
            "level_results": [],
        }

    score = sum(float(row["score"]) for row in level_results) / len(level_results)
    solved = all(bool(row["solved"]) for row in level_results)
    expanded = sum(int(row["expanded"]) for row in level_results)
    generated = sum(int(row["generated"]) for row in level_results)
    solution_length = sum(int(row["solution_length"]) for row in level_results)
    failed = [row for row in level_results if not row["solved"]]
    if failed:
        lead = failed[0]
        feedback = (
            f"Multi-level aggregate for {game_name}: solved {len(level_results) - len(failed)}/"
            f"{len(level_results)}, mean_score={score:.4f}. First failed level "
            f"{lead['level']} feedback:\n{lead.get('feedback', '')}"
        )
    else:
        feedback = (
            f"Multi-level aggregate for {game_name}: solved {len(level_results)}/"
            f"{len(level_results)}, mean_score={score:.4f}, expanded_total={expanded}."
        )

    payload = {
        "score": score,
        "solved": solved,
        "expanded": expanded,
        "generated": generated,
        "solution_length": solution_length,
        "feedback": feedback,
        "level_results": level_results,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "search_stats.json").write_text(json.dumps({
            "game": game_name,
            "levels": [row["level"] for row in level_results],
            "solved": solved,
            "score": score,
            "expanded": expanded,
            "generated": generated,
            "solution_length": solution_length,
            "level_results": level_results,
            "feedback": feedback,
        }, indent=2))
    return payload


def _should_accept_prompt_candidate(
    *,
    mean_score: float,
    solve_rate: float,
    best_mean_score: Optional[float],
    best_solve_rate: Optional[float],
) -> tuple[bool, str]:
    """Gate champion updates so a scalar gain cannot silently lose solves."""

    if best_mean_score is None or best_solve_rate is None:
        return True, "first_candidate"

    score_delta = mean_score - float(best_mean_score)
    solve_delta = solve_rate - float(best_solve_rate)
    if solve_delta >= 0.0 and score_delta > BEST_PROMPT_MIN_SCORE_DELTA:
        return True, "score_gain_without_solve_regression"
    if solve_delta > 0.0 and mean_score >= float(best_mean_score) - BEST_PROMPT_SCORE_BACKOFF_FOR_SOLVE_GAIN:
        return True, "solve_rate_gain_with_small_score_backoff"
    return False, (
        f"rejected: score_delta={score_delta:+.4f}, "
        f"solve_delta={solve_delta:+.4f}"
    )


def evaluate_game_levels(
    evaluator: PuzzleScriptEvaluator,
    game_name: str,
    game_text: str,
    heuristic_fn: Callable,
    level_budgets: Mapping[int, int],
    output_dir: Optional[Path] = None,
    blind_baselines: Optional[Mapping[int, dict[str, Any]]] = None,
    builtin_baselines: Optional[Mapping[int, dict[str, Any]]] = None,
    base_prompt_baselines: Optional[Mapping[int, dict[str, Any]]] = None,
    env_description: Optional[str] = None,
    heuristic_code: Optional[str] = None,
    reflection_lm: Any = None,
    astar_timeout_s: float = DEFAULT_ASTAR_TIMEOUT_S,
) -> dict[str, Any]:
    """Evaluate one heuristic on several levels and average their scores."""

    level_results: list[dict[str, Any]] = []
    for level_i, budget in level_budgets.items():
        level_output = output_dir / f"level-{level_i:02d}" if output_dir else None
        level_results.append(
            evaluate_one_game(
                evaluator,
                game_name,
                game_text,
                heuristic_fn,
                budget,
                level_i=level_i,
                output_dir=level_output,
                blind_baseline=(blind_baselines or {}).get(level_i),
                builtin_baseline=(builtin_baselines or {}).get(level_i),
                base_prompt_baseline=(base_prompt_baselines or {}).get(level_i),
                env_description=env_description,
                heuristic_code=heuristic_code,
                reflection_lm=reflection_lm,
                astar_timeout_s=astar_timeout_s,
            )
        )
    return _aggregate_level_results(
        game_name=game_name,
        level_results=level_results,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Synthesize heuristic from prompt text
# ---------------------------------------------------------------------------
def _invoke_heuristic_predictor(
    *,
    prompt_text: str,
    env_description: str,
    lm: Any,
) -> tuple[str, Optional[str]]:
    try:
        with dspy.context(lm=lm):
            pred = _heuristic_predictor(
                synthesis_prompt=prompt_text,
                env_description=env_description,
            )
        return str(pred.heuristic_code), None
    except Exception as e:
        return "", f"LLM call failed: {e}"


def _strip_outer_markdown_fences(code: str) -> str:
    cleaned = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def _compile_synthesized_heuristic(code: str) -> tuple[Optional[Callable], Optional[str]]:
    try:
        raw_fn = sanitize_and_compile_puzzlescript_heuristic(code)

        def heuristic_from_ctx(ctx: dict[str, Any]) -> float:
            return float(raw_fn(None, None, ctx))

        return heuristic_from_ctx, None
    except Exception as e:
        return None, f"Sanitization failed: {e}"


def _constant_return_issue(code: str) -> bool:
    try:
        tree = ast.parse(_strip_outer_markdown_fences(code))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "heuristic_cost_to_go":
            executable = [
                child for child in node.body
                if not isinstance(child, (ast.Expr, ast.Pass))
            ]
            return (
                len(executable) == 1
                and isinstance(executable[0], ast.Return)
                and isinstance(executable[0].value, ast.Constant)
                and isinstance(executable[0].value.value, (int, float))
            )
    return False


def _static_synthesis_issues(code: str, compile_error: Optional[str]) -> list[str]:
    issues: list[str] = []
    if compile_error:
        issues.append(compile_error)
    if "```" in code:
        issues.append("Output contains markdown fences/backticks; return plain Python code only.")
    if re.search(r"^\s*(from\s+\S+\s+import|import\s+\S+)", code, flags=re.MULTILINE):
        issues.append("Output contains imports; imports are not allowed.")
    if "def heuristic_cost_to_go" not in code:
        issues.append("Output does not define heuristic_cost_to_go.")
    if _constant_return_issue(code):
        issues.append("Function returns a single constant and provides no A* guidance.")
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", code):
        try:
            if abs(float(match.group(0))) >= 100_000:
                issues.append("Output uses very large numeric constants; avoid huge penalties unless proven safe.")
                break
        except ValueError:
            continue
    return issues


def _preflight_synthesis_issues(
    *,
    evaluator: Optional[PuzzleScriptEvaluator],
    game_text: Optional[str],
    level_i: int,
    heuristic_fn: Optional[Callable],
) -> list[str]:
    if evaluator is None or not game_text or heuristic_fn is None:
        return []
    try:
        json_str = evaluator.compile_game(game_text)
        compiled = json.loads(json_str)
        engine = evaluator.load_engine(json_str)
        engine.load_level(level_i)
        root_ctx = build_puzzlescript_ctx(engine, compiled)
        diagnostics = _sample_local_heuristic_diagnostics(engine, compiled, heuristic_fn)
    except Exception as exc:
        return [f"Preflight diagnostics failed before A*: {exc}"]

    issues: list[str] = []
    if (
        not bool(root_ctx.get("is_winning", False))
        and float(diagnostics.get("root_heuristic", 0.0)) == 0.0
    ):
        issues.append("Root state is non-winning but heuristic returns 0.0.")
    if diagnostics.get("constant_like") and int(diagnostics.get("n_successors", 0)) >= 2:
        issues.append("Immediate successors receive constant or near-constant values.")
    if diagnostics.get("penalty_dominated"):
        issues.append("Local successor values are dominated by huge penalties.")
    return issues


def _repair_prompt_text(prompt_text: str, issues: list[str]) -> str:
    return (
        prompt_text
        + "\n\nREPAIR MODE:\n"
        + "The previous heuristic output failed validation or preflight. Rewrite it once.\n"
        + "Output exactly one plain Python heuristic_cost_to_go function. No markdown, no imports, no prose.\n"
        + "Keep the function finite and state-varying. Use win_conditions_text, object_positions, "
        + "object counts, player-to-interaction distance, and score_normalized as conservative "
        + "fallbacks when mechanics are uncertain.\n"
        + "Issues to fix:\n- "
        + "\n- ".join(issues)
    )


def _repair_env_description(env_description: str, bad_code: str) -> str:
    return (
        env_description
        + "\n\nPrevious heuristic output that must be repaired:\n"
        + _strip_outer_markdown_fences(bad_code)[:12_000]
    )


def synthesize_heuristic_from_prompt(
    prompt_text: str,
    env_description: str,
    lm: Any,
    *,
    preflight_evaluator: Optional[PuzzleScriptEvaluator] = None,
    preflight_game_text: Optional[str] = None,
    preflight_level_i: int = 0,
) -> tuple[Optional[Callable], str, Optional[str]]:
    """Use the (GEPA-optimized) prompt to synthesize a heuristic."""
    code, call_error = _invoke_heuristic_predictor(
        prompt_text=prompt_text,
        env_description=env_description,
        lm=lm,
    )
    if call_error:
        return None, "", call_error

    heuristic_fn, compile_error = _compile_synthesized_heuristic(code)
    issues = _static_synthesis_issues(code, compile_error)
    if not issues:
        issues = _preflight_synthesis_issues(
            evaluator=preflight_evaluator,
            game_text=preflight_game_text,
            level_i=preflight_level_i,
            heuristic_fn=heuristic_fn,
        )
    if not issues:
        return heuristic_fn, _strip_outer_markdown_fences(code), None

    repair_code, repair_call_error = _invoke_heuristic_predictor(
        prompt_text=_repair_prompt_text(prompt_text, issues),
        env_description=_repair_env_description(env_description, code),
        lm=lm,
    )
    if repair_call_error:
        if heuristic_fn is not None and compile_error is None:
            return heuristic_fn, _strip_outer_markdown_fences(code), None
        return None, code, repair_call_error

    repaired_fn, repaired_compile_error = _compile_synthesized_heuristic(repair_code)
    repaired_issues = _static_synthesis_issues(repair_code, repaired_compile_error)
    if not repaired_issues:
        return repaired_fn, _strip_outer_markdown_fences(repair_code), None

    if heuristic_fn is not None and compile_error is None:
        return heuristic_fn, _strip_outer_markdown_fences(code), None
    return None, repair_code, "; ".join(repaired_issues)


def evaluate_prompt_per_game(
    *,
    evaluator: PuzzleScriptEvaluator,
    prompt_text: str,
    game_names: list[str],
    all_game_texts: Mapping[str, str],
    all_env_descs: Mapping[str, str],
    level_indices_by_game: Mapping[str, list[int]],
    max_expansions: int,
    lm: Any,
    blind_baselines: Optional[Mapping[str, Mapping[int, dict[str, Any]]]] = None,
    builtin_baselines: Optional[Mapping[str, Mapping[int, dict[str, Any]]]] = None,
    base_prompt_baselines: Optional[Mapping[str, Mapping[int, dict[str, Any]]]] = None,
    output_dir: Optional[Path] = None,
    reflection_lm: Any = None,
    astar_timeout_s: float = DEFAULT_ASTAR_TIMEOUT_S,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Synthesize and evaluate one heuristic per game from a shared prompt."""

    results: list[dict[str, Any]] = []
    heuristic_codes: dict[str, str] = {}
    for name in game_names:
        if name not in all_game_texts:
            continue
        env_desc = all_env_descs.get(name, name)
        heuristic_fn, code, error = synthesize_heuristic_from_prompt(
            prompt_text,
            env_desc,
            lm,
            preflight_evaluator=evaluator,
            preflight_game_text=all_game_texts[name],
            preflight_level_i=level_indices_by_game.get(name, [0])[0],
        )
        if error:
            print(f"    [{name}] final synthesis error: {error[:200]}")
            heuristic_fn = builtin_heuristic
            code = f"# FALLBACK: {error[:200]}"

        heuristic_codes[name] = code
        if output_dir is not None:
            game_output_dir = output_dir / name
            game_output_dir.mkdir(parents=True, exist_ok=True)
            (game_output_dir / "heuristic.py").write_text(code)
        else:
            game_output_dir = None

        final_budgets = {
            level_i: max_expansions
            for level_i in level_indices_by_game.get(name, [0])
        }
        result = evaluate_game_levels(
            evaluator,
            name,
            all_game_texts[name],
            heuristic_fn,
            final_budgets,
            output_dir=game_output_dir,
            blind_baselines=(blind_baselines or {}).get(name),
            builtin_baselines=(builtin_baselines or {}).get(name),
            base_prompt_baselines=(base_prompt_baselines or {}).get(name),
            env_description=env_desc,
            heuristic_code=code,
            reflection_lm=reflection_lm,
            astar_timeout_s=astar_timeout_s,
        )
        result["game"] = name
        results.append(result)
    return results, heuristic_codes


def evaluate_prompt_per_level(
    *,
    evaluator: PuzzleScriptEvaluator,
    prompt_text: str,
    examples: list[Mapping[str, Any]],
    all_game_texts: Mapping[str, str],
    all_level_env_descs: Mapping[str, Mapping[int, str]],
    max_expansions: int,
    lm: Any,
    blind_baselines: Optional[Mapping[str, Mapping[int, dict[str, Any]]]] = None,
    builtin_baselines: Optional[Mapping[str, Mapping[int, dict[str, Any]]]] = None,
    base_prompt_baselines: Optional[Mapping[str, Mapping[int, dict[str, Any]]]] = None,
    output_dir: Optional[Path] = None,
    reflection_lm: Any = None,
    astar_timeout_s: float = DEFAULT_ASTAR_TIMEOUT_S,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Synthesize and evaluate one heuristic per game level."""

    results: list[dict[str, Any]] = []
    heuristic_codes: dict[str, str] = {}
    for example in examples:
        name = str(example["game"])
        level_i = int(example["level"])
        budget = int(example.get("budget", max_expansions))
        if name not in all_game_texts:
            continue
        env_desc = all_level_env_descs.get(name, {}).get(level_i, name)
        heuristic_fn, code, error = synthesize_heuristic_from_prompt(
            prompt_text,
            env_desc,
            lm,
            preflight_evaluator=evaluator,
            preflight_game_text=all_game_texts[name],
            preflight_level_i=level_i,
        )
        level_key = f"{name}::level-{level_i:02d}"
        if error:
            print(f"    [{level_key}] final synthesis error: {error[:200]}")
            heuristic_fn = builtin_heuristic
            code = f"# FALLBACK: {error[:200]}"

        heuristic_codes[level_key] = code
        if output_dir is not None:
            level_output_dir = output_dir / name / f"level-{level_i:02d}"
            level_output_dir.mkdir(parents=True, exist_ok=True)
            (level_output_dir / "heuristic.py").write_text(code)
        else:
            level_output_dir = None

        try:
            result = evaluate_one_game(
                evaluator,
                name,
                all_game_texts[name],
                heuristic_fn,
                budget,
                output_dir=level_output_dir,
                level_i=level_i,
                blind_baseline=(blind_baselines or {}).get(name, {}).get(level_i),
                builtin_baseline=(builtin_baselines or {}).get(name, {}).get(level_i),
                base_prompt_baseline=(base_prompt_baselines or {}).get(name, {}).get(level_i),
                env_description=env_desc,
                heuristic_code=code,
                reflection_lm=reflection_lm,
                astar_timeout_s=astar_timeout_s,
            )
        except RuntimeError as e:
            if "Level index out of range" in str(e):
                print(f"    [{level_key}] skipping invalid level: {e}")
                continue
            raise
        result["game"] = name
        result["level"] = level_i
        result["example"] = level_key
        results.append(result)
    return results, heuristic_codes


# ---------------------------------------------------------------------------
# Main curriculum runner with GEPA
# ---------------------------------------------------------------------------
def run_curriculum(
    evaluator: PuzzleScriptEvaluator,
    train_jobs: list[dict],
    eval_jobs: list[dict],
    sd_path: Path,
    state_root: Path,
    baseline_root: Optional[Path],
    max_phase_iterations: int,
    max_expansions: int,
    llm_name: str,
    llm_max_tokens: int,
    levels_per_game: int,
    gepa_num_threads: int,
    max_gepa_expansions_per_level: int,
    astar_timeout_s: float,
) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "runs"
    logs_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "curriculum_state.json"
    baseline_state_root = (baseline_root or state_root).expanduser().resolve()
    baseline_state_root.mkdir(parents=True, exist_ok=True)

    # Configure LLM
    lm = dspy.LM(llm_name, max_tokens=llm_max_tokens)
    dspy.configure(lm=lm)
    cost_logger = LMCostLogger(lm, state_root)
    print(f"LLM: {llm_name} max_tokens={llm_max_tokens}")

    # Pre-compile all game texts and env descriptions
    all_game_texts, all_env_descs, all_level_env_descs, level_indices_by_game = (
        prepare_puzzlescript_inputs(
            evaluator=evaluator,
            train_jobs=train_jobs,
            eval_jobs=eval_jobs,
            sd_path=sd_path,
            levels_per_game=levels_per_game,
        )
    )

    # Phase schedule
    phase_schedule = build_curriculum_phase_schedule(train_jobs)
    total_phases = len(phase_schedule)

    # Load or init state
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        print(f"Resumed from {state_path}")
    else:
        state = {
            "current_phase": 1, "completed_phases": [],
            "phase_records": {}, "total_phases": total_phases,
            "phase_game_counts": [len(p) for p in phase_schedule],
            "global_iteration": 0, "stop_reason": None,
            "best_heuristic_code": None, "best_heuristic_codes": {},
            "best_prompt_text": None,
            "best_prompt_state": None,
        }

    # Load or compute baselines on ALL training levels. Budget =
    # floor(0.95 * blind_expanded) so a heuristic matching blind search exceeds
    # its GEPA budget and scores 0, forcing useful guidance.
    print("\n--- Training baselines (cache + shards) ---")
    print(f"  Baseline root: {baseline_state_root}")
    all_train_names = [str(e["name"]) for e in train_jobs if str(e["name"]) in all_game_texts]
    all_train_examples = build_training_level_examples(
        train_jobs,
        all_game_texts,
        level_indices_by_game,
    )
    baseline_signature = build_baseline_cache_signature(
        train_jobs=train_jobs,
        level_indices_by_game=level_indices_by_game,
        max_expansions=max_expansions,
        max_gepa_expansions_per_level=max_gepa_expansions_per_level,
        astar_timeout_s=astar_timeout_s,
        levels_per_game=levels_per_game,
        llm_name=llm_name,
        llm_max_tokens=llm_max_tokens,
    )
    (
        blind_baselines,
        builtin_baselines,
        base_prompt_baselines,
        per_game_budgets,
        loaded_baseline_paths,
    ) = load_cached_puzzlescript_baselines(baseline_state_root, baseline_signature)
    if loaded_baseline_paths:
        print(f"  Loaded {len(loaded_baseline_paths)} matching baseline cache/shard files.")

    missing_baseline_examples = missing_puzzlescript_baseline_examples(
        all_train_examples,
        blind_baselines=blind_baselines,
        builtin_baselines=builtin_baselines,
        base_prompt_baselines=base_prompt_baselines,
        per_game_budgets=per_game_budgets,
    )
    computed_missing_count = len(missing_baseline_examples)
    if missing_baseline_examples:
        print(f"  Computing {computed_missing_count} missing baseline level(s) locally.")
        (
            new_blind_baselines,
            new_builtin_baselines,
            new_base_prompt_baselines,
            new_per_game_budgets,
        ) = compute_puzzlescript_baselines_for_examples(
            evaluator=evaluator,
            examples=missing_baseline_examples,
            all_game_texts=all_game_texts,
            all_level_env_descs=all_level_env_descs,
            all_env_descs=all_env_descs,
            max_expansions=max_expansions,
            max_gepa_expansions_per_level=max_gepa_expansions_per_level,
            astar_timeout_s=astar_timeout_s,
            lm=lm,
        )
        _merge_nested_map(blind_baselines, new_blind_baselines)
        _merge_nested_map(builtin_baselines, new_builtin_baselines)
        _merge_nested_map(base_prompt_baselines, new_base_prompt_baselines)
        _merge_nested_map(per_game_budgets, new_per_game_budgets)
    else:
        print("  All training baselines were loaded from cache/shards.")

    baseline_cache_path = save_puzzlescript_baseline_cache(
        baseline_state_root,
        signature=baseline_signature,
        blind_baselines=blind_baselines,
        builtin_baselines=builtin_baselines,
        base_prompt_baselines=base_prompt_baselines,
        per_game_budgets=per_game_budgets,
        metadata={
            "n_loaded_files": len(loaded_baseline_paths),
            "n_missing_computed": computed_missing_count,
        },
    )
    print(f"  Merged baseline cache: {baseline_cache_path}")

    state["per_game_budgets"] = per_game_budgets
    state["level_indices_by_game"] = level_indices_by_game

    cost_logger.sync(
        "base_prompt_training_baseline",
        {
            "n_games": len(all_train_names),
            "n_level_examples": sum(len(level_indices_by_game.get(n, [])) for n in all_train_names),
            "n_loaded_baseline_files": len(loaded_baseline_paths),
            "n_missing_baselines_computed": computed_missing_count,
        },
    )

    current_phase = state["current_phase"]
    global_iteration = state["global_iteration"]
    best_prompt_state = state.get("best_prompt_state")
    best_prompt_text = state.get("best_prompt_text") or PUZZLESCRIPT_HEURISTIC_CONTRACT
    best_code = state.get("best_heuristic_code")
    best_codes_by_game = dict(state.get("best_heuristic_codes") or {})
    stop_reason = state["stop_reason"]
    run_counter = 0

    print(f"\n{'='*70}")
    print("GEPA PuzzleScript Heuristic Optimization")
    print(f"  Phases: {[len(p) for p in phase_schedule]} games")
    print(f"  Threshold: {PHASE_SOLVE_RATE_THRESHOLD}, Patience: {PHASE_EARLY_STOP_PATIENCE}")
    print(
        f"  Near-threshold advance: solve_rate >= {PHASE_NEAR_THRESHOLD} "
        f"after {PHASE_NEAR_THRESHOLD_PATIENCE} non-improving iterations"
    )
    print(
        "  GEPA metric-call cap: "
        f"min(iteration * trainset * 3, trainset * {GEPA_MAX_METRIC_CALLS_MULTIPLIER})"
    )
    print(f"  Max expansions (global): {max_expansions}, Max iters/phase: {max_phase_iterations}")
    print(f"  Max GEPA expansions per level: {max_gepa_expansions_per_level}")
    print(f"  A* timeout per level: {astar_timeout_s:.1f}s")
    print(f"  Levels per game: {levels_per_game}")
    print(f"  GEPA threads: {gepa_num_threads}")
    print(f"  Per-game/level GEPA budgets: {per_game_budgets}")
    print(f"{'='*70}")

    while stop_reason is None and current_phase <= total_phases:
        phase_entries = phase_schedule[current_phase - 1]
        phase_key = str(current_phase)
        active_names = [e["name"] for e in phase_entries if e["name"] in all_game_texts]
        n_games = len(active_names)
        active_level_examples = [
            {
                "game": name,
                "level": level_i,
                "budget": per_game_budgets.get(name, {}).get(level_i, max_expansions),
            }
            for name in active_names
            for level_i in level_indices_by_game.get(name, [0])
        ]
        n_examples = len(active_level_examples)
        is_final = current_phase >= total_phases

        records = state.setdefault("phase_records", {})
        if phase_key not in records:
            records[phase_key] = {
                "n_games": n_games, "n_examples": n_examples, "best_solve_rate": None,
                "max_observed_solve_rate": 0.0,
                "best_mean_score": None, "non_improving_streak": 0,
                "iterations": 0, "advanced": False, "completed": False,
                "stop_reason": None, "iteration_results": [],
            }
        rec = records[phase_key]
        phase_iter = rec["iterations"]

        # Build DSPy examples for GEPA
        trainset = []
        for level_example in active_level_examples:
            name = str(level_example["game"])
            level_i = int(level_example["level"])
            desc = all_level_env_descs.get(name, {}).get(level_i, all_env_descs.get(name, name))
            ex = dspy.Example(
                env_description=desc,
                heuristic_contract=PUZZLESCRIPT_HEURISTIC_CONTRACT,
                game_name=name,
                level_i=level_i,
                budget=int(level_example["budget"]),
            ).with_inputs("env_description", "heuristic_contract")
            trainset.append(ex)

        # Caches for GEPA metric
        score_cache: dict[int, float] = {}
        feedback_cache: dict[int, str] = {}
        metric_lock = threading.Lock()

        def metric(
            example: dspy.Example,
            prediction: dspy.Prediction,
            trace: Any = None,
            pred_name: Optional[str] = None,
            pred_trace: Any = None,
        ) -> float | ScoreWithFeedback:
            nonlocal run_counter
            del trace, pred_trace
            prediction_id = id(prediction)

            # Return cached score for reflection calls
            if pred_name is not None:
                with metric_lock:
                    cached_score = score_cache.get(prediction_id, 0.0)
                    cached_feedback = feedback_cache.get(prediction_id, "No feedback.")
                return ScoreWithFeedback(
                    score=cached_score,
                    feedback=cached_feedback,
                )

            game_name = getattr(example, "game_name", "unknown")
            level_i = int(getattr(example, "level_i", 0))
            budget = int(getattr(example, "budget", max_expansions))
            if game_name not in all_game_texts:
                return ScoreWithFeedback(score=0.0, feedback=f"Game {game_name} not found")

            prompt_text = getattr(prediction, "prompt_text", None)
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                prompt_text = PUZZLESCRIPT_HEURISTIC_CONTRACT

            # Synthesize heuristic using the GEPA-optimized prompt
            env_desc = all_level_env_descs.get(game_name, {}).get(
                level_i,
                all_env_descs.get(game_name, game_name),
            )
            heuristic_fn, code, error = synthesize_heuristic_from_prompt(
                prompt_text,
                env_desc,
                lm,
                preflight_evaluator=evaluator,
                preflight_game_text=all_game_texts[game_name],
                preflight_level_i=level_i,
            )

            with metric_lock:
                local_run_counter = run_counter
                run_counter += 1
            run_dir = logs_root / f"candidate-{local_run_counter:04d}-{game_name}-level-{level_i:02d}"

            if error:
                print(f"    [{game_name}] synthesis error: {error[:100]}")
                heuristic_fn = builtin_heuristic
                code = f"# FALLBACK: {error[:200]}"

            # Save candidate
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "heuristic.py").write_text(code)

            try:
                result = evaluate_one_game(
                    evaluator,
                    game_name,
                    all_game_texts[game_name],
                    heuristic_fn,
                    budget,
                    output_dir=run_dir,
                    level_i=level_i,
                    blind_baseline=blind_baselines.get(game_name, {}).get(level_i),
                    builtin_baseline=builtin_baselines.get(game_name, {}).get(level_i),
                    base_prompt_baseline=base_prompt_baselines.get(game_name, {}).get(level_i),
                    env_description=env_desc,
                    heuristic_code=code,
                    reflection_lm=lm,
                    astar_timeout_s=astar_timeout_s,
                )
            except Exception as e:
                result = {"score": 0.0, "feedback": f"Eval error: {e}",
                          "solved": False, "expanded": 0, "solution_length": 0}

            score = float(result["score"])
            feedback = result["feedback"]
            with metric_lock:
                score_cache[prediction_id] = score
                feedback_cache[prediction_id] = feedback

            solved_str = "Y" if result["solved"] else "N"
            print(f"    [{game_name} level={level_i}] score={score:.4f} solved={solved_str} "
                  f"expanded={result['expanded']} budget={budget}")

            return ScoreWithFeedback(score=score, feedback=feedback)

        # Build GEPA program and compiler
        print(f"\n{'='*70}")
        print(f"Phase {current_phase}/{total_phases}: {n_games} games, {n_examples} level examples, "
              f"iteration {phase_iter + 1}/{max_phase_iterations}")
        print(f"{'='*70}")

        program = PuzzleScriptPromptProgram(
            PUZZLESCRIPT_HEURISTIC_CONTRACT, best_prompt_state)

        gepa_log_dir = logs_root / f"phase-{current_phase:02d}-gepa"
        gepa_log_dir.mkdir(parents=True, exist_ok=True)

        # DSPy's GEPA resumes from `log_dir` until the run reaches the supplied
        # metric-call cap. The cap must therefore be cumulative across outer
        # phase iterations; a fixed cap causes resumed iterations to do no work.
        max_metric_calls = _phase_gepa_max_metric_calls(
            phase_iteration=phase_iter + 1,
            trainset_size=len(trainset),
        )

        compiler = dspy.GEPA(
            metric=metric,
            max_metric_calls=max_metric_calls,
            reflection_lm=lm,
            reflection_minibatch_size=1,
            track_stats=True,
            num_threads=gepa_num_threads,
            log_dir=str(gepa_log_dir),
        )

        print(f"  Running GEPA (max_metric_calls={max_metric_calls})...")
        calls_before_compile = int(cost_logger.summary().get("total_calls", 0))
        optimized = compiler.compile(program, trainset=trainset)
        compile_summary = cost_logger.sync(
            "gepa_compile",
            {
                "phase": current_phase,
                "iteration": phase_iter + 1,
                "n_games": n_games,
                "max_metric_calls": max_metric_calls,
            },
        )
        compile_new_calls = int(compile_summary.get("total_calls", 0)) - calls_before_compile

        if compile_new_calls == 0:
            rec["iterations"] += 1
            rec["non_improving_streak"] += 1
            phase_iter = rec["iterations"]
            global_iteration += 1
            state["global_iteration"] = global_iteration
            rec["iteration_results"].append({
                "iteration": phase_iter,
                "mean_score": rec.get("best_mean_score") or 0.0,
                "solve_rate": rec.get("best_solve_rate") or 0.0,
                "n_solved": None,
                "improved": False,
                "skipped_final_eval": True,
                "reason": "gepa_metric_call_cap_reached",
            })

            if not is_final:
                rec["advanced"] = True
                rec["completed"] = True
                rec["stop_reason"] = "advanced_after_gepa_metric_call_cap"
                if current_phase not in state["completed_phases"]:
                    state["completed_phases"].append(current_phase)
                current_phase += 1
                state["current_phase"] = current_phase
                print("  >>> Phase advanced because GEPA produced no new metric calls at the cap.")
            elif phase_iter >= max_phase_iterations:
                rec["completed"] = True
                rec["stop_reason"] = "phase_iteration_cap"
                stop_reason = "phase_iteration_cap"

            state["stop_reason"] = stop_reason
            state["llm_cost_summary"] = cost_logger.summary()
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            continue

        # Extract optimized prompt
        optimized_prompt_state = optimized.prompt_generator.dump_state()
        try:
            optimized_prompt_text = optimized.prompt_generator(
                optimized._build_rewrite_prompt())
        except Exception:
            optimized_prompt_text = best_prompt_text

        # Evaluate the shared optimized prompt by synthesizing one heuristic per level.
        print(f"\n  Evaluating optimized prompt on {n_examples} level examples...")
        final_eval_dir = logs_root / f"phase-{current_phase:02d}-iter-{phase_iter + 1:02d}-final"
        final_results, final_codes_by_level = evaluate_prompt_per_level(
            evaluator=evaluator,
            prompt_text=optimized_prompt_text,
            examples=active_level_examples,
            all_game_texts=all_game_texts,
            all_level_env_descs=all_level_env_descs,
            max_expansions=max_expansions,
            lm=lm,
            blind_baselines=blind_baselines,
            builtin_baselines=builtin_baselines,
            base_prompt_baselines=base_prompt_baselines,
            output_dir=final_eval_dir,
            reflection_lm=lm,
            astar_timeout_s=astar_timeout_s,
        )
        cost_logger.sync(
            "final_per_level_eval",
            {
                "phase": current_phase,
                "iteration": phase_iter + 1,
                "n_level_examples": n_examples,
            },
        )

        scores = []
        n_solved = 0
        for result in final_results:
            name = result["game"]
            level_i = result["level"]
            scores.append(result["score"])
            if result["solved"]:
                n_solved += 1
            solved_str = "Y" if result["solved"] else "N"
            print(f"    {name:<40} level={level_i:<2} score={result['score']:.4f} "
                  f"solved={solved_str} expanded={result['expanded']}")

        mean_score = sum(scores) / len(scores) if scores else 0.0
        solve_rate = n_solved / n_examples if n_examples else 0.0
        print(f"\n  Phase result: score={mean_score:.4f} solve_rate={solve_rate:.3f}")

        # Track improvement
        best = rec["best_mean_score"]
        best_solve = rec.get("best_solve_rate")
        improved, selection_reason = _should_accept_prompt_candidate(
            mean_score=mean_score,
            solve_rate=solve_rate,
            best_mean_score=best,
            best_solve_rate=best_solve,
        )
        rec["max_observed_solve_rate"] = max(
            float(rec.get("max_observed_solve_rate") or 0.0),
            solve_rate,
        )
        if improved:
            improved = True
            rec["best_mean_score"] = mean_score
            rec["best_solve_rate"] = solve_rate
            rec["non_improving_streak"] = 0
            best_codes_by_game = dict(final_codes_by_level)
            best_code = None
            best_prompt_text = optimized_prompt_text
            best_prompt_state = optimized_prompt_state
            state["best_heuristic_code"] = None
            state["best_heuristic_codes"] = best_codes_by_game
            state["best_prompt_text"] = optimized_prompt_text
            state["best_prompt_state"] = optimized_prompt_state
            state["best_prompt_selection"] = {
                "phase": current_phase,
                "iteration": phase_iter + 1,
                "mean_score": mean_score,
                "solve_rate": solve_rate,
                "reason": selection_reason,
            }
        else:
            rec["non_improving_streak"] += 1

        rec["iterations"] += 1
        phase_iter = rec["iterations"]
        global_iteration += 1
        state["global_iteration"] = global_iteration
        rec["iteration_results"].append({
            "iteration": phase_iter, "mean_score": mean_score,
            "solve_rate": solve_rate, "n_solved": n_solved,
            "improved": improved,
            "selection_reason": selection_reason,
        })

        near_threshold_plateau = (
            not is_final
            and solve_rate >= PHASE_NEAR_THRESHOLD
            and rec["non_improving_streak"] >= PHASE_NEAR_THRESHOLD_PATIENCE
        )

        # Phase advancement
        if not is_final and solve_rate >= PHASE_SOLVE_RATE_THRESHOLD:
            rec["advanced"] = True
            rec["completed"] = True
            rec["stop_reason"] = "advanced_to_next_phase"
            if current_phase not in state["completed_phases"]:
                state["completed_phases"].append(current_phase)
            current_phase += 1
            state["current_phase"] = current_phase
            print(f"  >>> Phase advanced! solve_rate={solve_rate:.3f} >= "
                  f"{PHASE_SOLVE_RATE_THRESHOLD}")
        elif near_threshold_plateau:
            rec["advanced"] = True
            rec["completed"] = True
            rec["stop_reason"] = "advanced_after_near_threshold_plateau"
            if current_phase not in state["completed_phases"]:
                state["completed_phases"].append(current_phase)
            current_phase += 1
            state["current_phase"] = current_phase
            print(
                "  >>> Phase advanced after near-threshold plateau: "
                f"solve_rate={solve_rate:.3f} >= {PHASE_NEAR_THRESHOLD} and "
                f"non_improving_streak={rec['non_improving_streak']} >= "
                f"{PHASE_NEAR_THRESHOLD_PATIENCE}"
            )
        elif is_final and phase_iter >= max_phase_iterations:
            rec["completed"] = True
            rec["stop_reason"] = "phase_iteration_cap"
            stop_reason = "phase_iteration_cap"
        elif not is_final:
            if rec["non_improving_streak"] >= PHASE_EARLY_STOP_PATIENCE:
                rec["advanced"] = True
                rec["completed"] = True
                rec["stop_reason"] = "advanced_after_patience_cap"
                if current_phase not in state["completed_phases"]:
                    state["completed_phases"].append(current_phase)
                current_phase += 1
                state["current_phase"] = current_phase
                print("  >>> Phase advanced after reaching non-improvement patience cap "
                      f"without hitting solve-rate threshold ({solve_rate:.3f} < "
                      f"{PHASE_SOLVE_RATE_THRESHOLD})")
            elif phase_iter >= max_phase_iterations:
                rec["advanced"] = True
                rec["completed"] = True
                rec["stop_reason"] = "advanced_after_iteration_cap"
                if current_phase not in state["completed_phases"]:
                    state["completed_phases"].append(current_phase)
                current_phase += 1
                state["current_phase"] = current_phase
                print("  >>> Phase advanced after reaching iteration cap "
                      f"without hitting solve-rate threshold ({solve_rate:.3f} < "
                      f"{PHASE_SOLVE_RATE_THRESHOLD})")

        state["stop_reason"] = stop_reason
        state["llm_cost_summary"] = cost_logger.summary()
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("Curriculum Complete")
    print(f"  Stop reason: {stop_reason or 'all phases completed'}")
    for pk, pr in state.get("phase_records", {}).items():
        print(f"  Phase {pk}: best_score={pr['best_mean_score']}, "
              f"solve_rate={pr['best_solve_rate']}, iters={pr['iterations']}")

    if best_codes_by_game:
        best_heuristics_dir = state_root / "best_heuristics"
        best_heuristics_dir.mkdir(parents=True, exist_ok=True)
        for name, code in sorted(best_codes_by_game.items()):
            safe_name = name.replace("::", "__").replace("/", "_")
            (best_heuristics_dir / f"{safe_name}.py").write_text(code)
        manifest_path = best_heuristics_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact": "per-level heuristics",
                    "shared_prompt": str(state_root / "best_prompt.txt"),
                    "examples": sorted(best_codes_by_game),
                },
                indent=2,
            )
        )
        print(f"  Best per-level heuristics: {best_heuristics_dir}")
    elif best_code:
        (state_root / "best_heuristic.py").write_text(best_code)
        print(f"  Best heuristic: {state_root / 'best_heuristic.py'}")
    if best_prompt_text:
        (state_root / "best_prompt.txt").write_text(best_prompt_text)
        print(f"  Best prompt: {state_root / 'best_prompt.txt'}")

    # Holdout
    if eval_jobs:
        print(f"\n--- Holdout ({len(eval_jobs)} games) ---")
        holdout_names = [entry["name"] for entry in eval_jobs if entry["name"] in all_game_texts]
        holdout_examples = [
            {
                "game": name,
                "level": level_i,
                "budget": max_expansions,
            }
            for name in holdout_names
            for level_i in level_indices_by_game.get(name, [0])
        ]
        print("  Evaluating base prompt on holdout for pairwise comparison...")
        holdout_base_results, _holdout_base_codes = evaluate_prompt_per_level(
            evaluator=evaluator,
            prompt_text=PUZZLESCRIPT_HEURISTIC_CONTRACT,
            examples=holdout_examples,
            all_game_texts=all_game_texts,
            all_level_env_descs=all_level_env_descs,
            max_expansions=max_expansions,
            lm=lm,
            output_dir=state_root / "holdout_base_prompt",
            astar_timeout_s=astar_timeout_s,
        )
        holdout_base_baselines: dict[str, dict[int, dict[str, Any]]] = {}
        for row in holdout_base_results:
            holdout_base_baselines.setdefault(str(row["game"]), {})[int(row["level"])] = row
        cost_logger.sync(
            "holdout_base_prompt_eval",
            {"n_games": len(holdout_names), "n_level_examples": len(holdout_examples)},
        )

        print("  Evaluating best prompt on holdout against base prompt...")
        holdout_results, _holdout_codes = evaluate_prompt_per_level(
            evaluator=evaluator,
            prompt_text=best_prompt_text,
            examples=holdout_examples,
            all_game_texts=all_game_texts,
            all_level_env_descs=all_level_env_descs,
            max_expansions=max_expansions,
            lm=lm,
            base_prompt_baselines=holdout_base_baselines,
            output_dir=state_root / "holdout_heuristics",
            astar_timeout_s=astar_timeout_s,
        )
        cost_logger.sync(
            "holdout_per_level_eval",
            {"n_games": len(holdout_names), "n_level_examples": len(holdout_examples)},
        )
        for r in holdout_results:
            name = r["game"]
            level_i = r["level"]
            solved_str = "Y" if r["solved"] else "N"
            print(
                f"  {name:<40} level={level_i:<2} score={r['score']:.4f} solved={solved_str}"
            )
    cost_logger.sync("run_complete")
    print("=" * 70)


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="GEPA PuzzleScript heuristic optimization")
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help=(
            "Directory containing shared baseline shards/cache. Defaults to "
            "--state-root. Use this for GEPA replica arrays so each replica has "
            "isolated optimizer state but reuses the same baselines."
        ),
    )
    parser.add_argument("--max-phase-iterations", type=int,
                        default=DEFAULT_MAX_PHASE_ITERATIONS)
    parser.add_argument("--max-expansions", type=int,
                        default=DEFAULT_ASTAR_MAX_EXPANSIONS)
    parser.add_argument("--max-gepa-expansions-per-level", type=int,
                        default=DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL)
    parser.add_argument("--astar-timeout-s", type=float,
                        default=DEFAULT_ASTAR_TIMEOUT_S)
    parser.add_argument("--llm", type=str, default=DEFAULT_LLM)
    parser.add_argument("--llm-max-tokens", type=int,
                        default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--script-doctor", type=Path,
                        default=SCRIPT_DOCTOR_PATH)
    parser.add_argument("--levels-per-game", type=int,
                        default=DEFAULT_LEVELS_PER_GAME)
    parser.add_argument("--gepa-num-threads", type=int,
                        default=DEFAULT_GEPA_NUM_THREADS)
    args = parser.parse_args()

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    train_jobs, eval_jobs = load_env_grid(args.env_grid)

    run_curriculum(
        evaluator=evaluator, train_jobs=train_jobs, eval_jobs=eval_jobs,
        sd_path=args.script_doctor, state_root=args.state_root,
        baseline_root=args.baseline_root,
        max_phase_iterations=args.max_phase_iterations,
        max_expansions=args.max_expansions, llm_name=args.llm,
        llm_max_tokens=args.llm_max_tokens,
        levels_per_game=args.levels_per_game,
        gepa_num_threads=max(1, args.gepa_num_threads),
        max_gepa_expansions_per_level=max(1, args.max_gepa_expansions_per_level),
        astar_timeout_s=max(1.0, args.astar_timeout_s),
    )


if __name__ == "__main__":
    main()
