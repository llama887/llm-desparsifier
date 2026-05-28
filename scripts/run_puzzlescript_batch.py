#!/usr/bin/env python3
"""Run GEPA heuristic optimization on PuzzleScript Sokoban environments.

Uses DSPy GEPA to optimize a prompt that causes an LLM to emit a Python
heuristic function. The heuristic guides A* search on PuzzleScript games
using the C++ engine for fast state transitions.

Curriculum: 5 games -> 10 games, mirroring the XLand pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Direct imports to avoid heavy __init__.py chains
sys.path.insert(0, str(_PROJECT_ROOT / "llm_desparsifier" / "search"))
from puzzle_evaluator import PuzzleScriptEvaluator
from puzzlescript_adapter import build_env_description, build_puzzlescript_ctx
from puzzlescript_astar import (
    PuzzleScriptSearchResult,
    blind_heuristic,
    builtin_heuristic,
    puzzlescript_astar,
)
from puzzlescript_sanitizer import sanitize_and_compile_puzzlescript_heuristic

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
import yaml

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

CURRICULUM_PHASE_GAME_COUNTS = (5, 10)
PHASE_SOLVE_RATE_THRESHOLD = 0.80
PHASE_EARLY_STOP_PATIENCE = 3
DEFAULT_MAX_PHASE_ITERATIONS = 10
DEFAULT_ASTAR_MAX_EXPANSIONS = 50_000
DEFAULT_LLM = "deepseek/deepseek-v4-pro"
DEFAULT_LLM_MAX_TOKENS = 32_000
DEFAULT_LEVELS_PER_GAME = 3
DEFAULT_GEPA_NUM_THREADS = 1

PUZZLESCRIPT_HEURISTIC_CONTRACT = """You are writing a heuristic function for A* search on a PuzzleScript grid puzzle.

Function signature: def heuristic_cost_to_go(ts, env_params, ctx) -> float

For PuzzleScript games, ts and env_params are None. Use only ctx.
ctx is a dict with these keys:
  ctx.get('object_positions'): dict mapping object name -> list of (x,y) tuples
  ctx.get('grid_width'): int, grid width
  ctx.get('grid_height'): int, grid height
  ctx.get('win_conditions_text'): str describing what must happen to win
  ctx.get('ascii_state'): str, text grid of current state
  ctx.get('is_winning'): bool, True if state is already won
  ctx.get('object_names'): list of all object type names

IMPORTANT - read the game's PuzzleScript rules carefully. Each game has unique
mechanics (gravity, swapping, collapsing floors, teleportation, pulling, etc.)
that make naive distance heuristics INEFFECTIVE. Your heuristic must reason
about the SPECIFIC mechanics of THIS game, not just measure distances.

Think about:
- What do the rules actually DO? (e.g. swap, collapse, gravity)
- What structural patterns make a state good or bad? (deadlocks, blocking)
- What game-specific features matter? (remaining safe tiles, vertical position)
- Are there configurations that are provably unsolvable?

DO NOT just compute Manhattan distances. That is the baseline and it performs
poorly on games with interesting mechanics. Encode game-specific insight.

Rules:
- Return a non-negative float (lower = closer to goal, 0.0 = already won)
- Return 0.0 if ctx.get('is_winning') is True
- No imports, no print, no exec, no open
- Only math builtins available (abs, min, max, sum, len, etc.)

=== WHAT NOT TO DO (negative example) ===

The following "blind heuristic" returns 0 for every state:

    def heuristic_cost_to_go(ts, env_params, ctx):
        return 0.0

This is equivalent to BFS (breadth-first search) -- it provides NO guidance to
A*. With this heuristic, A* degrades to uninformed search and wastes its
expansion budget exploring states blindly. Your heuristic is evaluated with a
budget TIGHTER than what blind search needs, so returning 0 (or any constant)
will FAIL. You MUST return values that meaningfully distinguish states closer
to the goal from states farther away.

Similarly, a heuristic that returns a constant or near-constant value for all
states (e.g., always returning grid_width + grid_height) is equally useless.
The heuristic must vary based on the actual game state.
"""


def load_local_env() -> None:
    """Load repo-local .env credentials when available."""

    if load_dotenv is not None:
        load_dotenv(_PROJECT_ROOT / ".env")


class LMCostLogger:
    """Append DSPy LM history costs and write cumulative GEPA cost summaries."""

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
            event = {
                "label": label,
                "timestamp": entry.get("timestamp"),
                "uuid": entry.get("uuid"),
                "model": entry.get("model"),
                "response_model": entry.get("response_model"),
                "model_type": entry.get("model_type"),
                "cost_usd": self._as_float(entry.get("cost")),
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
        for event in self._events:
            model = str(event.get("model") or event.get("response_model") or "unknown")
            row = by_model.setdefault(model, {"calls": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["cost_usd"] += self._as_float(event.get("cost_usd"))
        return {
            "total_calls": len(self._events),
            "total_cost_usd": sum(self._as_float(e.get("cost_usd")) for e in self._events),
            "by_model": by_model,
            "events_path": str(self.events_path),
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
    return (
        base_env_description
        + "\n\n"
        + f"Level of interest: {level_i}\n"
        + "Initial state for this level:\n"
        + str(ctx.get("ascii_state", ""))
        + "\n\n"
        + "Write the heuristic for this specific level while still using the rules above."
    )


def gepa_score(solved: bool, expanded: int, max_expansions: int) -> float:
    n = max_expansions
    s = expanded if solved else n + 1
    return ((n + 1) - s) / (n + 1)


def _extract_mechanics_hints(game_text: str) -> list[str]:
    """Return lightweight mechanic hints inferred from the raw PuzzleScript.

    Feedback should point the optimizer toward specific rule families when the
    heuristic behaves poorly. We keep the detection intentionally cheap and
    keyword-based because it runs inside GEPA evaluation.
    """

    lowered = game_text.lower()
    hints: list[str] = []
    keyword_hints = [
        ("pull", "pull interactions"),
        ("swap", "swap or permutation mechanics"),
        ("teleport", "teleport or relocation rules"),
        ("beam", "beam-style movement or transport rules"),
        ("mud", "terrain that changes movement cost or safety"),
        ("collapse", "collapsing or disappearing terrain"),
        ("fall", "gravity or falling behavior"),
        ("slide", "sliding or inertial movement"),
        ("ice", "slippery movement constraints"),
        ("again", "forced multi-step transitions"),
        ("no ", "negative rule clauses or exclusion constraints"),
    ]
    for keyword, hint in keyword_hints:
        if keyword in lowered and hint not in hints:
            hints.append(hint)
    return hints


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
    try:
        root_h = float(heuristic_fn(root_ctx))
    except Exception:
        root_h = 0.0
    root_h = max(0.0, root_h)

    has_action = not engine.has_metadata("noaction") if hasattr(engine, "has_metadata") else True
    n_actions = 5 if has_action else 4

    successors: list[dict[str, Any]] = []
    for action in range(n_actions):
        engine.restore_level(initial_backup)
        if not _process_action_with_again(engine, action):
            continue
        ctx = build_puzzlescript_ctx(engine, compiled_json)
        try:
            h_val = float(heuristic_fn(ctx))
        except Exception:
            h_val = 0.0
        h_val = max(0.0, h_val)
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

    heuristic_values = [entry["heuristic"] for entry in successors]
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
    blind_baseline: Optional[dict[str, Any]] = None,
    builtin_baseline: Optional[dict[str, Any]] = None,
) -> str:
    """Build structured feedback for GEPA reflection.

    The previous feedback was a short scalar summary. This version adds
    actionable observations about local heuristic shape, search efficiency, and
    likely mechanic families that deserve attention.
    """

    outcome_lines = [
        f"Game: {game_name}",
        f"Outcome: solved={result.solved} score={result.score:.4f}",
        (
            "Search stats: "
            f"expanded={result.expanded_states}/{max_expansions} "
            f"generated={result.generated_states} "
            f"solution_length={result.solution_length}"
        ),
    ]

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
    mechanics_hints = _extract_mechanics_hints(game_text)

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

    if not observed_issues:
        observed_issues.append("The run solved the game cleanly; focus on sharper mechanic-specific ranking to cut expansions further.")

    mechanics_lines: list[str] = []
    if mechanics_hints:
        mechanics_lines.append(
            "Rules appear to include "
            + ", ".join(mechanics_hints)
            + "; make sure the heuristic reasons about these mechanics explicitly."
        )
    else:
        mechanics_lines.append(
            "Focus on the game's win-condition objects, irreversible deadlocks, and movement constraints instead of generic distance only."
        )

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
        "Mechanics hypothesis:\n- " + "\n- ".join(mechanics_lines),
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
    env_description: Optional[str] = None,
    heuristic_code: Optional[str] = None,
    reflection_lm: Any = None,
) -> dict[str, Any]:
    """Compile game, run A* on one level, return result dict with feedback."""
    json_str = evaluator.compile_game(game_text)
    compiled = json.loads(json_str)
    engine = evaluator.load_engine(json_str)
    engine.load_level(level_i)
    diagnostics = _sample_local_heuristic_diagnostics(engine, compiled, heuristic_fn)
    engine.load_level(level_i)

    result = puzzlescript_astar(
        engine=engine, compiled_json=compiled,
        heuristic_fn=heuristic_fn, max_expansions=max_expansions,
    )

    deterministic_feedback = _build_feedback_report(
        game_name=game_name,
        game_text=game_text,
        result=result,
        max_expansions=max_expansions,
        diagnostics=diagnostics,
        blind_baseline=blind_baseline,
        builtin_baseline=builtin_baseline,
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
            "score": result.score, "time_s": result.time_s,
            "feedback_diagnostics": diagnostics,
            "trace_summary": result.trace_summary,
            "deterministic_feedback": deterministic_feedback,
            "feedback": feedback,
        }, indent=2))

    return {
        "score": result.score,
        "level": level_i,
        "solved": result.solved,
        "expanded": result.expanded_states,
        "generated": result.generated_states,
        "solution_length": result.solution_length,
        "feedback": feedback,
        "deterministic_feedback": deterministic_feedback,
        "feedback_diagnostics": diagnostics,
        "trace_summary": result.trace_summary,
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


def evaluate_game_levels(
    evaluator: PuzzleScriptEvaluator,
    game_name: str,
    game_text: str,
    heuristic_fn: Callable,
    level_budgets: Mapping[int, int],
    output_dir: Optional[Path] = None,
    blind_baselines: Optional[Mapping[int, dict[str, Any]]] = None,
    builtin_baselines: Optional[Mapping[int, dict[str, Any]]] = None,
    env_description: Optional[str] = None,
    heuristic_code: Optional[str] = None,
    reflection_lm: Any = None,
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
                env_description=env_description,
                heuristic_code=heuristic_code,
                reflection_lm=reflection_lm,
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
def synthesize_heuristic_from_prompt(
    prompt_text: str,
    env_description: str,
    lm: Any,
) -> tuple[Optional[Callable], str, Optional[str]]:
    """Use the (GEPA-optimized) prompt to synthesize a heuristic."""
    try:
        with dspy.context(lm=lm):
            pred = _heuristic_predictor(
                synthesis_prompt=prompt_text,
                env_description=env_description)
        code = pred.heuristic_code
    except Exception as e:
        return None, "", f"LLM call failed: {e}"

    try:
        raw_fn = sanitize_and_compile_puzzlescript_heuristic(code)

        def heuristic_from_ctx(ctx: dict[str, Any]) -> float:
            return float(raw_fn(None, None, ctx))

        return heuristic_from_ctx, code, None
    except Exception as e:
        return None, code, f"Sanitization failed: {e}"


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
    output_dir: Optional[Path] = None,
    reflection_lm: Any = None,
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
            env_description=env_desc,
            heuristic_code=code,
            reflection_lm=reflection_lm,
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
    output_dir: Optional[Path] = None,
    reflection_lm: Any = None,
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
            env_description=env_desc,
            heuristic_code=code,
            reflection_lm=reflection_lm,
        )
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
    max_phase_iterations: int,
    max_expansions: int,
    llm_name: str,
    llm_max_tokens: int,
    levels_per_game: int,
    gepa_num_threads: int,
) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    logs_root = state_root / "runs"
    logs_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "curriculum_state.json"

    # Configure LLM
    lm = dspy.LM(llm_name, max_tokens=llm_max_tokens)
    dspy.configure(lm=lm)
    cost_logger = LMCostLogger(lm, state_root)
    print(f"LLM: {llm_name} max_tokens={llm_max_tokens}")

    # Pre-compile all game texts and env descriptions
    all_game_texts: dict[str, str] = {}
    all_env_descs: dict[str, str] = {}
    all_level_env_descs: dict[str, dict[int, str]] = {}
    level_indices_by_game: dict[str, list[int]] = {}
    for entry in train_jobs + eval_jobs:
        name = entry["name"]
        text = load_game_text(name, sd_path)
        if text:
            all_game_texts[name] = text
            try:
                json_str = evaluator.compile_game(text)
                compiled = json.loads(json_str)
                engine = evaluator.load_engine(json_str)
                engine.load_level(0)
                n_levels = engine.get_num_levels()
                requested_levels = max(1, int(entry.get("levels", n_levels) or n_levels))
                level_count = min(n_levels, requested_levels, max(1, levels_per_game))
                level_indices_by_game[name] = list(range(level_count))
                base_desc = build_env_description(compiled, engine.get_id_dict(), text)
                all_env_descs[name] = base_desc
                all_level_env_descs[name] = {
                    level_i: build_level_env_description(base_desc, engine, compiled, level_i)
                    for level_i in level_indices_by_game[name]
                }
            except Exception as e:
                print(f"  [WARN] Could not compile {name}: {e}")

    # Phase schedule
    phase_schedule: list[list[dict]] = []
    for count in CURRICULUM_PHASE_GAME_COUNTS:
        phase_schedule.append(train_jobs[:count])
    if len(train_jobs) > CURRICULUM_PHASE_GAME_COUNTS[-1]:
        phase_schedule.append(train_jobs)
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
        }

    # Run blind baseline on ALL training levels and compute per-level GEPA budgets.
    # Budget = floor(0.95 * blind_expanded) so that a heuristic matching blind
    # search will exceed its budget and score 0, forcing GEPA to improve.
    print("\n--- Blind A* baseline (h=0) on ALL training levels ---")
    all_train_names = [e["name"] for e in train_jobs if e["name"] in all_game_texts]
    blind_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    builtin_baselines: dict[str, dict[int, dict[str, Any]]] = {}
    per_game_budgets: dict[str, dict[int, int]] = {}

    for name in all_train_names:
        blind_baselines[name] = {}
        per_game_budgets[name] = {}
        for level_i in level_indices_by_game.get(name, [0]):
            r = evaluate_one_game(
                evaluator, name, all_game_texts[name],
                blind_heuristic, max_expansions, level_i=level_i,
            )
            blind_baselines[name][level_i] = r
            if r["solved"] and r["expanded"] > 0:
                per_game_budgets[name][level_i] = max(math.floor(0.95 * r["expanded"]), 1)
            else:
                # Blind didn't solve -- keep global max as fallback
                per_game_budgets[name][level_i] = max_expansions
            print(
                f"  {name} level={level_i}: solved={r['solved']} expanded={r['expanded']} "
                f"score={r['score']:.4f} -> gepa_budget={per_game_budgets[name][level_i]}"
            )

    state["per_game_budgets"] = per_game_budgets
    state["level_indices_by_game"] = level_indices_by_game

    print("\n--- Built-in heuristic baseline on training levels ---")
    for name in all_train_names:
        builtin_baselines[name] = {}
        for level_i in level_indices_by_game.get(name, [0]):
            budget = per_game_budgets.get(name, {}).get(level_i, max_expansions)
            r = evaluate_one_game(
                evaluator, name, all_game_texts[name],
                builtin_heuristic, budget, level_i=level_i,
            )
            builtin_baselines[name][level_i] = r
            print(
                f"  {name} level={level_i}: solved={r['solved']} "
                f"expanded={r['expanded']} score={r['score']:.4f}"
            )

    current_phase = state["current_phase"]
    global_iteration = state["global_iteration"]
    best_prompt_state = None
    best_prompt_text = state.get("best_prompt_text") or PUZZLESCRIPT_HEURISTIC_CONTRACT
    best_code = state.get("best_heuristic_code")
    best_codes_by_game = dict(state.get("best_heuristic_codes") or {})
    stop_reason = state["stop_reason"]
    run_counter = 0

    print(f"\n{'='*70}")
    print("GEPA PuzzleScript Heuristic Optimization")
    print(f"  Phases: {[len(p) for p in phase_schedule]} games")
    print(f"  Threshold: {PHASE_SOLVE_RATE_THRESHOLD}, Patience: {PHASE_EARLY_STOP_PATIENCE}")
    print(f"  Max expansions (global): {max_expansions}, Max iters/phase: {max_phase_iterations}")
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
                prompt_text, env_desc, lm)

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
                    env_description=env_desc,
                    heuristic_code=code,
                    reflection_lm=lm,
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

        # GEPA gets max_metric_calls = trainset_size * 3 per iteration
        # (enough for 1 baseline + 2 candidates per job)
        max_metric_calls = len(trainset) * 3

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
        optimized = compiler.compile(program, trainset=trainset)
        cost_logger.sync(
            "gepa_compile",
            {
                "phase": current_phase,
                "iteration": phase_iter + 1,
                "n_games": n_games,
                "max_metric_calls": max_metric_calls,
            },
        )

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
            output_dir=final_eval_dir,
            reflection_lm=lm,
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
        improved = False
        best = rec["best_mean_score"]
        if best is None or mean_score > best:
            improved = True
            rec["best_mean_score"] = mean_score
            rec["non_improving_streak"] = 0
            best_codes_by_game = dict(final_codes_by_level)
            best_code = None
            best_prompt_text = optimized_prompt_text
            best_prompt_state = optimized_prompt_state
            state["best_heuristic_code"] = None
            state["best_heuristic_codes"] = best_codes_by_game
            state["best_prompt_text"] = optimized_prompt_text
        else:
            rec["non_improving_streak"] += 1

        if rec["best_solve_rate"] is None or solve_rate > rec["best_solve_rate"]:
            rec["best_solve_rate"] = solve_rate

        rec["iterations"] += 1
        phase_iter = rec["iterations"]
        global_iteration += 1
        state["global_iteration"] = global_iteration
        rec["iteration_results"].append({
            "iteration": phase_iter, "mean_score": mean_score,
            "solve_rate": solve_rate, "n_solved": n_solved,
            "improved": improved,
        })

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
        elif is_final and phase_iter >= max_phase_iterations:
            rec["completed"] = True
            rec["stop_reason"] = "phase_iteration_cap"
            stop_reason = "phase_iteration_cap"
        elif not is_final:
            if rec["non_improving_streak"] >= PHASE_EARLY_STOP_PATIENCE:
                rec["completed"] = True
                rec["stop_reason"] = "threshold_failure_early_stop"
                stop_reason = "threshold_failure_early_stop"
            elif phase_iter >= max_phase_iterations:
                rec["completed"] = True
                rec["stop_reason"] = "phase_iteration_cap"
                stop_reason = "phase_iteration_cap"

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
        holdout_results, _holdout_codes = evaluate_prompt_per_level(
            evaluator=evaluator,
            prompt_text=best_prompt_text,
            examples=holdout_examples,
            all_game_texts=all_game_texts,
            all_level_env_descs=all_level_env_descs,
            max_expansions=max_expansions,
            lm=lm,
            output_dir=state_root / "holdout_heuristics",
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
    parser.add_argument("--max-phase-iterations", type=int,
                        default=DEFAULT_MAX_PHASE_ITERATIONS)
    parser.add_argument("--max-expansions", type=int,
                        default=DEFAULT_ASTAR_MAX_EXPANSIONS)
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
        max_phase_iterations=args.max_phase_iterations,
        max_expansions=args.max_expansions, llm_name=args.llm,
        llm_max_tokens=args.llm_max_tokens,
        levels_per_game=args.levels_per_game,
        gepa_num_threads=max(1, args.gepa_num_threads),
    )


if __name__ == "__main__":
    main()
