#!/usr/bin/env python3
"""Small experiments for PuzzleScript heuristic generalization.

This script intentionally avoids GEPA/LLM calls. It evaluates saved heuristics
and baselines across multiple levels so we can test runner/evaluator changes
quickly before spending model calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SEARCH_ROOT = ROOT / "llm_desparsifier" / "search"
if str(SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_ROOT))

from puzzle_evaluator import PuzzleScriptEvaluator
from puzzlescript_adapter import build_puzzlescript_ctx
from puzzlescript_astar import blind_heuristic, builtin_heuristic, puzzlescript_astar
from puzzlescript_sanitizer import sanitize_and_compile_puzzlescript_heuristic


def load_game_text(name: str, sd_path: Path) -> str:
    for subdir in ("data/scraped_games", "custom_games"):
        path = sd_path / subdir / f"{name}.txt"
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"Game {name} not found under {sd_path}")


def load_jobs(path: Path, include_eval: bool) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text())
    jobs = list(raw.get("jobs", []))
    if include_eval:
        jobs.extend(raw.get("eval_jobs", []))
    return jobs


def load_heuristic(path: Path) -> Callable[..., float]:
    return sanitize_and_compile_puzzlescript_heuristic(path.read_text())


def adapt_heuristic(fn: Callable[..., float]) -> Callable[[dict[str, Any]], float]:
    def wrapped(ctx: dict[str, Any]) -> float:
        return float(fn(None, None, ctx))

    return wrapped


def strict_adapt_heuristic(fn: Callable[..., float], errors: list[str]) -> Callable[[dict[str, Any]], float]:
    def wrapped(ctx: dict[str, Any]) -> float:
        try:
            return float(fn(None, None, ctx))
        except Exception:
            errors.append(traceback.format_exc(limit=2))
            raise

    return wrapped


def score_result(solved: bool, expanded: int, budget: int) -> float:
    s = expanded if solved else budget + 1
    return ((budget + 1) - s) / (budget + 1)


def evaluate_level(
    evaluator: PuzzleScriptEvaluator,
    json_str: str,
    compiled: dict[str, Any],
    level: int,
    heuristic_fn: Callable[[dict[str, Any]], float],
    budget: int,
    strict: bool,
) -> dict[str, Any]:
    engine = evaluator.load_engine(json_str)
    engine.load_level(level)

    if strict:
        root_ctx = build_puzzlescript_ctx(engine, compiled)
        _ = heuristic_fn(root_ctx)
        engine.load_level(level)

    result = puzzlescript_astar(
        engine=engine,
        compiled_json=compiled,
        heuristic_fn=heuristic_fn,
        max_expansions=budget,
    )
    return {
        "solved": result.solved,
        "expanded": result.expanded_states,
        "generated": result.generated_states,
        "solution_length": result.solution_length,
        "score": score_result(result.solved, result.expanded_states, budget),
        "trace": result.trace_summary,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    solved = sum(1 for row in rows if row["solved"])
    return {
        "n": len(rows),
        "solved": solved,
        "solve_rate": solved / len(rows),
        "mean_score": sum(float(row["score"]) for row in rows) / len(rows),
        "mean_expanded": sum(int(row["expanded"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-grid", type=Path, default=ROOT / "configs/gepa_puzzlescript_envs.yaml")
    parser.add_argument("--script-doctor", type=Path, default=ROOT.parent / "script-doctor")
    parser.add_argument("--heuristic", type=Path, default=ROOT / "artifacts/gepa_puzzlescript_state_llm_feedback_20260415_1239/best_heuristic.py")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/puzzlescript_generalization/results.json")
    parser.add_argument("--max-games", type=int, default=5)
    parser.add_argument("--levels-per-game", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--games",
        default="",
        help="Comma-separated game names to evaluate after loading the env grid.",
    )
    parser.add_argument(
        "--methods",
        default="blind,builtin,heuristic",
        help="Comma-separated methods from: blind,builtin,heuristic.",
    )
    args = parser.parse_args()

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    raw_fn = load_heuristic(args.heuristic)
    errors: list[str] = []
    heuristic = strict_adapt_heuristic(raw_fn, errors) if args.strict else adapt_heuristic(raw_fn)
    jobs = load_jobs(args.env_grid, args.include_eval)
    requested_games = {name.strip() for name in args.games.split(",") if name.strip()}
    if requested_games:
        jobs = [job for job in jobs if job["name"] in requested_games]
    jobs = jobs[: args.max_games]
    requested_methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    method_map = {
        "blind": blind_heuristic,
        "builtin": builtin_heuristic,
        "heuristic": heuristic,
    }
    methods = [(name, method_map[name]) for name in requested_methods]

    all_rows: list[dict[str, Any]] = []
    by_baseline = {name: [] for name, _ in methods}
    for job in jobs:
        game = job["name"]
        game_text = load_game_text(game, args.script_doctor)
        json_str = evaluator.compile_game(game_text)
        compiled = json.loads(json_str)
        info = evaluator.get_game_info(json_str)
        n_levels = int(info["n_levels"])
        levels = list(range(min(n_levels, max(1, args.levels_per_game))))
        for level in levels:
            for name, fn in methods:
                try:
                    row = evaluate_level(
                        evaluator=evaluator,
                        json_str=json_str,
                        compiled=compiled,
                        level=level,
                        heuristic_fn=fn,
                        budget=args.budget,
                        strict=args.strict and name == "heuristic",
                    )
                except Exception as exc:
                    row = {
                        "solved": False,
                        "expanded": 0,
                        "generated": 0,
                        "solution_length": 0,
                        "score": 0.0,
                        "error": repr(exc),
                    }
                row.update({"game": game, "level": level, "method": name})
                by_baseline[name].append(row)
                all_rows.append(row)
                print(
                    f"{game} level={level} {name}: "
                    f"solved={row['solved']} expanded={row['expanded']} score={row['score']:.4f}"
                )

    payload = {
        "heuristic": str(args.heuristic),
        "budget": args.budget,
        "levels_per_game": args.levels_per_game,
        "strict": args.strict,
        "summaries": {name: summarize(rows) for name, rows in by_baseline.items()},
        "rows": all_rows,
        "strict_errors_sample": errors[:5],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summaries"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
