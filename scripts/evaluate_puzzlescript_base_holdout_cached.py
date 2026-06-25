#!/usr/bin/env python3
"""Fill missing base-prompt holdout results and plot against cached best results."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from dspy_cache_control import configure_dspy_cache, prepare_dspy_import

prepare_dspy_import("evaluate_puzzlescript_base_holdout_cached")
import dspy
configure_dspy_cache(dspy, "evaluate_puzzlescript_base_holdout_cached")

from evaluate_puzzlescript_best_prompt import (
    _aggregate,
    _plot_aggregate,
    _plot_by_game_score,
    _plot_by_game_score_loss_log,
    _prepare_examples,
)
from run_puzzlescript_batch import (
    DEFAULT_LLM_MAX_TOKENS,
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    evaluate_one_game,
    load_env_grid,
    load_game_text,
    load_local_env,
    synthesize_heuristic_from_prompt,
)

sys.path.insert(0, str(_PROJECT_ROOT / "llm_desparsifier" / "search"))
from puzzle_evaluator import PuzzleScriptEvaluator
from puzzlescript_adapter import build_env_description
from puzzlescript_astar import blind_heuristic, builtin_heuristic


def _row_from_stats(policy: str, stats_path: Path, max_expansions: int | None = None) -> dict[str, Any]:
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    game = str(data.get("game") or stats_path.parents[1].name)
    level = int(data.get("level", stats_path.parent.name.split("-")[-1]))
    solved = bool(data.get("solved"))
    expanded = int(data.get("expanded") or 0)
    cap = int(max_expansions or data.get("max_expansions") or 50000)
    gepa_score = max(0.0, ((cap + 1) - expanded) / (cap + 1)) if solved else 0.0
    return {
        "policy": policy,
        "game": game,
        "level": level,
        "example": f"{game}::level-{level:02d}",
        "solved": solved,
        "expanded": expanded,
        "generated": int(data.get("generated") or 0),
        "solution_length": int(data.get("solution_length") or 0),
        "result_score": float(data.get("score") or 0.0),
        "gepa_score": gepa_score,
    }


def _task_key(example: dict[str, Any]) -> str:
    return f"{example['game']}::level-{int(example['level']):02d}"


def _base_stats_path(output_dir: Path, game: str, level: int) -> Path:
    return output_dir / "base_prompt" / game / f"level-{level:02d}" / "search_stats.json"


def _evaluate_base_task(task: dict[str, Any]) -> dict[str, Any]:
    load_local_env()
    state_root = Path(task["state_root"])
    output_dir = Path(task["output_dir"])
    script_doctor = Path(task["script_doctor"])
    game = str(task["game"])
    level = int(task["level"])
    max_expansions = int(task["max_expansions"])
    llm_name = str(task["llm"])
    llm_max_tokens = int(task["llm_max_tokens"])

    stats_path = _base_stats_path(output_dir, game, level)
    if stats_path.exists():
        return _row_from_stats("base_prompt", stats_path, max_expansions)

    level_dir = stats_path.parent
    level_dir.mkdir(parents=True, exist_ok=True)
    try:
        evaluator = PuzzleScriptEvaluator(script_doctor)
        game_text = load_game_text(game, script_doctor)
        if not game_text:
            raise RuntimeError(f"missing game text for {game}")

        json_str = evaluator.compile_game(game_text)
        compiled = json.loads(json_str)
        engine = evaluator.load_engine(json_str)
        base_desc = build_env_description(compiled, engine.get_id_dict(), game_text)
        from run_puzzlescript_batch import build_level_env_description

        level_desc = build_level_env_description(base_desc, engine, compiled, level)
        lm = dspy.LM(llm_name, max_tokens=llm_max_tokens)
        dspy.configure(lm=lm)
        heuristic_fn, code, error = synthesize_heuristic_from_prompt(
            PUZZLESCRIPT_HEURISTIC_CONTRACT,
            level_desc,
            lm,
        )
        if heuristic_fn is None:
            heuristic_fn = builtin_heuristic

        if code:
            (level_dir / "heuristic.py").write_text(code, encoding="utf-8")
        if error:
            (level_dir / "synthesis_error.txt").write_text(error, encoding="utf-8")

        evaluate_one_game(
            evaluator,
            game,
            game_text,
            heuristic_fn,
            max_expansions,
            output_dir=level_dir,
            level_i=level,
            env_description=level_desc,
            heuristic_code=code,
        )
    except Exception as exc:
        (level_dir / "evaluation_error.txt").write_text(str(exc), encoding="utf-8")
        stats_path.write_text(
            json.dumps(
                {
                    "game": game,
                    "level": level,
                    "solved": False,
                    "expanded": max_expansions,
                    "generated": 0,
                    "solution_length": 0,
                    "score": 0.0,
                    "max_expansions": max_expansions,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return _row_from_stats("base_prompt", stats_path, max_expansions)


def _evaluate_policy_no_llm(
    *,
    label: str,
    examples: list[dict[str, Any]],
    output_dir: Path,
    script_doctor: Path,
    max_expansions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluator = PuzzleScriptEvaluator(script_doctor)
    cache: dict[str, tuple[str, str, str]] = {}
    heuristic = builtin_heuristic if label == "builtin" else blind_heuristic
    for example in examples:
        game = str(example["game"])
        level = int(example["level"])
        stats_path = output_dir / label / game / f"level-{level:02d}" / "search_stats.json"
        if stats_path.exists():
            rows.append(_row_from_stats(label, stats_path, max_expansions))
            continue
        if game not in cache:
            text = load_game_text(game, script_doctor)
            json_str = evaluator.compile_game(text)
            compiled = json.loads(json_str)
            engine = evaluator.load_engine(json_str)
            base_desc = build_env_description(compiled, engine.get_id_dict(), text)
            cache[game] = (text, json_str, base_desc)
        text, json_str, base_desc = cache[game]
        engine = evaluator.load_engine(json_str)
        compiled = json.loads(json_str)
        from run_puzzlescript_batch import build_level_env_description

        level_desc = build_level_env_description(base_desc, engine, compiled, level)
        level_dir = stats_path.parent
        level_dir.mkdir(parents=True, exist_ok=True)
        evaluate_one_game(
            evaluator,
            game,
            text,
            heuristic,
            max_expansions,
            output_dir=level_dir,
            level_i=level,
            env_description=level_desc,
            heuristic_code="",
        )
        rows.append(_row_from_stats(label, stats_path, max_expansions))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--env-grid", type=Path, default=Path("configs/gepa_puzzlescript_envs.yaml"))
    parser.add_argument("--script-doctor", type=Path, default=Path("/scratch/fyy2003/repos/script-doctor"))
    parser.add_argument("--levels-per-game", type=int, default=3)
    parser.add_argument("--max-expansions", type=int, default=10000)
    parser.add_argument("--llm", type=str, default="deepseek/deepseek-v4-pro")
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    load_local_env()
    output_dir = args.output_dir or (args.state_root / "best_vs_base_prompt_eval_holdout")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    _train_jobs, eval_jobs = load_env_grid(args.env_grid)
    examples, _game_texts, _level_descs = _prepare_examples(
        evaluator=evaluator,
        jobs=eval_jobs,
        sd_path=args.script_doctor,
        levels_per_game=args.levels_per_game,
    )
    print(f"Prepared {len(examples)} eval level examples.", flush=True)

    best_rows = [
        _row_from_stats("best_prompt", path, None)
        for path in sorted((args.state_root / "holdout_heuristics").glob("*/level-*/search_stats.json"))
    ]
    print(f"Loaded {len(best_rows)} cached best-prompt rows.", flush=True)

    tasks = []
    for example in examples:
        game = str(example["game"])
        level = int(example["level"])
        if _base_stats_path(output_dir, game, level).exists():
            continue
        tasks.append(
            {
                "state_root": str(args.state_root),
                "output_dir": str(output_dir),
                "script_doctor": str(args.script_doctor),
                "game": game,
                "level": level,
                "max_expansions": args.max_expansions,
                "llm": args.llm,
                "llm_max_tokens": args.llm_max_tokens,
            }
        )
    print(f"Missing base-prompt rows: {len(tasks)}", flush=True)

    if tasks:
        workers = max(1, min(args.workers, len(tasks)))
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_evaluate_base_task, tasks), start=1):
                solved = "Y" if row["solved"] else "N"
                print(
                    f"  [base_prompt {i}/{len(tasks)}] {row['game']:<45} "
                    f"level={row['level']:<2} solved={solved} expanded={row['expanded']} "
                    f"score={row['gepa_score']:.4f}",
                    flush=True,
                )

    base_rows = [
        _row_from_stats("base_prompt", path, args.max_expansions)
        for path in sorted((output_dir / "base_prompt").glob("*/level-*/search_stats.json"))
    ]
    builtin_rows = _evaluate_policy_no_llm(
        label="builtin",
        examples=examples,
        output_dir=output_dir,
        script_doctor=args.script_doctor,
        max_expansions=args.max_expansions,
    )
    blind_rows = _evaluate_policy_no_llm(
        label="blind",
        examples=examples,
        output_dir=output_dir,
        script_doctor=args.script_doctor,
        max_expansions=args.max_expansions,
    )

    rows = best_rows + base_rows + builtin_rows + blind_rows
    summary = _aggregate(rows)
    stats = {
        "source": "cached_best_plus_base_prompt",
        "state_root": str(args.state_root),
        "max_expansions": args.max_expansions,
        "levels_per_game": args.levels_per_game,
        "rows": rows,
        "summary": summary,
    }
    (output_dir / "best_prompt_eval_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot_aggregate(summary, output_dir / "holdout_comparison_aggregate.png")
    _plot_by_game_score(summary, output_dir / "holdout_comparison_by_env.png")
    _plot_by_game_score_loss_log(summary, output_dir / "holdout_comparison_by_env_score_loss_log.png")

    print("\nSummary:", flush=True)
    for policy, bucket in summary["by_policy"].items():
        print(
            f"  {policy:<12} solved={bucket['solved']}/{bucket['n']} "
            f"solve_rate={bucket['solve_rate']:.3f} mean_score={bucket['mean_score']:.4f} "
            f"mean_expanded={bucket['mean_expanded']:.1f}",
            flush=True,
        )
    print(f"Wrote stats and plots to {output_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
