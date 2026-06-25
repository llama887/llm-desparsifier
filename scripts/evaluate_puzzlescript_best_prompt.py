#!/usr/bin/env python3
"""Evaluate a saved PuzzleScript GEPA best prompt and plot comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from dspy_cache_control import configure_dspy_cache, prepare_dspy_import

prepare_dspy_import("evaluate_puzzlescript_best_prompt")
import dspy
configure_dspy_cache(dspy, "evaluate_puzzlescript_best_prompt")
import matplotlib.pyplot as plt

from run_puzzlescript_batch import (
    DEFAULT_ASTAR_MAX_EXPANSIONS,
    DEFAULT_ENV_GRID,
    DEFAULT_LEVELS_PER_GAME,
    DEFAULT_LLM,
    DEFAULT_LLM_MAX_TOKENS,
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    SCRIPT_DOCTOR_PATH,
    LMCostLogger,
    build_level_env_description,
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


def _load_best_prompt(state_root: Path) -> str:
    prompt_path = state_root / "best_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    state_path = state_root / "curriculum_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        prompt = state.get("best_prompt_text")
        if isinstance(prompt, str) and prompt.strip():
            prompt_path.write_text(prompt, encoding="utf-8")
            return prompt

    return PUZZLESCRIPT_HEURISTIC_CONTRACT


def _prepare_examples(
    *,
    evaluator: PuzzleScriptEvaluator,
    jobs: list[dict[str, Any]],
    sd_path: Path,
    levels_per_game: int,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[int, str]]]:
    examples: list[dict[str, Any]] = []
    game_texts: dict[str, str] = {}
    level_env_descs: dict[str, dict[int, str]] = {}

    for job in jobs:
        name = str(job["name"])
        text = load_game_text(name, sd_path)
        if not text:
            print(f"  [skip] {name}: game text not found")
            continue

        try:
            json_str = evaluator.compile_game(text)
            compiled = json.loads(json_str)
            engine = evaluator.load_engine(json_str)
            n_levels = int(engine.get_num_levels())
        except Exception as exc:
            print(f"  [skip] {name}: compile failed: {exc}")
            continue

        requested = int(job.get("levels", n_levels) or n_levels)
        level_count = min(n_levels, requested, max(1, levels_per_game))
        if level_count <= 0:
            print(f"  [skip] {name}: no levels")
            continue

        base_desc = build_env_description(compiled, engine.get_id_dict(), text)
        valid_level_descs: dict[int, str] = {}
        for level_i in range(level_count):
            try:
                valid_level_descs[level_i] = build_level_env_description(
                    base_desc, engine, compiled, level_i
                )
            except RuntimeError as exc:
                if "Level index out of range" in str(exc):
                    print(f"  [skip] {name} level={level_i}: {exc}")
                    continue
                raise

        if not valid_level_descs:
            print(f"  [skip] {name}: no loadable levels")
            continue

        game_texts[name] = text
        level_env_descs[name] = valid_level_descs
        for level_i in sorted(valid_level_descs):
            examples.append({"game": name, "level": level_i})

    return examples, game_texts, level_env_descs


def _score_result(result: dict[str, Any], max_expansions: int) -> float:
    if not result.get("solved"):
        return 0.0
    expanded = int(result.get("expanded", max_expansions))
    return max(0.0, ((max_expansions + 1) - expanded) / (max_expansions + 1))


def _evaluate_policy(
    *,
    label: str,
    evaluator: PuzzleScriptEvaluator,
    examples: list[dict[str, Any]],
    game_texts: dict[str, str],
    level_env_descs: dict[str, dict[int, str]],
    max_expansions: int,
    output_dir: Path,
    heuristic_factory: Callable[[str, int], tuple[Callable, str, str | None]],
    baseline_by_example: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        name = str(example["game"])
        level_i = int(example["level"])
        key = f"{name}::level-{level_i:02d}"
        level_dir = output_dir / label / name / f"level-{level_i:02d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        heuristic_fn, code, error = heuristic_factory(name, level_i)
        if code:
            (level_dir / "heuristic.py").write_text(code, encoding="utf-8")
        if error:
            print(f"  [{label}] {key}: synthesis error, falling back to built-in: {error[:160]}")
            heuristic_fn = builtin_heuristic

        result = evaluate_one_game(
            evaluator,
            name,
            game_texts[name],
            heuristic_fn,
            max_expansions,
            output_dir=level_dir,
            level_i=level_i,
            env_description=level_env_descs.get(name, {}).get(level_i),
            heuristic_code=code,
            base_prompt_baseline=(baseline_by_example or {}).get(key),
        )
        row = {
            "policy": label,
            "game": name,
            "level": level_i,
            "example": key,
            "solved": bool(result["solved"]),
            "expanded": int(result["expanded"]),
            "generated": int(result["generated"]),
            "solution_length": int(result["solution_length"]),
            "result_score": float(result["score"]),
            "gepa_score": _score_result(result, max_expansions),
        }
        rows.append(row)
        solved = "Y" if row["solved"] else "N"
        print(
            f"  [{label}] {name:<45} level={level_i:<2} "
            f"solved={solved} expanded={row['expanded']} score={row['gepa_score']:.4f}"
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, dict[str, Any]] = {}
    by_game: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        policy = row["policy"]
        game = row["game"]
        p = by_policy.setdefault(policy, {"n": 0, "solved": 0, "score_sum": 0.0, "expanded_sum": 0})
        p["n"] += 1
        p["solved"] += int(row["solved"])
        p["score_sum"] += float(row["gepa_score"])
        p["expanded_sum"] += int(row["expanded"])

        g = by_game.setdefault(game, {}).setdefault(
            policy, {"n": 0, "solved": 0, "score_sum": 0.0, "expanded_sum": 0}
        )
        g["n"] += 1
        g["solved"] += int(row["solved"])
        g["score_sum"] += float(row["gepa_score"])
        g["expanded_sum"] += int(row["expanded"])

    def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
        n = max(int(bucket["n"]), 1)
        return {
            "n": int(bucket["n"]),
            "solved": int(bucket["solved"]),
            "solve_rate": float(bucket["solved"]) / n,
            "mean_score": float(bucket["score_sum"]) / n,
            "mean_expanded": float(bucket["expanded_sum"]) / n,
        }

    return {
        "by_policy": {policy: finalize(bucket) for policy, bucket in by_policy.items()},
        "by_game": {
            game: {policy: finalize(bucket) for policy, bucket in policies.items()}
            for game, policies in by_game.items()
        },
    }


def _plot_aggregate(summary: dict[str, Any], output_path: Path) -> None:
    policies = [
        p
        for p in ("best_prompt", "base_prompt", "builtin", "blind")
        if p in summary["by_policy"]
    ]
    rates = [summary["by_policy"][p]["solve_rate"] for p in policies]
    colors_by_policy = {
        "best_prompt": "#2f6f73",
        "base_prompt": "#7c3aed",
        "builtin": "#8b6f47",
        "blind": "#6b7280",
    }
    colors = [colors_by_policy[p] for p in policies]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(policies, rates, color=colors)
    if rates and min(rates) >= 0.5:
        lower = max(0.0, min(rates) - 0.03)
    else:
        lower = 0.0
    ax.set_ylim(lower, 1.0)
    ax.set_ylabel("Solve rate")
    ax.set_title("Holdout Solve Rate (Zoomed)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.998, value + 0.003),
            f"{value:.1%}",
            ha="center",
        )
    if lower > 0.0:
        ax.text(
            0.0,
            1.02,
            f"Zoomed y-axis starts at {lower:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_by_game_score(summary: dict[str, Any], output_path: Path) -> None:
    games = sorted(summary["by_game"])
    policies = [
        p
        for p in ("best_prompt", "base_prompt", "builtin", "blind")
        if any(p in summary["by_game"][g] for g in games)
    ]
    width = 0.24
    x = list(range(len(games)))
    colors = {
        "best_prompt": "#2f6f73",
        "base_prompt": "#7c3aed",
        "builtin": "#8b6f47",
        "blind": "#6b7280",
    }

    fig_w = max(12.0, len(games) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    for i, policy in enumerate(policies):
        offset = (i - (len(policies) - 1) / 2) * width
        values = [summary["by_game"][game].get(policy, {}).get("mean_score", 0.0) for game in games]
        ax.bar([v + offset for v in x], values, width=width, label=policy, color=colors.get(policy))
    all_values = [
        summary["by_game"][game].get(policy, {}).get("mean_score", 0.0)
        for game in games
        for policy in policies
    ]
    positive_values = [value for value in all_values if value > 0.0]
    if positive_values:
        lower = max(0.0, min(positive_values) - 0.03)
    else:
        lower = 0.0
    ax.set_ylim(lower, 1.01)
    ax.set_ylabel("Mean GEPA score")
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, policy in enumerate(policies):
        offset = (i - (len(policies) - 1) / 2) * width
        zero_x = [
            v + offset
            for v, game in zip(x, games)
            if summary["by_game"][game].get(policy, {}).get("mean_score", 0.0) == 0.0
        ]
        if zero_x:
            ax.scatter(
                zero_x,
                [lower + 0.004] * len(zero_x),
                marker="v",
                s=26,
                color=colors.get(policy),
                edgecolor="black",
                linewidth=0.3,
                zorder=5,
            )
    if lower > 0.0:
        fig.text(
            0.99,
            0.91,
            f"y starts at {lower:.2f}; triangles mark zero-score failures",
            fontsize=9,
            ha="right",
            va="top",
        )
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Holdout Mean Score by Game (Zoomed)", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(policies))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_by_game_score_loss_log(summary: dict[str, Any], output_path: Path) -> None:
    games = sorted(summary["by_game"])
    policies = [
        p
        for p in ("best_prompt", "base_prompt", "builtin", "blind")
        if any(p in summary["by_game"][g] for g in games)
    ]
    width = 0.24
    x = list(range(len(games)))
    colors = {
        "best_prompt": "#2f6f73",
        "base_prompt": "#7c3aed",
        "builtin": "#8b6f47",
        "blind": "#6b7280",
    }

    fig_w = max(12.0, len(games) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    min_visible = 0.0001
    for i, policy in enumerate(policies):
        offset = (i - (len(policies) - 1) / 2) * width
        values = []
        for game in games:
            row = summary["by_game"][game].get(policy, {})
            mean_score = float(row.get("mean_score", 0.0))
            values.append(max(1.0 - mean_score, min_visible))
        ax.bar([v + offset for v in x], values, width=width, label=policy, color=colors.get(policy))
    ax.set_yscale("log")
    ax.set_ylim(min_visible, 1.0)
    ax.set_ylabel("Score loss = 1 - mean score, log scale")
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=45, ha="right")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.text(
        0.99,
        0.91,
        f"perfect scores clipped to {min_visible:g}",
        fontsize=9,
        ha="right",
        va="top",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Holdout Score Loss by Game (Lower Is Better)", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(policies))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("eval", "train"), default="eval")
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_ASTAR_MAX_EXPANSIONS)
    parser.add_argument("--levels-per-game", type=int, default=DEFAULT_LEVELS_PER_GAME)
    parser.add_argument("--llm", type=str, default=DEFAULT_LLM)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--script-doctor", type=Path, default=SCRIPT_DOCTOR_PATH)
    args = parser.parse_args()

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    train_jobs, eval_jobs = load_env_grid(args.env_grid)
    jobs = eval_jobs if args.split == "eval" else train_jobs
    output_dir = args.output_dir or (args.state_root / "best_prompt_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = _load_best_prompt(args.state_root)
    (output_dir / "best_prompt.txt").write_text(prompt_text, encoding="utf-8")

    examples, game_texts, level_env_descs = _prepare_examples(
        evaluator=evaluator,
        jobs=jobs,
        sd_path=args.script_doctor,
        levels_per_game=args.levels_per_game,
    )
    print(f"Prepared {len(examples)} {args.split} level examples across {len(game_texts)} games.")

    lm = dspy.LM(args.llm, max_tokens=args.llm_max_tokens)
    dspy.configure(lm=lm)
    cost_logger = LMCostLogger(lm, output_dir)

    def best_prompt_factory(name: str, level_i: int) -> tuple[Callable, str, str | None]:
        heuristic_fn, code, error = synthesize_heuristic_from_prompt(
            prompt_text,
            level_env_descs[name][level_i],
            lm,
            preflight_evaluator=evaluator,
            preflight_game_text=game_texts[name],
            preflight_level_i=level_i,
        )
        if heuristic_fn is None:
            return builtin_heuristic, code, error
        return heuristic_fn, code, None

    def base_prompt_factory(name: str, level_i: int) -> tuple[Callable, str, str | None]:
        heuristic_fn, code, error = synthesize_heuristic_from_prompt(
            PUZZLESCRIPT_HEURISTIC_CONTRACT,
            level_env_descs[name][level_i],
            lm,
            preflight_evaluator=evaluator,
            preflight_game_text=game_texts[name],
            preflight_level_i=level_i,
        )
        if heuristic_fn is None:
            return builtin_heuristic, code, error
        return heuristic_fn, code, None

    rows: list[dict[str, Any]] = []
    base_rows = _evaluate_policy(
        label="base_prompt",
        evaluator=evaluator,
        examples=examples,
        game_texts=game_texts,
        level_env_descs=level_env_descs,
        max_expansions=args.max_expansions,
        output_dir=output_dir,
        heuristic_factory=base_prompt_factory,
    )
    rows.extend(base_rows)
    base_baselines = {
        str(row["example"]): {
            "solved": bool(row["solved"]),
            "expanded": int(row["expanded"]),
            "generated": int(row["generated"]),
            "solution_length": int(row["solution_length"]),
            "score": float(row["result_score"]),
        }
        for row in base_rows
    }
    cost_logger.sync("base_prompt_eval", {"n_examples": len(examples), "split": args.split})

    rows.extend(
        _evaluate_policy(
            label="best_prompt",
            evaluator=evaluator,
            examples=examples,
            game_texts=game_texts,
            level_env_descs=level_env_descs,
            max_expansions=args.max_expansions,
            output_dir=output_dir,
            heuristic_factory=best_prompt_factory,
            baseline_by_example=base_baselines,
        )
    )
    cost_logger.sync("best_prompt_eval", {"n_examples": len(examples), "split": args.split})

    rows.extend(
        _evaluate_policy(
            label="builtin",
            evaluator=evaluator,
            examples=examples,
            game_texts=game_texts,
            level_env_descs=level_env_descs,
            max_expansions=args.max_expansions,
            output_dir=output_dir,
            heuristic_factory=lambda _name, _level: (builtin_heuristic, "", None),
        )
    )
    rows.extend(
        _evaluate_policy(
            label="blind",
            evaluator=evaluator,
            examples=examples,
            game_texts=game_texts,
            level_env_descs=level_env_descs,
            max_expansions=args.max_expansions,
            output_dir=output_dir,
            heuristic_factory=lambda _name, _level: (blind_heuristic, "", None),
        )
    )

    summary = _aggregate(rows)
    stats = {
        "split": args.split,
        "state_root": str(args.state_root),
        "max_expansions": args.max_expansions,
        "levels_per_game": args.levels_per_game,
        "rows": rows,
        "summary": summary,
        "llm_cost_summary": cost_logger.summary(),
    }
    (output_dir / "best_prompt_eval_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot_aggregate(summary, output_dir / "holdout_comparison_aggregate.png")
    _plot_by_game_score(summary, output_dir / "holdout_comparison_by_env.png")
    _plot_by_game_score_loss_log(summary, output_dir / "holdout_comparison_by_env_score_loss_log.png")
    cost_logger.sync("run_complete")

    print("\nSummary:")
    for policy, bucket in summary["by_policy"].items():
        print(
            f"  {policy:<12} solved={bucket['solved']}/{bucket['n']} "
            f"solve_rate={bucket['solve_rate']:.3f} mean_score={bucket['mean_score']:.4f} "
            f"mean_expanded={bucket['mean_expanded']:.1f}"
        )
    print(f"\nWrote stats: {output_dir / 'best_prompt_eval_stats.json'}")
    print(f"Wrote plots: {output_dir / 'holdout_comparison_aggregate.png'}")
    print(f"             {output_dir / 'holdout_comparison_by_env.png'}")


if __name__ == "__main__":
    main()
