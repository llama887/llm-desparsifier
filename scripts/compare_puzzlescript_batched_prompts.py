#!/usr/bin/env python3
"""Compare base and optimized PuzzleScript heuristic prompts on the holdout split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_puzzlescript_batched_gepa import (  # noqa: E402
    DEFAULT_ASTAR_TIMEOUT_S,
    DEFAULT_BASE_URL,
    DEFAULT_ENV_GRID,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_MAX_MODEL_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SCRIPT_DOCTOR,
    DEFAULT_SEARCH_ARRAY_STALL_TIMEOUT_S,
    HEURISTIC_COMPONENT,
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    OpenAITextClient,
    PuzzleScriptBatchedGEPAAdapter,
    PuzzleScriptEvaluator,
    SearchArrayConfig,
    build_level_tasks,
    load_env_grid,
    parse_extra_sbatch_args,
)


def output_summary(label: str, outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return aggregate score, solve, and error metrics for one prompt policy."""
    scores = [float(row.get("score", 0.0)) for row in outputs]
    solved = sum(1 for row in outputs if bool(row.get("solved", False)))
    errors = sum(1 for row in outputs if row.get("error") is not None)
    expanded_values = [
        float(row.get("expanded", 0.0))
        for row in outputs
        if row.get("expanded") is not None
    ]
    n_outputs = len(outputs)
    return {
        "label": label,
        "n": n_outputs,
        "score_mean": sum(scores) / n_outputs if n_outputs else 0.0,
        "solved": solved,
        "solve_rate": solved / n_outputs if n_outputs else 0.0,
        "result_errors": errors,
        "expanded_mean": (
            sum(expanded_values) / len(expanded_values) if expanded_values else 0.0
        ),
    }


def compare_prompt_outputs(
    *,
    base_outputs: Sequence[Mapping[str, Any]],
    optimized_outputs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build aggregate, per-level, and per-game holdout comparisons.

    Outputs are matched by `task_id`, which is stable because both prompt
    policies are evaluated over the same ordered holdout task list.
    """
    base_by_id = {int(row["task_id"]): row for row in base_outputs}
    optimized_by_id = {int(row["task_id"]): row for row in optimized_outputs}
    common_ids = sorted(set(base_by_id) & set(optimized_by_id))

    per_level: list[dict[str, Any]] = []
    for task_id in common_ids:
        base = base_by_id[task_id]
        optimized = optimized_by_id[task_id]
        base_score = float(base.get("score", 0.0))
        optimized_score = float(optimized.get("score", 0.0))
        base_solved = bool(base.get("solved", False))
        optimized_solved = bool(optimized.get("solved", False))
        per_level.append(
            {
                "task_id": task_id,
                "game": str(optimized.get("game", base.get("game", ""))),
                "level": int(optimized.get("level", base.get("level", 0))),
                "base_score": base_score,
                "optimized_score": optimized_score,
                "score_delta": optimized_score - base_score,
                "base_solved": base_solved,
                "optimized_solved": optimized_solved,
                "solve_delta": int(optimized_solved) - int(base_solved),
                "base_expanded": int(base.get("expanded", 0) or 0),
                "optimized_expanded": int(optimized.get("expanded", 0) or 0),
                "base_error": base.get("error"),
                "optimized_error": optimized.get("error"),
            }
        )

    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in per_level:
        by_game.setdefault(str(row["game"]), []).append(row)

    per_game: list[dict[str, Any]] = []
    for game, rows in sorted(by_game.items()):
        n_rows = len(rows)
        base_solved_count = sum(1 for row in rows if row["base_solved"])
        optimized_solved_count = sum(1 for row in rows if row["optimized_solved"])
        base_score = sum(float(row["base_score"]) for row in rows) / n_rows
        optimized_score = sum(float(row["optimized_score"]) for row in rows) / n_rows
        per_game.append(
            {
                "game": game,
                "n": n_rows,
                "base_score_mean": base_score,
                "optimized_score_mean": optimized_score,
                "score_delta": optimized_score - base_score,
                "base_solved": base_solved_count,
                "optimized_solved": optimized_solved_count,
                "solved_delta": optimized_solved_count - base_solved_count,
                "better_score_count": sum(1 for row in rows if row["score_delta"] > 0.0),
                "worse_score_count": sum(1 for row in rows if row["score_delta"] < 0.0),
                "new_solve_count": sum(1 for row in rows if row["solve_delta"] > 0),
                "lost_solve_count": sum(1 for row in rows if row["solve_delta"] < 0),
            }
        )

    base_summary = output_summary("base", base_outputs)
    optimized_summary = output_summary("optimized", optimized_outputs)
    aggregate = {
        "base": base_summary,
        "optimized": optimized_summary,
        "n_matched": len(per_level),
        "score_delta": optimized_summary["score_mean"] - base_summary["score_mean"],
        "relative_score_delta": (
            optimized_summary["score_mean"] / base_summary["score_mean"] - 1.0
            if base_summary["score_mean"]
            else None
        ),
        "solved_delta": optimized_summary["solved"] - base_summary["solved"],
        "solve_rate_delta": optimized_summary["solve_rate"] - base_summary["solve_rate"],
        "better_score_count": sum(1 for row in per_level if row["score_delta"] > 0.0),
        "worse_score_count": sum(1 for row in per_level if row["score_delta"] < 0.0),
        "equal_score_count": sum(1 for row in per_level if row["score_delta"] == 0.0),
        "new_solve_count": sum(1 for row in per_level if row["solve_delta"] > 0),
        "lost_solve_count": sum(1 for row in per_level if row["solve_delta"] < 0),
    }
    return aggregate, per_level, per_game


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_plots(
    *,
    output_dir: Path,
    per_game: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Write PNG plots that summarize base-vs-optimized holdout behavior."""
    rows = sorted(per_game, key=lambda row: float(row.get("score_delta", 0.0)))
    if not rows:
        return []

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["game"]) for row in rows]
    y_positions = list(range(len(rows)))
    colors = [
        "#1f8a4c" if float(row.get("score_delta", 0.0)) >= 0.0 else "#b23a3a"
        for row in rows
    ]
    height = max(4.5, min(18.0, 0.28 * len(rows) + 1.8))
    plot_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(
        y_positions,
        [float(row.get("score_delta", 0.0)) for row in rows],
        color=colors,
    )
    ax.axvline(0.0, color="#2f2f2f", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Optimized score - base score")
    ax.set_title("Holdout score delta by game")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    score_delta_path = output_dir / "holdout_score_delta_by_game.png"
    fig.savefig(score_delta_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(score_delta_path)

    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(
        y_positions,
        [int(row.get("solved_delta", 0)) for row in rows],
        color=[
            "#1f8a4c" if int(row.get("solved_delta", 0)) >= 0 else "#b23a3a"
            for row in rows
        ],
    )
    ax.axvline(0, color="#2f2f2f", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Optimized solved levels - base solved levels")
    ax.set_title("Holdout solve delta by game")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    solve_delta_path = output_dir / "holdout_solve_delta_by_game.png"
    fig.savefig(solve_delta_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(solve_delta_path)

    base_scores = [float(row.get("base_score_mean", 0.0)) for row in rows]
    optimized_scores = [float(row.get("optimized_score_mean", 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(
        base_scores,
        optimized_scores,
        s=[max(28.0, float(row.get("n", 1)) * 14.0) for row in rows],
        c=colors,
        alpha=0.82,
        edgecolors="#222222",
        linewidths=0.35,
    )
    low = min(base_scores + optimized_scores + [0.0])
    high = max(base_scores + optimized_scores + [1.0])
    pad = max(0.02, (high - low) * 0.06)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#555555", linewidth=1.0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Base score mean")
    ax.set_ylabel("Optimized score mean")
    ax.set_title("Holdout base vs optimized score by game")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    scatter_path = output_dir / "holdout_score_base_vs_optimized.png"
    fig.savefig(scatter_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(scatter_path)

    return plot_paths


def write_comparison_artifacts(
    *,
    state_root: Path,
    aggregate: Mapping[str, Any],
    per_level: Sequence[Mapping[str, Any]],
    per_game: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Persist machine-readable, CSV, and plot comparison artifacts."""
    plot_paths = write_comparison_plots(output_dir=state_root, per_game=per_game)
    aggregate_payload = {
        **dict(aggregate),
        "plot_paths": [str(path) for path in plot_paths],
    }
    (state_root / "comparison_summary.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (state_root / "per_level_comparison.json").write_text(
        json.dumps(list(per_level), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (state_root / "per_game_comparison.json").write_text(
        json.dumps(list(per_game), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_csv(state_root / "per_level_comparison.csv", per_level)
    _write_csv(state_root / "per_game_comparison.csv", per_game)
    return plot_paths


def _new_eval_dir(before: set[Path], candidate_eval_root: Path) -> Path:
    after = set(candidate_eval_root.glob("eval-*"))
    created = sorted(after - before)
    if len(created) == 1:
        return created[0]
    return max(after, key=lambda path: path.stat().st_mtime)


def evaluate_candidate(
    *,
    adapter: PuzzleScriptBatchedGEPAAdapter,
    tasks: list[Any],
    state_root: Path,
    label: str,
    prompt_text: str,
) -> tuple[list[dict[str, Any]], Path]:
    """Evaluate one prompt and return its merged per-level outputs and eval dir."""
    candidate_root = state_root / "candidate_evals"
    before = set(candidate_root.glob("eval-*")) if candidate_root.exists() else set()
    batch = adapter.evaluate(
        batch=tasks,
        candidate={HEURISTIC_COMPONENT: prompt_text},
        capture_traces=False,
    )
    eval_dir = _new_eval_dir(before, candidate_root)
    (eval_dir / "prompt_label.txt").write_text(label + "\n", encoding="utf-8")
    return [dict(row) for row in batch.outputs], eval_dir


def run_holdout_comparison(args: argparse.Namespace) -> None:
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)

    optimized_prompt = args.optimized_prompt.read_text(encoding="utf-8").strip()
    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    _train_jobs, eval_jobs = load_env_grid(args.env_grid)
    tasks = build_level_tasks(
        evaluator=evaluator,
        jobs=eval_jobs,
        script_doctor=args.script_doctor,
        levels_per_game=args.levels_per_game,
        budget=max(1, args.max_expansions),
    )
    if not tasks:
        raise RuntimeError("No holdout tasks were loadable.")

    (state_root / "holdout_tasks.json").write_text(
        json.dumps([asdict(task) for task in tasks], indent=2, sort_keys=True) + "\n",
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
    )

    print(f"[holdout] tasks={len(tasks)} state_root={state_root}", flush=True)
    base_outputs, base_dir = evaluate_candidate(
        adapter=adapter,
        tasks=tasks,
        state_root=state_root,
        label="base",
        prompt_text=PUZZLESCRIPT_HEURISTIC_CONTRACT,
    )
    optimized_outputs, optimized_dir = evaluate_candidate(
        adapter=adapter,
        tasks=tasks,
        state_root=state_root,
        label="optimized",
        prompt_text=optimized_prompt,
    )
    aggregate, per_level, per_game = compare_prompt_outputs(
        base_outputs=base_outputs,
        optimized_outputs=optimized_outputs,
    )
    aggregate = {
        **aggregate,
        "base_eval_dir": str(base_dir),
        "optimized_eval_dir": str(optimized_dir),
        "optimized_prompt_path": str(args.optimized_prompt),
    }
    plot_paths = write_comparison_artifacts(
        state_root=state_root,
        aggregate=aggregate,
        per_level=per_level,
        per_game=per_game,
    )
    print(
        "[holdout] base "
        f"score={aggregate['base']['score_mean']:.4f} "
        f"solved={aggregate['base']['solved']}/{aggregate['base']['n']}",
        flush=True,
    )
    print(
        "[holdout] optimized "
        f"score={aggregate['optimized']['score_mean']:.4f} "
        f"solved={aggregate['optimized']['solved']}/{aggregate['optimized']['n']}",
        flush=True,
    )
    print(
        "[holdout] delta "
        f"score={aggregate['score_delta']:.4f} "
        f"solved={aggregate['solved_delta']} "
        f"new_solves={aggregate['new_solve_count']} "
        f"lost_solves={aggregate['lost_solve_count']}",
        flush=True,
    )
    print(f"[holdout] summary={state_root / 'comparison_summary.json'}", flush=True)
    for path in plot_paths:
        print(f"[holdout] plot={path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and optimized PuzzleScript prompts on eval_jobs."
    )
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("artifacts/puzzlescript_holdout_compare"),
    )
    parser.add_argument("--script-doctor", type=Path, default=DEFAULT_SCRIPT_DOCTOR)
    parser.add_argument("--optimized-prompt", type=Path, required=True)
    parser.add_argument("--levels-per-game", type=int, default=0)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
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
    parser.add_argument("--search-array-concurrency", type=int, default=16)
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
    return parser.parse_args()


def main() -> None:
    run_holdout_comparison(parse_args())


if __name__ == "__main__":
    main()
