#!/usr/bin/env python3
"""Run a small heldout PuzzleScript model smoke test.

The full GEPA loop is too expensive for quick model triage. This script uses
the same PuzzleScript base prompt, synthesis adapter, search evaluator, and
reflection proposer as the batched GEPA workflow, but limits the workload to a
few heldout levels and one model-dependent feedback/proposal pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_puzzlescript_batched_gepa import (  # noqa: E402
    DEFAULT_ASTAR_TIMEOUT_S,
    DEFAULT_BASE_URL,
    DEFAULT_CANDIDATE_ERROR_PENALTY,
    DEFAULT_ENV_GRID,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOST_SOLVE_PENALTY,
    DEFAULT_MAX_MODEL_TOKENS,
    DEFAULT_NEW_SOLVE_BONUS,
    DEFAULT_PARTIAL_PROGRESS_WEIGHT,
    DEFAULT_SCORE_DELTA_CLIP,
    DEFAULT_SCORE_DELTA_WEIGHT,
    DEFAULT_SCRIPT_DOCTOR,
    DEFAULT_SEARCH_ARRAY_STALL_TIMEOUT_S,
    HEURISTIC_COMPONENT,
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    OpenAITextClient,
    PuzzleScriptBatchedGEPAAdapter,
    PuzzleScriptEvaluator,
    SearchArrayConfig,
    build_level_tasks,
    candidate_prompt_issue,
    is_candidate_error,
    load_env_grid,
    parse_extra_sbatch_args,
    reassign_task_ids,
    trace_classification,
    truncate_text,
)

DEFAULT_SMOKE_GAMES = (
    "sokoban_sanity",
    "No_Right_Turn_Sokoban",
    "Crates_and_Portals",
    "Gravity_Sokoban",
    "Ice_Cubes",
    "Beam_Islands",
)


def select_smoke_jobs(
    eval_jobs: Sequence[Mapping[str, Any]],
    requested_games: Sequence[str],
    *,
    max_games: int,
) -> list[dict[str, Any]]:
    """Return heldout jobs for a short model smoke test.

    Requested games are preserved in caller-provided order so two model jobs see
    the same mechanics mix. If some games are unavailable, the function fills
    the remaining slots from the eval split rather than shrinking the test.
    """

    available = {str(job["name"]): dict(job) for job in eval_jobs}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = max(1, max_games)

    for game in requested_games:
        if game in available and game not in seen:
            selected.append(dict(available[game]))
            seen.add(game)
        if len(selected) >= limit:
            return selected

    for job in eval_jobs:
        game = str(job["name"])
        if game in seen:
            continue
        selected.append(dict(job))
        seen.add(game)
        if len(selected) >= limit:
            break
    return selected


def _code_quality_flags(code: str) -> dict[str, bool]:
    lower_code = code.lower()
    return {
        "starts_with_contract_def": code.lstrip().startswith("def heuristic_cost_to_go"),
        "uses_object_positions": "object_positions" in lower_code,
        "uses_object_names": "object_names" in lower_code,
        "uses_win_text": "win_conditions" in lower_code or "wincondition" in lower_code,
        "uses_score_fallback": "score_normalized" in lower_code or "ctx['score']" in lower_code,
        "mentions_hardcoded_sokoban_roles": any(
            token in lower_code for token in ("crate", "box", "target")
        ),
    }


def summarize_smoke_outputs(
    outputs: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return aggregate solve, error, and code-shape diagnostics for a smoke run."""

    output_rows = list(outputs)
    trace_rows = list(trajectories)
    scores = [float(row.get("score", 0.0)) for row in output_rows]
    adjusted_scores = [
        float(row.get("adjusted_score", row.get("score", 0.0))) for row in output_rows
    ]
    expanded_values = [
        int(row["expanded"]) for row in output_rows if row.get("expanded") is not None
    ]
    codes = [str(trace.get("heuristic_code", "")) for trace in trace_rows]
    flags = [_code_quality_flags(code) for code in codes]
    n_outputs = len(output_rows)

    return {
        "n": n_outputs,
        "score_mean": sum(scores) / n_outputs if n_outputs else 0.0,
        "adjusted_score_mean": (
            sum(adjusted_scores) / n_outputs if n_outputs else 0.0
        ),
        "solved": sum(1 for row in output_rows if bool(row.get("solved", False))),
        "candidate_errors": sum(1 for row in output_rows if is_candidate_error(row)),
        "synthesis_errors": sum(1 for row in output_rows if row.get("synthesis_error") is not None),
        "result_errors": sum(1 for row in output_rows if row.get("error") is not None),
        "expanded_mean": (
            sum(expanded_values) / len(expanded_values) if expanded_values else 0.0
        ),
        "code_count": len(codes),
        "code_line_count_mean": (
            sum(len(code.splitlines()) for code in codes) / len(codes) if codes else 0.0
        ),
        "starts_with_contract_def": sum(
            1 for row in flags if row["starts_with_contract_def"]
        ),
        "uses_object_positions": sum(1 for row in flags if row["uses_object_positions"]),
        "uses_object_names": sum(1 for row in flags if row["uses_object_names"]),
        "uses_win_text": sum(1 for row in flags if row["uses_win_text"]),
        "uses_score_fallback": sum(1 for row in flags if row["uses_score_fallback"]),
        "mentions_hardcoded_sokoban_roles": sum(
            1 for row in flags if row["mentions_hardcoded_sokoban_roles"]
        ),
    }


def attach_self_baseline_metadata(
    trajectories: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> None:
    """Attach base-vs-base metadata so reflection labels are not misleading.

    A smoke run evaluates only the base prompt for one model. GEPA's reflection
    helpers expect candidate results to carry baseline metadata, so this marks
    each generated result as its own baseline. Solved levels become stable cases
    and unsolved levels become persistent failures, which matches the evidence
    available in this quick test.
    """

    baseline_by_key = {
        (str(row["game"]), int(row["level"])): row
        for row in outputs
    }
    for trace in trajectories:
        result = trace.get("result")
        if not isinstance(result, dict):
            continue
        baseline = baseline_by_key.get((str(result.get("game")), int(result.get("level", 0))))
        if baseline is None:
            continue
        result["baseline_score"] = float(baseline.get("score", 0.0))
        result["baseline_solved"] = bool(baseline.get("solved", False))
        result["baseline_error"] = baseline.get("error")
        result["baseline_expanded"] = baseline.get("expanded")
        result["baseline_generated"] = baseline.get("generated")
        result["baseline_solution_length"] = baseline.get("solution_length")
        result["baseline_partial_progress_score"] = baseline.get(
            "partial_progress_score",
            0.0,
        )
        result["baseline_feedback"] = baseline.get("feedback")


def build_trajectory_summaries(
    trajectories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return compact per-level records for manual model-output inspection."""

    summaries: list[dict[str, Any]] = []
    for trace in trajectories:
        task = trace.get("task", {})
        result = trace.get("result", {})
        code = str(trace.get("heuristic_code", ""))
        summaries.append(
            {
                "task": {
                    "game": task.get("game"),
                    "level": task.get("level"),
                    "budget": task.get("budget"),
                },
                "classification": trace_classification(trace),
                "solved": bool(result.get("solved", False)),
                "score": float(result.get("score", 0.0)),
                "adjusted_score": float(result.get("adjusted_score", result.get("score", 0.0))),
                "expanded": result.get("expanded"),
                "generated": result.get("generated"),
                "solution_length": result.get("solution_length"),
                "error": result.get("error"),
                "synthesis_error": trace.get("synthesis_error"),
                "heuristic_code_path": trace.get("heuristic_code_path"),
                "heuristic_code_preview": truncate_text(code, 2000),
                "feedback": truncate_text(str(result.get("feedback", "")), 2000),
            }
        )
    return summaries


def build_feedback_probe_prompt(
    *,
    model: str,
    summary: Mapping[str, Any],
    trajectory_summaries: Sequence[Mapping[str, Any]],
    reflection_records: Sequence[Mapping[str, Any]],
) -> str:
    """Build a short model-dependent audit prompt for feedback-quality triage."""

    compact_payload = {
        "model_under_test": model,
        "aggregate_summary": dict(summary),
        "level_results": list(trajectory_summaries),
        "reflection_records": list(reflection_records),
    }
    return (
        "You are auditing a local model for PuzzleScript GEPA prompt optimization.\n"
        "Given this small heldout base-prompt run, write a concise technical audit.\n"
        "Cover: generated-code quality, search behavior, likely failure causes, "
        "and 3 prompt changes that would be safe to try. Use only evidence in the "
        "payload and avoid game-specific memorization.\n\n"
        "Payload:\n"
        + truncate_text(json.dumps(compact_payload, indent=2, sort_keys=True, default=str), 24_000)
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _limited_tasks(tasks: Sequence[Any], max_tasks: int) -> list[Any]:
    if max_tasks <= 0:
        return list(tasks)
    return list(tasks)[:max_tasks]


def run_model_smoke(args: argparse.Namespace) -> None:
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    _train_jobs, eval_jobs = load_env_grid(args.env_grid)
    requested_games = args.games or list(DEFAULT_SMOKE_GAMES)
    jobs = select_smoke_jobs(
        eval_jobs,
        requested_games,
        max_games=args.max_games,
    )
    tasks = build_level_tasks(
        evaluator=evaluator,
        jobs=jobs,
        script_doctor=args.script_doctor,
        levels_per_game=args.levels_per_game,
        budget=max(1, args.max_expansions),
    )
    tasks = reassign_task_ids(_limited_tasks(tasks, args.max_tasks))
    if not tasks:
        raise RuntimeError("No heldout smoke tasks were loadable.")

    _write_json(state_root / "smoke_jobs.json", jobs)
    _write_json(state_root / "smoke_tasks.json", [asdict(task) for task in tasks])

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
        global_lost_solve_gate_penalty=args.global_lost_solve_gate_penalty,
        global_net_solve_loss_gate_penalty=args.global_net_solve_loss_gate_penalty,
    )

    candidate = {HEURISTIC_COMPONENT: PUZZLESCRIPT_HEURISTIC_CONTRACT}
    print(
        f"[smoke] evaluating model={args.model} tasks={len(tasks)} "
        f"state_root={state_root}",
        flush=True,
    )
    eval_batch = adapter.evaluate(
        batch=tasks,
        candidate=candidate,
        capture_traces=True,
    )
    outputs = [dict(row) for row in eval_batch.outputs]
    trajectories = list(eval_batch.trajectories or [])
    attach_self_baseline_metadata(trajectories, outputs)

    summary = summarize_smoke_outputs(outputs, trajectories)
    trajectory_summaries = build_trajectory_summaries(trajectories)
    reflective_dataset = adapter.make_reflective_dataset(
        candidate,
        eval_batch,
        [HEURISTIC_COMPONENT],
    )
    reflection_records = reflective_dataset.get(HEURISTIC_COMPONENT, [])

    _write_json(state_root / "base_outputs.json", outputs)
    _write_json(state_root / "trajectory_summaries.json", trajectory_summaries)
    _write_json(state_root / "reflection_records.json", reflection_records)

    proposed_prompt = ""
    proposed_prompt_error = None
    started_at = time.monotonic()
    try:
        proposal = adapter.propose_new_texts(
            candidate,
            reflective_dataset,
            [HEURISTIC_COMPONENT],
        )
        proposed_prompt = proposal.get(HEURISTIC_COMPONENT, "")
    except Exception as exc:  # pragma: no cover - records external LLM failures.
        proposed_prompt_error = str(exc)
    proposal_latency_s = time.monotonic() - started_at
    (state_root / "proposed_prompt.txt").write_text(
        proposed_prompt + "\n",
        encoding="utf-8",
    )
    proposal_issue = candidate_prompt_issue(proposed_prompt) if proposed_prompt else "empty proposal"
    _write_json(
        state_root / "proposal_summary.json",
        {
            "latency_s": proposal_latency_s,
            "error": proposed_prompt_error,
            "prompt_chars": len(proposed_prompt),
            "prompt_issue": proposal_issue,
        },
    )

    feedback_probe_prompt = build_feedback_probe_prompt(
        model=args.model,
        summary=summary,
        trajectory_summaries=trajectory_summaries,
        reflection_records=reflection_records,
    )
    (state_root / "feedback_probe_prompt.txt").write_text(
        feedback_probe_prompt,
        encoding="utf-8",
    )
    feedback_probe_response = ""
    feedback_probe_error = None
    started_at = time.monotonic()
    try:
        feedback_probe_response = llm.complete(feedback_probe_prompt)
    except Exception as exc:  # pragma: no cover - records external LLM failures.
        feedback_probe_error = str(exc)
    feedback_probe_latency_s = time.monotonic() - started_at
    (state_root / "feedback_probe_response.txt").write_text(
        feedback_probe_response + "\n",
        encoding="utf-8",
    )

    full_summary = {
        **summary,
        "model": args.model,
        "state_root": str(state_root),
        "tasks": [
            {"game": task.game, "level": task.level, "budget": task.budget}
            for task in tasks
        ],
        "proposal": {
            "latency_s": proposal_latency_s,
            "error": proposed_prompt_error,
            "prompt_chars": len(proposed_prompt),
            "prompt_issue": proposal_issue,
        },
        "feedback_probe": {
            "latency_s": feedback_probe_latency_s,
            "error": feedback_probe_error,
            "response_chars": len(feedback_probe_response),
        },
    }
    _write_json(state_root / "model_smoke_summary.json", full_summary)
    print(
        "[smoke] summary "
        f"score={summary['score_mean']:.4f} "
        f"solved={summary['solved']}/{summary['n']} "
        f"candidate_errors={summary['candidate_errors']} "
        f"proposal_issue={proposal_issue}",
        flush=True,
    )
    print(f"[smoke] summary_path={state_root / 'model_smoke_summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--script-doctor", type=Path, default=DEFAULT_SCRIPT_DOCTOR)
    parser.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="Heldout game names to try first; defaults to a mechanics-diverse subset.",
    )
    parser.add_argument("--max-games", type=int, default=6)
    parser.add_argument("--levels-per-game", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=6)
    parser.add_argument("--max-expansions", type=int, default=10_000)
    parser.add_argument("--astar-timeout-s", type=float, default=DEFAULT_ASTAR_TIMEOUT_S)
    parser.add_argument("--model", type=str, default=os.getenv("LOCAL_LLM_MODEL", "openai/gpt-oss-120b"))
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
    parser.add_argument("--llm-concurrency", type=int, default=4)
    parser.add_argument("--submit-search-array", action="store_true")
    parser.add_argument(
        "--search-array-script",
        type=Path,
        default=Path("sbatch/evaluate_puzzlescript_search_array.s"),
    )
    parser.add_argument("--search-array-count", type=int, default=8)
    parser.add_argument("--search-array-concurrency", type=int, default=8)
    parser.add_argument("--search-poll-interval-s", type=float, default=10.0)
    parser.add_argument(
        "--search-array-stall-timeout-s",
        type=float,
        default=DEFAULT_SEARCH_ARRAY_STALL_TIMEOUT_S,
    )
    parser.add_argument("--extra-sbatch-args", type=str, default="")
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
        "--global-lost-solve-gate-penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--global-net-solve-loss-gate-penalty",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    run_model_smoke(parse_args())


if __name__ == "__main__":
    main()
