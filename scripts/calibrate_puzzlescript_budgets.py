#!/usr/bin/env python3
"""Measure blind (h=0) A* search effort for every PuzzleScript training level.

The efficiency objective needs a fixed reference point. Comparing a candidate
prompt's expansions against the previous prompt's expansions cannot support a
"search became more efficient" claim on its own, and a flat expansion budget
across levels of very different difficulty leaves easy levels with no
efficiency pressure and hard levels with no chance of solving at all.

This runs the existing legacy-A* route with `heuristic_cost_to_go` pinned to
0.0, which is exactly blind uniform-cost search, and records what each level
costs. Downstream that reference gives two things:

  * per-level expansion budgets, set as a multiple of blind effort instead of a
    flat constant, and
  * the headline metric, log2(blind_expansions / candidate_expansions), which
    is a speedup against a fixed baseline rather than against a moving prompt.

The measurement reuses `evaluate_manifest_shard`, so calibration and training
run through the same evaluation path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_puzzlescript_batched_gepa import (  # noqa: E402
    PuzzleScriptEvaluator,
    build_level_tasks,
    evaluate_manifest_shard,
    evaluate_manifest_shards_locally,
    load_env_grid,
    publish_search_pool_manifest,
    wait_for_shards,
)

BLIND_HEURISTIC = '''def heuristic_cost_to_go(ts, env_params, ctx):
    """Blind search: every state looks equally promising."""
    return 0.0
'''


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-grid", type=Path, default=Path("configs/gepa_puzzlescript_envs.yaml"))
    parser.add_argument("--script-doctor", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("configs/puzzlescript_blind_reference.json"))
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--levels-per-game", type=int, default=0)
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=200_000,
        help="Ceiling for the blind measurement. Levels that do not solve "
        "inside it are recorded as unsolved with the observed counts.",
    )
    parser.add_argument("--astar-timeout-s", type=float, default=600.0)
    parser.add_argument("--task-wall-timeout-s", type=float, default=900.0)
    parser.add_argument("--include-eval-jobs", action="store_true")
    parser.add_argument("--array-count", type=int, default=64)
    parser.add_argument("--search-pool-dir", type=Path, default=None)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--stall-timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--local-workers",
        type=int,
        default=0,
        help="Evaluate shards in-process with this many workers instead of "
        "using a Slurm search pool.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    script_doctor = args.script_doctor.expanduser().resolve()

    train_jobs, val_jobs, eval_jobs = load_env_grid(
        args.env_grid.expanduser().resolve()
    )
    # Validation games need calibrated references too.
    train_jobs = [*train_jobs, *val_jobs]
    jobs = list(train_jobs) + (list(eval_jobs) if args.include_eval_jobs else [])
    print(f"[calibrate] jobs={len(jobs)} (train={len(train_jobs)} eval_included={args.include_eval_jobs})", flush=True)

    evaluator = PuzzleScriptEvaluator(script_doctor)
    tasks = build_level_tasks(
        evaluator=evaluator,
        jobs=jobs,
        script_doctor=script_doctor,
        levels_per_game=args.levels_per_game,
        budget=int(args.max_expansions),
    )
    print(f"[calibrate] levels={len(tasks)}", flush=True)
    if not tasks:
        print("[calibrate] no levels to calibrate", file=sys.stderr)
        return 2

    blind_path = state_root / "blind_heuristic.py"
    blind_path.write_text(BLIND_HEURISTIC, encoding="utf-8")

    shard_dir = state_root / "search_shards"
    manifest_path = state_root / "search_manifest.json"
    array_count = max(1, min(int(args.array_count), len(tasks)))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = {
            "task_id": int(task.task_id),
            "game": task.game,
            "level": int(task.level),
            "budget": int(args.max_expansions),
            "game_text_path": task.game_text_path,
            "heuristic_code_path": str(blind_path),
            "replicate": 0,
        }
        rows.append(row)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_doctor": str(script_doctor),
        "astar_timeout_s": float(args.astar_timeout_s),
        "task_wall_timeout_s": float(args.task_wall_timeout_s),
        "shard_dir": str(shard_dir),
        "array_count": array_count,
        "tasks": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[calibrate] manifest={manifest_path} array_count={array_count}", flush=True)

    if args.search_pool_dir is not None:
        pointer = publish_search_pool_manifest(args.search_pool_dir.expanduser().resolve(), manifest_path)
        print(f"[calibrate] published manifest via {pointer}", flush=True)
        wait_for_shards(
            shard_dir=shard_dir,
            array_count=array_count,
            poll_interval_s=float(args.poll_interval_s),
            stall_timeout_s=float(args.stall_timeout_s),
        )
    elif args.local_workers > 1:
        evaluate_manifest_shards_locally(
            manifest_path=manifest_path,
            array_count=array_count,
            missing_indices=list(range(array_count)),
            max_workers=int(args.local_workers),
        )
    else:
        for index in range(array_count):
            evaluate_manifest_shard(
                manifest_path=manifest_path,
                array_index=index,
                array_count=array_count,
            )
            print(f"[calibrate] shard {index + 1}/{array_count} done", flush=True)

    results: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("task-*.json")):
        payload = json.loads(shard.read_text(encoding="utf-8"))
        results.extend(payload["results"] if isinstance(payload, dict) else payload)

    reference: dict[str, dict[str, Any]] = {}
    solved = 0
    for row in results:
        key = f"{row['game']}::{int(row['level'])}"
        is_solved = bool(row.get("solved"))
        solved += int(is_solved)
        reference[key] = {
            "game": row["game"],
            "level": int(row["level"]),
            "blind_solved": is_solved,
            "blind_expanded": int(row.get("expanded", 0) or 0),
            "blind_generated": int(row.get("generated", 0) or 0),
            "blind_solution_length": int(row.get("solution_length", 0) or 0),
            "blind_time_s": float(row.get("time_s", 0.0) or 0.0),
            "measurement_ceiling": int(args.max_expansions),
            "terminated_reason": (row.get("trace_summary") or {}).get("terminated_reason"),
        }

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_grid": str(args.env_grid),
        "max_expansions": int(args.max_expansions),
        "astar_timeout_s": float(args.astar_timeout_s),
        "level_count": len(reference),
        "blind_solved_count": solved,
        "levels": reference,
    }
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[calibrate] wrote {out}: levels={len(reference)} blind_solved={solved}", flush=True)

    expansions = sorted(
        v["blind_expanded"] for v in reference.values() if v["blind_solved"]
    )
    if expansions:
        def q(p: float) -> int:
            return expansions[min(len(expansions) - 1, int(p * len(expansions)))]

        print(
            f"[calibrate] blind expansions on solved levels: "
            f"min={expansions[0]} p25={q(0.25)} median={q(0.5)} "
            f"p75={q(0.75)} p95={q(0.95)} max={expansions[-1]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
