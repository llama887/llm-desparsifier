#!/usr/bin/env python3
"""Seed the synthesis cache from artifacts a previous run already paid for.

Synthesis is the expensive half of an evaluation, and a run that was stopped
part-way still leaves every artifact it produced on disk. This rebuilds the
task set exactly as that run built it, recomputes each artifact's cache key
from the inputs the agent saw, and stores it. A later run with the same prompt
and workspace then reuses the work instead of buying it again.

The task-construction flags must match the run being harvested. A mismatch
changes the key, so entries simply never match later; nothing is silently
mis-attributed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_puzzlescript_batched_gepa import (  # noqa: E402
    PUZZLESCRIPT_HEURISTIC_CONTRACT,
    PuzzleScriptEvaluator,
    SynthesisCache,
    build_codex_synthesis_workspace,
    build_level_tasks,
    build_synthesis_prompt,
    load_blind_reference,
    load_env_grid,
    synthesis_cache_key,
    validate_heuristic_code,
)

ARTIFACT_RE = re.compile(r"^(\d+)-(?P<game>.+)-level-(?P<level>\d+)-rep-(?P<rep>\d+)\.py$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Prior run's state root.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--script-doctor", type=Path, required=True)
    parser.add_argument("--env-grid", type=Path, default=Path("configs/gepa_puzzlescript_envs.yaml"))
    parser.add_argument("--blind-reference", type=Path, default=None)
    parser.add_argument("--blind-budget-multiplier", type=float, default=2.0)
    parser.add_argument("--levels-per-game", type=int, default=0)
    parser.add_argument("--fallback-budget", type=int, default=10000)
    parser.add_argument("--sibling-level-holdout", action="store_true")
    parser.add_argument("--sibling-seed", type=int, default=0)
    parser.add_argument("--require-blind-reference", action="store_true")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    script_doctor = args.script_doctor.expanduser().resolve()

    artifacts: dict[tuple[str, int, int], Path] = {}
    for path in sorted(args.run_root.expanduser().resolve().glob("candidate_evals/*/heuristics/*.py")):
        match = ARTIFACT_RE.match(path.name)
        if not match:
            continue
        key = (match.group("game"), int(match.group("level")), int(match.group("rep")))
        artifacts.setdefault(key, path)
    print(f"[backfill] artifacts on disk: {len(artifacts)}")
    if not artifacts:
        print("[backfill] nothing to harvest", file=sys.stderr)
        return 2

    train_jobs, _ = load_env_grid(args.env_grid.expanduser().resolve())
    reference = load_blind_reference(args.blind_reference)
    evaluator = PuzzleScriptEvaluator(script_doctor)
    tasks = build_level_tasks(
        evaluator=evaluator,
        jobs=train_jobs,
        script_doctor=script_doctor,
        levels_per_game=args.levels_per_game,
        budget=int(args.fallback_budget),
        sibling_level_holdout=bool(args.sibling_level_holdout),
        sibling_seed=int(args.sibling_seed),
        blind_reference=reference,
        blind_budget_multiplier=float(args.blind_budget_multiplier),
        require_blind_reference=bool(args.require_blind_reference),
    )
    print(f"[backfill] reconstructed tasks: {len(tasks)}")

    cache = SynthesisCache(None if args.dry_run else args.cache_dir)
    stored = matched = invalid = 0
    unmatched: list[tuple[str, int, int]] = []
    for task in tasks:
        workspace = build_codex_synthesis_workspace(task, script_doctor=script_doctor)
        prompt = build_synthesis_prompt(PUZZLESCRIPT_HEURISTIC_CONTRACT, task)
        for replicate in range(max(1, int(args.replicates))):
            path = artifacts.get((task.game, int(task.level), replicate))
            if path is None:
                unmatched.append((task.game, int(task.level), replicate))
                continue
            code = path.read_text(encoding="utf-8")
            if not code.strip():
                continue
            if validate_heuristic_code(code) is not None:
                invalid += 1
                continue
            matched += 1
            key = synthesis_cache_key(
                prompt=prompt,
                workspace_files=workspace,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                agentic=True,
                replicate=replicate,
            )
            if not args.dry_run:
                cache.put(key, code)
            stored += 1

    print(
        f"[backfill] matched={matched} stored={stored} invalid_artifacts={invalid} "
        f"not_yet_synthesized={len(unmatched)}"
    )
    if args.dry_run:
        print("[backfill] dry run: nothing written")
    else:
        print(f"[backfill] cache dir: {args.cache_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
