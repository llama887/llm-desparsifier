#!/usr/bin/env python3
"""Prepare PuzzleScript GEPA baselines as independent Slurm array shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from dspy_cache_control import configure_dspy_cache, prepare_dspy_import

prepare_dspy_import("prepare_puzzlescript_baselines")
import dspy
configure_dspy_cache(dspy, "prepare_puzzlescript_baselines")
from run_puzzlescript_batch import (
    DEFAULT_ASTAR_MAX_EXPANSIONS,
    DEFAULT_ASTAR_TIMEOUT_S,
    DEFAULT_ENV_GRID,
    DEFAULT_LEVELS_PER_GAME,
    DEFAULT_LLM,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL,
    DEFAULT_STATE_ROOT,
    SCRIPT_DOCTOR_PATH,
    PuzzleScriptEvaluator,
    build_baseline_cache_signature,
    build_training_level_examples,
    compute_puzzlescript_baselines_for_examples,
    load_env_grid,
    load_local_env,
    prepare_puzzlescript_inputs,
    save_puzzlescript_baseline_shard,
)


def _optional_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute one shard of PuzzleScript GEPA training baselines."
    )
    parser.add_argument("--env-grid", type=Path, default=DEFAULT_ENV_GRID)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--script-doctor", type=Path, default=SCRIPT_DOCTOR_PATH)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_ASTAR_MAX_EXPANSIONS)
    parser.add_argument(
        "--max-gepa-expansions-per-level",
        type=int,
        default=DEFAULT_MAX_GEPA_EXPANSIONS_PER_LEVEL,
    )
    parser.add_argument("--astar-timeout-s", type=float, default=DEFAULT_ASTAR_TIMEOUT_S)
    parser.add_argument("--levels-per-game", type=int, default=DEFAULT_LEVELS_PER_GAME)
    parser.add_argument("--llm", type=str, default=DEFAULT_LLM)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Zero-based worker index. Defaults to SLURM_ARRAY_TASK_ID minus SLURM_ARRAY_TASK_MIN.",
    )
    parser.add_argument(
        "--array-count",
        type=int,
        default=None,
        help="Total worker count. Defaults to SLURM_ARRAY_TASK_COUNT.",
    )
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = _parse_args()

    slurm_task_id = _optional_env_int("SLURM_ARRAY_TASK_ID")
    slurm_task_min = _optional_env_int("SLURM_ARRAY_TASK_MIN") or 0
    slurm_task_count = _optional_env_int("SLURM_ARRAY_TASK_COUNT")
    raw_task_id = args.array_index if args.array_index is not None else (slurm_task_id or 0)
    array_count = args.array_count if args.array_count is not None else (slurm_task_count or 1)
    array_index = raw_task_id if args.array_index is not None else raw_task_id - slurm_task_min
    if array_count <= 0:
        raise ValueError("--array-count must be > 0")
    if array_index < 0 or array_index >= array_count:
        raise ValueError(
            f"array index {array_index} is outside [0, {array_count}); "
            f"raw_task_id={raw_task_id} slurm_task_min={slurm_task_min}"
        )

    evaluator = PuzzleScriptEvaluator(args.script_doctor)
    train_jobs, _eval_jobs = load_env_grid(args.env_grid)
    all_game_texts, all_env_descs, all_level_env_descs, level_indices_by_game = (
        prepare_puzzlescript_inputs(
            evaluator=evaluator,
            train_jobs=train_jobs,
            eval_jobs=[],
            sd_path=args.script_doctor,
            levels_per_game=args.levels_per_game,
        )
    )
    all_examples = build_training_level_examples(
        train_jobs,
        all_game_texts,
        level_indices_by_game,
    )
    assigned_examples = all_examples[array_index::array_count]
    print(
        f"[baseline-array] task {array_index}/{array_count}: "
        f"{len(assigned_examples)} of {len(all_examples)} level example(s)"
    )

    signature = build_baseline_cache_signature(
        train_jobs=train_jobs,
        level_indices_by_game=level_indices_by_game,
        max_expansions=args.max_expansions,
        max_gepa_expansions_per_level=max(1, args.max_gepa_expansions_per_level),
        astar_timeout_s=max(1.0, args.astar_timeout_s),
        levels_per_game=args.levels_per_game,
        llm_name=args.llm,
        llm_max_tokens=args.llm_max_tokens,
    )

    lm = dspy.LM(args.llm, max_tokens=args.llm_max_tokens)
    dspy.configure(lm=lm)
    blind_baselines, builtin_baselines, base_prompt_baselines, per_game_budgets = (
        compute_puzzlescript_baselines_for_examples(
            evaluator=evaluator,
            examples=assigned_examples,
            all_game_texts=all_game_texts,
            all_level_env_descs=all_level_env_descs,
            all_env_descs=all_env_descs,
            max_expansions=args.max_expansions,
            max_gepa_expansions_per_level=max(1, args.max_gepa_expansions_per_level),
            astar_timeout_s=max(1.0, args.astar_timeout_s),
            lm=lm,
        )
    )
    shard_name = f"task-{array_index:04d}-of-{array_count:04d}"
    shard_path = save_puzzlescript_baseline_shard(
        args.state_root,
        shard_name,
        signature=signature,
        blind_baselines=blind_baselines,
        builtin_baselines=builtin_baselines,
        base_prompt_baselines=base_prompt_baselines,
        per_game_budgets=per_game_budgets,
        metadata={
            "array_index": array_index,
            "array_count": array_count,
            "raw_task_id": raw_task_id,
            "n_assigned_examples": len(assigned_examples),
            "assigned_examples": assigned_examples,
        },
    )
    print(f"[baseline-array] wrote {shard_path}")


if __name__ == "__main__":
    main()
