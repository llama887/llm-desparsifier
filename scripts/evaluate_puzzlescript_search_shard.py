#!/usr/bin/env python3
"""Evaluate one CPU shard from a batched PuzzleScript GEPA manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from run_puzzlescript_batched_gepa import evaluate_manifest_shard


def _optional_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    env_manifest = os.environ.get("EVAL_MANIFEST")
    parser = argparse.ArgumentParser(description="Evaluate one PuzzleScript search shard.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(env_manifest) if env_manifest else None,
        required=env_manifest is None,
    )
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--array-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slurm_task_id = _optional_env_int("SLURM_ARRAY_TASK_ID")
    slurm_task_min = _optional_env_int("SLURM_ARRAY_TASK_MIN") or 0
    slurm_task_count = _optional_env_int("SLURM_ARRAY_TASK_COUNT")
    search_array_count = _optional_env_int("SEARCH_ARRAY_COUNT")

    raw_task_id = args.array_index if args.array_index is not None else (slurm_task_id or 0)
    array_index = raw_task_id if args.array_index is not None else raw_task_id - slurm_task_min
    array_count = (
        args.array_count
        if args.array_count is not None
        else (search_array_count or slurm_task_count or 1)
    )
    shard_path = evaluate_manifest_shard(
        manifest_path=args.manifest,
        array_index=array_index,
        array_count=array_count,
    )
    print(f"[search-shard] wrote {shard_path}")


if __name__ == "__main__":
    main()
