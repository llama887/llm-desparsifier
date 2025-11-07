"""Legacy entry point retained for manual runs.

This script now delegates to the refactored RL pipeline API.
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Callable, Optional

from llm_desparsifier.rewards import RewardGenerator
from llm_desparsifier.rl.pipeline import TrainingResult, run_dense_and_sparse

DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "artifacts", "baseline_run")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL training with LLM-shaped rewards.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for logs, plots, and generated reward snapshots (default: %(default)s).",
    )
    parser.add_argument(
        "--env-id",
        default="XLand-MiniGrid-R4-9x9",
        help="Environment ID passed to xminigrid.make.",
    )
    parser.add_argument(
        "--benchmark-id",
        default="trivial-1m",
        help="Benchmark identifier used for sampling rulesets.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000_000,
        help="Total timesteps to train (distributed across devices).",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Random seed for the training rollout RNG.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Random seed for evaluation rollouts.",
    )
    parser.add_argument(
        "--ctx-fn",
        default=None,
        help=(
            "Optional override for the context function as 'package.module:function'. "
            "Defaults to auto-selection based on env id."
        ),
    )
    parser.add_argument(
        "--compare-dense-vs-sparse",
        action="store_true",
        help=(
            "If set, sequentially trains dense- and sparse-reward agents in one job and "
            "stores their artifacts under OUTPUT_DIR/dense and OUTPUT_DIR/sparse."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("dense", "sparse"),
        default="dense",
        help=(
            "Reward mode to run when --compare-dense-vs-sparse is not set. "
            "Defaults to dense (LLM-shaped)."
        ),
    )
    return parser.parse_args()


def _maybe_resolve_ctx_fn(dotted: Optional[str]) -> Optional[Callable]:
    if not dotted:
        return None
    if ":" not in dotted:
        raise ValueError("--ctx-fn must be of the form 'package.module:function'")
    module_name, func_name = dotted.rsplit(":", 1)
    module = importlib.import_module(module_name)
    ctx_fn = getattr(module, func_name, None)
    if ctx_fn is None:
        raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'")
    return ctx_fn


def main():
    args = _parse_args()
    output_dir = os.path.abspath(args.output_dir)
    reward_generator = RewardGenerator()

    config_override = {
        "env_id": args.env_id,
        "benchmark_id": args.benchmark_id,
        "total_timesteps": args.total_timesteps,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
    }

    ctx_fn_override = _maybe_resolve_ctx_fn(args.ctx_fn)

    results = run_dense_and_sparse(
        reward_generator,
        output_dir=output_dir,
        ctx_fn=ctx_fn_override,
        config_override=config_override,
        compare_dense_vs_sparse=args.compare_dense_vs_sparse,
        default_reward_mode=args.reward_mode,
    )

    if len(results) == 1:
        result = results[0]
        print("Training complete (mode=", result.reward_mode, "). Key metrics:", result.final_metrics, sep="")
        print("Artifacts:", result.artifacts)
    else:
        for result in results:
            print(
                f"Training complete (mode={result.reward_mode}). Key metrics: {result.final_metrics}"
            )
            print("Artifacts:", result.artifacts)


if __name__ == "__main__":
    main()
