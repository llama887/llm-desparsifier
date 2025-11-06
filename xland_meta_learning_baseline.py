"""Legacy entry point retained for manual runs.

This script now delegates to the refactored RL pipeline API.
"""

from __future__ import annotations

import os
import argparse

from llm_desparsifier.rewards import RewardGenerator
from llm_desparsifier.rl.pipeline import TrainingResult, run_training_with_reward
from llm_desparsifier.utils import extract_xland_ctx

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
    return parser.parse_args()


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

    result: TrainingResult = run_training_with_reward(
        reward_generator,
        output_dir=output_dir,
        ctx_fn=extract_xland_ctx,
        config_override=config_override,
    )

    print("Training complete. Key metrics:", result.final_metrics)
    print("Artifacts:", result.artifacts)


if __name__ == "__main__":
    main()
