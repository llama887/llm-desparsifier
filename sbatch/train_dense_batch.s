#!/bin/bash
#SBATCH --job-name=gepa_onpolicy
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo $SLURM_SUBMIT_DIR
mkdir -p logs

export XLAND_MINIGRID_DATA="$PWD/.xland_minigrid"
mkdir -p "$XLAND_MINIGRID_DATA"

export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$XDG_CACHE_HOME"

export WANDB_API_KEY=4eebac5d4dc88793e64cfb18af3233657db3aeda
export WANDB_DATA_DIR=$SCRATCH/wandb_cache
mkdir -p "$WANDB_DATA_DIR"
export WANDB_DIR="${WANDB_DATA_DIR}/runs"
mkdir -p "$WANDB_DIR"


BASE_STATE_ROOT="${STATE_ROOT:-$PWD/artifacts/gepa_state}"

uv sync
uv run scripts/run_reward_batch.py --state-root "${BASE_STATE_ROOT}-2.5pro" --llm "@vertex-ai-3e806d/gemini-2.5-pro"
uv run scripts/run_reward_batch.py --state-root "${BASE_STATE_ROOT}-o3mini" --llm "@o3-mini-5791cb/o3-mini"
