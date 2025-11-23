#!/bin/bash
#SBATCH --job-name=gepa_onpolicy
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

STATE_ROOT="${STATE_ROOT:-$PWD/artifacts/gepa_state}"

uv sync
uv run scripts/run_reward_batch.py --state-root "$STATE_ROOT"
