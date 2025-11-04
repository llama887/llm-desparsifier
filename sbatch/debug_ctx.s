#!/bin/bash
#SBATCH --job-name=ctx_probe
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Keep benchmark assets alongside the project.
export XLAND_MINIGRID_DATA="$PWD/.xland_minigrid"
mkdir -p "$XLAND_MINIGRID_DATA"

# Local cache to avoid polluting $HOME on shared nodes.
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$XDG_CACHE_HOME"

# Ensure dependencies match pyproject/uv lock.
uv sync

# Run the lightweight context probe.
uv run python tests/debug_ctx.py
