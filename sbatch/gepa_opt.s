#!/bin/bash
#SBATCH --job-name=gepa_opt
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export JAX_PLATFORMS=cpu
STATE_ROOT="${STATE_ROOT:-$PWD/artifacts/gepa_state}"

uv sync
uv run python scripts/run_gepa_opt.py --state-root "$STATE_ROOT"
