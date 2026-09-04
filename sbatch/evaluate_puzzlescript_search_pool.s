#!/bin/bash
#
#SBATCH --job-name=puzzlescript-search-pool
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/llm-desparsifier
#SBATCH --output=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%A_%a.out
#SBATCH --error=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${SEARCH_POOL_DIR:?SEARCH_POOL_DIR is required}"
SEARCH_POOL_SIZE="${SEARCH_POOL_SIZE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SEARCH_ARRAY_SCRIPT="${SEARCH_ARRAY_SCRIPT:-sbatch/evaluate_puzzlescript_search_array.s}"
POLL_S="${SEARCH_POOL_POLL_S:-2}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
READY_DIR="$SEARCH_POOL_DIR/ready"
mkdir -p "$READY_DIR" sbatch/logs
touch "$READY_DIR/$INDEX"

last_manifest=""
while [ ! -e "$SEARCH_POOL_DIR/stop" ]; do
    if [ -s "$SEARCH_POOL_DIR/current_manifest" ]; then
        IFS= read -r manifest < "$SEARCH_POOL_DIR/current_manifest"
        if [ -n "$manifest" ] && [ "$manifest" != "$last_manifest" ]; then
            EVAL_MANIFEST="$manifest" \
            SEARCH_ARRAY_COUNT="$SEARCH_POOL_SIZE" \
            bash "$SEARCH_ARRAY_SCRIPT"
            last_manifest="$manifest"
        fi
    fi
    sleep "$POLL_S"
done

echo "[search-pool] worker=$INDEX stopped"
