#!/bin/bash
#SBATCH --job-name=cx-sk-search-code-holdout
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/llm-desparsifier
#SBATCH --output=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/search_code/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/search_code/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/search_code
[ -f .env ] && { set -a; source .env; set +a; }

SD_PATH="$SLURM_SUBMIT_DIR/../script-doctor"
NODE_DIR="/scratch/fyy2003/node-v20.18.0-linux-x64"
export PATH="$SD_PATH/.venv/bin:$NODE_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1
[ -x "$SD_PATH/.venv/bin/python" ] || { echo "missing script-doctor venv" >&2; exit 2; }
command -v codex >/dev/null || { echo "codex CLI not found" >&2; exit 2; }
codex login status

RUN_STATE_ROOT="${STATE_ROOT:?STATE_ROOT is required}"
HOLDOUT_ROOT="${OUTPUT_DIR:?OUTPUT_DIR is required}"
POOL_SIZE="${ARRAY_SIZE:-64}"
POOL_DIR="$HOLDOUT_ROOT/search_pool_${SLURM_JOB_ID}"
mkdir -p "$HOLDOUT_ROOT" "$POOL_DIR/ready"

cleanup() {
    touch "$POOL_DIR/stop"
    [ -n "${POOL_JOB_ID:-}" ] && scancel "$POOL_JOB_ID" 2>/dev/null || true
}
trap cleanup EXIT

POOL_JOB_ID=$(sbatch --parsable \
    --job-name="cx-sk-pool-${SLURM_JOB_ID}" \
    --array="0-$((POOL_SIZE - 1))%$POOL_SIZE" \
    --cpus-per-task=1 --mem=2G \
    --export="ALL,SEARCH_POOL_DIR=$POOL_DIR,SEARCH_POOL_SIZE=$POOL_SIZE,SEARCH_ARRAY_SCRIPT=sbatch/evaluate_puzzlescript_search_array.s" \
    sbatch/evaluate_puzzlescript_search_pool.s)
echo "[search-pool] job_id=$POOL_JOB_ID size=$POOL_SIZE"
deadline=$((SECONDS + 1800))
while [ "$(find "$POOL_DIR/ready" -type f | wc -l)" -lt "$POOL_SIZE" ]; do
    [ "$SECONDS" -lt "$deadline" ] || { echo "search pool did not become ready" >&2; exit 2; }
    sleep 10
done

"$SD_PATH/.venv/bin/python" -u scripts/compare_puzzlescript_batched_prompts.py \
    --env-grid configs/gepa_puzzlescript_envs.yaml \
    --script-doctor "$SD_PATH" \
    --state-root "$HOLDOUT_ROOT" \
    --optimized-prompt "$RUN_STATE_ROOT/best_prompt.txt" \
    --max-expansions "${MAX_EXPANSIONS:-50000}" \
    --astar-timeout-s 30 \
    --llm-timeout-s 900 \
    --llm-concurrency 4 \
    --synthesis-replicates "${REPLICATES:-10}" \
    --synthesis-backend codex-cli \
    --synthesis-codex-model "${LUNA_MODEL:-gpt-5.6-luna}" \
    --synthesis-agentic \
    --codex-reasoning-effort "${LUNA_EFFORT:-high}" \
    --search-array-count "$POOL_SIZE" \
    --search-array-concurrency "$POOL_SIZE" \
    --search-pool-dir "$POOL_DIR" \
    --search-poll-interval-s 2 \
    --search-array-stall-timeout-s 600
