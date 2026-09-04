#!/bin/bash
#
#SBATCH --job-name=gallery-gepa-codex
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/llm-desparsifier
#SBATCH --output=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs
[ -f .env ] && { set -a; source .env; set +a; }

SD_PATH="${SLURM_SUBMIT_DIR}/../script-doctor"
NODE_DIR="/scratch/fyy2003/node-v20.18.0-linux-x64"
export PATH="$SD_PATH/.venv/bin:$NODE_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

[ -x "$SD_PATH/.venv/bin/python" ] || { echo "missing $SD_PATH/.venv" >&2; exit 2; }
command -v codex >/dev/null || { echo "codex CLI not found" >&2; exit 2; }
codex login status

RUN_STATE_ROOT="${STATE_ROOT:-$PWD/artifacts/gepa_gallery_luna_agentic_sol_${SLURM_JOB_ID}}"
ENV_GRID="${ENV_GRID:-configs/gepa_puzzlescript_gallery_random_20260723.yaml}"
POOL_SIZE="${SEARCH_POOL_SIZE:-96}"
POOL_DIR="$RUN_STATE_ROOT/search_pool_${SLURM_JOB_ID}"
mkdir -p "$RUN_STATE_ROOT" "$POOL_DIR/ready"

cleanup() {
    touch "$POOL_DIR/stop"
    [ -n "${POOL_JOB_ID:-}" ] && scancel "$POOL_JOB_ID" 2>/dev/null || true
}
trap cleanup EXIT

POOL_JOB_ID=$(sbatch --parsable \
    --array="0-$((POOL_SIZE - 1))%$POOL_SIZE" \
    --cpus-per-task="${SEARCH_CPUS_PER_WORKER:-1}" \
    --mem="${SEARCH_WORKER_MEM:-2G}" \
    --export="ALL,SEARCH_POOL_DIR=$POOL_DIR,SEARCH_POOL_SIZE=$POOL_SIZE,SEARCH_ARRAY_SCRIPT=sbatch/evaluate_puzzlescript_search_array.s" \
    sbatch/evaluate_puzzlescript_search_pool.s)
echo "[search-pool] job_id=$POOL_JOB_ID size=$POOL_SIZE cpus_per_worker=${SEARCH_CPUS_PER_WORKER:-1}"

deadline=$((SECONDS + ${SEARCH_POOL_START_TIMEOUT_S:-1800}))
while [ "$(find "$POOL_DIR/ready" -type f | wc -l)" -lt "$POOL_SIZE" ]; do
    [ "$SECONDS" -lt "$deadline" ] || { echo "search pool did not become ready" >&2; exit 2; }
    sleep 10
done
echo "[search-pool] all workers ready"

COMMON_ARGS=(
    --env-grid "$ENV_GRID"
    --script-doctor "$SD_PATH"
    --levels-per-game "${LEVELS_PER_GAME:-3}"
    --astar-timeout-s "${ASTAR_TIMEOUT_S:-120}"
    --llm-timeout-s "${LLM_TIMEOUT_S:-900}"
    --llm-concurrency "${LLM_CONCURRENCY:-8}"
    --synthesis-backend codex-cli
    --synthesis-codex-model "${SYNTHESIS_CODEX_MODEL:-gpt-5.6-luna}"
    --synthesis-agentic
    --codex-executable "${CODEX_EXECUTABLE:-codex}"
    --codex-reasoning-effort "${CODEX_REASONING_EFFORT:-high}"
    --search-array-script sbatch/evaluate_puzzlescript_search_array.s
    --search-array-count "$POOL_SIZE"
    --search-array-concurrency "$POOL_SIZE"
    --search-pool-dir "$POOL_DIR"
    --search-poll-interval-s "${SEARCH_POLL_INTERVAL_S:-2}"
    --search-array-stall-timeout-s "${SEARCH_ARRAY_STALL_TIMEOUT_S:-600}"
)

"$SD_PATH/.venv/bin/python" -u scripts/run_puzzlescript_batched_gepa.py \
    "${COMMON_ARGS[@]}" \
    --state-root "$RUN_STATE_ROOT" \
    --max-gepa-expansions-per-level "${MAX_GEPA_EXPANSIONS_PER_LEVEL:-50000}" \
    --reflection-backend codex-cli \
    --codex-model "${CODEX_MODEL:-gpt-5.6-sol}" \
    --reflection-artifact-tools \
    --optimize-full-prompt \
    --synthesis-replicates "${SYNTHESIS_REPLICATES:-5}" \
    --lost-solve-penalty "${LOST_SOLVE_PENALTY:-20}" \
    --new-solve-bonus "${NEW_SOLVE_BONUS:-20}" \
    --score-delta-weight 0 \
    --common-solve-efficiency-clip 0.25 \
    --global-lost-solve-gate-penalty 0 \
    --global-net-solve-loss-gate-penalty 0 \
    --val-split dev \
    --dev-fraction "${DEV_FRACTION:-0.33}" \
    --max-gepa-iterations "${MAX_GEPA_ITERATIONS:-20}" \
    --max-metric-calls "${MAX_METRIC_CALLS:-0}"

"$SD_PATH/.venv/bin/python" -u scripts/compare_puzzlescript_batched_prompts.py \
    "${COMMON_ARGS[@]}" \
    --state-root "$RUN_STATE_ROOT/holdout" \
    --optimized-prompt "$RUN_STATE_ROOT/best_prompt.txt" \
    --max-expansions "${MAX_EXPANSIONS:-50000}" \
    --synthesis-replicates "${HOLDOUT_SYNTHESIS_REPLICATES:-5}"
