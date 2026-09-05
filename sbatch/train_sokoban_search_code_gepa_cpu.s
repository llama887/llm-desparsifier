#!/bin/bash
#SBATCH --job-name=cx-sk-search-code
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
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
# Per-search heap cap. A generated artifact that blows past this fails as one
# scored task instead of OOM-killing the controller or an array worker.
export LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB="${LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB:-3072}"
[ -x "$SD_PATH/.venv/bin/python" ] || { echo "missing script-doctor venv" >&2; exit 2; }
command -v codex >/dev/null || { echo "codex CLI not found" >&2; exit 2; }
codex login status

MODE="${CONFIG:-smoke}"
case "$MODE" in smoke|train|full) ;; *) echo "CONFIG must be smoke, train, or full" >&2; exit 2;; esac
RUN_STATE_ROOT="${STATE_ROOT:?STATE_ROOT is required}"
POOL_SIZE="${ARRAY_SIZE:-$([ "$MODE" = smoke ] && echo 8 || echo 64)}"
# SLURM_JOB_PARTITION can report a partition alias that is not itself a
# valid submission target, so allow an explicit override.
POOL_PARTITION="${POOL_PARTITION:-${SLURM_JOB_PARTITION:?missing controller partition}}"
if [ "$MODE" = smoke ] || [ "$POOL_PARTITION" = cpu_short ]; then
    POOL_TIME=04:00:00
else
    POOL_TIME=2-00:00:00
fi
POOL_DIR="$RUN_STATE_ROOT/search_pool_${SLURM_JOB_ID}"
mkdir -p "$RUN_STATE_ROOT" "$POOL_DIR/ready"
printf '{"holdout_in_optimizer":false,"reflection_shell_tools":false,"synthesis_workspace":"temporary","eval_jobs_used_after_gepa_only":true}\n' > "$RUN_STATE_ROOT/holdout_boundary_audit.json"

cleanup() {
    # scancel first, and let nothing here abort the handler. The previous order
    # created the stop file first, so when the controller died of an exhausted
    # disk quota the touch failed, set -e aborted cleanup, and the pool array
    # outlived its controller by ~16 hours. A cleanup path must never depend on
    # the resource whose exhaustion triggered it.
    if [ -n "${POOL_JOB_ID:-}" ]; then
        scancel "$POOL_JOB_ID" 2>/dev/null || true
    fi
    touch "$POOL_DIR/stop" 2>/dev/null || true
}
trap cleanup EXIT

POOL_JOB_ID=$(sbatch --parsable \
    --job-name="cx-sk-pool-${SLURM_JOB_ID}" \
    --partition="$POOL_PARTITION" \
    --array="0-$((POOL_SIZE - 1))%$POOL_SIZE" \
    --cpus-per-task=1 --mem=6G --time="$POOL_TIME" \
    --export="ALL,SEARCH_POOL_DIR=$POOL_DIR,SEARCH_POOL_SIZE=$POOL_SIZE,SEARCH_ARRAY_SCRIPT=sbatch/evaluate_puzzlescript_search_array.s" \
    sbatch/evaluate_puzzlescript_search_pool.s)
echo "[search-pool] job_id=$POOL_JOB_ID size=$POOL_SIZE"
deadline=$((SECONDS + 21600))
while [ "$(find "$POOL_DIR/ready" -type f | wc -l)" -lt "$POOL_SIZE" ]; do
    [ "$SECONDS" -lt "$deadline" ] || { echo "search pool did not become ready" >&2; exit 2; }
    sleep 10
done

COMMON_ARGS=(
    --env-grid "${ENV_GRID:-configs/gepa_puzzlescript_envs.yaml}"
    --script-doctor "$SD_PATH"
    --astar-timeout-s 30
    --llm-timeout-s 900
    --llm-concurrency "${LLM_CONCURRENCY:-32}"
    --synthesis-backend codex-cli
    --synthesis-codex-model "${LUNA_MODEL:-gpt-5.6-sol}"
    --synthesis-agentic
    --codex-reasoning-effort "${LUNA_EFFORT:-high}"
    --search-array-count "$POOL_SIZE"
    --search-array-concurrency "$POOL_SIZE"
    --search-pool-dir "$POOL_DIR"
    --search-poll-interval-s 2
    --search-array-stall-timeout-s 600
)

BLIND_REFERENCE="${BLIND_REFERENCE:-configs/puzzlescript_blind_reference.json}"
[ -f "$BLIND_REFERENCE" ] || { echo "missing blind reference: $BLIND_REFERENCE" >&2; exit 2; }

RUN_ARGS=(
    --state-root "$RUN_STATE_ROOT"
    --blind-reference "$BLIND_REFERENCE"
    --blind-budget-multiplier "${BLIND_BUDGET_MULTIPLIER:-2}"
    --objective "${OBJECTIVE:-blind-relative-time}"
    --sibling-level-holdout
    --synthesis-cache-dir "${SYNTHESIS_CACHE_DIR:-artifacts/synthesis_cache}"
    --require-blind-reference
    --reflection-backend codex-cli
    --codex-model "${GEPA_MODEL:-gpt-5.6-sol}"
    --no-reflection-artifact-tools
    --optimize-full-prompt
    --min-reference-seconds "${MIN_REFERENCE_SECONDS:-1.0}"
    --unsolved-log2 "${UNSOLVED_LOG2:--3.0}"
    --speedup-clip "${SPEEDUP_CLIP:-14.0}"
    --slow-solve-clip "${SLOW_SOLVE_CLIP:-2.0}"
    --seed "${SEED:-0}"
)

# Levels blind search ran out of budget on are the ones a smarter search has to
# win. Measuring only what plain A* already solves cannot show that novelty or
# diversity-driven search helps, because on those levels A* is the right answer.
if [ "${INCLUDE_FRONTIER:-1}" = "1" ]; then
    RUN_ARGS+=(--include-frontier-levels)
fi

if [ "$MODE" = smoke ]; then
    RUN_ARGS+=(
        --training-targets-file "${TRAINING_TARGETS:-configs/gepa_overfit_no_right_turn_sokoban_level_03.json}"
        --max-gepa-expansions-per-level 50000
        --synthesis-replicates "${REPLICATES:-2}"
        --val-split train
        --max-gepa-iterations "${ITERATIONS:-2}"
    )
else
    RUN_ARGS+=(
        --levels-per-game "${LEVELS_PER_GAME:-0}"
        --max-gepa-expansions-per-level "${FALLBACK_EXPANSIONS:-10000}"
        --synthesis-replicates "${REPLICATES:-5}"
        --val-split dev
        --dev-fraction "${DEV_FRACTION:-0.4}"
        --reflection-minibatch-size "${MINIBATCH:-24}"
        --max-gepa-iterations "${ITERATIONS:-20}"
    )
fi

"$SD_PATH/.venv/bin/python" -u scripts/run_puzzlescript_batched_gepa.py \
    "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"

if [ "$MODE" = full ]; then
    "$SD_PATH/.venv/bin/python" -u scripts/compare_puzzlescript_batched_prompts.py \
        "${COMMON_ARGS[@]}" \
        --state-root "$RUN_STATE_ROOT/untouched_holdout" \
        --optimized-prompt "$RUN_STATE_ROOT/best_prompt.txt" \
        --max-expansions "${MAX_EXPANSIONS:-50000}" \
        --synthesis-replicates 10
fi
