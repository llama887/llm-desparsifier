#!/bin/bash
#
#SBATCH --job-name=puzzlescript-batched-gepa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/llm-desparsifier
#SBATCH --output=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs

if [ -f .env ]; then
    set -a; source .env; set +a
fi

SD_PATH="${SLURM_SUBMIT_DIR}/../script-doctor"
NODE_DIR="/scratch/fyy2003/node-v20.18.0-linux-x64"
NODE_VERSION="v20.18.0"
SETUP_LOCK="$SLURM_SUBMIT_DIR/sbatch/logs/puzzlescript_setup.lock"

(
    flock 9

    if [ ! -f "$NODE_DIR/bin/node" ]; then
        echo "[setup] Downloading Node.js $NODE_VERSION..."
        curl -sL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
            | tar -xJ -C "$(dirname "$NODE_DIR")"
    fi
    export PATH="$NODE_DIR/bin:$PATH"
    echo "[setup] Node.js: $(node --version)"

    if [ ! -d "$SD_PATH" ]; then
        echo "[setup] Cloning script-doctor..."
        git clone https://github.com/smearle/script-doctor.git "$SD_PATH"
    fi

    if [ ! -d "$SD_PATH/PuzzleScript" ]; then
        echo "[setup] Cloning PuzzleScript engine source..."
        git clone https://github.com/increpare/PuzzleScript.git "$SD_PATH/PuzzleScript"
    fi

    if [ ! -d "$SD_PATH/.venv" ]; then
        echo "[setup] Creating script-doctor venv..."
        cd "$SD_PATH"
        uv venv --python 3.12
        uv pip install jax lark numpy py-cpuinfo pybind11 imageio setuptools wheel \
            python-dotenv chex openai tiktoken einops flax hydra-core Pillow javascript pyyaml \
            matplotlib "gepa>=0.0.7" torch
        cd "$SLURM_SUBMIT_DIR"
    else
        uv pip install --python "$SD_PATH/.venv/bin/python" "gepa>=0.0.7" openai pyyaml javascript matplotlib torch
    fi

    if ! ls "$SD_PATH"/puzzlescript_cpp/_puzzlescript_cpp*.so &>/dev/null; then
        echo "[setup] Building C++ PuzzleScript extension..."
        cd "$SD_PATH"
        .venv/bin/python setup_cpp.py build_ext --inplace
        cd "$SLURM_SUBMIT_DIR"
    fi

    mkdir -p "$SD_PATH/data/game_trees" "$SD_PATH/data/pretty_trees" "$SD_PATH/data/simplified_games"
) 9>"$SETUP_LOCK"

export PATH="$SD_PATH/.venv/bin:$NODE_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

if [ -n "${STATE_ROOT:-}" ]; then
    RUN_STATE_ROOT="$STATE_ROOT"
elif [ -n "${RUN_STATE_ROOT:-}" ]; then
    RUN_STATE_ROOT="$RUN_STATE_ROOT"
else
    RUN_STATE_ROOT="$PWD/artifacts/gepa_puzzlescript_batched_${SLURM_JOB_ID:-manual}"
fi
mkdir -p "$RUN_STATE_ROOT"

# Keep model and JIT caches off the home filesystem. vLLM/FlashInfer/Triton can
# write large compiled artifacts during startup, and home quotas are too small
# for first-time H100 model launches.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_STATE_ROOT/xdg_cache}"
export HF_HOME="${HF_HOME:-$RUN_STATE_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$RUN_STATE_ROOT/flashinfer_workspace_base}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$RUN_STATE_ROOT/flashinfer_workspace}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_STATE_ROOT/triton_cache}"
export TORCH_HOME="${TORCH_HOME:-$RUN_STATE_ROOT/torch_home}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUN_STATE_ROOT/torchinductor_cache}"
mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HF_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$FLASHINFER_WORKSPACE_BASE" \
    "$FLASHINFER_WORKSPACE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCH_HOME" \
    "$TORCHINDUCTOR_CACHE_DIR"

# Low-priority top-up work. These defaults follow the Puffer-Soccer heartbeat
# pattern and keep the device visibly active while CPU arrays are running.
export GPU_HEARTBEAT_TARGET_UTILIZATION="${GPU_HEARTBEAT_TARGET_UTILIZATION:-70}"
export GPU_HEARTBEAT_CHECK_INTERVAL="${GPU_HEARTBEAT_CHECK_INTERVAL:-0.2}"
export GPU_HEARTBEAT_MATRIX_SIZE="${GPU_HEARTBEAT_MATRIX_SIZE:-6144}"
export GPU_HEARTBEAT_UTILIZATION_TOLERANCE="${GPU_HEARTBEAT_UTILIZATION_TOLERANCE:-3}"
export GPU_HEARTBEAT_MIN_COMPUTE_SECONDS="${GPU_HEARTBEAT_MIN_COMPUTE_SECONDS:-0.10}"
export GPU_HEARTBEAT_MAX_COMPUTE_SECONDS="${GPU_HEARTBEAT_MAX_COMPUTE_SECONDS:-1.20}"
export GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS="${GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS:-0.03}"
export GPU_HEARTBEAT_MATMULS_PER_CHUNK="${GPU_HEARTBEAT_MATMULS_PER_CHUNK:-8}"
export GPU_HEARTBEAT_DTYPE="${GPU_HEARTBEAT_DTYPE:-bfloat16}"
export GPU_HEARTBEAT_LOG_INTERVAL_S="${GPU_HEARTBEAT_LOG_INTERVAL_S:-60}"

cleanup() {
    if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
        kill "$HEARTBEAT_PID"
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nice -n 19 "$SD_PATH/.venv/bin/python" -u sbatch/gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "[gpu] Started GPU heartbeat with PID: $HEARTBEAT_PID"

export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-openai/gpt-oss-120b}"
if [ -z "${VLLM_PORT:-}" ]; then
    VLLM_PORT_BASE="${VLLM_PORT_BASE:-10000}"
    VLLM_PORT_SPACING="${VLLM_PORT_SPACING:-20}"
    VLLM_PORT_RANGE="${VLLM_PORT_RANGE:-2500}"
    export VLLM_PORT="$((VLLM_PORT_BASE + ((${SLURM_JOB_ID:-0} % VLLM_PORT_RANGE) * VLLM_PORT_SPACING)))"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
if [ -z "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        VLLM_TENSOR_PARALLEL_SIZE="${SLURM_GPUS_ON_NODE:-2}"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _visible_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
        VLLM_TENSOR_PARALLEL_SIZE="${#_visible_gpu_ids[@]}"
    else
        VLLM_TENSOR_PARALLEL_SIZE=2
    fi
    export VLLM_TENSOR_PARALLEL_SIZE
fi

if [ "${START_VLLM:-1}" = "1" ]; then
    if ! command -v vllm >/dev/null 2>&1; then
        if [ "${INSTALL_VLLM:-1}" = "1" ]; then
            echo "[vllm] Installing vLLM into $SD_PATH/.venv"
            uv pip install --python "$SD_PATH/.venv/bin/python" "${VLLM_PACKAGE:-vllm>=0.17.0}"
        else
            echo "[vllm] vllm command not found and INSTALL_VLLM=0" >&2
            exit 2
        fi
    fi

    echo "[vllm] Starting $LOCAL_LLM_MODEL on $OPENAI_BASE_URL"
    vllm serve "$LOCAL_LLM_MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-65536}" \
        --shutdown-timeout "${VLLM_SHUTDOWN_TIMEOUT:-30}" \
        ${VLLM_EXTRA_ARGS:-} &
    VLLM_PID=$!

    "$SD_PATH/.venv/bin/python" - <<'PY'
import os
import time
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
deadline = time.time() + float(os.environ.get("VLLM_STARTUP_TIMEOUT_S", "1800"))
url = base.rsplit("/v1", 1)[0] + "/health"
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f"[vllm] healthy at {url}")
                raise SystemExit(0)
    except Exception as exc:
        print(f"[vllm] waiting for server: {exc}")
        time.sleep(10)
raise SystemExit(f"vLLM did not become healthy before deadline: {url}")
PY
fi

SEARCH_ARRAY_SCRIPT="${SEARCH_ARRAY_SCRIPT:-sbatch/evaluate_puzzlescript_search_array.s}"

echo "[run] state_root=$RUN_STATE_ROOT"
echo "[run] model=$LOCAL_LLM_MODEL base_url=$OPENAI_BASE_URL"
echo "[run] search_array_count=${SEARCH_ARRAY_COUNT:-101} concurrency=${SEARCH_ARRAY_CONCURRENCY:-16}"
echo "[run] search_extra_sbatch_args=${SEARCH_EXTRA_SBATCH_ARGS:-}"
echo "[run] val_split=${VAL_SPLIT:-dev} dev_fraction=${DEV_FRACTION:-0.25}"

"$SD_PATH/.venv/bin/python" -u scripts/run_puzzlescript_batched_gepa.py \
    --env-grid "${ENV_GRID:-configs/gepa_puzzlescript_envs.yaml}" \
    --state-root "$RUN_STATE_ROOT" \
    --script-doctor "$SD_PATH" \
    --levels-per-game "${LEVELS_PER_GAME:-0}" \
    --max-expansions "${MAX_EXPANSIONS:-50000}" \
    --max-gepa-expansions-per-level "${MAX_GEPA_EXPANSIONS_PER_LEVEL:-10000}" \
    --astar-timeout-s "${ASTAR_TIMEOUT_S:-30}" \
    --model "$LOCAL_LLM_MODEL" \
    --openai-base-url "$OPENAI_BASE_URL" \
    --openai-api-key "$OPENAI_API_KEY" \
    --max-model-tokens "${MAX_MODEL_TOKENS:-8192}" \
    --temperature "${LLM_TEMPERATURE:-0.0}" \
    --top-p "${LLM_TOP_P:-0.95}" \
    --llm-timeout-s "${LLM_TIMEOUT_S:-600}" \
    --llm-concurrency "${LLM_CONCURRENCY:-16}" \
    --submit-search-array \
    --search-array-script "$SEARCH_ARRAY_SCRIPT" \
    --search-array-count "${SEARCH_ARRAY_COUNT:-101}" \
    --search-array-concurrency "${SEARCH_ARRAY_CONCURRENCY:-16}" \
    --search-poll-interval-s "${SEARCH_POLL_INTERVAL_S:-15}" \
    --search-array-stall-timeout-s "${SEARCH_ARRAY_STALL_TIMEOUT_S:-300}" \
    --extra-sbatch-args "${SEARCH_EXTRA_SBATCH_ARGS:-}" \
    --val-split "${VAL_SPLIT:-dev}" \
    --dev-fraction "${DEV_FRACTION:-0.25}" \
    --guard-levels "${GUARD_LEVELS:-Aperture_Science_Sokoban_Testing_Initiative:5;Beam_Islands:3;Gravity_Sokoban:6,9;Ice_Cubes:5,7,14,26;Inswaption:7,18;Memory_Push:0,2;Not_Normal_Crates:5,11,12;Sokoban_Dungeon:1;Sokocross:0,5;where_did_all_this_ice_come_from_:3,6}" \
    --max-gepa-iterations "${MAX_GEPA_ITERATIONS:-16}" \
    --max-metric-calls "${MAX_METRIC_CALLS:-0}" \
    --lost-solve-penalty "${LOST_SOLVE_PENALTY:-8.0}" \
    --new-solve-bonus "${NEW_SOLVE_BONUS:-1.0}" \
    --candidate-error-penalty "${CANDIDATE_ERROR_PENALTY:-2.0}" \
    --score-delta-weight "${SCORE_DELTA_WEIGHT:-1.0}" \
    --score-delta-clip "${SCORE_DELTA_CLIP:-0.5}" \
    --partial-progress-weight "${PARTIAL_PROGRESS_WEIGHT:-0.05}" \
    --global-lost-solve-gate-penalty "${GLOBAL_LOST_SOLVE_GATE_PENALTY:-${LOST_SOLVE_PENALTY:-8.0}}" \
    --global-net-solve-loss-gate-penalty "${GLOBAL_NET_SOLVE_LOSS_GATE_PENALTY:-${LOST_SOLVE_PENALTY:-8.0}}" \
    --reflection-minibatch-size "${REFLECTION_MINIBATCH_SIZE:-0}" \
    --initial-gepa-addendum "${INITIAL_GEPA_ADDENDUM:-}" \
    --seed "${GEPA_SEED:-0}"

if [ "${RUN_HOLDOUT_COMPARE:-1}" = "1" ]; then
    BEST_PROMPT_PATH="${BEST_PROMPT_PATH:-$RUN_STATE_ROOT/best_prompt.txt}"
    HOLDOUT_STATE_ROOT="${HOLDOUT_STATE_ROOT:-$RUN_STATE_ROOT/holdout_compare}"
    if [ ! -s "$BEST_PROMPT_PATH" ]; then
        echo "[holdout] missing optimized prompt at $BEST_PROMPT_PATH" >&2
        exit 2
    fi
    mkdir -p "$HOLDOUT_STATE_ROOT"
    echo "[holdout] comparing base prompt against $BEST_PROMPT_PATH"
    echo "[holdout] state_root=$HOLDOUT_STATE_ROOT"
    "$SD_PATH/.venv/bin/python" -u scripts/compare_puzzlescript_batched_prompts.py \
        --env-grid "${ENV_GRID:-configs/gepa_puzzlescript_envs.yaml}" \
        --state-root "$HOLDOUT_STATE_ROOT" \
        --script-doctor "$SD_PATH" \
        --optimized-prompt "$BEST_PROMPT_PATH" \
        --levels-per-game "${HOLDOUT_LEVELS_PER_GAME:-${LEVELS_PER_GAME:-0}}" \
        --max-expansions "${HOLDOUT_MAX_EXPANSIONS:-${MAX_EXPANSIONS:-50000}}" \
        --astar-timeout-s "${HOLDOUT_ASTAR_TIMEOUT_S:-${ASTAR_TIMEOUT_S:-30}}" \
        --model "$LOCAL_LLM_MODEL" \
        --openai-base-url "$OPENAI_BASE_URL" \
        --openai-api-key "$OPENAI_API_KEY" \
        --max-model-tokens "${HOLDOUT_MAX_MODEL_TOKENS:-${MAX_MODEL_TOKENS:-8192}}" \
        --temperature "${HOLDOUT_LLM_TEMPERATURE:-${LLM_TEMPERATURE:-0.0}}" \
        --top-p "${HOLDOUT_LLM_TOP_P:-${LLM_TOP_P:-0.95}}" \
        --llm-timeout-s "${HOLDOUT_LLM_TIMEOUT_S:-${LLM_TIMEOUT_S:-600}}" \
        --llm-concurrency "${HOLDOUT_LLM_CONCURRENCY:-${LLM_CONCURRENCY:-16}}" \
        --submit-search-array \
        --search-array-script "$SEARCH_ARRAY_SCRIPT" \
        --search-array-count "${HOLDOUT_SEARCH_ARRAY_COUNT:-${SEARCH_ARRAY_COUNT:-101}}" \
        --search-array-concurrency "${HOLDOUT_SEARCH_ARRAY_CONCURRENCY:-${SEARCH_ARRAY_CONCURRENCY:-16}}" \
        --search-poll-interval-s "${HOLDOUT_SEARCH_POLL_INTERVAL_S:-${SEARCH_POLL_INTERVAL_S:-15}}" \
        --search-array-stall-timeout-s "${HOLDOUT_SEARCH_ARRAY_STALL_TIMEOUT_S:-${SEARCH_ARRAY_STALL_TIMEOUT_S:-300}}" \
        --extra-sbatch-args "${HOLDOUT_SEARCH_EXTRA_SBATCH_ARGS:-${SEARCH_EXTRA_SBATCH_ARGS:-}}"
fi
