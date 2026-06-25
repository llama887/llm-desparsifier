#!/bin/bash
#
#SBATCH --job-name=puzzlescript-gepa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%2
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/llm-desparsifier
#SBATCH --output=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%A_%a.out
#SBATCH --error=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$SLURM_SUBMIT_DIR"
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
            "dspy>=3.0.3"
        cd "$SLURM_SUBMIT_DIR"
    fi

    if ! ls "$SD_PATH"/puzzlescript_cpp/_puzzlescript_cpp*.so &>/dev/null; then
        echo "[setup] Building C++ PuzzleScript extension..."
        cd "$SD_PATH"
        .venv/bin/python setup_cpp.py build_ext --inplace
        cd "$SLURM_SUBMIT_DIR"
    fi

    mkdir -p "$SD_PATH/data/game_trees" "$SD_PATH/data/pretty_trees" "$SD_PATH/data/simplified_games"
) 9>"$SETUP_LOCK"

export PATH="$NODE_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

if [ -n "${STATE_ROOT:-}" ]; then
    GROUP_STATE_ROOT="$STATE_ROOT"
    echo "[run] Using explicit STATE_ROOT group=$GROUP_STATE_ROOT"
else
    GROUP_STATE_ROOT="$PWD/artifacts/gepa_puzzlescript_state_replicas_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
    echo "[run] Using fresh STATE_ROOT group=$GROUP_STATE_ROOT"
fi

BASELINE_STATE_ROOT="${BASELINE_ROOT:-$GROUP_STATE_ROOT}"
REPLICA_ID="${SLURM_ARRAY_TASK_ID:-0}"
REPLICA_STATE_ROOT="$GROUP_STATE_ROOT/gepa_replicas/replica-${REPLICA_ID}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$REPLICA_STATE_ROOT/dspy_cache}"
export DSPY_DISABLE_DISK_CACHE="${DSPY_DISABLE_DISK_CACHE:-1}"
mkdir -p "$DSPY_CACHEDIR"

echo "[run] Replica $REPLICA_ID state root: $REPLICA_STATE_ROOT"
echo "[run] Shared baseline root: $BASELINE_STATE_ROOT"
echo "[run] GEPA threads: ${GEPA_NUM_THREADS:-8} allocated cpus: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[run] DSPy cache: $DSPY_CACHEDIR"
echo "[run] DSPY_DISABLE_DISK_CACHE=$DSPY_DISABLE_DISK_CACHE"

"$SD_PATH/.venv/bin/python" scripts/run_puzzlescript_batch.py \
    --env-grid configs/gepa_puzzlescript_envs.yaml \
    --state-root "$REPLICA_STATE_ROOT" \
    --baseline-root "$BASELINE_STATE_ROOT" \
    --max-phase-iterations "${MAX_PHASE_ITERATIONS:-10}" \
    --max-expansions "${MAX_EXPANSIONS:-50000}" \
    --max-gepa-expansions-per-level "${MAX_GEPA_EXPANSIONS_PER_LEVEL:-10000}" \
    --astar-timeout-s "${ASTAR_TIMEOUT_S:-30}" \
    --levels-per-game "${LEVELS_PER_GAME:-0}" \
    --gepa-num-threads "${GEPA_NUM_THREADS:-8}" \
    --llm "deepseek/deepseek-v4-pro" \
    --llm-max-tokens "${LLM_MAX_TOKENS:-384000}" \
    --script-doctor "$SD_PATH"
