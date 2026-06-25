#!/bin/bash
#
#SBATCH --job-name=puzzlescript-baselines
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=FAIL
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
    BASE_STATE_ROOT="$STATE_ROOT"
    echo "[run] Using explicit STATE_ROOT=$BASE_STATE_ROOT"
else
    BASE_STATE_ROOT="$PWD/artifacts/gepa_puzzlescript_state_baselines_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
    echo "[run] Using fresh STATE_ROOT=$BASE_STATE_ROOT"
fi

BASELINE_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$BASE_STATE_ROOT/dspy_cache/baseline-${BASELINE_TASK_ID}}"
export DSPY_DISABLE_DISK_CACHE="${DSPY_DISABLE_DISK_CACHE:-1}"
mkdir -p "$DSPY_CACHEDIR"
echo "[run] DSPy cache: $DSPY_CACHEDIR"
echo "[run] DSPY_DISABLE_DISK_CACHE=$DSPY_DISABLE_DISK_CACHE"

"$SD_PATH/.venv/bin/python" scripts/prepare_puzzlescript_baselines.py \
    --env-grid configs/gepa_puzzlescript_envs.yaml \
    --state-root "$BASE_STATE_ROOT" \
    --max-expansions "${MAX_EXPANSIONS:-50000}" \
    --max-gepa-expansions-per-level "${MAX_GEPA_EXPANSIONS_PER_LEVEL:-10000}" \
    --astar-timeout-s "${ASTAR_TIMEOUT_S:-30}" \
    --levels-per-game "${LEVELS_PER_GAME:-0}" \
    --llm "deepseek/deepseek-v4-pro" \
    --llm-max-tokens "${LLM_MAX_TOKENS:-384000}" \
    --script-doctor "$SD_PATH"
