#!/bin/bash
#SBATCH --job-name=cx-sk-calibrate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/llm-desparsifier
#SBATCH --output=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/search_code/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/llm-desparsifier/sbatch/logs/search_code/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/search_code

SD_PATH="$SLURM_SUBMIT_DIR/../script-doctor"
NODE_DIR="/scratch/fyy2003/node-v20.18.0-linux-x64"
export PATH="$SD_PATH/.venv/bin:$NODE_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

# Each worker self-limits, so a pathological level fails as one measurement
# instead of taking the whole calibration down. 16 workers x 3 GiB < 64 GiB.
export LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB="${LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB:-3072}"

[ -x "$SD_PATH/.venv/bin/python" ] || { echo "missing script-doctor venv" >&2; exit 2; }

STATE_ROOT="${STATE_ROOT:-artifacts/blind_calibration_$(date +%Y%m%d)}"
OUT="${OUT:-configs/puzzlescript_blind_reference.json}"
MAX_EXPANSIONS="${MAX_EXPANSIONS:-50000}"

echo "[calibrate] state_root=$STATE_ROOT out=$OUT ceiling=$MAX_EXPANSIONS"

"$SD_PATH/.venv/bin/python" -u scripts/calibrate_puzzlescript_budgets.py \
    --env-grid configs/gepa_puzzlescript_envs.yaml \
    --script-doctor "$SD_PATH" \
    --state-root "$STATE_ROOT" \
    --out "$OUT" \
    --levels-per-game 0 \
    --max-expansions "$MAX_EXPANSIONS" \
    --astar-timeout-s 600 \
    --task-wall-timeout-s 900 \
    --array-count 64 \
    --local-workers 16 \
    --include-eval-jobs
