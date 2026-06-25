#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p sbatch/logs artifacts

STATE_ROOT="${STATE_ROOT:-$PWD/artifacts/gepa_puzzlescript_state_$(date +%Y%m%d_%H%M%S)}"
BASELINE_ARRAY_SIZE="${BASELINE_ARRAY_SIZE:-24}"
BASELINE_ARRAY_CONCURRENCY="${BASELINE_ARRAY_CONCURRENCY:-4}"

if [ "$BASELINE_ARRAY_SIZE" -lt 1 ]; then
    echo "BASELINE_ARRAY_SIZE must be >= 1" >&2
    exit 1
fi
if [ "$BASELINE_ARRAY_CONCURRENCY" -lt 1 ]; then
    echo "BASELINE_ARRAY_CONCURRENCY must be >= 1" >&2
    exit 1
fi

array_max=$((BASELINE_ARRAY_SIZE - 1))
array_spec="0-${array_max}%${BASELINE_ARRAY_CONCURRENCY}"

echo "[submit] STATE_ROOT=$STATE_ROOT"
echo "[submit] baseline array=$array_spec"
baseline_job=$(
    STATE_ROOT="$STATE_ROOT" sbatch --parsable \
        --array="$array_spec" \
        sbatch/prepare_puzzlescript_baselines_array.s
)
baseline_job_id="${baseline_job%%;*}"
echo "[submit] baseline job: $baseline_job"

gepa_job=$(
    STATE_ROOT="$STATE_ROOT" sbatch --parsable \
        --dependency="afterok:${baseline_job_id}" \
        sbatch/train_puzzlescript_batch.s
)
echo "[submit] GEPA job: $gepa_job"
echo "[submit] GEPA will merge baseline shards from $STATE_ROOT/baseline_shards"
