#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p sbatch/logs artifacts

STATE_ROOT="${STATE_ROOT:-$PWD/artifacts/gepa_puzzlescript_parallel_$(date +%Y%m%d_%H%M%S)}"
BASELINE_ROOT="${BASELINE_ROOT:-$STATE_ROOT}"
BASELINE_ARRAY_SIZE="${BASELINE_ARRAY_SIZE:-24}"
BASELINE_ARRAY_CONCURRENCY="${BASELINE_ARRAY_CONCURRENCY:-4}"
GEPA_SHARD_COUNT="${GEPA_SHARD_COUNT:-${GEPA_REPLICA_COUNT:-4}}"
GEPA_SHARD_CONCURRENCY="${GEPA_SHARD_CONCURRENCY:-${GEPA_REPLICA_CONCURRENCY:-2}}"
GEPA_REPLICA_COUNT="$GEPA_SHARD_COUNT"
GEPA_REPLICA_CONCURRENCY="$GEPA_SHARD_CONCURRENCY"
GEPA_NUM_THREADS="${GEPA_NUM_THREADS:-8}"
GEPA_CPUS_PER_TASK="${GEPA_CPUS_PER_TASK:-$GEPA_NUM_THREADS}"
GEPA_MEM="${GEPA_MEM:-96G}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"

if [ "$GEPA_REPLICA_COUNT" -lt 1 ]; then
    echo "GEPA_REPLICA_COUNT must be >= 1" >&2
    exit 1
fi
if [ "$GEPA_REPLICA_CONCURRENCY" -lt 1 ]; then
    echo "GEPA_REPLICA_CONCURRENCY must be >= 1" >&2
    exit 1
fi
if [ "$GEPA_NUM_THREADS" -lt 1 ]; then
    echo "GEPA_NUM_THREADS must be >= 1" >&2
    exit 1
fi
if [ "$GEPA_CPUS_PER_TASK" -lt 1 ]; then
    echo "GEPA_CPUS_PER_TASK must be >= 1" >&2
    exit 1
fi

gepa_max=$((GEPA_REPLICA_COUNT - 1))
gepa_array_spec="0-${gepa_max}%${GEPA_REPLICA_CONCURRENCY}"

echo "[submit] STATE_ROOT group=$STATE_ROOT"
echo "[submit] BASELINE_ROOT=$BASELINE_ROOT"
echo "[submit] GEPA shards=$gepa_array_spec"
echo "[submit] GEPA threads/shard=$GEPA_NUM_THREADS cpus/shard=$GEPA_CPUS_PER_TASK mem/shard=$GEPA_MEM"

dependency_args=()
if [ "$SKIP_BASELINES" != "1" ]; then
    if [ "$BASELINE_ARRAY_SIZE" -lt 1 ]; then
        echo "BASELINE_ARRAY_SIZE must be >= 1" >&2
        exit 1
    fi
    if [ "$BASELINE_ARRAY_CONCURRENCY" -lt 1 ]; then
        echo "BASELINE_ARRAY_CONCURRENCY must be >= 1" >&2
        exit 1
    fi
    baseline_max=$((BASELINE_ARRAY_SIZE - 1))
    baseline_array_spec="0-${baseline_max}%${BASELINE_ARRAY_CONCURRENCY}"
    echo "[submit] baseline array=$baseline_array_spec"
    baseline_job=$(
        STATE_ROOT="$BASELINE_ROOT" sbatch --parsable \
            --array="$baseline_array_spec" \
            sbatch/prepare_puzzlescript_baselines_array.s
    )
    baseline_job_id="${baseline_job%%;*}"
    dependency_args=(--dependency="afterok:${baseline_job_id}")
    echo "[submit] baseline job: $baseline_job"
else
    echo "[submit] SKIP_BASELINES=1; GEPA replicas will read existing baseline cache/shards."
fi

gepa_job=$(
    STATE_ROOT="$STATE_ROOT" BASELINE_ROOT="$BASELINE_ROOT" GEPA_NUM_THREADS="$GEPA_NUM_THREADS" sbatch --parsable \
        "${dependency_args[@]}" \
        --cpus-per-task="$GEPA_CPUS_PER_TASK" \
        --mem="$GEPA_MEM" \
        --array="$gepa_array_spec" \
        sbatch/train_puzzlescript_gepa_array.s
)
echo "[submit] GEPA replica array job: $gepa_job"
echo "[submit] Replica state roots will be under $STATE_ROOT/gepa_replicas/"
