# PuzzleScript Slurm Arrays

Use the array path when local runs spend too long in baseline setup. The
baseline array computes blind, built-in, and base-prompt baselines for disjoint
game/level subsets, writes shards under one `STATE_ROOT`, then GEPA reads the
merged baseline cache and only recomputes missing levels.

```bash
BASELINE_ARRAY_SIZE=24 \
BASELINE_ARRAY_CONCURRENCY=4 \
GEPA_NUM_THREADS=8 \
bash sbatch/submit_puzzlescript_array_pipeline.sh
```

That pipeline runs one GEPA optimizer process after the baseline shards finish.
For more GEPA parallelism, use the replica-array launcher. Each Slurm array task
gets an isolated GEPA state directory and runs its own DSPy GEPA thread pool,
while sharing the same baseline cache/root:

```bash
BASELINE_ARRAY_SIZE=24 \
BASELINE_ARRAY_CONCURRENCY=4 \
GEPA_SHARD_COUNT=4 \
GEPA_SHARD_CONCURRENCY=2 \
GEPA_NUM_THREADS=8 \
bash sbatch/submit_puzzlescript_parallel_gepa_pipeline.sh
```

This is parallel prompt search by independent GEPA replicas. It is intentionally
not multiple processes writing one shared DSPy GEPA `log_dir`, because that
would race on optimizer state and logs. After the array runs, rank the replicas:

```bash
python scripts/summarize_puzzlescript_gepa_replicas.py "$STATE_ROOT" --promote-best
```

To start only GEPA replicas against an existing baseline root:

```bash
STATE_ROOT="$PWD/artifacts/gepa_puzzlescript_parallel_manual" \
BASELINE_ROOT="$PWD/artifacts/gepa_puzzlescript_state_20260618_141528" \
SKIP_BASELINES=1 \
GEPA_SHARD_COUNT=4 \
GEPA_SHARD_CONCURRENCY=2 \
GEPA_NUM_THREADS=8 \
bash sbatch/submit_puzzlescript_parallel_gepa_pipeline.sh
```

Useful knobs:

- `STATE_ROOT`: set this when resuming or when manually submitting baseline and
  GEPA jobs. Both jobs must use the same value.
- `BASELINE_ARRAY_SIZE`: number of baseline shard tasks.
- `BASELINE_ARRAY_CONCURRENCY`: Slurm `%N` throttle. Keep this conservative if
  the base-prompt LLM provider rate limits concurrent requests.
- `GEPA_NUM_THREADS`: thread count passed to DSPy GEPA inside the main job.
  Match this to `--cpus-per-task` in `sbatch/train_puzzlescript_batch.s`.
- `GEPA_SHARD_COUNT`: number of independent GEPA replica tasks for
  `submit_puzzlescript_parallel_gepa_pipeline.sh`.
- `GEPA_SHARD_CONCURRENCY`: Slurm `%N` throttle for GEPA replica tasks.
- `GEPA_CPUS_PER_TASK`: CPU allocation per GEPA replica. Defaults to
  `GEPA_NUM_THREADS`.
- `GEPA_MEM`: memory allocation per GEPA replica. Defaults to `96G`.
- `LEVELS_PER_GAME`: keep `0` for all selected loadable levels, or use a small
  number for fast smoke runs.
- `DSPY_DISABLE_DISK_CACHE`: defaults to `1`. Keep this enabled on the cluster;
  DSPy will use no SQLite disk cache, avoiding `locking protocol` failures.
  Set it to `0` only when running on storage with reliable SQLite locking.
- `DSPY_CACHEDIR`: only used when `DSPY_DISABLE_DISK_CACHE=0`. Each baseline
  shard and GEPA replica gets a private default path; do not point multiple
  workers at one shared DSPy cache directory.

Manual equivalent:

```bash
export STATE_ROOT="$PWD/artifacts/gepa_puzzlescript_state_manual"
baseline_job=$(sbatch --parsable --array=0-23%4 sbatch/prepare_puzzlescript_baselines_array.s)
sbatch --dependency=afterok:${baseline_job%%;*} sbatch/train_puzzlescript_batch.s
```

The main runner writes the merged cache to:

```text
$STATE_ROOT/puzzlescript_baselines.json
```

Array shards live in:

```text
$STATE_ROOT/baseline_shards/
```

If cache settings change, for example `LEVELS_PER_GAME`, expansion caps, timeout,
LLM name, or prompt contract, stale shards are ignored automatically.
