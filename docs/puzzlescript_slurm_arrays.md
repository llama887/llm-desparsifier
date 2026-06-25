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

## Batched Local-LLM GEPA

Use the batched local path when API latency or DSPy internals become the
bottleneck. This path runs one GPU controller job for the whole GEPA run. The
controller keeps a local OpenAI-compatible model endpoint alive, starts the GPU
heartbeat for the entire allocation, batches heuristic synthesis across all
active levels, waits while CPU arrays run A* search, then returns to the GPU for
GEPA reflection/proposal.

```bash
STATE_ROOT="$PWD/artifacts/gepa_puzzlescript_batched_manual" \
LOCAL_LLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct" \
SEARCH_ARRAY_COUNT=101 \
SEARCH_ARRAY_CONCURRENCY=64 \
LLM_CONCURRENCY=16 \
sbatch sbatch/train_puzzlescript_batched_gepa_gpu.s
```

The GPU job submits `sbatch/evaluate_puzzlescript_search_array.s` internally for
each GEPA evaluation batch. Each CPU array task reads the manifest written under
`$STATE_ROOT/candidate_evals/.../search_manifest.json`, evaluates its assigned
level slice, and writes a shard under that evaluation directory. The controller
merges those shards before returning scores and text feedback to standalone
`gepa.optimize`.

Useful knobs:

- `START_VLLM`: defaults to `1`. Set to `0` if you already launched an
  OpenAI-compatible endpoint and only want the controller to call it.
- `INSTALL_VLLM`: defaults to `1` when `START_VLLM=1` and `vllm` is not already
  on `PATH`.
- `VLLM_MAX_MODEL_LEN`: defaults to `32768`, which is safer for single-H100
  smoke runs than the model's full context window.
- `VLLM_EXTRA_ARGS`: appended to `vllm serve`, for example tensor-parallel or
  quantization flags.
- `REFLECTION_MINIBATCH_SIZE`: defaults to `0`, meaning every GEPA reflection
  round evaluates all active levels as one batch.
- `MAX_METRIC_CALLS`: defaults to a budget derived from the number of levels and
  `MAX_GEPA_ITERATIONS`.
- `SEARCH_EXTRA_SBATCH_ARGS`: optional extra arguments appended to the internal
  CPU-array `sbatch` call.
- `HF_HOME`, `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`, and related cache
  variables default under `STATE_ROOT` so first-time model/JIT downloads do not
  hit home-directory quota limits.
- `VLLM_PORT`: defaults to a job-specific high port derived from `SLURM_JOB_ID`
  to avoid node-local collisions; set it explicitly if you need a fixed port.

For local smoke tests without Slurm, omit `--submit-search-array` and point the
controller at an already running compatible endpoint:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=EMPTY \
uv run python scripts/run_puzzlescript_batched_gepa.py \
  --state-root artifacts/gepa_puzzlescript_batched_smoke \
  --script-doctor ../script-doctor \
  --levels-per-game 1 \
  --max-gepa-iterations 1 \
  --search-array-count 4
```
