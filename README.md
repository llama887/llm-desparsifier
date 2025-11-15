# llm-desparsifier
Prompt optimizing LLMs to desparsify RL environments.

## Quickstart
1. Ensure you have [uv](https://github.com/astral-sh/uv) installed.
2. Sync the environment (only needed after dependency changes):
   ```bash
   uv sync
   ```
3. Launch training (defaults to dense reward only):
   ```bash
   uv run python xland_meta_learning_baseline.py \
     --output-dir artifacts/local-run \
     --reward-mode dense
   ```
4. To run the dense-vs-sparse comparison workflow in a single job, add:
   ```bash
   uv run python xland_meta_learning_baseline.py \
     --output-dir artifacts/compare-run \
     --compare-dense-vs-sparse
   ```

Both runs emit per-mode artifacts under `artifacts/...` including the ground-truth CSV/summary files and the combined `plots/dense_vs_sparse_gt.png` plot.

## SLURM / Cluster Jobs
Use the bundled sbatch script, which defaults to running the comparison workflow:
```bash
sbatch sbatch/train.s
```
Environment variables:
- `OUTPUT_DIR`: override the artifact root (default `artifacts/runs/$SLURM_JOB_ID`).
- `COMPARE_DENSE_VS_SPARSE`: set to `0` to disable the comparison flag.
- `REWARD_MODE`: when comparison is disabled, choose `dense` or `sparse` as the run mode.

The script automatically sets `XLAND_MINIGRID_DATA` and `XDG_CACHE_HOME` inside the job and uses `uv run` for every invocation.

## GEPA Automation Workflow
The GEPA loop runs as two alternating SLURM jobs that read/write shared state under `artifacts/gepa_state`.

### State Layout
- `active_prompt.json`: current `RewardSynthesizer` state (`constraints_text` + DSPy predictor dump). GPU jobs load this automatically.
- `iter-<timestamp>/runs/<job_name>`: per-environment training artifacts, identical to the single-run outputs.
- `iter-<timestamp>/train_dense.jsonl`: JSONL dataset containing env description, reward code, sparse-return curves, component curves, and reflection text for each job.
- `iter-<timestamp>/metadata.json`: provenance info (prompt source, env grid, timestamps).
- Marker files:
  - `ready_for_gepa`: GPU batch finished logging and dataset is ready.
  - `prompt_ready`: GEPA optimization finished and wrote `optimized_prompt.json` for that iteration.

### GPU Batch (Dense Training)
```
sbatch sbatch/train_dense_batch.s
```
This job:
1. Syncs dependencies, sets cache directories, and determines `STATE_ROOT` (defaults to `artifacts/gepa_state`).
2. Reads `active_prompt.json` (or the fallback base prompt) to seed the `RewardGenerator`.
3. Loads environment specs from `configs/gepa_envs.yaml` and runs each job with dense rewards only.
4. Logs `train_dense.jsonl`, `metadata.json`, and `ready_for_gepa` in a fresh `iter-*/` directory.

> **Cold start:** When the state directory is empty, run the GPU batch first. It seeds `active_prompt.json` with the default prompt and produces the initial dataset that GEPA needs. Subsequent iterations alternate GPU → CPU.

### CPU Batch (GEPA Optimization)
```
sbatch sbatch/gepa_opt.s
```
This job:
1. Forces `JAX_PLATFORMS=cpu`, syncs dependencies, and points at the same `STATE_ROOT`.
2. Finds the latest iteration with `ready_for_gepa` but no `prompt_ready` and loads its dataset.
3. Builds DSPy examples with sparse-return curves + reflection text, then runs `dspy.GEPA` using `o3-mini` via Portkey for both the program LM and reflection LM.
4. Writes the optimized synthesizer state to `active_prompt.json` (atomic swap) and `iter-*/optimized_prompt.json`, then touches `prompt_ready`.

Repeat the two sbatch commands to iterate on prompts without any manual copying. The GPU job always consumes the newest optimized prompt; the CPU job always optimizes the newest dataset.
