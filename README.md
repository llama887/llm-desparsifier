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
GEPA now runs on-policy inside a single job. For each GEPA candidate prompt, we run the existing RL training loop (same budget as before), capture the achieved reward plus Eureka-style reflection, and let GEPA mutate the prompt immediately—no intermediate JSONL datasets or marker files.

### State Layout
- `active_prompt.json`: current `RewardSynthesizer` state (`constraints_text` + DSPy predictor dump). Jobs load and overwrite this atomically.
- `gepa_runs/`: per-candidate training artifacts written during a GEPA session (one subdir per candidate).
- `gepa_runs/gepa_stats.json`: DSPy GEPA stats from the latest session.

### Run it
```
sbatch sbatch/train_dense_batch.s
```
This job:
1. Syncs dependencies, sets cache directories, and determines `STATE_ROOT` (defaults to `artifacts/gepa_state`).
2. Loads `active_prompt.json` (or the base prompt) to seed the reward synthesizer.
3. Loads environment specs from `configs/gepa_envs.yaml`.
4. Invokes GEPA; each candidate prompt is evaluated by running the full RL training/eval loop with the same budget as before. Feedback comes from `build_reward_reflection` on the candidate’s own rollout.
5. Writes the updated prompt to `active_prompt.json` and logs GEPA stats under `gepa_runs/`.

Repeat the single sbatch command to iterate on prompts; the job always consumes the latest `active_prompt.json` and replaces it with the newly optimized state.
