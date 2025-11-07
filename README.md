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
