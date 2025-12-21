#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Keep XLand-MiniGrid data in the project, not $HOME
export XLAND_MINIGRID_DATA="$PWD/.xland_minigrid"
mkdir -p "$XLAND_MINIGRID_DATA"

# Keep general caches local too
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$XDG_CACHE_HOME"

export WANDB_API_KEY=4eebac5d4dc88793e64cfb18af3233657db3aeda
export WANDB_DATA_DIR=$SCRATCH/wandb_cache
mkdir -p "$WANDB_DATA_DIR"
export WANDB_DIR="${WANDB_DATA_DIR}/runs"
mkdir -p "$WANDB_DIR"

# If you installed JAX **CUDA wheels** via `uv add jax[cuda12]` or `[cuda13]`,
# you usually don't need a site CUDA module; you DO need a new-enough driver.
# If your cluster forces module loads for the driver env, load the *driver* module only.
# module load cuda/12.2  # Only if required for driver runtime on your cluster

# Avoid picking up CUDA "stubs" that can shadow the real driver libcuda.so.1
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vE '/cuda.*/lib64/stubs' | paste -sd: -)"
fi

# Optional: temporarily force CPU to unblock if GPU init fails
# export JAX_PLATFORMS=cpu

# Resolve env (no-op after first run unless pyproject changed)
uv sync

# Diagnostics: driver + JAX devices + which libs the JAX extension links to
echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== JAX info ==="
uv run -- python - <<'PY'
import os, inspect, jax, jaxlib
print("JAX:", jax.__version__, "jaxlib:", jaxlib.__version__)
print("Devices:", jax.devices())
# Show linkage for the core extension to verify libcuda resolution
import jaxlib as jl, os, inspect
ext = os.path.join(os.path.dirname(inspect.getfile(jl)), "xla_extension.so")
print("xla_extension.so:", ext)
try:
    # Avoid os.fork() from subprocess in a multi-threaded JAX process:
    # use posix_spawn so the child is created without inheriting problematic thread state.
    pid = os.posix_spawn("ldd", ["ldd", ext], os.environ)
    _, status = os.waitpid(pid, 0)
    if status != 0:
        print(f"ldd exited with status {status}")
except AttributeError:
    # Fallback for platforms lacking posix_spawn; keep old behavior as last resort.
    import subprocess
    subprocess.run(["ldd", ext], check=False)
except Exception as e:
    print("ldd failed:", e)
PY

# Run training (dense vs sparse comparison by default)
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/artifacts/runs/${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR"

COMPARE_DENSE_VS_SPARSE=${COMPARE_DENSE_VS_SPARSE:-1}
REWARD_MODE=${REWARD_MODE:-dense}

COMPARE_FLAG=()
if [[ "$COMPARE_DENSE_VS_SPARSE" == "1" ]]; then
  COMPARE_FLAG+=("--compare-dense-vs-sparse")
fi

uv run xland_meta_learning_baseline.py \
  --output-dir "$OUTPUT_DIR" \
  --reward-mode "$REWARD_MODE" \
  "${COMPARE_FLAG[@]}"
