#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Keep XLand-MiniGrid data in the project, not $HOME
export XLAND_MINIGRID_DATA="$PWD/.xland_minigrid"
mkdir -p "$XLAND_MINIGRID_DATA"

# Keep general caches local too
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$XDG_CACHE_HOME"

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
import os, inspect, jax, jaxlib, subprocess
print("JAX:", jax.__version__, "jaxlib:", jaxlib.__version__)
print("Devices:", jax.devices())
# Show linkage for the core extension to verify libcuda resolution
import jaxlib as jl, os, inspect
ext = os.path.join(os.path.dirname(inspect.getfile(jl)), "xla_extension.so")
print("xla_extension.so:", ext)
try:
    subprocess.run(["ldd", ext], check=False)
except Exception as e:
    print("ldd failed:", e)
PY

# Run training
uv run xland_meta_learning_baseline.py
