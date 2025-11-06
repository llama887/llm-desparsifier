"""Manual regression check for the dense-reward context pipeline.

This script mirrors the training setup so we can verify that:
1. The XLand MiniGrid environment exposes the same benchmark/ruleset used during training.
2. The custom `extract_xland_ctx` function produces the keys required by the synthesized dense reward.
3. The dense reward output diverges from the sparse ground-truth reward once context is available.

Run with: `python tests/debug_ctx.py`
Any assertion failure means the ctx plumbing broke or the task definition changed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make project modules importable when invoked from subdirectories / SLURM.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer local cache/data directories so the script can run on shared nodes.
DATA_ROOT = ROOT / "data" / "xland_minigrid"
os.environ.setdefault("XLAND_MINIGRID_DATA", str(DATA_ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

import jax
import jax.numpy as jnp

import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper

from llm_desparsifier.rewards import sanitize_and_compile
from llm_desparsifier.rl.wrappers import DesparsifyRewardWrapper
from llm_desparsifier.utils import extract_xland_ctx

# Keep these constants in sync with training so the probe exercises the same task.
BENCHMARK_NAME = "trivial-1m"
RULESET_INDEX = 0

# Required context keys for the current dense reward implementation.
CTX_KEYS = (
    "yellow_square_pos",
    "green_ball_pos",
    "agent_pos",
    "agent_direction",
    "step_num",
    "is_carrying",
    "carried_item",
)

# Sentinel used by the extractor when an object is missing; we assert it never appears.
MISSING_POS = jnp.array([-1, -1], dtype=jnp.int32)


def _candidate_reward_paths() -> list[Path]:
    runs_dir = ROOT / "artifacts" / "runs"
    if runs_dir.exists():
        run_candidates = sorted(
            runs_dir.glob("*/dense_reward_synthesized.py"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    else:
        run_candidates = []
    base_candidates = [
        ROOT / "artifacts" / "generated_rewards" / "dense_reward_synthesized.py",
        ROOT / "artifacts" / "baseline_run" / "dense_reward_synthesized.py",
        ROOT / "dense_reward_synthesized.py",
    ]
    return run_candidates + base_candidates


def _safe_extras(ts):
    """Return extras mapping if present (RewardTimeStep) else None."""
    return getattr(ts, "extras", None)


def _print_with_values(label: str, ts) -> None:
    """Helper to surface both rewards for quick human inspection."""
    extras = _safe_extras(ts)
    if extras is None:
        extras_msg = "extras=None"
    else:
        dense = float(jax.device_get(extras["dense_reward"]))
        sparse = float(jax.device_get(extras["ground_truth_reward"]))
        extras_msg = f"dense={dense:.3f}, ground_truth={sparse:.3f}"
    reward = float(jax.device_get(ts.reward))
    step_type = int(jax.device_get(ts.step_type))
    print(f"{label}: reward={reward:.3f}, step_type={step_type}, {extras_msg}")


def _load_dense_code() -> str:
    candidates = [path for path in _candidate_reward_paths() if path.exists()]
    if not candidates:
        search_list = "\n".join(str(path) for path in _candidate_reward_paths())
        raise RuntimeError(
            "dense_reward_synthesized.py not found in expected artifact directories:\n"
            f"{search_list}\nRun the reward generator before this check."
        )
    chosen = candidates[0]
    print(f"Using generated reward from: {chosen}")
    return chosen.read_text(encoding="utf-8")


def main() -> None:
    # 1) Build the environment stack exactly like training does.
    env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    benchmark = xminigrid.load_benchmark(BENCHMARK_NAME)
    env_params = env_params.replace(ruleset=benchmark.get_ruleset(RULESET_INDEX))
    env = GymAutoResetWrapper(env)

    # 2) Compile the synthesized dense reward exactly like training does.
    dense_code = _load_dense_code()
    generated_dense_reward = sanitize_and_compile(dense_code)

    rng = jax.random.PRNGKey(0)
    ts = env.reset(env_params, rng)
    _print_with_values("raw.reset", ts)

    # Advance a few steps on the raw env for baseline comparison.
    for idx in range(3):
        ts = env.step(env_params, ts, int(idx % env.num_actions(env_params)))
        _print_with_values(f"raw.step[{idx}]", ts)

    # Capture ctx and rewards while using the same dense_fn as training.
    ctx_snapshots: list[dict[str, object]] = []
    dense_values: list[float] = []
    sparse_values: list[float] = []

    def dense_probe(env_params, ts_prev, action, ts_next, ctx):
        host_ctx = {k: jax.device_get(v) for k, v in ctx.items()}
        ctx_snapshots.append(host_ctx)
        dense_val = generated_dense_reward(env_params, ts_prev, action, ts_next, ctx)
        dense_values.append(float(jax.device_get(dense_val)))
        sparse_values.append(float(jax.device_get(ts_next.reward)))
        return dense_val

    wrapped = DesparsifyRewardWrapper(env, dense_fn=dense_probe, ctx_fn=extract_xland_ctx)
    ts = wrapped.reset(env_params, rng)
    _print_with_values("wrapped.reset", ts)

    for idx in range(3):
        ts = wrapped.step(env_params, ts, int(idx % wrapped.num_actions(env_params)))
        _print_with_values(f"wrapped.step[{idx}]", ts)

    # 3) Assertions documenting the invariants we rely on downstream.
    assert ctx_snapshots, "Dense reward was never invoked; ctx pipeline is broken."
    for snapshot in ctx_snapshots:
        # Every required key must be present.
        missing = [key for key in CTX_KEYS if key not in snapshot]
        assert not missing, f"Missing ctx keys: {missing}"

        # Positions should be found (tasks expect these objects to exist).
        for key in ("yellow_square_pos", "green_ball_pos"):
            value = jnp.asarray(snapshot[key])
            assert value.shape == (2,), f"{key} has unexpected shape {value.shape}"
            assert not jnp.array_equal(value, MISSING_POS), f"{key} not found in grid (value {value})."

    # Confirm dense reward diverges from sparse at least once (indicates shaping signal).
    diffs = [
        abs(dense - sparse) > 1e-6
        for dense, sparse in zip(dense_values, sparse_values)
    ]
    assert any(diffs), "Dense reward never diverged from sparse reward; shaping ineffective."

    print("All ctx assertions passed. Dense reward differs from sparse as expected.")


if __name__ == "__main__":
    main()
