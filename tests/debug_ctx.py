"""Manual regression check for the dense-reward context pipeline.

This script mirrors the training setup so we can verify that:
1. The XLand MiniGrid environment exposes the same benchmark/ruleset used during training.
2. The custom `extract_xland_ctx` function produces the keys required by the synthesized dense reward.
3. The dense reward output diverges from the sparse ground-truth reward once context is available.

Run with: `python tests/debug_ctx.py`
Any assertion failure means the ctx plumbing broke or the task definition changed.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
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
BASE_CTX_KEYS = (
    "yellow_square_pos",
    "green_ball_pos",
    "agent_pos",
    "agent_direction",
    "step_num",
    "is_carrying",
    "carried_item",
)

CTX_KEYS = BASE_CTX_KEYS + tuple(f"{key}_prev" for key in BASE_CTX_KEYS)

# Sentinel used by the extractor when an object is missing; we assert it never appears.
MISSING_POS = jnp.array([-1, -1], dtype=jnp.int32)

FALLBACK_ACTION_MACRO: tuple[int, ...] = (
    1,  # turn_right
    0,  # move_forward
    0,  # move_forward
    3,  # pick_up
    2,  # turn_left
    0,  # move_forward
    4,  # put_down
)


def _has_ctx_variation(snapshots: list[dict[str, object]], key: str) -> bool:
    if len(snapshots) < 2:
        return False
    baseline = jnp.asarray(snapshots[0][key])
    for snapshot in snapshots[1:]:
        value = jnp.asarray(snapshot[key])
        if not jnp.array_equal(value, baseline):
            return True
    return False


def _apply_action_sequence(wrapped_env, env_params, ts, actions, step_counter):
    for action in actions:
        ts = wrapped_env.step(env_params, ts, int(action))
        _print_with_values(f"wrapped.step[{step_counter}]", ts)
        step_counter += 1
        if bool(jax.device_get(ts.last() > 0)):
            break
    return ts, step_counter


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


def _load_dense_code(dense_path: Path | None, allow_search: bool) -> str:
    if dense_path is not None:
        if not dense_path.exists():
            raise FileNotFoundError(f"Specified dense reward file not found: {dense_path}")
        chosen = dense_path
    else:
        default_path = ROOT / "artifacts" / "baseline_run" / "dense_reward_synthesized.py"
        warnings.warn(
            "--dense-path not provided; defaulting to artifacts/baseline_run/dense_reward_synthesized.py",
            stacklevel=2,
        )
        if default_path.exists():
            chosen = default_path
        elif allow_search:
            candidates = [path for path in _candidate_reward_paths() if path.exists()]
            if not candidates:
                search_list = "\n".join(str(path) for path in _candidate_reward_paths())
                raise RuntimeError(
                    "dense_reward_synthesized.py not found in expected artifact directories:\n"
                    f"{search_list}\nRun the reward generator or pass --dense-path."
                )
            chosen = candidates[0]
            print(f"Falling back to heuristic search; using {chosen}")
        else:
            raise FileNotFoundError(
                "Default dense reward artifact missing. Run training to generate it or pass --dense-path."
            )

    print(f"Using generated reward from: {chosen}")
    code = chosen.read_text(encoding="utf-8")
    preview = "\n".join(code.splitlines()[:100])
    print("Dense reward preview (first 100 lines):")
    print(preview)
    print("---- end dense preview ----")
    return code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug dense reward context plumbing.")
    parser.add_argument(
        "--dense-path",
        type=Path,
        default=None,
        help="Explicit path to dense_reward_synthesized.py (defaults to artifacts/baseline_run).",
    )
    parser.add_argument(
        "--allow-search-fallback",
        action="store_true",
        help="Search legacy artifact locations if the default path is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # 1) Build the environment stack exactly like training does.
    env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    benchmark = xminigrid.load_benchmark(BENCHMARK_NAME)
    env_params = env_params.replace(ruleset=benchmark.get_ruleset(RULESET_INDEX))
    env = GymAutoResetWrapper(env)

    # 2) Compile the synthesized dense reward exactly like training does.
    dense_code = _load_dense_code(args.dense_path, allow_search=args.allow_search_fallback)
    generated_dense_reward = sanitize_and_compile(dense_code)

    rng = jax.random.PRNGKey(0)
    rng, raw_reset_key = jax.random.split(rng)
    ts = env.reset(env_params, raw_reset_key)
    _print_with_values("raw.reset", ts)

    # Advance a few steps on the raw env for baseline comparison.
    for idx in range(3):
        action = int(idx % env.num_actions(env_params))
        ts = env.step(env_params, ts, action)
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
    rng, wrapped_reset_key = jax.random.split(rng)
    ts = wrapped.reset(env_params, wrapped_reset_key)
    _print_with_values("wrapped.reset", ts)

    # Explore an episode worth of steps to gather meaningful sequences.
    max_steps = 32
    step_counter = 0
    for _ in range(max_steps):
        rng, action_rng = jax.random.split(rng)
        action = int(jax.random.randint(action_rng, (), 0, wrapped.num_actions(env_params)))
        ts = wrapped.step(env_params, ts, action)
        _print_with_values(f"wrapped.step[{step_counter}]", ts)
        step_counter += 1
        if bool(jax.device_get(ts.last() > 0)):
            break

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

    # Sanity-check that previous-step slices line up with actual history.
    for idx in range(1, len(ctx_snapshots)):
        current = ctx_snapshots[idx]
        previous = ctx_snapshots[idx - 1]
        for key in BASE_CTX_KEYS:
            prev_key = f"{key}_prev"
            cur_prev = jnp.asarray(current[prev_key])
            prev_val = jnp.asarray(previous[key])
            assert jnp.array_equal(cur_prev, prev_val), f"{prev_key} mismatch at step {idx}"

    # Emit a short summary of context evolution for manual inspection.
    preview = min(6, len(ctx_snapshots))
    print("Context preview (first few steps):")
    for idx in range(preview):
        snapshot = ctx_snapshots[idx]
        yellow = jnp.asarray(snapshot["yellow_square_pos"])
        yellow_prev = jnp.asarray(snapshot["yellow_square_pos_prev"])
        green = jnp.asarray(snapshot["green_ball_pos"])
        green_prev = jnp.asarray(snapshot["green_ball_pos_prev"])
        rel_prev = green_prev - yellow_prev
        rel = green - yellow
        agent = jnp.asarray(snapshot["agent_pos"])
        agent_prev = jnp.asarray(snapshot["agent_pos_prev"])
        print(
            f"  step {idx}: agent={agent.tolist()} (prev={agent_prev.tolist()}), "
            f"yellow={yellow.tolist()} (prev={yellow_prev.tolist()}), "
            f"green={green.tolist()} (prev={green_prev.tolist()}), "
            f"rel={rel.tolist()} (prev={rel_prev.tolist()})"
        )

    agent_changed = _has_ctx_variation(ctx_snapshots, "agent_pos")
    yellow_changed = _has_ctx_variation(ctx_snapshots, "yellow_square_pos")
    green_changed = _has_ctx_variation(ctx_snapshots, "green_ball_pos")
    context_changed = agent_changed or yellow_changed or green_changed

    if not context_changed:
        print("Context positions never changed during random rollout; executing fallback action macro.")
        ts, step_counter = _apply_action_sequence(
            wrapped, env_params, ts, FALLBACK_ACTION_MACRO, step_counter
        )
        agent_changed = _has_ctx_variation(ctx_snapshots, "agent_pos")
        yellow_changed = _has_ctx_variation(ctx_snapshots, "yellow_square_pos")
        green_changed = _has_ctx_variation(ctx_snapshots, "green_ball_pos")
        context_changed = agent_changed or yellow_changed or green_changed

    if not context_changed:
        raise AssertionError(
            "Context positions never changed even after fallback macro; "
            "dense reward cannot be evaluated under a static trajectory."
        )

    dense_arr = jnp.asarray(dense_values)
    sparse_arr = jnp.asarray(sparse_values)

    assert dense_arr.size > 0, "Dense reward sequence is empty."
    assert sparse_arr.size == dense_arr.size, "Sparse sequence length mismatch."

    # Compute correlation for diagnostics (safe guard against constant arrays).
    dense_var = jnp.var(dense_arr)
    sparse_var = jnp.var(sparse_arr)
    if float(dense_var) > 0.0 and float(sparse_var) > 0.0:
        corr = jnp.corrcoef(dense_arr, sparse_arr)[0, 1]
    else:
        corr = jnp.nan

    dense_min = float(dense_arr.min())
    dense_max = float(dense_arr.max())
    sparse_min = float(sparse_arr.min())
    sparse_max = float(sparse_arr.max())

    diff_arr = dense_arr - sparse_arr
    diff_min = float(diff_arr.min())
    diff_max = float(diff_arr.max())
    mae = float(jnp.mean(jnp.abs(diff_arr)))

    print(
        "Reward diagnostics:\n"
        f"  dense range: [{dense_min:.3f}, {dense_max:.3f}]\n"
        f"  sparse range: [{sparse_min:.3f}, {sparse_max:.3f}]\n"
        f"  diff range: [{diff_min:.3f}, {diff_max:.3f}], MAE={mae:.3f}\n"
        f"  correlation: {float(corr):.4f}"
    )

    # Confirm dense reward diverges from sparse at least once (indicates shaping signal).
    diffs = [
        abs(dense - sparse) > 1e-6
        for dense, sparse in zip(dense_values, sparse_values)
    ]
    assert any(diffs), "Dense reward never diverged from sparse reward; shaping ineffective."

    def _is_affine_transform(dense_seq: jnp.ndarray, sparse_seq: jnp.ndarray, atol: float = 1e-3) -> bool:
        sparse_var = jnp.var(sparse_seq)
        if float(sparse_var) < 1e-8:
            return True
        sparse_mean = jnp.mean(sparse_seq)
        dense_mean = jnp.mean(dense_seq)
        cov = jnp.mean((sparse_seq - sparse_mean) * (dense_seq - dense_mean))
        slope = cov / sparse_var
        intercept = dense_mean - slope * sparse_mean
        residual = dense_seq - (slope * sparse_seq + intercept)
        return float(jnp.max(jnp.abs(residual))) < atol

    assert not _is_affine_transform(dense_arr, sparse_arr), (
        "Dense reward appears to be an affine transform of sparse reward; "
        "shaping signal is ineffective."
    )

    print("All ctx assertions passed. Dense reward differs from sparse beyond a simple affine transform.")


if __name__ == "__main__":
    main()
