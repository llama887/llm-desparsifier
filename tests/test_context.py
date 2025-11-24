from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
xminigrid = pytest.importorskip("xminigrid")
from xminigrid.wrappers import GymAutoResetWrapper

from llm_desparsifier.utils.context import extract_xland_ctx, _extract_state_snapshot


def _freeze_to_dict(value):
    try:
        return dict(value)
    except TypeError:
        return value


def _to_numpy(value):
    return np.asarray(jax.device_get(value))


def test_extract_xland_ctx_includes_previous_snapshot_after_initial_step():
    env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    env = GymAutoResetWrapper(env)

    rng = jax.random.PRNGKey(0)
    ts0 = env.reset(env_params, rng)
    ts_prev = env.step(env_params, ts0, jnp.int32(0))
    ts_next = env.step(env_params, ts_prev, jnp.int32(1))

    ctx = extract_xland_ctx(env_params, ts_prev, ts_next)

    base_keys = (
        "yellow_square_pos",
        "green_ball_pos",
        "agent_pos",
        "agent_direction",
        "step_num",
        "is_carrying",
        "carried_item",
    )

    prev_snapshot = _extract_state_snapshot(ts_prev)
    next_snapshot = _extract_state_snapshot(ts_next)

    for key in base_keys:
        assert key in ctx, f"{key} missing from context"
        prev_key = f"{key}_prev"
        assert prev_key in ctx, f"{prev_key} missing from context"

        npt.assert_array_equal(_to_numpy(ctx[key]), _to_numpy(next_snapshot[key]))
        npt.assert_array_equal(_to_numpy(ctx[prev_key]), _to_numpy(prev_snapshot[key]))

    step_num = int(_to_numpy(ctx["step_num"]))
    step_num_prev = int(_to_numpy(ctx["step_num_prev"]))
    assert step_num == step_num_prev + 1, "step_num should increment between timesteps"

    object_positions = _freeze_to_dict(ctx["object_positions"])
    assert "yellow_square" in object_positions
    assert "yellow square" in object_positions
    assert "green_ball" in object_positions
    npt.assert_array_equal(
        _to_numpy(object_positions["yellow_square"]),
        _to_numpy(next_snapshot["yellow_square_pos"]),
    )


def test_extract_xland_ctx_clones_snapshot_on_reset():
    env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    env = GymAutoResetWrapper(env)

    rng = jax.random.PRNGKey(1)
    ts_reset = env.reset(env_params, rng)
    ts_next = env.step(env_params, ts_reset, jnp.int32(0))

    ctx = extract_xland_ctx(env_params, ts_reset, ts_next)

    base_keys = (
        "yellow_square_pos",
        "green_ball_pos",
        "agent_pos",
        "agent_direction",
        "step_num",
        "is_carrying",
        "carried_item",
    )

    for key in base_keys:
        prev_key = f"{key}_prev"
        npt.assert_array_equal(_to_numpy(ctx[key]), _to_numpy(ctx[prev_key]))
