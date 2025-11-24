"""Context extraction utilities shared across environments.

Functions should follow the signature::

    def ctx_fn(env_params, ts_prev, ts_next) -> dict[str, jax.Array]:

and return JAX-friendly arrays only so they can be used under `jax.jit`/`jax.vmap`.

For `extract_xland_ctx`, the contract is:

* Always emit the base keys for the *current* timestep first.
* Append the same keys with a ``_prev`` suffix so dense rewards can compute deltas.
* When the previous timestep corresponds to a reset (`step_type == FIRST`),
  duplicate the current snapshot for all ``_prev`` keys so rewards never see
  stale context leaking across auto-resets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flax.core.frozen_dict import freeze
from xminigrid.core.constants import Colors, Tiles

__all__ = ["extract_xland_ctx"]


def _find_first_match(mask: jnp.ndarray) -> jnp.ndarray:
    """Return the first `(row, col)` where `mask` is True, or [-1, -1] if none."""
    mask_flat = mask.reshape(-1)
    coords = jnp.stack(
        jnp.meshgrid(
            jnp.arange(mask.shape[0], dtype=jnp.int32),
            jnp.arange(mask.shape[1], dtype=jnp.int32),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    idx = jnp.argmax(mask_flat.astype(jnp.int32))
    default = jnp.array([-1, -1], dtype=jnp.int32)

    def _select(i):
        return jax.lax.dynamic_index_in_dim(coords, i, axis=0, keepdims=False)

    return jax.lax.cond(jnp.any(mask_flat), _select, lambda _: default, idx)


def _unwrap_timestep(ts):
    """Return the underlying dm_env TimeStep when wrapped by RewardTimeStep."""
    return getattr(ts, "base", ts)


def _is_reset(ts) -> bool:
    """Best-effort detection of dm_env.StepType.FIRST."""
    step_type = getattr(ts, "step_type", None)
    if step_type is None:
        return False
    value = step_type
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass
    try:
        return int(value) == 0
    except Exception:
        return False


def _build_lookup_tables():
    tile_lookup = {}
    for name, value in Tiles.__dict__.items():
        if name.startswith("_"):
            continue
        if not isinstance(value, int):
            continue
        tile_lookup[value] = name.lower()
    color_lookup = {}
    for name, value in Colors.__dict__.items():
        if name.startswith("_"):
            continue
        if not isinstance(value, int):
            continue
        color_lookup[value] = name.lower()
    return tile_lookup, color_lookup


_TILE_ID_TO_NAME, _COLOR_ID_TO_NAME = _build_lookup_tables()
_OBJECT_TILE_IDS = {
    Tiles.BALL,
    Tiles.SQUARE,
    Tiles.PYRAMID,
    Tiles.GOAL,
    Tiles.KEY,
    Tiles.HEX,
    Tiles.STAR,
}


def _extract_object_positions(tile_layer, color_layer):
    positions = {}
    for tile_id, tile_name in _TILE_ID_TO_NAME.items():
        if tile_id not in _OBJECT_TILE_IDS:
            continue
        tile_mask = tile_layer == tile_id
        for color_id, color_name in _COLOR_ID_TO_NAME.items():
            mask = jnp.logical_and(tile_mask, color_layer == color_id)
            snake_key = f"{color_name}_{tile_name}"
            value = _find_first_match(mask)
            positions[snake_key] = value
            spaced_key = snake_key.replace("_", " ")
            positions[spaced_key] = value
    return positions


def _extract_state_snapshot(ts):
    """Return a dictionary of spatial features for the given timestep."""
    grid = ts.state.grid
    tile_layer = grid[..., 0]
    color_layer = grid[..., 1]

    yellow_square_mask = jnp.logical_and(
        tile_layer == Tiles.SQUARE,
        color_layer == Colors.YELLOW,
    )
    green_ball_mask = jnp.logical_and(
        tile_layer == Tiles.BALL,
        color_layer == Colors.GREEN,
    )

    yellow_square_pos = _find_first_match(yellow_square_mask)
    green_ball_pos = _find_first_match(green_ball_mask)

    agent_pos = jnp.asarray(ts.state.agent.position, dtype=jnp.int32)
    agent_dir = jnp.asarray(ts.state.agent.direction, dtype=jnp.int32)
    step_num = jnp.asarray(ts.state.step_num, dtype=jnp.int32)

    pocket = jnp.asarray(ts.state.agent.pocket, dtype=jnp.int32)
    empty_pocket = jnp.array([Tiles.EMPTY, Colors.EMPTY], dtype=jnp.int32)
    is_carrying = jnp.logical_not(jnp.all(pocket == empty_pocket))

    snapshot = {
        "yellow_square_pos": yellow_square_pos,
        "green_ball_pos": green_ball_pos,
        "agent_pos": agent_pos,
        "agent_direction": agent_dir,
        "step_num": step_num,
        "is_carrying": is_carrying,
        "carried_item": pocket,
    }
    snapshot["object_positions"] = freeze(_extract_object_positions(tile_layer, color_layer))
    return snapshot


def extract_xland_ctx(env_params, ts_prev, ts_next):
    """Build a dense-reward context dictionary from consecutive timesteps."""
    del env_params

    ts_prev_unwrapped = _unwrap_timestep(ts_prev)
    ts_next_unwrapped = _unwrap_timestep(ts_next)

    prev_snapshot = _extract_state_snapshot(ts_prev_unwrapped)
    next_snapshot = _extract_state_snapshot(ts_next_unwrapped)

    if _is_reset(ts_prev_unwrapped):
        prev_snapshot = next_snapshot

    ctx = dict(next_snapshot)
    # Attach previous-step features with `_prev` suffix so dense rewards can compute deltas.
    for key, value in prev_snapshot.items():
        ctx[f"{key}_prev"] = value

    return ctx
