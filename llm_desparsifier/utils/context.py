"""Context extraction utilities shared across environments.

Functions should follow the signature::

    def ctx_fn(env_params, ts_prev, ts_next) -> dict[str, jax.Array]:

and return JAX-friendly arrays only so they can be used under `jax.jit`/`jax.vmap`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


def extract_xland_ctx(env_params, ts_prev, ts_next):
    """Build a dense-reward context dictionary from XLand MiniGrid timesteps."""
    del env_params, ts_prev

    grid = ts_next.state.grid
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

    agent_pos = jnp.asarray(ts_next.state.agent.position, dtype=jnp.int32)
    agent_dir = jnp.asarray(ts_next.state.agent.direction, dtype=jnp.int32)
    step_num = jnp.asarray(ts_next.state.step_num, dtype=jnp.int32)

    pocket = jnp.asarray(ts_next.state.agent.pocket, dtype=jnp.int32)
    empty_pocket = jnp.array([Tiles.EMPTY, Colors.EMPTY], dtype=jnp.int32)
    is_carrying = jnp.logical_not(jnp.all(pocket == empty_pocket))

    return {
        "yellow_square_pos": yellow_square_pos,
        "green_ball_pos": green_ball_pos,
        "agent_pos": agent_pos,
        "agent_direction": agent_dir,
        "step_num": step_num,
        "is_carrying": is_carrying,
        "carried_item": pocket,
    }
