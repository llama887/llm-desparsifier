"""Dense-reward context helpers shared across reward synthesis and replay.

This module rebuilds the historical XLand `ctx` contract expected by the dense
reward pipeline. It is needed because reward generation, training-time wrappers,
and older tests still rely on previous/current state snapshots, while the search
package now also exposes a separate heuristic-only context. It differs from the
search adapter by providing dense-reward features such as `_prev` fields and
egocentric observations instead of only the current full-information state.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from xminigrid.core.constants import Colors, Tiles


def _build_name_lookup(constants_cls: Any) -> dict[int, str]:
    """Build a value-to-name lookup for XLand integer constants.

    This helper normalizes the `Tiles` and `Colors` constant bags into plain
    dictionaries so snapshot extraction can recover stable object-key names. It
    is needed because the dense-reward context exposes object positions keyed by
    semantic names, and it differs from using the raw integer IDs by keeping the
    downstream reward code readable and consistent with prompt text.
    """

    lookup: dict[int, str] = {}
    for name, value in constants_cls.__dict__.items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        lookup[int(value)] = name.lower()
    return lookup


_TILE_NAME_LOOKUP = _build_name_lookup(Tiles)
_COLOR_NAME_LOOKUP = _build_name_lookup(Colors)
_MISSING_POS = jnp.asarray((-1, -1), dtype=jnp.int32)
_DEFAULT_OBJECT_KEYS = ("yellow_square", "green_ball")


def _extract_object_positions(grid: Any) -> dict[str, Any]:
    """Extract semantic object-position entries from the full symbolic grid.

    This helper scans the entire world state and records each colored object
    under both underscore and space-separated aliases. It is needed because the
    dense-reward context historically allowed rewards to look up objects via
    either naming style, and it differs from visibility-only extraction by
    always using the full grid so rewards can compute exact progress signals.
    """

    grid_arr = jnp.asarray(grid)
    tile_layer = jnp.asarray(grid_arr[..., 0], dtype=jnp.int32)
    color_layer = jnp.asarray(grid_arr[..., 1], dtype=jnp.int32)
    object_positions: dict[str, Any] = {}
    for row in range(int(tile_layer.shape[0])):
        for col in range(int(tile_layer.shape[1])):
            tile_id = int(tile_layer[row, col])
            color_id = int(color_layer[row, col])
            if tile_id in (
                int(Tiles.EMPTY),
                int(Tiles.FLOOR),
                int(Tiles.WALL),
            ):
                continue
            tile_name = _TILE_NAME_LOOKUP.get(tile_id, f"tile_{tile_id}")
            color_name = _COLOR_NAME_LOOKUP.get(color_id, f"color_{color_id}")
            key = f"{color_name}_{tile_name}"
            pos = jnp.asarray((row, col), dtype=jnp.int32)
            object_positions[key] = pos
            object_positions[key.replace("_", " ")] = pos
    return object_positions


def _extract_carried_item(state: Any) -> str | None:
    """Return a readable carried-item name when the agent pocket is non-empty.

    This helper preserves the older context fields `is_carrying` and
    `carried_item`. It is needed because some synthesized rewards inspect the
    carried object semantically, and it differs from exposing raw pocket arrays
    by decoding them into the same color/tile naming scheme used elsewhere.
    """

    pocket = getattr(getattr(state, "agent"), "pocket", None)
    if pocket is None:
        return None
    pocket_arr = jnp.asarray(pocket, dtype=jnp.int32).reshape(-1)
    if pocket_arr.size < 2:
        return None
    tile_id = int(pocket_arr[0])
    color_id = int(pocket_arr[1])
    if tile_id == 0:
        return None
    tile_name = _TILE_NAME_LOOKUP.get(tile_id, f"tile_{tile_id}")
    color_name = _COLOR_NAME_LOOKUP.get(color_id, f"color_{color_id}")
    return f"{color_name}_{tile_name}"


def _extract_state_snapshot(timestep: Any) -> dict[str, Any]:
    """Extract the legacy dense-reward snapshot fields from one timestep.

    This helper materializes the state description that the dense-reward
    wrapper historically exposed for both current and previous timesteps. It is
    needed because the public `extract_xland_ctx` function composes two such
    snapshots into one `ctx` mapping, and it differs from the heuristic-only
    adapter by including egocentric observation tensors and previous-step-ready
    aliases.
    """

    state = getattr(timestep, "state")
    observation = jnp.asarray(getattr(timestep, "observation"))
    object_positions = _extract_object_positions(getattr(state, "grid"))
    carried_item = _extract_carried_item(state)
    snapshot: dict[str, Any] = {
        "agent_pos": jnp.asarray(getattr(getattr(state, "agent"), "position")),
        "agent_direction": jnp.asarray(getattr(getattr(state, "agent"), "direction")),
        "step_num": jnp.asarray(getattr(state, "step_num")),
        "is_carrying": jnp.asarray(carried_item is not None),
        "carried_item": carried_item,
        "observation": observation,
        "observation_tile_ids": observation[..., 0],
        "observation_color_ids": observation[..., 1],
        "object_positions": object_positions,
        "visible_object_positions": dict(object_positions),
    }
    for key, value in object_positions.items():
        if " " in key:
            continue
        snapshot[f"{key}_pos"] = jnp.asarray(value, dtype=jnp.int32)
    for key in _DEFAULT_OBJECT_KEYS:
        snapshot.setdefault(f"{key}_pos", _MISSING_POS)
        object_positions.setdefault(key, _MISSING_POS)
        object_positions.setdefault(
            key.replace("_", " "),
            _MISSING_POS,
        )
        snapshot["visible_object_positions"].setdefault(
            key,
            _MISSING_POS,
        )
    return snapshot


def extract_xland_ctx(env_params: Any, ts_prev: Any, ts_next: Any) -> dict[str, Any]:
    """Build the dense-reward `ctx` mapping with current and previous snapshots.

    This function restores the dense-reward contract used by synthesized reward
    code: current-state fields, `_prev` copies from the preceding timestep, and
    object-position dictionaries for semantic lookups. It is needed because the
    reward wrapper still calls into `llm_desparsifier.utils.extract_xland_ctx`,
    and it differs from the search adapter by duplicating state into
    previous/current views. When `ts_prev` is the initial reset state, the
    current snapshot is intentionally cloned from that reset state so callers do
    not observe a fabricated "previous step" transition before the first action.

    Args:
        env_params: Environment parameters for the current task instance. The
            dense-reward context no longer needs this directly, but it remains
            part of the stable public signature consumed by wrappers.
        ts_prev: Previous timestep.
        ts_next: Current timestep.

    Returns:
        Context mapping containing current fields plus matching `_prev` fields.
    """

    del env_params

    prev_snapshot = _extract_state_snapshot(ts_prev)
    current_snapshot = (
        prev_snapshot
        if int(jnp.asarray(getattr(getattr(ts_prev, "state"), "step_num"))) == 0
        else _extract_state_snapshot(ts_next)
    )

    ctx = dict(current_snapshot)
    for key, value in prev_snapshot.items():
        ctx[f"{key}_prev"] = value
    return ctx


__all__ = ["_extract_state_snapshot", "extract_xland_ctx"]
