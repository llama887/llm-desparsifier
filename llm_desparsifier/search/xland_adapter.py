"""XLand-specific search adapter helpers.

This module converts XLand benchmark jobs into concrete deterministic task
instances and extracts the runtime `ctx` mapping exposed to synthesized
heuristics. It is needed because the search backend should consume encoded task
information without knowing about GEPA job configs or prompt contracts, and it
differs from the legacy reward context path by exposing only current-state
full-information fields that are safe for heuristic evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import xminigrid
from xminigrid.core.constants import Colors, Tiles

from llm_desparsifier.heuristics.prompting import (
    describe_ruleset_for_heuristic,
    extract_goal_description_from_ruleset_text,
)

ACTION_NAMES = (
    "move_forward",
    "turn_right",
    "turn_left",
    "pick_up",
    "put_down",
    "toggle",
)
DEFAULT_RULESET_INDEX = 42


def _build_name_lookup(constants_cls: Any) -> dict[int, str]:
    """Build a value-to-name lookup table for XLand integer constants.

    This helper normalizes the `Tiles` and `Colors` classes into plain Python
    dictionaries. It is needed because these classes behave like constant bags
    rather than standard enums under static analysis, and it differs from
    accessing `name` attributes directly by working consistently for both
    runtime extraction and linting.
    """

    lookup: dict[int, str] = {}
    for name, value in constants_cls.__dict__.items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        lookup[int(value)] = name.lower()
    return lookup


_TILE_NAME_LOOKUP = _build_name_lookup(Tiles)
_COLOR_NAME_LOOKUP = _build_name_lookup(Colors)


@dataclass(frozen=True)
class XLandTaskInstance:
    """Concrete deterministic XLand search task derived from one seed.

    This dataclass records both the environment state needed to execute search
    and the replay metadata needed to reconstruct the exact instance later. It
    is needed because the heuristic runner evaluates many seeds per job and must
    persist one representative seed for video replay, and it differs from raw
    `xminigrid` objects by keeping only serializable metadata.
    """

    env_id: str
    benchmark_id: str
    seed: int
    ruleset_seed: int | None
    ruleset_text: str
    reset_key: list[int]
    goal_description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the task instance.

        This helper is used for `task_instance.json`. It is needed because the
        replay script reconstructs environments from persisted metadata rather
        than live Python objects, and it differs from `asdict(...)` only by
        making the intended artifact contract explicit at the call site.
        """

        return asdict(self)


def _key_to_list(key: jax.Array) -> list[int]:
    """Convert a JAX PRNG key into JSON-friendly raw words.

    This helper preserves deterministic reset material for replay artifacts. It
    is needed because JAX typed keys are not directly serializable, and it
    differs from naive conversion by using `jax.random.key_data(...)` to support
    typed keys correctly.
    """

    arr = jnp.asarray(jax.random.key_data(key), dtype=jnp.uint32).reshape(-1)
    return [int(value) for value in arr.tolist()]


def _resolve_ruleset(
    *,
    benchmark: Any,
    deterministic_rulesets: bool,
    fixed_ruleset_seed: int | None,
    ruleset_key: jax.Array,
) -> tuple[Any, int | None]:
    """Resolve the benchmark ruleset for one deterministic task instance.

    This helper keeps prompt construction, search execution, and replay
    reconstruction aligned on the same ruleset selection policy. It is needed
    because GEPA training may use either deterministic or sampled benchmark
    rulesets, and it differs from the runner by focusing only on one seed at a
    time.
    """

    if deterministic_rulesets:
        if fixed_ruleset_seed is None:
            return benchmark.get_ruleset(DEFAULT_RULESET_INDEX), None
        return benchmark.sample_ruleset(jax.random.key(fixed_ruleset_seed)), fixed_ruleset_seed
    return benchmark.sample_ruleset(ruleset_key), None


def build_task_instance(
    *,
    env_id: str,
    benchmark_id: str,
    seed: int,
    deterministic_rulesets: bool,
    fixed_ruleset_seed: int | None,
) -> tuple[Any, Any, Any, Any, Any, XLandTaskInstance]:
    """Build one concrete XLand environment instance for search evaluation.

    This function constructs the environment, resolves the benchmark ruleset,
    resets the initial state, and returns replay metadata. It is needed because
    every per-seed search run requires a deterministic task instance plus the
    exact reset key used for replay, and it differs from the legacy evaluator by
    omitting all reward-wrapper logic.
    """

    env, env_params = xminigrid.make(env_id)
    benchmark = xminigrid.load_benchmark(benchmark_id)
    rng = jax.random.key(seed)
    rng, ruleset_key, reset_key = jax.random.split(rng, 3)
    ruleset, ruleset_seed = _resolve_ruleset(
        benchmark=benchmark,
        deterministic_rulesets=deterministic_rulesets,
        fixed_ruleset_seed=fixed_ruleset_seed,
        ruleset_key=ruleset_key,
    )
    env_params = env_params.replace(ruleset=ruleset)
    try:
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
    except Exception:
        reset_fn = env.reset
        step_fn = env.step
    root_timestep = reset_fn(env_params, reset_key)
    ruleset_text = describe_ruleset_for_heuristic(env, env_params)
    goal_description = extract_goal_description_from_ruleset_text(ruleset_text)
    task_instance = XLandTaskInstance(
        env_id=env_id,
        benchmark_id=benchmark_id,
        seed=seed,
        ruleset_seed=ruleset_seed,
        ruleset_text=ruleset_text,
        reset_key=_key_to_list(reset_key),
        goal_description=goal_description,
    )
    return env, env_params, step_fn, root_timestep, reset_key, task_instance


def _to_int_tuple(value: Any) -> tuple[int, ...]:
    """Convert an array-like value into a tuple of Python ints.

    This helper normalizes JAX arrays into stable immutable values for `ctx`.
    It is needed because the heuristic contract intentionally exposes plain
    Python tuples and ints rather than array types, and it differs from raw
    `tolist()` by always returning a tuple.
    """

    arr = jnp.asarray(value, dtype=jnp.int32).reshape(-1)
    return tuple(int(item) for item in arr.tolist())


def _extract_object_positions(
    grid: Any,
) -> tuple[dict[str, tuple[int, int]], dict[str, dict[str, Any]], tuple[tuple[int, int], ...]]:
    """Extract object locations and metadata from the full symbolic grid.

    This helper scans the global grid and builds the `object_positions`,
    `object_metadata`, and `static_walls` fields exposed to synthesized
    heuristics. It is needed because lower-bound heuristics should reason from
    full state rather than egocentric observations, and it differs from the old
    reward context extractor by omitting previous-step and visibility-only data.

    The heuristic-only pipeline now uses the human-readable object names shown
    in prompt text and ruleset feedback, such as `"red key"` instead of
    `"red_key"`. This is needed because synthesized heuristics are instructed
    with space-delimited names and silently degrade to `0.0` when the runtime
    context exports a different key format. To keep older underscore-based code
    working, the helper exports both spellings and points them at the same
    position and metadata payloads.
    """

    grid_arr = jnp.asarray(grid)
    tile_layer = jnp.asarray(grid_arr[..., 0], dtype=jnp.int32)
    color_layer = jnp.asarray(grid_arr[..., 1], dtype=jnp.int32)
    object_positions: dict[str, tuple[int, int]] = {}
    object_metadata: dict[str, dict[str, Any]] = {}
    static_walls: list[tuple[int, int]] = []
    for row in range(int(tile_layer.shape[0])):
        for col in range(int(tile_layer.shape[1])):
            tile_id = int(tile_layer[row, col])
            color_id = int(color_layer[row, col])
            if tile_id == int(Tiles.WALL):
                static_walls.append((row, col))
                continue
            if tile_id in (int(Tiles.EMPTY), int(Tiles.FLOOR)):
                continue
            tile_name = _TILE_NAME_LOOKUP.get(tile_id, f"tile_{tile_id}")
            color_name = _COLOR_NAME_LOOKUP.get(color_id, f"color_{color_id}")
            spaced_key = f"{color_name} {tile_name}"
            underscored_key = f"{color_name}_{tile_name}"
            metadata = {
                "tile": tile_name,
                "color": color_name,
                "row": row,
                "col": col,
            }
            object_positions[spaced_key] = (row, col)
            object_positions[underscored_key] = (row, col)
            object_metadata[spaced_key] = metadata
            object_metadata[underscored_key] = metadata
    return object_positions, object_metadata, tuple(static_walls)


def build_heuristic_ctx(
    *,
    ts: Any,
    env_params: Any,
    env_id: str,
    benchmark_id: str,
    ruleset_text: str,
    goal_description: str,
) -> dict[str, Any]:
    """Build the full-information `ctx` mapping used by synthesized heuristics.

    This function extracts the exact contract described in the refactor plan. It
    is needed because the generated heuristic code and runtime validation must
    agree on one stable interface, and it differs from the legacy reward
    context by exposing only current-state symbolic data plus static task
    metadata.

    The current heuristic corpus already contains many synthesized functions
    that read `agent_state["pos"]` while newer documentation favors the more
    explicit `agent_state["position"]`. This helper therefore exposes both keys
    as aliases to the same tuple so resumed runs, old artifacts, and newly
    generated prompts all observe a backward-compatible position field instead
    of silently degrading into `h=0.0` blind search.
    """

    state = getattr(ts, "state")
    grid = getattr(state, "grid")
    object_positions, object_metadata, static_walls = _extract_object_positions(grid)
    carrying = _to_int_tuple(getattr(state.agent, "pocket"))
    agent_position = _to_int_tuple(getattr(state.agent, "position"))
    ctx = {
        "env_id": env_id,
        "benchmark_id": benchmark_id,
        "ruleset_text": ruleset_text,
        "grid_shape": (
            int(getattr(env_params, "height")),
            int(getattr(env_params, "width")),
        ),
        "action_names": ACTION_NAMES,
        "step_cost": 1,
        "goal_description": goal_description,
        "agent_state": {
            "position": agent_position,
            "pos": agent_position,
            "direction": int(jnp.asarray(getattr(state.agent, "direction"))),
            "carrying": carrying,
        },
        "object_positions": object_positions,
        "object_metadata": object_metadata,
        "static_walls": static_walls,
        "task_features": {
            "grid_type": str(getattr(env_params, "grid_type", "unknown")),
            "max_steps": int(getattr(env_params, "max_steps", 0)),
            "goal_description": goal_description,
        },
    }
    return ctx


__all__ = [
    "ACTION_NAMES",
    "DEFAULT_RULESET_INDEX",
    "XLandTaskInstance",
    "build_heuristic_ctx",
    "build_task_instance",
]
