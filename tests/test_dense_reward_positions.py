from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
xminigrid = pytest.importorskip("xminigrid")
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.core.constants import Colors, Tiles
text_render = pytest.importorskip("xminigrid.rendering.text_render")

from llm_desparsifier.utils.context import extract_xland_ctx


def _to_numpy(value):
    return np.asarray(jax.device_get(value))


def _make_timestep(grid, step_num=0, agent_pos=(0, 0), agent_dir=0, step_type=1):
    grid = jnp.asarray(grid, dtype=jnp.int32)
    agent = SimpleNamespace(
        position=jnp.asarray(agent_pos, dtype=jnp.int32),
        direction=jnp.asarray(agent_dir, dtype=jnp.int32),
        pocket=jnp.asarray([Tiles.EMPTY, Colors.EMPTY], dtype=jnp.int32),
    )
    state = SimpleNamespace(grid=grid, agent=agent, step_num=jnp.asarray(step_num, dtype=jnp.int32))
    return SimpleNamespace(state=state, step_type=step_type)


def _build_color_vocab():
    colors = set()
    for name, value in Colors.__dict__.items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        colors.add(name.lower())
    if "gray" in colors:
        colors.add("grey")
    return colors


def _build_tile_vocab():
    object_tiles = {"ball", "square", "pyramid", "goal", "key", "hex", "star"}
    tiles = set()
    for name, value in Tiles.__dict__.items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        tile_name = name.lower()
        if tile_name in object_tiles:
            tiles.add(tile_name)
    return tiles


def _extract_ruleset_text(ruleset):
    buf = io.StringIO()
    with redirect_stdout(buf):
        text_render.print_ruleset(ruleset)
    return buf.getvalue()


def _extract_init_object_keys(ruleset_text):
    # Parse "INIT TILES" section for color+tile mentions.
    lines = [ln.strip() for ln in ruleset_text.splitlines()]
    init_lines = []
    in_init = False
    for ln in lines:
        if not ln:
            continue
        key = ln.upper().rstrip(":")
        if key.startswith("INIT TILES"):
            in_init = True
            continue
        if key in {"GOAL", "RULES"}:
            in_init = False
            continue
        if in_init:
            init_lines.append(ln)

    color_vocab = _build_color_vocab()
    tile_vocab = _build_tile_vocab()

    object_keys = set()
    for ln in init_lines:
        tokens = re.sub(r"[^a-zA-Z_\\s]", " ", ln.lower()).split()
        for color in color_vocab:
            if color not in tokens:
                continue
            for tile in tile_vocab:
                if tile in tokens:
                    object_keys.add(f"{color}_{tile}")
    return object_keys


def test_ctx_lookup_does_not_fall_back_when_object_exists():
    """Ensure ctx.get lookups used in dense_reward retrieve real object coords."""
    grid = np.zeros((3, 3, 2), dtype=np.int32)
    grid[1, 1, 0] = Tiles.SQUARE
    grid[1, 1, 1] = Colors.YELLOW
    grid[2, 0, 0] = Tiles.BALL
    grid[2, 0, 1] = Colors.GREEN

    ts_prev = _make_timestep(grid, step_num=1, step_type=1)
    ts_next = _make_timestep(grid, step_num=2, step_type=1)

    ctx = extract_xland_ctx(None, ts_prev, ts_next)

    fallback = jnp.array([-1, -1], dtype=jnp.int32)
    obj_pos = ctx.get("object_positions", {})

    yellow_square = obj_pos.get("yellow_square", ctx.get("yellow_square_pos", fallback))
    green_ball = obj_pos.get("green_ball", ctx.get("green_ball_pos", fallback))
    yellow_square_prev = obj_pos.get(
        "yellow_square_prev", ctx.get("yellow_square_pos_prev", fallback)
    )
    green_ball_prev = obj_pos.get(
        "green_ball_prev", ctx.get("green_ball_pos_prev", fallback)
    )

    expected = {
        "yellow_square_pos": jnp.array([1, 1], dtype=jnp.int32),
        "green_ball_pos": jnp.array([2, 0], dtype=jnp.int32),
        "yellow_square_pos_prev": jnp.array([1, 1], dtype=jnp.int32),
        "green_ball_pos_prev": jnp.array([2, 0], dtype=jnp.int32),
    }

    for name, value in {
        "yellow_square_pos": yellow_square,
        "green_ball_pos": green_ball,
        "yellow_square_pos_prev": yellow_square_prev,
        "green_ball_pos_prev": green_ball_prev,
    }.items():
        assert not np.array_equal(
            _to_numpy(value), _to_numpy(fallback)
        ), f"{name} unexpectedly fell back to [-1, -1]\nctx: {ctx}"
        assert np.array_equal(_to_numpy(value), _to_numpy(expected[name]))


def test_ctx_lookup_matches_ruleset_init_objects():
    """Use real XLand rulesets to confirm ctx lookups for declared init objects."""
    env_id = "XLand-MiniGrid-R1-11x11"
    benchmark_id = "trivial-1m"

    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    benchmark = xminigrid.load_benchmark(benchmark_id)

    rng = jax.random.key(0)
    ruleset = None
    object_keys = set()
    # Find a ruleset that declares at least one explicit object in INIT TILES.
    for key in jax.random.split(rng, num=50):
        candidate = benchmark.sample_ruleset(key)
        ruleset_text = _extract_ruleset_text(candidate)
        object_keys = _extract_init_object_keys(ruleset_text)
        if object_keys:
            ruleset = candidate
            break

    if ruleset is None:
        pytest.skip("No ruleset with explicit init objects found in sample.")

    env_params = env_params.replace(ruleset=ruleset)
    reset_key = jax.random.key(123)
    ts_prev = env.reset(env_params, reset_key)
    ts_next = env.step(env_params, ts_prev, jnp.int32(0))
    ctx = extract_xland_ctx(env_params, ts_prev, ts_next)

    fallback = jnp.array([-1, -1], dtype=jnp.int32)
    obj_pos = ctx.get("object_positions", {})

    for key in sorted(object_keys):
        print("key:", key)
        value = obj_pos.get(key, fallback)
        assert not np.array_equal(
            _to_numpy(value), _to_numpy(fallback)
        ), f"{key} unexpectedly fell back to [-1, -1]\nctx: {ctx}"

