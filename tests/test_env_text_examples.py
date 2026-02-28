from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

xminigrid = pytest.importorskip("xminigrid")
text_render = pytest.importorskip("xminigrid.rendering.text_render")

from llm_desparsifier.rewards import parser as parser_mod


def _ruleset_summary(env_params) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        text_render.print_ruleset(env_params.ruleset)
    return buf.getvalue().strip()


def _underscore(name: str) -> str:
    return name.lower().replace(" ", "_")


def _goal_object_keys(goal_line: str) -> list[str]:
    match = parser_mod._GOAL_AGENT_HOLD_RE.search(goal_line)
    if match:
        return [_underscore(match.group(1).strip())]

    match = parser_mod._GOAL_AGENT_NEAR_DIR_RE.search(goal_line)
    if match:
        return [_underscore(match.group(2).strip())]

    match = parser_mod._GOAL_AGENT_NEAR_RE.search(goal_line)
    if match:
        return [_underscore(match.group(1).strip())]

    match = parser_mod._GOAL_TILE_NEAR_DIR_RE.search(goal_line)
    if match:
        return [
            _underscore(match.group(2).strip()),
            _underscore(match.group(3).strip()),
        ]

    match = parser_mod._GOAL_TILE_NEAR_RE.search(goal_line)
    if match:
        return [
            _underscore(match.group(1).strip()),
            _underscore(match.group(2).strip()),
        ]

    return []


def test_env_text_inline_examples_match_ruleset():
    env, env_params = xminigrid.make("XLand-MiniGrid-R1-11x11")
    try:
        benchmark = xminigrid.load_benchmark("trivial-1m")
        env_params = env_params.replace(ruleset=benchmark.get_ruleset(0))
    except Exception as exc:  # pragma: no cover - optional data dependency
        pytest.skip(f"benchmark ruleset unavailable: {exc}")

    summary = _ruleset_summary(env_params)
    goal_line, _rule_lines, init_lines = parser_mod._parse_ruleset_text(summary)

    env_text = parser_mod.describe_ruleset(env, env_params)

    assert 'ctx.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))' in env_text
    assert 'ctx.get("object_positions", {})' in env_text
    assert 'ctx.get("visible_object_positions", {})' in env_text
    assert 'ctx.get("visible_object_positions_prev", {})' in env_text
    assert 'ctx.get("observation", ts_next.observation).astype(jnp.int32)' in env_text
    assert 'ctx.get("observation_prev", ts_prev.observation).astype(jnp.int32)' in env_text
    assert "1=turn_right (clockwise)" in env_text
    assert "2=turn_left (counterclockwise)" in env_text
    assert "obs[..., 0] is tile id and obs[..., 1] is color id" in env_text

    if init_lines:
        init_keys = [_underscore(obj) for obj in init_lines]
        example_key = init_keys[0]
        assert f'.get("{example_key}", jnp.array([-1, -1], dtype=jnp.int32))' in env_text
        for key in init_keys[1:]:
            assert f'"{key}"' in env_text

    if goal_line:
        goal_keys = _goal_object_keys(goal_line)
        for key in goal_keys:
            assert f'.get("{key}", jnp.array([-1, -1], dtype=jnp.int32))' in env_text


def test_goal_sentences_flip_agent_direction_tokens_to_world_relation() -> None:
    """Ensure directional agent goals use correct world-relative wording.

    This regression test verifies that parser-side natural-language goals reflect
    the true XMiniGrid directional semantics for `AgentNear*Goal` variants. It
    is needed because the benchmark token names encode direction from an inverse
    operand perspective, and it differs from end-to-end env-text tests by
    exercising `_goal_sentences` directly with synthetic goal lines.
    """

    direction_expectations = {
        "Up": "down",
        "Right": "left",
        "Down": "up",
        "Left": "right",
    }
    for goal_token, expected_relation in direction_expectations.items():
        task, success = parser_mod._goal_sentences(
            f"AgentNear{goal_token}Goal(green pyramid)",
            agent_pos_example='ctx.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))',
            obj_positions_example='ctx.get("object_positions", {})',
        )
        assert f"immediately {expected_relation} of the green pyramid" in task
        assert f"one cell {expected_relation} of the green pyramid" in success


def test_goal_sentences_flip_tile_direction_tokens_to_world_relation() -> None:
    """Ensure directional tile goals use correct world-relative wording.

    This test protects against left/right and up/down inversions in generated
    objective text for `TileNear*Goal(a, b)` expressions. It is needed because
    incorrect language can steer both human play and LLM reward shaping toward
    the wrong spatial relation, and it differs from benchmark-driven tests by
    validating deterministic parser output with explicit expected phrases.
    """

    task, success = parser_mod._goal_sentences(
        "TileNearLeftGoal(blue square, green pyramid)",
        agent_pos_example='ctx.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))',
        obj_positions_example='ctx.get("object_positions", {})',
    )

    assert "immediately right of the green pyramid" in task
    assert "one cell right of the green pyramid" in success
    assert "first column is exactly one greater than the second" in success
