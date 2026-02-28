from __future__ import annotations

from llm_desparsifier.rewards.reward_key_diagnostics import (
    build_reward_object_key_diagnostics,
)


def test_reward_key_diagnostics_extracts_alias_and_chained_get_keys() -> None:
    """Extract keys from both alias-based and chained map access styles.

    This test verifies that diagnostics capture object keys regardless of
    whether the reward code first assigns `ctx.get("object_positions", {})` to a
    local alias or performs direct chained lookups. It is needed because LLM
    reward code commonly mixes both styles in one function.
    """

    reward_code = """
def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    object_positions = ctx.get("object_positions", {})
    blue_square = object_positions.get("blue_square", jnp.array([-1, -1], dtype=jnp.int32))
    red_key = ctx.get("visible_object_positions", {}).get("red_key", jnp.array([-1, -1], dtype=jnp.int32))
    visible_prev = ctx.get("visible_object_positions_prev", {})
    _ = visible_prev.get("green_pyramid", jnp.array([-1, -1], dtype=jnp.int32))
    reward_components = {"progress": jnp.asarray(0.0, dtype=jnp.float32)}
    return reward_components["progress"], reward_components
"""
    env_text = '"blue_square" "green_pyramid" "red_key"'

    diagnostics = build_reward_object_key_diagnostics(reward_code, env_text)

    assert diagnostics.referenced_object_keys == (
        "blue_square",
        "green_pyramid",
        "red_key",
    )
    assert diagnostics.missing_from_task == ()


def test_reward_key_diagnostics_handles_markdown_fenced_code() -> None:
    """Support reward artifacts saved as Markdown-fenced Python blocks.

    This test guards parsing for the common artifact format where reward code is
    wrapped in triple backticks. It is needed so diagnostics can run directly on
    persisted reward files without pre-cleaning.
    """

    reward_code = """```python
def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    object_positions = ctx.get("object_positions", {})
    _ = object_positions.get("blue_square", jnp.array([-1, -1], dtype=jnp.int32))
    reward_components = {"progress": jnp.asarray(0.0, dtype=jnp.float32)}
    return reward_components["progress"], reward_components
```"""
    env_text = '"blue_square"'

    diagnostics = build_reward_object_key_diagnostics(reward_code, env_text)

    assert diagnostics.referenced_object_keys == ("blue_square",)
    assert diagnostics.missing_from_task == ()


def test_reward_key_diagnostics_reports_missing_object_keys() -> None:
    """Report reward lookups that do not appear in task description text.

    This regression-style test models the failure case where dense reward code
    targets `red_key` while the task description only includes `blue_square` and
    `green_pyramid`. It is needed to ensure GEPA feedback receives deterministic
    mismatch diagnostics for exactly this bug class.
    """

    reward_code = """
def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    object_positions = ctx.get("object_positions", {})
    _ = object_positions.get("red_key", jnp.array([-1, -1], dtype=jnp.int32))
    reward_components = {"progress": jnp.asarray(0.0, dtype=jnp.float32)}
    return reward_components["progress"], reward_components
"""
    env_text = (
        'Your task is to place the "blue_square" immediately left of the '
        '"green_pyramid".'
    )

    diagnostics = build_reward_object_key_diagnostics(reward_code, env_text)

    assert diagnostics.referenced_object_keys == ("red_key",)
    assert diagnostics.task_object_keys == ("blue_square", "green_pyramid")
    assert diagnostics.missing_from_task == ("red_key",)
