from __future__ import annotations

import textwrap

import jax.numpy as jnp
import pytest

from llm_desparsifier.rewards.sanitizer import sanitize_and_compile


def test_sanitize_and_compile_allows_simple_reward():
    code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            zeros = jnp.asarray(0.0, dtype=jnp.float32)
            reward_components = {"progress": zeros}
            return zeros, reward_components
        """
    )
    fn = sanitize_and_compile(code)
    result = fn(None, None, None, None, {})
    assert isinstance(result, tuple)
    assert result[0].shape == ()
    assert isinstance(result[1], dict)
    assert set(result[1].keys()) == {"progress"}


def test_sanitize_and_compile_blocks_imports():
    bad_code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            import os
            zeros = jnp.asarray(0.0, dtype=jnp.float32)
            reward_components = {"progress": zeros}
            return zeros, reward_components
        """
    )
    with pytest.raises(ValueError):
        sanitize_and_compile(bad_code)


def test_sanitize_and_compile_forbids_ctx_subscript():
    bad_code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            obj_pos = ctx["object_positions"]
            zeros = jnp.asarray(0.0, dtype=jnp.float32)
            reward_components = {"progress": zeros}
            return zeros, reward_components
        """
    )
    with pytest.raises(ValueError):
        sanitize_and_compile(bad_code)


def test_sanitizer_allows_dict_get_calls():
    code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            object_positions = ctx.get("object_positions", {})
            yellow = object_positions.get("yellow_square", jnp.array([0, 0], dtype=jnp.int32))
            zeros = jnp.asarray(yellow[0], dtype=jnp.float32)
            reward_components = {"progress": zeros}
            return zeros, reward_components
        """
    )
    fn = sanitize_and_compile(code)
    result = fn(None, None, None, None, {"object_positions": {"yellow_square": jnp.array([1, 1])}})
    assert result[0].shape == ()


def test_sanitizer_blocks_disallowed_methods():
    bad_code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            object_positions = ctx.get("object_positions", {})
            _ = object_positions.items()
            zeros = jnp.asarray(0.0, dtype=jnp.float32)
            reward_components = {"progress": zeros}
            return zeros, reward_components
        """
    )
    with pytest.raises(ValueError):
        sanitize_and_compile(bad_code)


def test_sanitizer_strips_fenced_code_with_language_tag():
    code = textwrap.dedent(
        """
        ```python   
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            zeros = jnp.asarray(0.0, dtype=jnp.float32)
            reward_components = {"progress": zeros}
            return zeros, reward_components
        ```   
        """
    )
    fn = sanitize_and_compile(code)
    result = fn(None, None, None, None, {})
    assert result[0].shape == ()


def test_sanitizer_strips_fenced_code_with_leading_whitespace():
    code = textwrap.dedent(
        """
            ```python
            def dense_reward(env_params, ts_prev, action, ts_next, ctx):
                zeros = jnp.asarray(0.0, dtype=jnp.float32)
                reward_components = {"progress": zeros}
                return zeros, reward_components
            ```
        """
    )
    fn = sanitize_and_compile(code)
    result = fn(None, None, None, None, {})
    assert result[0].shape == ()
