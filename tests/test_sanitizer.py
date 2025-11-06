from __future__ import annotations

import textwrap

import jax.numpy as jnp
import pytest

from llm_desparsifier.rewards.sanitizer import sanitize_and_compile


def test_sanitize_and_compile_allows_simple_reward():
    code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            return jnp.asarray(0.0, dtype=jnp.float32)
        """
    )
    fn = sanitize_and_compile(code)
    result = fn(None, None, None, None, {})
    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()


def test_sanitize_and_compile_blocks_imports():
    bad_code = textwrap.dedent(
        """
        def dense_reward(env_params, ts_prev, action, ts_next, ctx):
            import os
            return jnp.asarray(0.0, dtype=jnp.float32)
        """
    )
    with pytest.raises(ValueError):
        sanitize_and_compile(bad_code)
