from __future__ import annotations

import pytest

from llm_desparsifier.rewards.sanitizer import sanitize_and_compile


def test_sanitizer_allows_astype_on_expression():
    code = (
        "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
        "    step_num = ctx.get('step_num', jnp.array(0, dtype=jnp.int32))\n"
        "    progress = jnp.where(step_num > 0, step_num, jnp.array(0)).astype(jnp.float32)\n"
        "    reward_components = {'progress': progress}\n"
        "    total_reward = reward_components['progress']\n"
        "    return total_reward, reward_components\n"
    )
    dense_reward = sanitize_and_compile(code)
    assert callable(dense_reward)
    assert getattr(dense_reward, '__reward_component_keys__', None) == ("progress",)


def test_sanitizer_allows_inline_reward_components_return():
    code = (
        "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
        "    progress = jnp.asarray(0.0, dtype=jnp.float32)\n"
        "    penalty = jnp.asarray(0.0, dtype=jnp.float32)\n"
        "    total_reward = progress + penalty\n"
        "    return total_reward, {'progress': progress, 'penalty': penalty}\n"
    )
    dense_reward = sanitize_and_compile(code)
    assert callable(dense_reward)
    assert getattr(dense_reward, '__reward_component_keys__', None) == ("progress", "penalty")


def test_sanitizer_rejects_inconsistent_return_keys():
    code = (
        "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
        "    progress = jnp.asarray(0.0, dtype=jnp.float32)\n"
        "    total_reward = progress\n"
        "    if True:\n"
        "        return total_reward, {'progress': progress}\n"
        "    return total_reward, {'penalty': progress}\n"
    )
    with pytest.raises(ValueError, match="reward_components keys must remain constant"):
        sanitize_and_compile(code)
