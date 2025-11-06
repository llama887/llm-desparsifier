"""Compatibility layer for legacy imports."""

from __future__ import annotations

import jax.numpy as jnp

from llm_desparsifier.rl.wrappers import DesparsifyRewardWrapper, RewardTimeStep

__all__ = ["RewardTimeStep", "DesparsifyRewardWrapper", "dummy_dense_reward"]


def dummy_dense_reward(ts_prev, action, ts_next):
    """Placeholder dense reward that mirrors the sparse signal."""
    ones = jnp.ones_like(ts_next.reward)
    zeros = jnp.full_like(ts_next.reward, 0.0)
    return jnp.where(ts_next.last() > 0, zeros, zeros)
