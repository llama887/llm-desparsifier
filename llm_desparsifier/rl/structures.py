"""Shared training data structures for the RL pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


class Transition(struct.PyTreeNode):
    """Single PPO transition captured during rollout."""

    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class RolloutStats(struct.PyTreeNode):
    """Aggregate statistics from evaluation rollouts."""

    reward: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0))
    ground_truth_reward: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0))
    length: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))
    episodes: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))


__all__ = ["Transition", "RolloutStats"]
