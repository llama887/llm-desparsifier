# file: reward_wrapper.py
from dataclasses import replace as _py_replace
import inspect
from collections.abc import Mapping

from flax import struct

try:
    from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
except ImportError:  # pragma: no cover - flax always available during training
    FrozenDict = dict  # type: ignore[misc,assignment]

    def freeze(value):
        return value

    def unfreeze(value):
        return dict(value)


@struct.dataclass
class RewardTimeStep(struct.PyTreeNode):
    """Wrapper that keeps the original dm_env.TimeStep immutable but adds extras."""

    base: object
    extras: Mapping = struct.field(default_factory=dict)

    def replace(self, **kwargs):
        extras = kwargs.pop("extras", self.extras)
        new_base = self.base.replace(**kwargs)
        return RewardTimeStep(new_base, extras)

    def __getattr__(self, name):
        return getattr(self.base, name)

class DesparsifyRewardWrapper:
    """Overwrite env reward with a dense heuristic but keep the original env API."""
    def __init__(self, env, dense_fn, ctx_fn=None):
        """
        dense_fn: fn(ts_prev, action, ts_next) or fn(env_params, ts_prev, action, ts_next, ctx)
        ctx_fn: optional fn(env_params, ts_prev, ts_next) -> dict[str, jax.Array]
                Must be JAX-compatible if used under jit/vmap.
        """
        self.env = env
        self.dense_fn = dense_fn
        self.ctx_fn = ctx_fn

        # detect arity once (jit-friendly)
        self._dense_nargs = len(inspect.signature(dense_fn).parameters)

    def _unwrap(self, ts):
        return ts.base if isinstance(ts, RewardTimeStep) else ts

    def _augment_extras(self, ts, original_reward, dense_reward):
        source = ts.extras if isinstance(ts, RewardTimeStep) else getattr(ts, "extras", None)
        if source is not None:
            extras = source
            if extras is None:
                extras_out = {}
            elif isinstance(extras, FrozenDict):
                extras_out = unfreeze(extras)
            elif isinstance(extras, Mapping):
                extras_out = dict(extras)
            else:
                copy_fn = getattr(extras, "copy", None)
                if callable(copy_fn):
                    extras_out = copy_fn()
                else:
                    try:
                        extras_out = dict(extras)
                    except TypeError:
                        extras_out = {"_wrapped_extras": extras}
        else:
            extras_out = {}
        extras_out["ground_truth_reward"] = original_reward
        extras_out["dense_reward"] = dense_reward
        return freeze(extras_out)

    def _wrap_timestep(self, ts, original_reward, dense_reward):
        extras = self._augment_extras(ts, original_reward, dense_reward)
        if hasattr(ts, "replace"):
            base = ts.replace(reward=dense_reward)
        else:
            base = _py_replace(ts, reward=dense_reward)
        return RewardTimeStep(base=base, extras=extras)

    def reset(self, env_params, key):
        ts = self.env.reset(env_params, key)
        reward = ts.reward if hasattr(ts, "reward") else 0.0
        return self._wrap_timestep(ts, reward, reward)

    def step(self, env_params, ts, action):
        ts_base = self._unwrap(ts)
        ts_next = self.env.step(env_params, ts_base, action)
        original_reward = ts_next.reward
        if self.ctx_fn is not None:
            ctx = self.ctx_fn(env_params, ts, ts_next)
        else:
            extras = getattr(ts_next, "extras", None)
            ctx = extras if extras is not None else {}

        # Call the user/LLM reward in a way that preserves JIT traces.
        if self._dense_nargs == 3:
            r_dense = self.dense_fn(ts, action, ts_next)
        elif self._dense_nargs == 5:
            r_dense = self.dense_fn(env_params, ts, action, ts_next, ctx)
        else:
            # Safe fallback – keep env reward (sparse) rather than crash inside jit
            r_dense = ts_next.reward

        return self._wrap_timestep(ts_next, original_reward, r_dense)

    # Convenience passthroughs
    def num_actions(self, env_params):
        return self.env.num_actions(env_params)

    def observation_shape(self, env_params):
        return self.env.observation_shape(env_params)

    def render(self, env_params, ts):
        return self.env.render(env_params, self._unwrap(ts))

    def __getattr__(self, name):
        return getattr(self.env, name)



import jax.numpy as jnp

def dummy_dense_reward(ts_prev, action, ts_next):
    ones = jnp.ones_like(ts_next.reward)
    zeros = jnp.full_like(ts_next.reward, 0.0)
    return jnp.where(ts_next.last() > 0, zeros, zeros)
