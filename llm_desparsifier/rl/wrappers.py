"""Environment wrappers for integrating dense rewards."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace as _py_replace

import jax.numpy as jnp
from flax import struct

try:
    from flax.core.frozen_dict import freeze
except ImportError:  # pragma: no cover - flax always available during training
    def freeze(value):
        return value


@struct.dataclass
class RewardTimeStep(struct.PyTreeNode):
    """Keeps the original dm_env.TimeStep immutable while attaching extras."""

    base: object
    extras: Mapping = struct.field(default_factory=dict)

    def __getattr__(self, name):
        return getattr(self.base, name)


class DesparsifyRewardWrapper:
    """Overwrite env rewards with dense functions while preserving the env API."""

    def __init__(self, env, dense_fn, ctx_fn=None):
        """
        Args:
            env: Base environment following the dm_env API.
            dense_fn: Callable with signature
                `(ts_prev, action, ts_next)` or
                `(env_params, ts_prev, action, ts_next, ctx)`.
            ctx_fn: Optional callable returning a context dict from
                `(env_params, ts_prev, ts_next)`. Must be JAX-compatible if used under jit/vmap.
        """
        self.env = env
        self.dense_fn = dense_fn
        self.ctx_fn = ctx_fn
        self._dense_nargs = len(inspect.signature(dense_fn).parameters)
        self._component_keys = tuple(getattr(dense_fn, "__reward_component_keys__", ()))
        self._component_template = freeze({name: jnp.float32(0.0) for name in self._component_keys}) if self._component_keys else None

    def _augment_extras(self, ts, original_reward, dense_reward, reward_components):
        source = ts.extras if isinstance(ts, RewardTimeStep) else getattr(ts, "extras", None)
        if isinstance(source, Mapping):
            extras_out = dict(source)
        elif source is not None:
            copy_fn = getattr(source, "copy", None)
            if callable(copy_fn):
                extras_out = copy_fn()
            else:
                try:
                    extras_out = dict(source)
                except TypeError:
                    extras_out = {"_wrapped_extras": source}
        else:
            extras_out = {}
        extras_out["ground_truth_reward"] = original_reward
        extras_out["dense_reward"] = dense_reward
        normalized = self._normalize_reward_components(reward_components)
        if normalized is not None:
            extras_out["reward_components"] = normalized
        elif self._component_template is not None and "reward_components" not in extras_out:
            extras_out["reward_components"] = self._component_template
        return freeze(extras_out)

    def _wrap_timestep(self, ts, original_reward, dense_reward, reward_components=None):
        extras = self._augment_extras(ts, original_reward, dense_reward, reward_components)
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
        ts_base = ts.base if isinstance(ts, RewardTimeStep) else ts
        ts_next = self.env.step(env_params, ts_base, action)
        original_reward = ts_next.reward
        if self.ctx_fn is not None:
            ctx = self.ctx_fn(env_params, ts, ts_next)
        else:
            extras = getattr(ts_next, "extras", None)
            ctx = extras if extras is not None else {}

        if self._dense_nargs == 3:
            dense_output = self.dense_fn(ts, action, ts_next)
        elif self._dense_nargs == 5:
            dense_output = self.dense_fn(env_params, ts, action, ts_next, ctx)
        else:  # fallback: keep sparse reward
            dense_output = ts_next.reward

        reward_components = None
        if isinstance(dense_output, tuple) and len(dense_output) == 2:
            dense_reward, reward_components = dense_output
            if reward_components is not None and not isinstance(reward_components, Mapping):
                raise TypeError("reward_components must be a mapping of component_name -> value")
        else:
            dense_reward = dense_output
        return self._wrap_timestep(ts_next, original_reward, dense_reward, reward_components)

    def _normalize_reward_components(self, reward_components):
        if reward_components is None:
            return None
        frozen = freeze(reward_components)
        if not self._component_keys:
            return frozen
        actual_keys = tuple(frozen.keys())
        expected_set = set(self._component_keys)
        if set(actual_keys) != expected_set:
            missing = expected_set - set(actual_keys)
            extra = set(actual_keys) - expected_set
            raise ValueError(
                "reward_components keys must remain constant. "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        ordered = {name: frozen[name] for name in self._component_keys}
        return freeze(ordered)

    def num_actions(self, env_params):
        return self.env.num_actions(env_params)

    def observation_shape(self, env_params):
        return self.env.observation_shape(env_params)

    def render(self, env_params, ts):
        ts_base = ts.base if isinstance(ts, RewardTimeStep) else ts
        return self.env.render(env_params, ts_base)

    def __getattr__(self, name):
        return getattr(self.env, name)


__all__ = ["RewardTimeStep", "DesparsifyRewardWrapper"]
