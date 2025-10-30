# file: reward_wrapper.py
from dataclasses import replace as _py_replace
import inspect

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

    def reset(self, env_params, key):
        return self.env.reset(env_params, key)

    def step(self, env_params, ts, action):
        ts_next = self.env.step(env_params, ts, action)
        if self.ctx_fn is not None:
            ctx = self.ctx_fn(env_params, ts, ts_next)
        else:
            # Empty PyTree is fine for JAX if never used
            ctx = {}

        # Call the user/LLM reward in a way that preserves JIT traces.
        if self._dense_nargs == 3:
            r_dense = self.dense_fn(ts, action, ts_next)
        elif self._dense_nargs == 5:
            r_dense = self.dense_fn(env_params, ts, action, ts_next, ctx)
        else:
            # Safe fallback – keep env reward (sparse) rather than crash inside jit
            r_dense = ts_next.reward

        if hasattr(ts_next, "replace"):
            return ts_next.replace(reward=r_dense)
        return _py_replace(ts_next, reward=r_dense)

    # Convenience passthroughs
    def num_actions(self, env_params):
        return self.env.num_actions(env_params)

    def observation_shape(self, env_params):
        return self.env.observation_shape(env_params)

    def render(self, env_params, ts):
        return self.env.render(env_params, ts)

    def __getattr__(self, name):
        return getattr(self.env, name)



import jax.numpy as jnp

def dummy_dense_reward(ts_prev, action, ts_next):
    ones = jnp.ones_like(ts_next.reward)
    zeros = jnp.full_like(ts_next.reward, 0.0)
    return jnp.where(ts_next.last() > 0, zeros, zeros)
