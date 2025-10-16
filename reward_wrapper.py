from dataclasses import replace

class DesparsifyRewardWrapper:
    """Overwrite env reward with a dense heuristic but keep the original env API."""
    def __init__(self, env, dense_fn):
        self.env = env
        self.dense_fn = dense_fn

    # === Core step/reset (vectorizable, JIT-safe) ===
    def reset(self, env_params, key):
        return self.env.reset(env_params, key)

    def step(self, env_params, ts, action):
        ts_next = self.env.step(env_params, ts, action)
        r_dense = self.dense_fn(ts, action, ts_next)
        # Prefer Flax struct's .replace if available, else fallback to dataclasses.replace
        if hasattr(ts_next, "replace"):
            return ts_next.replace(reward=r_dense)
        return _py_replace(ts_next, reward=r_dense)

    # === Forward the env API you use elsewhere ===
    def num_actions(self, env_params):
        return self.env.num_actions(env_params)

    def observation_shape(self, env_params):
        return self.env.observation_shape(env_params)

    # Optional but handy if you ever call it through the wrapper
    def render(self, env_params, ts):
        return self.env.render(env_params, ts)

    # Generic passthrough for anything else (safe outside JIT traces)
    def __getattr__(self, name):
        # Only called if attribute not found on wrapper
        return getattr(self.env, name)

import jax.numpy as jnp

def dummy_dense_reward(ts_prev, action, ts_next):
    ones = jnp.ones_like(ts_next.reward)
    zeros = jnp.full_like(ts_next.reward, 0.0)
    return jnp.where(ts_next.last() > 0, zeros, zeros)
