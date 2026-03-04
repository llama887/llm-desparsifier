from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from llm_desparsifier.rl import eval as eval_mod
from llm_desparsifier.rl.eval import GroundTruthEvalConfig, _key_to_list


def test_key_to_list_supports_typed_prng_keys() -> None:
    """Ensure typed JAX keys serialize without crashing trajectory capture.

    This test guards against regressions where trajectory serialization attempts
    to cast typed keys (`key<fry>`) directly to `uint32`, which raises
    `Cannot convert_element_type from key<fry> to uint32` on modern JAX. It is
    needed because GEPA candidate runs can spend minutes training before hitting
    eval, and a key-serialization crash at that point silently prevents
    `eval_trajectory.json` from being written. It differs from broad integration
    tests by validating the exact low-level conversion helper that replay
    tooling depends on.
    """

    key = jax.random.key(123)
    expected = [int(v) for v in jax.random.key_data(key).reshape(-1).tolist()]
    assert _key_to_list(key) == expected


def test_key_to_list_supports_legacy_uint32_prng_keys() -> None:
    """Verify legacy uint32 PRNG keys remain compatible with serialization.

    This test ensures `_key_to_list` keeps working for call sites that still use
    `jax.random.PRNGKey`, which returns classic `uint32[2]` arrays. It is needed
    because replay payloads may be produced by mixed key styles during library
    transitions, and it differs from the typed-key test by validating backward
    compatibility rather than typed-key correctness.
    """

    key = jax.random.PRNGKey(456)
    expected = [int(v) for v in jax.random.key_data(key).reshape(-1).tolist()]
    assert _key_to_list(key) == expected


def test_run_ground_truth_eval_uses_one_ruleset_and_many_reset_seeds(
    monkeypatch,
) -> None:
    """Verify sparse evaluation reuses one task while varying reset randomness.

    This test exercises `run_ground_truth_eval` directly with deterministic
    rulesets enabled and a fixed ruleset seed. It is needed because the GEPA
    contract depends on evaluating multiple rollouts from the same task
    semantics (`env_text`) while still exploring different episode initial
    states via new reset keys. It differs from the key-serialization tests
    above by validating the higher-level rollout contract rather than a helper.
    """

    fixed_ruleset = {"ruleset_id": "fixed-task"}
    reset_records: list[dict[str, Any]] = []

    class DummyBenchmark:
        """Benchmark stub that exposes deterministic ruleset selection."""

        def sample_ruleset(self, _key):
            """Return the same sentinel ruleset for fixed-ruleset evaluation."""

            return fixed_ruleset

    class DummyEnvParams:
        """Tiny EnvParams replacement with the `replace` API used in eval."""

        def __init__(self, ruleset=None):
            self.ruleset = ruleset

        def replace(self, **kwargs):
            """Return a new params object with requested field updates."""

            return DummyEnvParams(ruleset=kwargs.get("ruleset", self.ruleset))

    class DummyTimeStep:
        """Minimal timestep object that ends after one step."""

        def __init__(self, reward: float, done: bool):
            self.reward = jnp.asarray(reward, dtype=jnp.float32)
            self.observation = jnp.zeros((1,), dtype=jnp.float32)
            self._done = done

        def last(self):
            """Expose the terminal flag with the real timestep API."""

            return self._done

    class DummyEnv:
        """Environment stub that records reset keys and ruleset identity."""

        def reset(self, episode_params, reset_key):
            """Record the reset key used for the episode and start at reward 0."""

            reset_records.append(
                {
                    "ruleset": episode_params.ruleset,
                    "reset_key": _key_to_list(reset_key),
                }
            )
            return DummyTimeStep(reward=0.0, done=False)

        def step(self, _episode_params, _timestep, _action):
            """Terminate immediately with a positive sparse reward."""

            return DummyTimeStep(reward=1.0, done=True)

    class DummyDist:
        """Policy distribution stub that always picks the same action."""

        def sample(self, seed):
            """Ignore RNG and return one deterministic action."""

            _ = seed
            return jnp.asarray([0], dtype=jnp.int32)

    class DummyModel:
        """Policy stub that satisfies the eval harness API."""

        def initialize_carry(self, _batch_size):
            """Return a placeholder hidden state."""

            return None

        def apply(self, _params, _obs, hidden):
            """Return a deterministic distribution and unchanged hidden state."""

            return DummyDist(), None, hidden

    monkeypatch.setattr(
        eval_mod, "_build_eval_env", lambda _cfg: (DummyEnv(), DummyEnvParams(), DummyBenchmark())
    )
    monkeypatch.setattr(eval_mod, "describe_ruleset", lambda _env, params: f"env-text:{params.ruleset['ruleset_id']}")
    monkeypatch.setattr(eval_mod.jax, "jit", lambda fn: fn)

    cfg = GroundTruthEvalConfig(
        env_id="XLand-MiniGrid-R1-9x9",
        benchmark_id="trivial-1m",
        num_episodes=3,
        seed=7,
        deterministic_rulesets=True,
        fixed_ruleset_seed=11,
        capture_trajectory=True,
        trajectory_episode_index=0,
    )
    result = eval_mod.run_ground_truth_eval(
        train_state=type("State", (), {"params": object()})(),
        model=DummyModel(),
        cfg=cfg,
    )

    assert result.returns == [1.0, 1.0, 1.0]
    assert len(reset_records) == 3
    assert all(record["ruleset"] is fixed_ruleset for record in reset_records)
    assert len({tuple(record["reset_key"]) for record in reset_records}) == 3
    assert result.trajectory is not None
    assert result.trajectory["env_text"] == "env-text:fixed-task"
    assert result.trajectory["fixed_ruleset_seed"] == 11
    assert result.trajectory["ruleset_key"] is None
