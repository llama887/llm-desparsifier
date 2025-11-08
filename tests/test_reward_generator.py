from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from llm_desparsifier.rewards.generator import RewardGenerator


class _DummySynth:
    def __call__(self, env_description: str, constraints: str) -> str:
        self.env_description = env_description
        self.constraints = constraints
        return (
            "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
            "    progress = jnp.asarray(0.0, dtype=jnp.float32)\n"
            "    reward_components = {'progress': progress}\n"
            "    return progress, reward_components\n"
        )


class _TestableRewardGenerator(RewardGenerator):
    def __post_init__(self) -> None:
        # Skip DSPy configuration for tests.
        if self.lm is None:
            self.lm = object()


def test_reward_generator_invokes_components():
    synth = _DummySynth()
    captured: dict[str, Any] = {}

    def fake_describe(env, env_params):
        return "env description"

    def fake_sanitize(code: str):
        captured["code"] = code

        def dense_reward(*_args, **_kwargs):
            return 42.0, {"progress": 42.0}

        return dense_reward

    generator = _TestableRewardGenerator(
        synthesizer=synth,
        describe_fn=fake_describe,
        sanitize_fn=fake_sanitize,
        lm=object(),
        verbose=False,
    )

    dense_fn, emitted = generator.generate(env=object(), env_params=object())

    assert emitted == captured["code"]
    assert synth.env_description == "env description"
    assert "dense_reward" in emitted
    assert dense_fn(None, None, None, None, {})[0] == 42.0
