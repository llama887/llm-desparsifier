from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

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


class _RetrySynth:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, env_description: str, constraints: str) -> str:
        self.calls.append(constraints)
        return (
            "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
            "    progress = jnp.asarray(1.0, dtype=jnp.float32)\n"
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
    )

    dense_fn, emitted = generator.generate(env=object(), env_params=object())

    assert emitted == captured["code"]
    assert synth.env_description == "env description"
    assert "dense_reward" in emitted
    assert dense_fn(None, None, None, None, {})[0] == 42.0


def test_reward_generator_retries_until_success():
    synth = _RetrySynth()

    attempt_counter = {"count": 0}

    def fake_sanitize(code: str):
        attempt_counter["count"] += 1
        if attempt_counter["count"] == 1:
            raise ValueError("use ctx.get(...) instead of ctx[]")

        def dense_reward(*_args, **_kwargs):
            return 7.0, {"progress": 7.0}

        return dense_reward

    generator = _TestableRewardGenerator(
        synthesizer=synth,
        describe_fn=lambda *_: "env",
        sanitize_fn=fake_sanitize,
        lm=object(),
        max_sanitize_attempts=3,
    )

    dense_fn, _ = generator.generate(env=object(), env_params=object())

    assert attempt_counter["count"] == 2
    assert len(synth.calls) == 2
    retry_prompt = synth.calls[1]
    assert "### Sanitizer retry guidance" in retry_prompt
    assert "Attempt 1" in retry_prompt
    assert "use ctx.get" in retry_prompt
    assert dense_fn(None, None, None, None, {})[0] == 7.0

def test_reward_generator_raises_after_max_attempts():
    synth = _RetrySynth()

    def always_fail(_code: str):
        raise ValueError("still invalid")

    generator = _TestableRewardGenerator(
        synthesizer=synth,
        describe_fn=lambda *_: "env",
        sanitize_fn=always_fail,
        lm=object(),
        max_sanitize_attempts=2,
    )

    with pytest.raises(RuntimeError) as excinfo:
        generator.generate(env=object(), env_params=object())

    message = str(excinfo.value)
    assert "Failed to sanitize" in message
    assert "Attempt 1" in message and "Attempt 2" in message


def test_reward_generator_preserves_env_text_in_retry_loop():
    xminigrid = pytest.importorskip("xminigrid")

    class _CaptureEnvSynth:
        def __init__(self) -> None:
            self.env_descriptions: list[str] = []

        def __call__(self, env_description: str, constraints: str) -> str:
            self.env_descriptions.append(env_description)
            return (
                "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
                "    progress = jnp.asarray(1.0, dtype=jnp.float32)\n"
                "    reward_components = {'progress': progress}\n"
                "    return progress, reward_components\n"
            )

    env, env_params = xminigrid.make("XLand-MiniGrid-R1-11x11")
    synth = _CaptureEnvSynth()
    attempt_counter = {"count": 0}

    def flaky_sanitize(_code: str):
        attempt_counter["count"] += 1
        if attempt_counter["count"] == 1:
            raise ValueError("simulated failure")

        def dense_reward(*_args, **_kwargs):
            return 1.0, {"progress": 1.0}

        return dense_reward

    generator = _TestableRewardGenerator(
        synthesizer=synth,
        sanitize_fn=flaky_sanitize,
        lm=object(),
        max_sanitize_attempts=2,
    )

    dense_fn, _ = generator.generate(env=env, env_params=env_params)

    assert dense_fn(None, None, None, None, {})[0] == 1.0
    assert len(synth.env_descriptions) == 2
    assert synth.env_descriptions[0] == synth.env_descriptions[1]
    env_text = synth.env_descriptions[0]
    assert "XLand MiniGrid world" in env_text
    assert 'ctx.get("object_positions", {})' in env_text
