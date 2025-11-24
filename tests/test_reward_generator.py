from __future__ import annotations

import tempfile
from pathlib import Path
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
        verbose=False,
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

    with tempfile.TemporaryDirectory() as tmpdir:
        failure_dir = Path(tmpdir)
        generator = _TestableRewardGenerator(
            synthesizer=synth,
            describe_fn=lambda *_: "env",
            sanitize_fn=fake_sanitize,
            lm=object(),
            verbose=False,
            max_sanitize_attempts=3,
            failure_artifact_dir=failure_dir,
        )

        dense_fn, _ = generator.generate(env=object(), env_params=object())

        assert attempt_counter["count"] == 2
        assert len(synth.calls) == 2
        retry_prompt = synth.calls[1]
        assert "### Sanitizer retry guidance" in retry_prompt
        assert "Attempt 1" in retry_prompt
        assert "use ctx.get" in retry_prompt
        assert dense_fn(None, None, None, None, {})[0] == 7.0

        code_files = list(failure_dir.glob("attempt-01-*.py"))
        err_files = list(failure_dir.glob("attempt-01-*.err.txt"))
        feedback_files = list(failure_dir.glob("attempt-01-*.feedback.md"))
        assert code_files and err_files and feedback_files


def test_reward_generator_raises_after_max_attempts():
    synth = _RetrySynth()

    def always_fail(_code: str):
        raise ValueError("still invalid")

    with tempfile.TemporaryDirectory() as tmpdir:
        failure_dir = Path(tmpdir)
        generator = _TestableRewardGenerator(
            synthesizer=synth,
            describe_fn=lambda *_: "env",
            sanitize_fn=always_fail,
            lm=object(),
            verbose=False,
            max_sanitize_attempts=2,
            failure_artifact_dir=failure_dir,
        )

        with pytest.raises(RuntimeError) as excinfo:
            generator.generate(env=object(), env_params=object())

        message = str(excinfo.value)
        assert "Failed to sanitize" in message
        assert "Attempt 1" in message and "Attempt 2" in message
        assert len(list(failure_dir.glob("attempt-*.py"))) == 2
