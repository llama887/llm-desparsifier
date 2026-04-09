from __future__ import annotations

import json

from llm_desparsifier.rewards.generator import (
    RewardGenerator,
    persist_generated_reward_artifacts,
)


class _StubSynthesizer:
    """Return deterministic reward code without invoking a real LM backend."""

    def __init__(self, code: str) -> None:
        self._code = code

    def __call__(self, _env_description: str, _constraints: str) -> str:
        return self._code


def test_reward_generator_returns_canonical_sanitized_payload() -> None:
    code = """```python
def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    zeros = jnp.asarray(0.0, dtype=jnp.float32)
    reward_components = {"progress": zeros}
    return zeros, reward_components
```"""
    generator = RewardGenerator(
        synthesizer=_StubSynthesizer(code),
        constraints_text="constraints",
        describe_fn=lambda *_args: (
            'Your task is to move next to the "white_key". '
            'Success when the agent is adjacent to the "white_key".'
        ),
        lm=object(),
        include_sanitizer_code_on_retry=False,
    )

    generated = generator.generate(object(), object())

    assert generated.raw_code.startswith("```python")
    assert generated.sanitized_code.startswith("def dense_reward")
    assert generated.component_keys == ("progress",)
    assert generated.validation.status == "ok"
    assert generated.validation.failure_reason is None
    assert generated.validation.diagnostics["missing_from_task"] == []


def test_reward_generator_reports_task_mismatch_in_validation() -> None:
    code = (
        "def dense_reward(env_params, ts_prev, action, ts_next, ctx):\n"
        "    object_positions = ctx.get('object_positions', {})\n"
        "    red_key = object_positions.get('red_key', jnp.array([-1, -1], dtype=jnp.int32))\n"
        "    progress = red_key[0].astype(jnp.float32)\n"
        "    reward_components = {'progress': progress}\n"
        "    return progress, reward_components\n"
    )
    generator = RewardGenerator(
        synthesizer=_StubSynthesizer(code),
        constraints_text="constraints",
        describe_fn=lambda *_args: (
            'Your task is to move next to the "white_key". '
            'Success when the agent is adjacent to the "white_key".'
        ),
        lm=object(),
        include_sanitizer_code_on_retry=False,
    )

    generated = generator.generate(object(), object())

    assert generated.validation.status == "invalid_task_mismatch"
    assert generated.validation.failure_reason is not None
    assert "red_key" in generated.validation.failure_reason
    assert generated.validation.diagnostics["missing_from_task"] == ["red_key"]


def test_persist_generated_reward_artifacts_writes_canonical_and_raw_files(
    tmp_path,
) -> None:
    code = """```python
def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    zeros = jnp.asarray(0.0, dtype=jnp.float32)
    reward_components = {"progress": zeros}
    return zeros, reward_components
```"""
    generator = RewardGenerator(
        synthesizer=_StubSynthesizer(code),
        constraints_text="constraints",
        describe_fn=lambda *_args: '"blue_square"',
        lm=object(),
        include_sanitizer_code_on_retry=False,
    )
    generated = generator.generate(object(), object())

    artifact_paths = persist_generated_reward_artifacts(tmp_path, generated)

    assert artifact_paths["dense_reward_path"].endswith("dense_reward_synthesized.py")
    assert artifact_paths["dense_reward_raw_response"].endswith(
        "dense_reward_raw_response.txt"
    )
    assert artifact_paths["reward_validation"].endswith("reward_validation.json")
    assert (tmp_path / "dense_reward_synthesized.py").read_text(
        encoding="utf-8"
    ).startswith("def dense_reward")
    assert (tmp_path / "dense_reward_raw_response.txt").read_text(
        encoding="utf-8"
    ).startswith("```python")
    validation_payload = json.loads(
        (tmp_path / "reward_validation.json").read_text(encoding="utf-8")
    )
    assert validation_payload["status"] == "ok"
    assert validation_payload["component_keys"] == ["progress"]
