from __future__ import annotations

from typing import Any, Dict

import pytest

from llm_desparsifier.rewards.reflection import build_reward_reflection


class StubReflectionModule:
    def __init__(self, text: str = "stub reflection") -> None:
        self.text = text
        self.last_kwargs: Dict[str, Any] | None = None

    def __call__(self, **kwargs: Any):
        self.last_kwargs = kwargs

        class _Prediction:
            def __init__(self, reflection: str) -> None:
                self.reflection = reflection

        return _Prediction(self.text)


class FailingReflectionModule:
    def __call__(self, **kwargs: Any):  # pragma: no cover - used to exercise fallback
        raise RuntimeError("failed to contact LLM")


def sample_run_record() -> Dict[str, Any]:
    return {
        "env_id": "XLand-MiniGrid-R4-9x9",
        "benchmark_id": "trivial-1m",
        "reward_code": "def dense_reward(...)",
        "sparse_return_curve": [0.0, 0.4, 0.8, 1.0],
        "component_curves": {
            "progress": [0.0, 0.3, 0.7, 0.9],
            "penalty": [0.0, -0.2, -0.4, -0.1],
        },
        "final_metrics": {"ground_truth_return": 1.0, "dense_return": 0.8},
    }


def test_build_reward_reflection_uses_module_inputs():
    module = StubReflectionModule("actionable reflection")
    record = sample_run_record()

    text = build_reward_reflection(record, reflection_module=module)

    assert text == "actionable reflection"
    assert module.last_kwargs is not None
    assert "Sparse reward checkpoints" in module.last_kwargs["sparse_curve_summary"]
    assert "progress" in module.last_kwargs["component_curve_summary"]


def test_build_reward_reflection_fallback_on_failure():
    module = FailingReflectionModule()
    record = sample_run_record()

    text = build_reward_reflection(record, reflection_module=module)

    assert "Fallback reward reflection" in text
    assert "failed to contact LLM" in text
