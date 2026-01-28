from __future__ import annotations

import numpy as np

from llm_desparsifier.rl.pipeline import extract_component_logs


def test_extract_component_logs_filters_eval_components() -> None:
    """Verify component log extraction returns only prefixed component series.

    This test builds a synthetic `loss_info` mapping that mixes eval component
    metrics with unrelated keys, then asserts that `extract_component_logs`
    returns only the component entries with their original series intact. It is
    needed to ensure reward reflection sees correct component curves without
    leaking unrelated evaluation metrics, and it differs from integration tests
    by validating the pure extraction logic without running PPO.
    """
    loss_info = {
        "eval/component_progress": np.asarray([1.0, 2.0], dtype=np.float32),
        "eval/component_penalty": np.asarray([0.0, -1.0], dtype=np.float32),
        "eval/returns_mean": np.asarray([3.0, 4.0], dtype=np.float32),
    }

    component_logs = extract_component_logs(loss_info)

    assert set(component_logs.keys()) == {"penalty", "progress"}
    np.testing.assert_allclose(component_logs["progress"], [1.0, 2.0])
    np.testing.assert_allclose(component_logs["penalty"], [0.0, -1.0])
