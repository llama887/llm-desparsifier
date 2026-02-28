from __future__ import annotations

import json
from pathlib import Path

from llm_desparsifier.rewards.behavior_summary import (
    summarize_trajectory_behavior,
    summarize_trajectory_behavior_from_path,
)


def test_behavior_summary_flags_pickup_putdown_churn() -> None:
    """Verify manipulation-heavy traces expose churn-oriented diagnostics.

    This test covers the failure mode where a policy repeatedly picks up and
    drops objects without clear task progress. It is needed because this churn
    pattern is one of the core reasons to add trajectory-aware reflection
    feedback, and it differs from general smoke tests by asserting that the
    summary emits explicit manipulation/churn metrics and event sketches.
    """

    trajectory = {
        "actions": [3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4],
    }

    text = summarize_trajectory_behavior(trajectory)

    assert "manipulation_rate=" in text
    assert "pickup_putdown_churn=" in text
    assert "Event sketch" in text
    assert "pick_up" in text
    assert "put_down" in text


def test_behavior_summary_navigation_trace_reports_low_manipulation() -> None:
    """Verify navigation-dominant traces produce low manipulation diagnostics.

    This test validates that action summaries remain discriminative when the
    trajectory mostly turns and moves forward. It is needed to ensure the new
    feedback does not over-report manipulation issues, and it differs from the
    churn case by asserting a low manipulation ratio and non-trivial
    forward/turn balance output.
    """

    trajectory = {
        "actions": [0, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0],
    }

    text = summarize_trajectory_behavior(trajectory)

    assert "manipulation_rate=0.000" in text
    assert "forward_turn_balance=" in text


def test_behavior_summary_uses_correct_turn_labels_for_action_ids() -> None:
    """Ensure action ids 1/2 are labeled right/left with correct semantics.

    This test verifies that trajectory summaries map action id 1 to
    `turn_right` and action id 2 to `turn_left`, matching XMiniGrid's action
    implementation. It is needed because an inverted mapping corrupts reflection
    diagnostics and can mislead prompt updates, and it differs from generic
    summary tests by asserting turn-direction label correctness.
    """

    trajectory = {
        "actions": [1, 2, 1, 2],
    }

    text = summarize_trajectory_behavior(trajectory)

    assert "turn_right:2 (50.0%)" in text
    assert "turn_left:2 (50.0%)" in text


def test_behavior_summary_from_path_handles_missing_or_malformed_payload(
    tmp_path: Path,
) -> None:
    """Verify path-based loading returns fallback text for bad artifacts.

    This test ensures metric evaluation remains robust when trajectory artifacts
    are missing or malformed. It is needed because GEPA scoring should continue
    even if trace serialization fails, and it differs from direct summarizer
    tests by covering filesystem loading and JSON parse failures.
    """

    missing_text = summarize_trajectory_behavior_from_path(tmp_path / "missing.json")
    assert "artifact missing" in missing_text

    broken_path = tmp_path / "broken.json"
    broken_path.write_text("not-json", encoding="utf-8")
    broken_text = summarize_trajectory_behavior_from_path(broken_path)
    assert "failed to parse" in broken_text

    ok_path = tmp_path / "ok.json"
    ok_path.write_text(json.dumps({"actions": [0, 1, 2]}), encoding="utf-8")
    ok_text = summarize_trajectory_behavior_from_path(ok_path)
    assert "Behavior summary" in ok_text
