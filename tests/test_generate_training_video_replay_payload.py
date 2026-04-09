from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_video_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "generate_training_video.py"
    spec = importlib.util.spec_from_file_location("generate_training_video", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_replay_payload_supports_heuristic_artifacts(tmp_path: Path) -> None:
    video_mod = _load_video_module()
    run_dir = tmp_path / "candidate-0001-job"
    run_dir.mkdir()
    (run_dir / "task_instance.json").write_text(
        json.dumps(
            {
                "env_id": "XLand-MiniGrid-R1-9x9",
                "benchmark_id": "trivial-1m",
                "seed": 123,
                "ruleset_text": "goal text",
                "goal_description": "goal text",
                "reset_key": [1, 2],
                "reset_payload": {"reset_key": [1, 2]},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "astar_plan.json").write_text(
        json.dumps({"actions": [0, 1, 2]}),
        encoding="utf-8",
    )
    (run_dir / "astar_search_stats.json").write_text(
        json.dumps({"aggregate_stats": {"job_score": 0.5}}),
        encoding="utf-8",
    )
    payload = video_mod._load_replay_payload(run_dir)
    assert payload["actions"] == [0, 1, 2]
    assert payload["env_seed"] == 123
    assert payload["env_text"] == "goal text"
