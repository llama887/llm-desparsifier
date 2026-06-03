from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import json


def _load_run_puzzlescript_batch():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "run_puzzlescript_batch.py"
    spec = importlib.util.spec_from_file_location("run_puzzlescript_batch", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthesize_heuristic_wraps_three_argument_contract(monkeypatch):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    class _DummyPredictor:
        def __call__(self, **_kwargs):
            return SimpleNamespace(
                heuristic_code=(
                    "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                    "    if ts is not None or env_params is not None:\n"
                    "        return 999.0\n"
                    "    return ctx.get('value', 0) + 2\n"
                )
            )

    class _DummyContext:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(run_puzzlescript_batch, "_heuristic_predictor", _DummyPredictor())
    monkeypatch.setattr(run_puzzlescript_batch.dspy, "context", _DummyContext)

    fn, code, error = run_puzzlescript_batch.synthesize_heuristic_from_prompt(
        "prompt",
        "env",
        lm=object(),
    )

    assert error is None
    assert "heuristic_cost_to_go" in code
    assert fn({"value": 3}) == 5.0


def test_phase_gepa_max_metric_calls_is_cumulative():
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    assert run_puzzlescript_batch._phase_gepa_max_metric_calls(
        phase_iteration=1,
        trainset_size=29,
    ) == 87
    assert run_puzzlescript_batch._phase_gepa_max_metric_calls(
        phase_iteration=2,
        trainset_size=29,
    ) == 174
    assert run_puzzlescript_batch._phase_gepa_max_metric_calls(
        phase_iteration=20,
        trainset_size=29,
    ) == 1740


def test_evaluate_prompt_per_game_synthesizes_with_each_env_description(monkeypatch):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()
    synthesis_calls = []
    eval_calls = []

    def _fake_synthesize(prompt_text, env_description, lm):
        synthesis_calls.append((prompt_text, env_description, lm))

        def _heuristic(_ctx):
            return 0.0

        return _heuristic, f"# code for {env_description}", None

    def _fake_evaluate_game_levels(
        evaluator,
        game_name,
        game_text,
        heuristic_fn,
        level_budgets,
        **kwargs,
    ):
        eval_calls.append(
            {
                "game_name": game_name,
                "game_text": game_text,
                "level_budgets": dict(level_budgets),
                "heuristic_code": kwargs["heuristic_code"],
            }
        )
        return {
            "score": 1.0,
            "solved": True,
            "expanded": 1,
            "generated": 1,
            "solution_length": 1,
            "feedback": "ok",
        }

    monkeypatch.setattr(
        run_puzzlescript_batch,
        "synthesize_heuristic_from_prompt",
        _fake_synthesize,
    )
    monkeypatch.setattr(
        run_puzzlescript_batch,
        "evaluate_game_levels",
        _fake_evaluate_game_levels,
    )

    results, codes = run_puzzlescript_batch.evaluate_prompt_per_game(
        evaluator=object(),
        prompt_text="shared prompt",
        game_names=["game_a", "game_b"],
        all_game_texts={"game_a": "text a", "game_b": "text b"},
        all_env_descs={"game_a": "desc a", "game_b": "desc b"},
        level_indices_by_game={"game_a": [0], "game_b": [0, 1]},
        max_expansions=50,
        lm="lm",
    )

    assert synthesis_calls == [
        ("shared prompt", "desc a", "lm"),
        ("shared prompt", "desc b", "lm"),
    ]
    assert [row["game"] for row in results] == ["game_a", "game_b"]
    assert codes == {"game_a": "# code for desc a", "game_b": "# code for desc b"}
    assert eval_calls == [
        {
            "game_name": "game_a",
            "game_text": "text a",
            "level_budgets": {0: 50},
            "heuristic_code": "# code for desc a",
        },
        {
            "game_name": "game_b",
            "game_text": "text b",
            "level_budgets": {0: 50, 1: 50},
            "heuristic_code": "# code for desc b",
        },
    ]


def test_evaluate_prompt_per_level_synthesizes_with_each_level_description(monkeypatch):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()
    synthesis_calls = []
    eval_calls = []

    def _fake_synthesize(prompt_text, env_description, lm):
        synthesis_calls.append((prompt_text, env_description, lm))

        def _heuristic(_ctx):
            return 0.0

        return _heuristic, f"# code for {env_description}", None

    def _fake_evaluate_one_game(
        evaluator,
        game_name,
        game_text,
        heuristic_fn,
        max_expansions,
        **kwargs,
    ):
        eval_calls.append(
            {
                "game_name": game_name,
                "game_text": game_text,
                "max_expansions": max_expansions,
                "level_i": kwargs["level_i"],
                "heuristic_code": kwargs["heuristic_code"],
            }
        )
        return {
            "score": 1.0,
            "level": kwargs["level_i"],
            "solved": True,
            "expanded": 1,
            "generated": 1,
            "solution_length": 1,
            "feedback": "ok",
        }

    monkeypatch.setattr(
        run_puzzlescript_batch,
        "synthesize_heuristic_from_prompt",
        _fake_synthesize,
    )
    monkeypatch.setattr(
        run_puzzlescript_batch,
        "evaluate_one_game",
        _fake_evaluate_one_game,
    )

    results, codes = run_puzzlescript_batch.evaluate_prompt_per_level(
        evaluator=object(),
        prompt_text="shared prompt",
        examples=[
            {"game": "game_a", "level": 0, "budget": 10},
            {"game": "game_a", "level": 1, "budget": 20},
        ],
        all_game_texts={"game_a": "text a"},
        all_level_env_descs={"game_a": {0: "desc a0", 1: "desc a1"}},
        max_expansions=50,
        lm="lm",
    )

    assert synthesis_calls == [
        ("shared prompt", "desc a0", "lm"),
        ("shared prompt", "desc a1", "lm"),
    ]
    assert [row["example"] for row in results] == [
        "game_a::level-00",
        "game_a::level-01",
    ]
    assert codes == {
        "game_a::level-00": "# code for desc a0",
        "game_a::level-01": "# code for desc a1",
    }
    assert eval_calls == [
        {
            "game_name": "game_a",
            "game_text": "text a",
            "max_expansions": 10,
            "level_i": 0,
            "heuristic_code": "# code for desc a0",
        },
        {
            "game_name": "game_a",
            "game_text": "text a",
            "max_expansions": 20,
            "level_i": 1,
            "heuristic_code": "# code for desc a1",
        },
    ]


def test_lm_cost_logger_records_incremental_history(tmp_path):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()
    lm = SimpleNamespace(history=[])
    logger = run_puzzlescript_batch.LMCostLogger(lm, tmp_path)

    lm.history.append(
        {
            "timestamp": "t1",
            "uuid": "u1",
            "model": "gemini/test",
            "cost": "0.125",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    )
    first = logger.sync("compile", {"phase": 1})
    assert first["total_calls"] == 1
    assert first["total_cost_usd"] == 0.125

    lm.history.append(
        {
            "timestamp": "t2",
            "uuid": "u2",
            "model": "gemini/test",
            "cost": 0.25,
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
    )
    second = logger.sync("eval")
    assert second["total_calls"] == 2
    assert second["total_cost_usd"] == 0.375
    assert second["by_model"]["gemini/test"]["calls"] == 2
    assert second["by_model"]["gemini/test"]["cost_usd"] == 0.375

    events = [
        json.loads(line)
        for line in (tmp_path / "llm_cost_events.jsonl").read_text().splitlines()
    ]
    assert [event["label"] for event in events] == ["compile", "eval"]
    assert events[0]["extra_phase"] == 1
    summary = json.loads((tmp_path / "llm_cost_summary.json").read_text())
    assert summary["total_calls"] == 2
