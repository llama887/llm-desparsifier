from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


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


def test_synthesize_heuristic_repairs_invalid_output_once(monkeypatch):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()
    calls = []

    class _DummyPredictor:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    heuristic_code=(
                        "import os\n"
                        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                        "    return 0.0\n"
                    )
                )
            return SimpleNamespace(
                heuristic_code=(
                    "def heuristic_cost_to_go(ts, env_params, ctx):\n"
                    "    if ctx.get('is_winning'):\n"
                    "        return 0.0\n"
                    "    return 1.0 + (1.0 - ctx.get('score_normalized', 0.0))\n"
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
    assert len(calls) == 2
    assert "import os" not in code
    assert fn({"is_winning": False, "score_normalized": 0.25}) == 1.75


def test_phase_gepa_max_metric_calls_is_cumulative_and_capped():
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
    ) == 348


def test_curriculum_phase_schedule_skips_easy_warmup():
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    jobs = [{"name": f"game_{i}"} for i in range(19)]
    schedule = run_puzzlescript_batch.build_curriculum_phase_schedule(jobs)

    assert [len(phase) for phase in schedule] == [10, 15, 19]
    assert schedule[-1] == jobs

    exact_schedule = run_puzzlescript_batch.build_curriculum_phase_schedule(jobs[:15])
    assert [len(phase) for phase in exact_schedule] == [10, 15]


def test_pairwise_metric_penalizes_lost_solves_and_devalues_both_failed():
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    lost = run_puzzlescript_batch._pairwise_gepa_metric(
        candidate={"solved": False, "expanded": 1000},
        base_prompt_baseline={"solved": True, "expanded": 20},
        max_expansions=1000,
    )
    both_failed = run_puzzlescript_batch._pairwise_gepa_metric(
        candidate={"solved": False, "expanded": 1000},
        base_prompt_baseline={"solved": False, "expanded": 1000},
        max_expansions=1000,
    )

    assert lost["outcome_class"] == "lost_solve"
    assert lost["metric"] < 0.0
    assert both_failed["outcome_class"] == "both_failed"
    assert both_failed["metric"] <= 0.03


def test_prompt_selection_rejects_score_gain_with_solve_regression():
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    accepted, reason = run_puzzlescript_batch._should_accept_prompt_candidate(
        mean_score=0.91,
        solve_rate=0.80,
        best_mean_score=0.90,
        best_solve_rate=0.90,
    )

    assert accepted is False
    assert "solve_delta=-0.1000" in reason


def test_filter_loadable_level_indices_drops_engine_rejections(capsys):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    class _Engine:
        def load_level(self, level_i):
            if level_i in {2, 4}:
                raise RuntimeError(f"Level index out of range: {level_i}")

    assert run_puzzlescript_batch.filter_loadable_level_indices(
        _Engine(),
        [0, 1, 2, 3, 4],
        "test_game",
    ) == [0, 1, 3]

    captured = capsys.readouterr()
    assert "Skipping test_game level=2" in captured.out
    assert "Skipping test_game level=4" in captured.out


def test_baseline_cache_merges_matching_shards_and_finds_missing(tmp_path):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()
    signature = {"version": 1, "example": "matching"}

    run_puzzlescript_batch.save_puzzlescript_baseline_shard(
        tmp_path,
        "task-0",
        signature=signature,
        blind_baselines={"game_a": {0: {"score": 0.1}}},
        builtin_baselines={"game_a": {0: {"score": 0.2}}},
        base_prompt_baselines={"game_a": {0: {"score": 0.3}}},
        per_game_budgets={"game_a": {0: 7}},
    )
    run_puzzlescript_batch.save_puzzlescript_baseline_shard(
        tmp_path,
        "task-1",
        signature=signature,
        blind_baselines={"game_b": {1: {"score": 0.4}}},
        builtin_baselines={"game_b": {1: {"score": 0.5}}},
        base_prompt_baselines={"game_b": {1: {"score": 0.6}}},
        per_game_budgets={"game_b": {1: 11}},
    )
    run_puzzlescript_batch.save_puzzlescript_baseline_shard(
        tmp_path,
        "stale",
        signature={"version": 999},
        blind_baselines={"stale": {0: {"score": 1.0}}},
        builtin_baselines={"stale": {0: {"score": 1.0}}},
        base_prompt_baselines={"stale": {0: {"score": 1.0}}},
        per_game_budgets={"stale": {0: 1}},
    )

    blind, builtin, base_prompt, budgets, loaded_paths = (
        run_puzzlescript_batch.load_cached_puzzlescript_baselines(tmp_path, signature)
    )

    assert len(loaded_paths) == 2
    assert blind["game_a"][0]["score"] == 0.1
    assert builtin["game_b"][1]["score"] == 0.5
    assert base_prompt["game_b"][1]["score"] == 0.6
    assert budgets["game_a"][0] == 7
    assert "stale" not in blind

    missing = run_puzzlescript_batch.missing_puzzlescript_baseline_examples(
        [
            {"game": "game_a", "level": 0},
            {"game": "game_b", "level": 1},
            {"game": "game_c", "level": 2},
        ],
        blind_baselines=blind,
        builtin_baselines=builtin,
        base_prompt_baselines=base_prompt,
        per_game_budgets=budgets,
    )
    assert missing == [{"game": "game_c", "level": 2}]


def test_evaluate_prompt_per_game_synthesizes_with_each_env_description(monkeypatch):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()
    synthesis_calls = []
    eval_calls = []

    def _fake_synthesize(prompt_text, env_description, lm, **_kwargs):
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

    def _fake_synthesize(prompt_text, env_description, lm, **_kwargs):
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


def test_local_heuristic_diagnostics_sanitizes_non_finite_values(monkeypatch):
    run_puzzlescript_batch = _load_run_puzzlescript_batch()

    class _Engine:
        def __init__(self):
            self.action = None

        def backup_level(self):
            return None

        def restore_level(self, _backup):
            self.action = None

        def has_metadata(self, _name):
            return True

    def _fake_process_action(engine, action):
        engine.action = action
        return action in {0, 1, 2}

    def _fake_ctx(engine, _compiled):
        return {
            "action": engine.action,
            "score_normalized": 1.0 if engine.action == 1 else 0.0,
            "is_winning": False,
        }

    def _heuristic(ctx):
        return {
            None: math.inf,
            0: math.inf,
            1: math.nan,
            2: -5.0,
        }[ctx["action"]]

    monkeypatch.setattr(
        run_puzzlescript_batch,
        "_process_action_with_again",
        _fake_process_action,
    )
    monkeypatch.setattr(run_puzzlescript_batch, "build_puzzlescript_ctx", _fake_ctx)

    diagnostics = run_puzzlescript_batch._sample_local_heuristic_diagnostics(
        _Engine(),
        {},
        _heuristic,
    )

    assert diagnostics["root_heuristic"] == 0.0
    assert diagnostics["heuristic_range"] == 0.0
    assert diagnostics["heuristic_stddev"] == 0.0
