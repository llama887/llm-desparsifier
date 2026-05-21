from __future__ import annotations

import importlib.util
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
