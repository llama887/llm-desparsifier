from __future__ import annotations

import pytest

from llm_desparsifier.heuristics.sanitizer import sanitize_and_compile_heuristic


def test_sanitize_and_compile_heuristic_accepts_simple_ctx_based_function() -> None:
    code = (
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    if ctx.get('goal_description') == 'solved':\n"
        "        return 0.0\n"
        "    return float(max(0, 3))\n"
    )
    heuristic = sanitize_and_compile_heuristic(code)
    assert heuristic(None, None, {"goal_description": "solved"}) == pytest.approx(0.0)


def test_sanitize_and_compile_heuristic_rejects_imports() -> None:
    code = (
        "import math\n"
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    return 0.0\n"
    )
    with pytest.raises(ValueError, match="imports"):
        sanitize_and_compile_heuristic(code)


def test_sanitize_and_compile_heuristic_rejects_ctx_subscript() -> None:
    code = (
        "def heuristic_cost_to_go(ts, env_params, ctx):\n"
        "    return float(ctx['goal_description'])\n"
    )
    with pytest.raises(ValueError, match="ctx.get"):
        sanitize_and_compile_heuristic(code)
