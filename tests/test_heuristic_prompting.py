from __future__ import annotations

from types import SimpleNamespace

from llm_desparsifier.heuristics import prompting


def test_heuristic_prompt_includes_ctx_access_rules() -> None:
    assert 'ctx.get("key")' in prompting.BASE_HEURISTIC_PROMPT
    assert 'Do not use `ctx["key"]`.' in prompting.BASE_HEURISTIC_PROMPT
    assert "agent_state.get(...)" in prompting.BASE_HEURISTIC_PROMPT
    assert 'mapping[\'name\']' in prompting.BASE_HEURISTIC_PROMPT


def test_describe_ruleset_for_heuristic_includes_ctx_access_rules(monkeypatch) -> None:
    monkeypatch.setattr(
        prompting,
        "_render_ruleset_text",
        lambda env_params: "GOAL:\nAgentNear(red key)\n",
    )
    env_params = SimpleNamespace(height=11, width=11, max_steps=363, grid_type="R1")

    description = prompting.describe_ruleset_for_heuristic(object(), env_params)

    assert 'ctx.get("key")' in description
    assert 'Do not use `ctx["key"]`.' in description
    assert "object_positions.get(...)" in description
    assert 'mapping[\'name\']' in description
