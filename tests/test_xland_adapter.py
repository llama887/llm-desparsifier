from __future__ import annotations

from types import SimpleNamespace

import pytest

jnp = pytest.importorskip("jax.numpy")

from xminigrid.core.constants import Colors, Tiles

from llm_desparsifier.heuristics.prompting import extract_goal_description_from_ruleset_text
from llm_desparsifier.search.xland_adapter import build_heuristic_ctx


def test_build_heuristic_ctx_exposes_backward_compatible_agent_position_aliases() -> None:
    """Ensure the runtime heuristic context exposes both supported agent position keys.

    This regression test protects the heuristic-only A* path against silent
    contract drift between prompt examples and runtime context assembly. It is
    needed because recent synthesized heuristics relied on `agent_state["pos"]`
    while the adapter had started exporting only `agent_state["position"]`, and
    it differs from broader integration tests by directly asserting the exact
    aliasing guarantee required for backward-compatible heuristic execution.
    """

    agent = SimpleNamespace(
        position=jnp.asarray([2, 3], dtype=jnp.int32),
        direction=jnp.asarray(1, dtype=jnp.int32),
        pocket=jnp.asarray([-1, -1], dtype=jnp.int32),
    )
    state = SimpleNamespace(agent=agent, grid=jnp.zeros((3, 3, 2), dtype=jnp.int32))
    timestep = SimpleNamespace(state=state)
    env_params = SimpleNamespace(height=3, width=3, grid_type="R1", max_steps=25)

    ctx = build_heuristic_ctx(
        ts=timestep,
        env_params=env_params,
        env_id="XLand-MiniGrid-R1-3x3",
        benchmark_id="dummy-benchmark",
        ruleset_text="GOAL:\nAgentNear(red key)\n",
        goal_description="AgentNear(red key)",
    )

    agent_state = ctx["agent_state"]
    assert agent_state["position"] == (2, 3)
    assert agent_state["pos"] == (2, 3)
    assert agent_state["position"] == agent_state["pos"]


def test_build_heuristic_ctx_exposes_space_and_underscore_object_aliases() -> None:
    """Ensure heuristic object lookups match both prompt and legacy key formats.

    This regression test locks down the object-key contract used by heuristic
    synthesis. It is needed because prompt text and feedback describe objects as
    `"red key"` while some older helpers and fixtures still reference
    `"red_key"`, and it differs from the agent-position alias test by
    validating the specific object-position compatibility layer that prevents
    synthesized heuristics from degenerating into blind search.
    """

    grid = jnp.zeros((3, 3, 2), dtype=jnp.int32)
    grid = grid.at[1, 2, 0].set(int(Tiles.KEY))
    grid = grid.at[1, 2, 1].set(int(Colors.RED))
    agent = SimpleNamespace(
        position=jnp.asarray([0, 0], dtype=jnp.int32),
        direction=jnp.asarray(0, dtype=jnp.int32),
        pocket=jnp.asarray([-1, -1], dtype=jnp.int32),
    )
    state = SimpleNamespace(agent=agent, grid=grid)
    timestep = SimpleNamespace(state=state)
    env_params = SimpleNamespace(height=3, width=3, grid_type="R1", max_steps=25)

    ctx = build_heuristic_ctx(
        ts=timestep,
        env_params=env_params,
        env_id="XLand-MiniGrid-R1-3x3",
        benchmark_id="dummy-benchmark",
        ruleset_text="GOAL:\nAgentNear(red key)\n",
        goal_description="AgentNear(red key)",
    )

    object_positions = ctx["object_positions"]
    object_metadata = ctx["object_metadata"]
    assert object_positions["red key"] == (1, 2)
    assert object_positions["red_key"] == (1, 2)
    assert object_positions["red key"] == object_positions["red_key"]
    assert object_metadata["red key"] == object_metadata["red_key"]


def test_extract_goal_description_from_ruleset_text_ignores_prefix_headers() -> None:
    """Ensure task metadata uses the actual `GOAL:` section, not summary headers.

    This regression test protects `task_instance.json` serialization from
    storing unrelated prefix lines such as `Grid shape: ...` as the top-level
    goal description. It is needed because the heuristic runner prepends a
    search-oriented header block ahead of the raw XLand ruleset text, and it
    differs from the context-shape tests above by validating the shared
    text-parsing contract used by prompt construction and replay artifacts.
    """

    ruleset_text = (
        "Grid shape: 11 x 11\n"
        "Grid type: R1\n"
        "Goal description: AgentNear(red key)\n"
        "Full ruleset:\n"
        "GOAL:\n"
        "AgentNear(red key)\n"
    )

    assert extract_goal_description_from_ruleset_text(ruleset_text) == "AgentNear(red key)"
