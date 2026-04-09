from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from llm_desparsifier.search.astar import plan_with_astar
from llm_desparsifier.search.evaluator import astar_score, run_astar_with_reward


def _load_video_module():
    """Load the video script module from disk for regression comparison tests.

    This helper imports `scripts/generate_training_video.py` as an isolated
    module so unit tests can compare the script-local wrapper against the shared
    planner implementation. It is needed because the `scripts` directory is not
    a Python package, and it differs from a standard import by constructing a
    module spec directly from the file path.
    """

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_training_video.py"
    spec = importlib.util.spec_from_file_location("generate_training_video_module_astar", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_astar_score_prefers_solved_then_fewer_expansions() -> None:
    """Ensure the planner score matches the requested lexicographic ordering.

    This test locks down the A* GEPA objective so future changes cannot make an
    unsolved run outrank a solved run or let additional state expansions improve
    the score. It is needed because GEPA depends on a scalar ordering, and it
    differs from integration tests by validating only the score function.
    """

    solved_fast = astar_score(solved=True, expanded_states=10, max_expansions=100)
    solved_slow = astar_score(solved=True, expanded_states=40, max_expansions=100)
    unsolved_fast = astar_score(solved=False, expanded_states=10, max_expansions=100)

    assert solved_fast > solved_slow
    assert solved_slow > unsolved_fast


def test_shared_planner_matches_video_wrapper() -> None:
    """Verify the video wrapper preserves the shared planner for baseline mode.

    This regression test compares the shared `plan_with_astar` function against
    the script-local `_plan_with_astar` wrapper when dense guidance is disabled.
    It is needed because the video script intentionally swaps to a local
    `h_only` planner for dense-guided visualization, and it differs from the
    dense-mode path by asserting exact parity only for the no-heuristic
    baseline that still delegates straight to the shared planner.
    """

    video_mod = _load_video_module()

    class DummyTimestep:
        def __init__(self, pos: int, dense_reward: float, sparse_reward: float, done: bool) -> None:
            self.state = {"pos": jnp.asarray(pos, dtype=jnp.int32)}
            self.reward = jnp.asarray(dense_reward, dtype=jnp.float32)
            self.extras = {"ground_truth_reward": jnp.asarray(sparse_reward, dtype=jnp.float32)}
            self._done = done

        def last(self) -> jax.Array:
            return jnp.asarray(self._done)

    class DummyEnv:
        def num_actions(self, _env_params: Any) -> int:
            return 2

    def step_fn(_env_params: Any, timestep: DummyTimestep, action: Any) -> DummyTimestep:
        pos = int(jnp.asarray(timestep.state["pos"]))
        action_id = int(jnp.asarray(action))
        if pos >= 3:
            return DummyTimestep(pos, dense_reward=0.0, sparse_reward=1.0, done=True)
        if action_id == 0:
            next_pos = pos + 1
            solved = next_pos >= 3
            return DummyTimestep(
                next_pos,
                dense_reward=1.0 if not solved else 2.0,
                sparse_reward=1.0 if solved else 0.0,
                done=solved,
            )
        return DummyTimestep(pos + 10, dense_reward=-1.0, sparse_reward=0.0, done=False)

    root = DummyTimestep(pos=0, dense_reward=0.0, sparse_reward=0.0, done=False)
    shared_plan = plan_with_astar(
        env=DummyEnv(),
        env_params=object(),
        step_fn=step_fn,
        root_timestep=root,
        use_dense_heuristic=False,
        max_nodes=128,
        max_expansions=128,
    )
    wrapped_plan = video_mod._plan_with_astar(
        env=DummyEnv(),
        env_params=object(),
        step_fn=step_fn,
        root_timestep=root,
        use_dense_heuristic=False,
        max_nodes=128,
        max_expansions=128,
    )

    assert wrapped_plan.actions == shared_plan.actions
    assert wrapped_plan.search_stats == shared_plan.search_stats


def test_run_astar_with_reward_writes_replayable_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure the A* evaluator emits the replay artifacts expected downstream.

    This test runs the shared A* evaluator on a small deterministic stub env and
    asserts that it writes `eval_trajectory.json` with the saved action plan and
    replay seeds. It is needed because the video and debug tooling read the same
    artifact shape regardless of backend, and it differs from planner tests by
    validating evaluator-side artifact generation.
    """

    class DummyEnvParams:
        def __init__(self) -> None:
            self.ruleset = None

        def replace(self, **kwargs: Any) -> "DummyEnvParams":
            new = DummyEnvParams()
            new.ruleset = kwargs.get("ruleset", self.ruleset)
            return new

    class DummyTimestep:
        def __init__(self, pos: int, dense_reward: float, sparse_reward: float, done: bool) -> None:
            self.state = {"pos": jnp.asarray(pos, dtype=jnp.int32)}
            self.reward = jnp.asarray(dense_reward, dtype=jnp.float32)
            self.extras = {
                "ground_truth_reward": jnp.asarray(sparse_reward, dtype=jnp.float32),
                "reward_components": {},
            }
            self._done = done

        def last(self) -> jax.Array:
            return jnp.asarray(self._done)

    class DummyEnv:
        def reset(self, _env_params: Any, _reset_key: Any) -> DummyTimestep:
            return DummyTimestep(0, dense_reward=0.0, sparse_reward=0.0, done=False)

        def step(self, _env_params: Any, timestep: DummyTimestep, action: Any) -> DummyTimestep:
            pos = int(jnp.asarray(timestep.state["pos"]))
            action_id = int(jnp.asarray(action))
            if action_id == 0:
                next_pos = pos + 1
                solved = next_pos >= 2
                return DummyTimestep(
                    next_pos,
                    dense_reward=1.0,
                    sparse_reward=1.0 if solved else 0.0,
                    done=solved,
                )
            return DummyTimestep(pos + 10, dense_reward=-1.0, sparse_reward=0.0, done=False)

        def num_actions(self, _env_params: Any) -> int:
            return 2

    class DummyBenchmark:
        def sample_ruleset(self, _key: Any) -> str:
            return "sampled-ruleset"

        def get_ruleset(self, _index: int) -> str:
            return "default-ruleset"

    class DummyRewardGenerator:
        def __init__(self) -> None:
            self.last_env_description = "deterministic env"

        def generate(self, _env: Any, _env_params: Any):
            def dense_reward(*_args: Any, **_kwargs: Any):
                return jnp.asarray(0.0), {}

            setattr(dense_reward, "__reward_component_keys__", ())
            return dense_reward, "def dense_reward(*args, **kwargs):\n    return 0.0, {}\n"

    import llm_desparsifier.search.evaluator as evaluator_mod

    monkeypatch.setattr(evaluator_mod.xminigrid, "make", lambda _env_id: (DummyEnv(), DummyEnvParams()))
    monkeypatch.setattr(evaluator_mod.xminigrid, "load_benchmark", lambda _benchmark_id: DummyBenchmark())
    monkeypatch.setattr(evaluator_mod, "GymAutoResetWrapper", lambda env: env)
    monkeypatch.setattr(evaluator_mod, "DesparsifyRewardWrapper", lambda env, **_kwargs: env)

    run_dir = tmp_path / "astar-run"
    result = run_astar_with_reward(
        DummyRewardGenerator(),
        output_dir=str(run_dir),
        config_override={
            "env_id": "XLand-MiniGrid-R1-9x9",
            "benchmark_id": "trivial-1m",
            "eval_seed": 11,
            "deterministic_rulesets": True,
        },
        max_nodes=32,
        max_expansions=32,
        reward_mode="dense",
        use_dense_heuristic=True,
    )

    assert result.solved is True
    trajectory_path = run_dir / "eval_trajectory.json"
    assert trajectory_path.exists()
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert payload["actions"] == [0, 0]
    assert payload["reset_key"]
    assert payload["env_text"] == "deterministic env"
