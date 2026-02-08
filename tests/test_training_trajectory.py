from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, List

import jax
import jax.numpy as jnp

from llm_desparsifier.rl import pipeline
from llm_desparsifier.rl.eval import GroundTruthEvalConfig, GroundTruthEvalResult


def test_run_training_with_reward_fallback_trajectory(monkeypatch) -> None:
    """Verify a fallback trajectory is captured and temp dirs are removed.

    This test simulates a training run where the primary ground-truth evaluation
    returns no trajectory, then asserts that a one-episode fallback rollout is
    invoked to create `eval_trajectory.json`. It is needed to guarantee that
    downstream video tooling always has a replayable trace, and it differs from
    integration tests by stubbing out PPO compilation and evaluation to keep the
    test fast and deterministic while still exercising the trajectory logic.
    """

    class DummyEnv:
        """Minimal env stub that supplies the action space size."""

        def num_actions(self, _env_params: Any) -> int:
            """Return a fixed action count for evaluation model setup."""

            return 3

    class DummyRewardGenerator:
        """Stub reward generator to satisfy the training entrypoint signature."""

        def generate(self, *_args: Any, **_kwargs: Any):
            """Raise if called because make_states is stubbed in this test."""

            raise AssertionError("DummyRewardGenerator.generate should not be called")

    def fake_make_states(
        *_args: Any, **_kwargs: Any
    ) -> tuple[Any, DummyEnv, Any, Any, Any, Any, str, str, Any]:
        """Return lightweight state objects without touching real envs."""

        rng = jax.random.key(0)
        return (
            rng,
            DummyEnv(),
            object(),
            None,
            "init_hstate",
            "train_state",
            "def dense_reward(ts_prev, action, ts_next):\n    return 0.0, {}\n",
            "",
            None,
        )

    def fake_make_train(*_args: Any, **_kwargs: Any):
        """Return a dummy train function that yields fixed loss metrics."""

        loss_info = {
            "eval/returns_mean": jnp.asarray([1.0]),
            "eval/ground_truth_returns_mean": jnp.asarray([0.0]),
            "eval/returns_abs_gap_mean": jnp.asarray([1.0]),
        }

        class _DummyLowerable:
            """Minimal object that mimics the JAX lowering/compilation API."""

            def lower(self, *_args: Any, **_kwargs: Any) -> "_DummyLowerable":
                """Return self to mirror the real lowering workflow."""

                return self

            def compile(self):
                """Return a callable that yields a static train_info payload."""

                def _compiled(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                    """Return synthetic training outputs for the pipeline."""

                    return {"loss_info": loss_info, "state": "trained_state"}

                return _compiled

        return _DummyLowerable()

    call_cfgs: List[GroundTruthEvalConfig] = []
    fallback_payload = {
        "version": 1,
        "env_id": "XLand-MiniGrid-R1-8x8",
        "benchmark_id": "trivial-1m",
        "deterministic_rulesets": False,
        "ruleset_index": None,
        "ruleset_key": [1, 2],
        "reset_key": [3, 4],
        "episode_index": 0,
        "episode_length": 1,
        "episode_return": 0.0,
        "actions": [1],
        "num_eval_episodes": 1,
        "eval_seed": 42,
        "env_seed": 42,
        "env_text": "Synthetic task description used by fallback test.",
        "img_obs": False,
    }

    def fake_run_ground_truth_eval(
        *_args: Any, cfg: GroundTruthEvalConfig, **_kwargs: Any
    ) -> GroundTruthEvalResult:
        """Return no trajectory on the first call, then a fallback payload."""

        call_cfgs.append(cfg)
        if len(call_cfgs) == 1:
            return GroundTruthEvalResult(
                returns=[0.0],
                lengths=[1],
                mean_return=0.0,
                std_return=0.0,
                total_steps=1,
                frames=None,
                trajectory=None,
            )
        return GroundTruthEvalResult(
            returns=[1.0],
            lengths=[1],
            mean_return=1.0,
            std_return=0.0,
            total_steps=1,
            frames=None,
            trajectory=fallback_payload,
        )

    def fake_replicate(value: Any, _devices: Any = None) -> Any:
        """Return inputs unchanged to avoid device replication in unit tests."""

        return value

    monkeypatch.setattr(pipeline, "make_states", fake_make_states)
    monkeypatch.setattr(pipeline, "make_train", fake_make_train)
    monkeypatch.setattr(pipeline, "run_ground_truth_eval", fake_run_ground_truth_eval)
    monkeypatch.setattr(pipeline, "replicate", fake_replicate)
    monkeypatch.setattr(pipeline, "unreplicate", lambda value: value)
    monkeypatch.setattr(pipeline.jax, "block_until_ready", lambda value: value)

    num_devices = pipeline.jax.local_device_count()
    config_override = {
        "env_id": "XLand-MiniGrid-R1-8x8",
        "benchmark_id": "trivial-1m",
        "num_envs": num_devices,
        "num_steps_per_env": 1,
        "num_steps_per_update": 1,
        "total_timesteps": num_devices,
        "eval_num_envs": num_devices,
        "eval_num_episodes": 0,
        "eval_seed": 42,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "candidate-run"
        result = pipeline.run_training_with_reward(
            DummyRewardGenerator(),
            output_dir=str(output_dir),
            config_override=config_override,
            reward_mode="dense",
        )

        trajectory_path = output_dir / "eval_trajectory.json"
        assert trajectory_path.exists()
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        assert payload["actions"] == fallback_payload["actions"]
        assert payload["env_seed"] == fallback_payload["env_seed"]
        assert payload["env_text"] == fallback_payload["env_text"]
        assert result.artifacts["eval_trajectory"] == str(trajectory_path)
        assert result.final_metrics["solve_rate"] == 0.0
        assert len(call_cfgs) == 2
        assert call_cfgs[0].num_episodes == 0
        assert call_cfgs[1].num_episodes == 1
        assert call_cfgs[1].capture_trajectory is True

    assert not Path(tmpdir).exists()


def test_run_training_with_reward_primary_trajectory(monkeypatch) -> None:
    """Verify primary ground-truth eval trajectories are saved without fallback.

    This test simulates a successful first-call trajectory capture and asserts
    that `eval_trajectory.json` is still written while avoiding the fallback
    eval pass. It is needed because trajectory persistence should work in the
    common success path, not only in the fallback branch. It differs from the
    fallback test above by returning a trajectory on the initial evaluation call
    and asserting only one eval invocation occurs.
    """

    class DummyEnv:
        """Minimal env stub that supplies the action space size."""

        def num_actions(self, _env_params: Any) -> int:
            """Return a fixed action count for evaluation model setup."""

            return 3

    class DummyRewardGenerator:
        """Stub reward generator to satisfy the training entrypoint signature."""

        def generate(self, *_args: Any, **_kwargs: Any):
            """Raise if called because make_states is stubbed in this test."""

            raise AssertionError("DummyRewardGenerator.generate should not be called")

    def fake_make_states(
        *_args: Any, **_kwargs: Any
    ) -> tuple[Any, DummyEnv, Any, Any, Any, Any, str, str, Any]:
        """Return lightweight state objects without touching real envs."""

        rng = jax.random.key(0)
        return (
            rng,
            DummyEnv(),
            object(),
            None,
            "init_hstate",
            "train_state",
            "def dense_reward(ts_prev, action, ts_next):\n    return 0.0, {}\n",
            "",
            None,
        )

    def fake_make_train(*_args: Any, **_kwargs: Any):
        """Return a dummy train function that yields fixed loss metrics."""

        loss_info = {
            "eval/returns_mean": jnp.asarray([1.0]),
            "eval/ground_truth_returns_mean": jnp.asarray([0.0]),
            "eval/returns_abs_gap_mean": jnp.asarray([1.0]),
        }

        class _DummyLowerable:
            """Minimal object that mimics the JAX lowering/compilation API."""

            def lower(self, *_args: Any, **_kwargs: Any) -> "_DummyLowerable":
                """Return self to mirror the real lowering workflow."""

                return self

            def compile(self):
                """Return a callable that yields a static train_info payload."""

                def _compiled(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                    """Return synthetic training outputs for the pipeline."""

                    return {"loss_info": loss_info, "state": "trained_state"}

                return _compiled

        return _DummyLowerable()

    call_cfgs: List[GroundTruthEvalConfig] = []
    primary_payload = {
        "version": 1,
        "env_id": "XLand-MiniGrid-R1-8x8",
        "benchmark_id": "trivial-1m",
        "deterministic_rulesets": False,
        "ruleset_index": None,
        "ruleset_key": [11, 12],
        "reset_key": [13, 14],
        "episode_index": 0,
        "episode_length": 2,
        "episode_return": 1.0,
        "actions": [1, 2],
        "num_eval_episodes": 1,
        "eval_seed": 99,
        "env_seed": 99,
        "env_text": "Primary trajectory payload for direct save path.",
        "img_obs": False,
    }

    def fake_run_ground_truth_eval(
        *_args: Any, cfg: GroundTruthEvalConfig, **_kwargs: Any
    ) -> GroundTruthEvalResult:
        """Return a trajectory immediately to exercise primary save behavior."""

        call_cfgs.append(cfg)
        return GroundTruthEvalResult(
            returns=[1.0],
            lengths=[2],
            mean_return=1.0,
            std_return=0.0,
            total_steps=2,
            frames=None,
            trajectory=primary_payload,
        )

    def fake_replicate(value: Any, _devices: Any = None) -> Any:
        """Return inputs unchanged to avoid device replication in unit tests."""

        return value

    monkeypatch.setattr(pipeline, "make_states", fake_make_states)
    monkeypatch.setattr(pipeline, "make_train", fake_make_train)
    monkeypatch.setattr(pipeline, "run_ground_truth_eval", fake_run_ground_truth_eval)
    monkeypatch.setattr(pipeline, "replicate", fake_replicate)
    monkeypatch.setattr(pipeline, "unreplicate", lambda value: value)
    monkeypatch.setattr(pipeline.jax, "block_until_ready", lambda value: value)

    num_devices = pipeline.jax.local_device_count()
    config_override = {
        "env_id": "XLand-MiniGrid-R1-8x8",
        "benchmark_id": "trivial-1m",
        "num_envs": num_devices,
        "num_steps_per_env": 1,
        "num_steps_per_update": 1,
        "total_timesteps": num_devices,
        "eval_num_envs": num_devices,
        "eval_num_episodes": 1,
        "eval_seed": 99,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "candidate-run"
        result = pipeline.run_training_with_reward(
            DummyRewardGenerator(),
            output_dir=str(output_dir),
            config_override=config_override,
            reward_mode="dense",
        )

        trajectory_path = output_dir / "eval_trajectory.json"
        assert trajectory_path.exists()
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        assert payload["actions"] == primary_payload["actions"]
        assert payload["env_seed"] == primary_payload["env_seed"]
        assert payload["env_text"] == primary_payload["env_text"]
        assert result.artifacts["eval_trajectory"] == str(trajectory_path)
        assert len(call_cfgs) == 1
