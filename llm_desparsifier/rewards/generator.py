"""High-level reward generation orchestrator."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import dspy
import jax.numpy as jnp

from llm_desparsifier.rewards.llm_client import configure_portkey_lm
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT, describe_ruleset
from llm_desparsifier.rewards.sanitizer import sanitize_and_compile


@dataclass
class _AttemptRecord:
    code: str
    error_text: str
    timestamp: str


class RewardSynthesis(dspy.Signature):
    """LLM signature for synthesizing dense reward code."""

    env_description: str = dspy.InputField()
    constraints: str = dspy.InputField()
    reward_code: str = dspy.OutputField(desc="Only one Python function named dense_reward(...)")


class RewardSynthesizer(dspy.Module):
    """DSPy module for reward synthesis."""

    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(RewardSynthesis)

    def forward(self, env_description: str, constraints: str) -> str:
        out = self.gen(env_description=env_description, constraints=constraints)
        return out.reward_code


@dataclass
class RewardGenerator:
    """Generate dense rewards by prompting an LLM and sanitizing the output."""

    synthesizer: RewardSynthesizer = field(default_factory=RewardSynthesizer)
    constraints_text: str = CONSTRAINTS_TEXT
    describe_fn: Callable[[object, object], str] = describe_ruleset
    sanitize_fn: Callable[[str], Callable] = sanitize_and_compile
    lm: Optional[dspy.LM] = None
    verbose: bool = True
    max_sanitize_attempts: int = 10
    failure_artifact_dir: Optional[Path] = None

    def __post_init__(self):
        if self.lm is None:
            self.lm = configure_portkey_lm()
        else:
            dspy.configure(lm=self.lm)
        if self.max_sanitize_attempts < 1:
            raise ValueError("max_sanitize_attempts must be >= 1")

    def generate(self, env, env_params) -> Tuple[Callable, str]:
        """Return `(dense_fn, emitted_code)` for the given environment setup."""
        env_text = self.describe_fn(env, env_params)
        attempt_history: List[_AttemptRecord] = []
        for attempt_idx in range(1, self.max_sanitize_attempts + 1):
            feedback_block = ""
            if attempt_history:
                feedback_block = self._build_feedback_block(attempt_history)
            constraints = f"{self.constraints_text}{feedback_block}"
            code = self.synthesizer(env_text, constraints)

            if self.verbose:
                print("\n==== Generated dense_reward candidate (pre-sanitize) ====\n")
                print(code)
                print("\nEnvironment Description: \n", env_text)
                print("\n=========================================================\n")

            try:
                dense_fn = self.sanitize_fn(code)
                self._run_smoke_test(dense_fn)
            except ValueError as exc:
                error_text = f"{exc.__class__.__name__}: {exc}"
                timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                record = _AttemptRecord(code=code, error_text=error_text, timestamp=timestamp)
                attempt_history.append(record)
                feedback_snapshot = self._build_feedback_block(attempt_history)
                self._persist_failed_attempt(record, attempt_idx, feedback_snapshot)
                if attempt_idx >= self.max_sanitize_attempts:
                    message = self._format_retry_failure(attempt_history)
                    raise RuntimeError(message) from exc
                continue

            if self.verbose:
                print("\n\n----\n")
                print("Dense Function: \n", dense_fn)
                print("\n----\n\n")

            return dense_fn, code

        # This line is unreachable, but satisfies type checkers.
        raise RuntimeError("Reward generation loop exited unexpectedly")

    def _run_smoke_test(self, dense_fn: Callable) -> None:
        """Execute a minimal, eager reward call to catch obvious runtime errors.

        The test is intentionally lightweight (single call, small tensors) to avoid
        noticeable overhead. Any exception is converted to ValueError so it enters
        the existing retry path.
        """

        class _SmokeTimeStep:
            def __init__(self):
                self.reward = jnp.asarray(0.0)
                self.observation = jnp.zeros((1, 1, 1))

            def last(self):
                return False

        dummy_env_params = type("DummyEnvParams", (), {
            "height": 1,
            "width": 1,
            "view_size": 1,
            "max_steps": 1,
            "ruleset": None,
        })()

        ts_prev = _SmokeTimeStep()
        ts_next = _SmokeTimeStep()
        action = jnp.asarray(0, dtype=jnp.int32)
        ctx = {
            "agent_pos": jnp.asarray([0, 0]),
            "agent_direction": jnp.asarray(0),
            "step_num": jnp.asarray(0),
            "is_carrying": jnp.asarray(0),
            "carried_item": jnp.asarray(-1),
            "yellow_square_pos": jnp.asarray([0, 0]),
            "green_ball_pos": jnp.asarray([1, 0]),
            "object_positions": {
                "yellow_square": jnp.asarray([0, 0]),
                "green_ball": jnp.asarray([1, 0]),
            },
        }

        try:
            dense_fn(dummy_env_params, ts_prev, action, ts_next, ctx)
        except Exception as exc:
            raise ValueError(f"Smoke test failed: {exc}") from exc

    def _build_feedback_block(self, attempts: List[_AttemptRecord]) -> str:
        if not attempts:
            return ""
        lines: List[str] = [
            "\n\n### Sanitizer retry guidance",
            f"You have already failed {len(attempts)} time(s). Review each failure carefully before emitting new code.",
            "",
            "| Attempt | Timestamp (UTC) | Error |",
            "| --- | --- | --- |",
        ]
        for idx, record in enumerate(attempts, start=1):
            lines.append(f"| {idx} | {record.timestamp} | `{record.error_text}` |")
        lines.append("")
        lines.append("Detailed failure summaries:")
        for idx, record in enumerate(attempts, start=1):
            lines.append(
                f"\n#### Attempt {idx}\n"
                f"Error: {record.error_text}\n"
                f"```python\n{record.code.strip()}\n```\n"
            )
        lines.append(
            "Checklist:\n"
            "- Use ctx.get(...) with explicit fallbacks for every optional key.\n"
            "- Only call jnp.* or jax.* primitives; dict.get is the only permitted Python method for ctx-derived maps.\n"
            "- Return (total_reward, reward_components) with scalar JAX arrays for every component.\n"
            "- Avoid extra imports, global state, or non-JAX math operations.\n"
        )
        lines.append("Rewrite dense_reward so it satisfies all original constraints and fixes every issue above.")
        return "\n".join(lines)

    def _persist_failed_attempt(self, record: _AttemptRecord, attempt_idx: int, feedback_block: str) -> None:
        if not self.failure_artifact_dir:
            return
        try:
            failure_dir = Path(self.failure_artifact_dir)
            failure_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"attempt-{attempt_idx:02d}-{record.timestamp}"
            (failure_dir / f"{prefix}.py").write_text(record.code, encoding="utf-8")
            (failure_dir / f"{prefix}.err.txt").write_text(record.error_text, encoding="utf-8")
            if feedback_block:
                (failure_dir / f"{prefix}.feedback.md").write_text(feedback_block, encoding="utf-8")
        except Exception:
            # Failing to write diagnostics should not abort reward generation.
            pass

    @staticmethod
    def _format_retry_failure(attempt_history: List[_AttemptRecord]) -> str:
        lines = [
            f"Failed to sanitize dense_reward after {len(attempt_history)} attempt(s).",
            "Errors encountered:",
        ]
        for idx, record in enumerate(attempt_history, start=1):
            lines.append(f"  Attempt {idx}: {record.error_text}")
        return "\n".join(lines)


def create_reward_generator(**kwargs) -> RewardGenerator:
    """Helper to instantiate a RewardGenerator with optional overrides."""
    return RewardGenerator(**kwargs)


__all__ = ["RewardGenerator", "RewardSynthesizer", "create_reward_generator"]
