"""High-level reward generation orchestrator."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import dspy

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
    reward_code: str = dspy.OutputField(
        desc="Only one Python function named dense_reward(...)"
    )


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
    max_sanitize_attempts: int = 10
    include_sanitizer_code_on_retry: bool = True
    sanitizer_code_context: Optional[str] = field(default=None, init=False)
    last_attempt_history: List[_AttemptRecord] = field(default_factory=list, init=False)
    # Sticky cache of the most recent environment description sent to the LLM.
    # Used downstream for reflections so the feedback LM knows the goal/ruleset.
    last_env_description: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.lm is None:
            self.lm = configure_portkey_lm()
            # Configure DSPy only when we own the LM; avoids thread ownership errors
            # when a caller supplies an already-configured LM from another thread.
            dspy.configure(lm=self.lm)
        if self.max_sanitize_attempts < 1:
            raise ValueError("max_sanitize_attempts must be >= 1")
        if self.include_sanitizer_code_on_retry:
            self.sanitizer_code_context = self._load_sanitizer_context()

    def _log_event(self, event: str, **fields: Any) -> None:
        """Emit structured JSON logs for reward synthesis stages.

        This helper prints a single JSON line describing reward-generation
        progress so slow LLM calls or sanitizer retries are visible in stdout.
        It is needed because reward synthesis happens inside JAX training
        orchestration with few other logs, and it differs from ad-hoc prints by
        including consistent metadata (timestamps, attempt indices) and by
        coercing non-serializable values to strings for reliability.
        """
        payload: dict[str, Any] = {
            "event": event,
            "component": "reward_generator",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        payload.update(fields)
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)

    def generate(self, env, env_params) -> Tuple[Callable, str]:
        """Generate a dense reward function and its source code for an environment.

        This routine builds an environment description, queries the reward LLM,
        sanitizes and compiles the returned code, and retries with sanitizer
        feedback when validation fails. It is needed because downstream PPO
        training requires a callable reward function while GEPA needs the
        emitted source for logging and reflection. It differs from lower-level
        sanitizer utilities by orchestrating LLM calls, retry history, and
        metadata capture in one place.
        """
        env_text = self.describe_fn(env, env_params)
        self.last_env_description = env_text
        attempt_history: List[_AttemptRecord] = []
        self.last_attempt_history = []
        total_start = time.time()
        self._log_event(
            "reward_generate_start",
            env_desc_len=len(env_text),
            max_attempts=self.max_sanitize_attempts,
            include_sanitizer_code=bool(self.include_sanitizer_code_on_retry),
        )
        for attempt_idx in range(1, self.max_sanitize_attempts + 1):
            feedback_block = ""
            if attempt_history:
                feedback_block = self._build_feedback_block(attempt_history)
            sanitizer_block = ""
            if (
                attempt_history
                and self.include_sanitizer_code_on_retry
                and self.sanitizer_code_context
            ):
                sanitizer_block = (
                    "\n\n### Sanitizer source (latest attempt only)\n"
                    "Use this code to understand the exact constraints enforced during sanitization.\n"
                    f"```python\n{self.sanitizer_code_context}\n```"
                )
            constraints = f"{self.constraints_text}{feedback_block}{sanitizer_block}"
            self._log_event(
                "attempt_start",
                attempt_idx=attempt_idx,
                max_attempts=self.max_sanitize_attempts,
                env_desc_len=len(env_text),
                constraints_len=len(constraints),
                has_retry_feedback=bool(attempt_history),
                has_sanitizer_context=bool(sanitizer_block),
            )
            # Use thread-local DSPy settings to avoid cross-thread configure errors.
            llm_start = time.time()
            self._log_event("llm_call_start", attempt_idx=attempt_idx)
            with dspy.settings.context(lm=self.lm):
                code = self.synthesizer(env_text, constraints)
            llm_elapsed = time.time() - llm_start
            code_sha16 = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
            self._log_event(
                "llm_call_end",
                attempt_idx=attempt_idx,
                elapsed_sec=round(llm_elapsed, 4),
                code_len=len(code),
                code_sha16=code_sha16,
            )

            sanitize_start = time.time()
            self._log_event("sanitize_start", attempt_idx=attempt_idx)
            try:
                dense_fn = self.sanitize_fn(code)
            except (ValueError, SyntaxError) as exc:
                sanitize_elapsed = time.time() - sanitize_start
                error_text = f"{exc.__class__.__name__}: {exc}"
                timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
                record = _AttemptRecord(
                    code=code, error_text=error_text, timestamp=timestamp
                )
                attempt_history.append(record)
                self._log_event(
                    "sanitize_failure",
                    attempt_idx=attempt_idx,
                    elapsed_sec=round(sanitize_elapsed, 4),
                    error=error_text,
                )
                if attempt_idx >= self.max_sanitize_attempts:
                    self.last_attempt_history = attempt_history
                    message = self._format_retry_failure(attempt_history)
                    raise RuntimeError(message) from exc
                self._log_event(
                    "retrying",
                    attempt_idx=attempt_idx,
                    next_attempt=attempt_idx + 1,
                )
                continue

            sanitize_elapsed = time.time() - sanitize_start
            self._log_event(
                "sanitize_success",
                attempt_idx=attempt_idx,
                elapsed_sec=round(sanitize_elapsed, 4),
            )
            self.last_attempt_history = attempt_history
            self._log_event(
                "reward_generate_end",
                attempt_idx=attempt_idx,
                total_elapsed_sec=round(time.time() - total_start, 4),
                code_sha16=code_sha16,
            )
            return dense_fn, code

        # This line is unreachable, but satisfies type checkers.
        raise RuntimeError("Reward generation loop exited unexpectedly")

    def _build_feedback_block(self, attempts: List[_AttemptRecord]) -> str:
        if not attempts:
            return ""
        lines: List[str] = [
            "\n\n### Sanitizer retry guidance",
            f"You have already failed {len(attempts)} time(s). Review each failure carefully before emitting new code.",
            "During the generation of the dense reward functions, we encountered these errors during sanitation:",
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
            "- Return (total_reward, reward_components) or (total_reward, { ... }) with scalar JAX arrays for every component.\n"
            "- Avoid extra imports, global state, or non-JAX math operations.\n"
        )
        lines.append(
            "Rewrite dense_reward so it satisfies all original constraints and fixes every issue above."
        )
        return "\n".join(lines)

    @staticmethod
    def _load_sanitizer_context() -> str:
        sanitizer_path = Path(__file__).with_name("sanitizer.py")
        return sanitizer_path.read_text(encoding="utf-8").strip()

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
