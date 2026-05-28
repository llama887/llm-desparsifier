"""High-level reward generation orchestrator."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

import dspy

from llm_desparsifier.rewards.llm_client import configure_deepseek_lm
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT, describe_ruleset
from llm_desparsifier.rewards.reward_key_diagnostics import (
    RewardObjectKeyDiagnostics,
    build_reward_object_key_diagnostics,
)
from llm_desparsifier.rewards.sanitizer import (
    SanitizedRewardResult,
    sanitize_reward_code,
)


@dataclass
class _AttemptRecord:
    code: str
    error_text: str
    timestamp: str


@dataclass(frozen=True)
class GeneratedRewardValidation:
    """Structured validation summary for one generated dense reward.

    This payload records the canonical source hashes, component-key contract,
    and semantic key-alignment diagnostics for a generated reward. It is needed
    because downstream evaluators now persist reward-validation artifacts and
    hard-fail semantically invalid candidates before expensive training, and it
    differs from sanitizer-only results by including task-alignment status in
    addition to syntax/AST validation.
    """

    status: str
    failure_reason: Optional[str]
    raw_code_sha16: str
    sanitized_code_sha16: str
    component_keys: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation payload for artifact writing.

        This helper centralizes the persisted validation schema so both the RL
        and A* evaluators can emit the same `reward_validation.json` contract.
        It is needed because the validation payload mixes tuples, strings, and
        nested mappings, and it differs from callers manually constructing dicts
        by keeping the artifact fields aligned with the dataclass.
        """

        return {
            "status": self.status,
            "failure_reason": self.failure_reason,
            "raw_code_sha16": self.raw_code_sha16,
            "sanitized_code_sha16": self.sanitized_code_sha16,
            "component_keys": list(self.component_keys),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class GeneratedReward:
    """Canonical generated dense reward payload consumed by evaluators.

    This record packages the compiled dense reward callable, raw LM response,
    canonical sanitized source, and structured validation metadata for one
    generation attempt. It is needed because artifact writers, GEPA gating, and
    reflection now need more than the callable alone, and it differs from the
    legacy tuple return by making the reward-generation contract explicit.
    """

    dense_fn: Callable[..., Any]
    raw_code: str
    sanitized_code: str
    component_keys: tuple[str, ...]
    validation: GeneratedRewardValidation


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

    synthesizer: Callable[[str, str], str] = field(default_factory=RewardSynthesizer)
    constraints_text: str = CONSTRAINTS_TEXT
    describe_fn: Callable[[object, object], str] = describe_ruleset
    sanitize_fn: Callable[[str], SanitizedRewardResult] = sanitize_reward_code
    lm: Optional[dspy.LM] = None
    max_sanitize_attempts: int = 10
    include_sanitizer_code_on_retry: bool = True
    sanitizer_code_context: Optional[str] = field(default=None, init=False)
    last_attempt_history: List[_AttemptRecord] = field(default_factory=list, init=False)
    # Sticky cache of the most recent environment description sent to the LLM.
    # Used downstream for reflections so the feedback LM knows the goal/ruleset.
    last_env_description: Optional[str] = field(default=None, init=False)
    # Sticky cache of the most recent successfully parsed reward payload so
    # downstream evaluators can reuse validation details when runs fail early.
    last_generated_reward: Optional[GeneratedReward] = field(default=None, init=False)

    def __post_init__(self):
        if self.lm is None:
            self.lm = configure_deepseek_lm()
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

    def generate(self, env, env_params) -> GeneratedReward:
        """Generate one canonical dense reward payload for an environment.

        This routine builds an environment description, queries the reward LLM,
        sanitizes and compiles the returned code, performs immediate
        reward/task-alignment validation, and retries with sanitizer feedback
        when structural validation fails. It is needed because downstream PPO
        and A* evaluators now need a compiled function plus canonical artifact
        metadata, and it differs from lower-level sanitizer utilities by
        orchestrating LM calls, retry history, and validation payload assembly
        in one place.
        """
        env_text = self.describe_fn(env, env_params)
        self.last_env_description = env_text
        attempt_history: List[_AttemptRecord] = []
        self.last_attempt_history = []
        self.last_generated_reward = None
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
                sanitized_reward = self.sanitize_fn(code)
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
            generated_reward = self._build_generated_reward(
                raw_code=code,
                sanitized_reward=sanitized_reward,
                env_description=env_text,
            )
            self.last_generated_reward = generated_reward
            self._log_event(
                "reward_generate_end",
                attempt_idx=attempt_idx,
                total_elapsed_sec=round(time.time() - total_start, 4),
                code_sha16=code_sha16,
                validation_status=generated_reward.validation.status,
            )
            return generated_reward

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

    def _build_generated_reward(
        self,
        *,
        raw_code: str,
        sanitized_reward: SanitizedRewardResult,
        env_description: str,
    ) -> GeneratedReward:
        """Assemble the canonical generated reward payload after sanitization.

        This helper computes stable hashes and semantic task-alignment
        diagnostics for a reward that already passed AST validation. It is
        needed because evaluators and artifact writers need one authoritative
        payload shape, and it differs from `sanitize_reward_code` by attaching
        environment-specific validation derived from the task description.
        """
        raw_code_sha16 = hashlib.sha256(raw_code.encode("utf-8")).hexdigest()[:16]
        sanitized_code_sha16 = hashlib.sha256(
            sanitized_reward.sanitized_code.encode("utf-8")
        ).hexdigest()[:16]
        status = "ok"
        failure_reason: Optional[str] = None
        diagnostics_payload: dict[str, Any]
        try:
            diagnostics = build_reward_object_key_diagnostics(
                sanitized_reward.sanitized_code,
                env_description,
            )
            diagnostics_payload = _reward_key_diagnostics_to_dict(diagnostics)
            missing_from_task = diagnostics_payload.get("missing_from_task", [])
            if missing_from_task:
                status = "invalid_task_mismatch"
                failure_reason = (
                    "Reward/task mismatch detected. Missing object keys from task "
                    f"description: {missing_from_task}."
                )
        except Exception as exc:
            status = "diagnostics_error"
            failure_reason = (
                "Reward/task diagnostics failed before evaluation: "
                f"{exc.__class__.__name__}: {exc}"
            )
            diagnostics_payload = {
                "referenced_object_keys": [],
                "task_object_keys": [],
                "missing_from_task": [],
                "diagnostics_error": str(exc),
            }
        validation = GeneratedRewardValidation(
            status=status,
            failure_reason=failure_reason,
            raw_code_sha16=raw_code_sha16,
            sanitized_code_sha16=sanitized_code_sha16,
            component_keys=sanitized_reward.component_keys,
            diagnostics=diagnostics_payload,
        )
        return GeneratedReward(
            dense_fn=sanitized_reward.dense_reward,
            raw_code=raw_code,
            sanitized_code=sanitized_reward.sanitized_code,
            component_keys=sanitized_reward.component_keys,
            validation=validation,
        )


def create_reward_generator(**kwargs) -> RewardGenerator:
    """Helper to instantiate a RewardGenerator with optional overrides."""
    return RewardGenerator(**kwargs)


def _reward_key_diagnostics_to_dict(
    diagnostics: RewardObjectKeyDiagnostics,
) -> dict[str, Any]:
    """Convert reward-key diagnostics into the shared artifact/logging schema.

    This helper centralizes the mapping from the typed diagnostics dataclass to
    a plain dictionary used across GEPA feedback, result objects, and
    validation-artifact writes. It is needed because callers need stable JSON
    shapes, and it differs from `asdict`-style conversions by preserving the
    field names already used elsewhere in the repo.
    """

    return {
        "referenced_object_keys": list(diagnostics.referenced_object_keys),
        "task_object_keys": list(diagnostics.task_object_keys),
        "missing_from_task": list(diagnostics.missing_from_task),
    }


def persist_generated_reward_artifacts(
    output_dir: Path,
    generated_reward: GeneratedReward,
) -> dict[str, str]:
    """Persist canonical reward-generation artifacts for one evaluation run.

    This helper writes the sanitized executable reward source, the raw LM
    response, and the structured validation payload into a shared artifact
    layout used by both PPO and A* evaluators. It is needed because the
    pipeline must now preserve canonical executable code separately from raw
    model text, and it differs from the legacy inline writes by guaranteeing
    that every caller emits the same filenames and validation schema.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    dense_reward_path = output_dir / "dense_reward_synthesized.py"
    raw_response_path = output_dir / "dense_reward_raw_response.txt"
    validation_path = output_dir / "reward_validation.json"
    validation_payload = (
        generated_reward.validation.to_dict()
        if generated_reward.validation is not None
        else {
            "status": "ok",
            "failure_reason": None,
            "raw_code_sha256": "",
            "sanitized_code_sha256": "",
            "component_keys": list(generated_reward.component_keys),
            "reward_object_key_diagnostics": None,
        }
    )
    dense_reward_path.write_text(
        generated_reward.sanitized_code + "\n",
        encoding="utf-8",
    )
    raw_response_path.write_text(generated_reward.raw_code, encoding="utf-8")
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "dense_reward_path": str(dense_reward_path),
        "dense_reward_raw_response": str(raw_response_path),
        "reward_validation": str(validation_path),
    }


__all__ = [
    "GeneratedReward",
    "GeneratedRewardValidation",
    "RewardGenerator",
    "RewardSynthesizer",
    "create_reward_generator",
    "persist_generated_reward_artifacts",
]
