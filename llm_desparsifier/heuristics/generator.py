"""High-level heuristic generation orchestrator.

This module handles prompt execution, sanitizer retries, and source capture for
the heuristic-only GEPA pipeline. It is needed because the evaluator requires
both a callable heuristic and the emitted source code, and it differs from the
legacy reward generator by targeting a single admissible-leaning scalar
heuristic contract.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import dspy

from .llm_client import configure_gemini_lm
from .prompting import BASE_HEURISTIC_PROMPT, HEURISTIC_CONTRACT_TEXT
from .sanitizer import sanitize_and_compile_heuristic


@dataclass
class _AttemptRecord:
    code: str
    error_text: str
    timestamp: str


class HeuristicSynthesis(dspy.Signature):
    """DSPy signature for heuristic synthesis."""

    env_description: str = dspy.InputField()
    heuristic_contract: str = dspy.InputField()
    constraints: str = dspy.InputField()
    heuristic_code: str = dspy.OutputField(
        desc="Only one Python function named heuristic_cost_to_go(ts, env_params, ctx)"
    )


class HeuristicSynthesizer(dspy.Module):
    """DSPy module that emits heuristic source code.

    This module wraps one `dspy.Predict` call so synthesis state can still be
    optimized by GEPA. It is needed because the heuristic prompt is a proper
    program under optimization rather than a one-off raw completion, and it
    differs from the prompt rewriter by emitting code instead of prompt text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gen = dspy.Predict(HeuristicSynthesis)

    def forward(
        self,
        env_description: str,
        heuristic_contract: str,
        constraints: str,
    ) -> str:
        """Generate heuristic code from environment and contract text.

        This method forwards the three synthesis inputs directly to the model.
        It is needed because prompt optimization controls the `constraints`
        field while the environment and contract vary per job, and it differs
        from the top-level generator by returning raw model text without any
        sanitization or retry logic.
        """

        out = self.gen(
            env_description=env_description,
            heuristic_contract=heuristic_contract,
            constraints=constraints,
        )
        return out.heuristic_code


@dataclass
class HeuristicGenerator:
    """Generate admissible-leaning heuristics and sanitize the result.

    This class orchestrates LLM calls plus sanitizer retries and exposes the
    final callable together with the emitted source code. It is needed because
    the search evaluator must compile generated code once per job and still save
    retry diagnostics for GEPA feedback, and it differs from calling DSPy
    directly by owning sanitizer retry state and artifact-friendly metadata.
    """

    synthesizer: HeuristicSynthesizer = field(default_factory=HeuristicSynthesizer)
    constraints_text: str = BASE_HEURISTIC_PROMPT
    heuristic_contract: str = HEURISTIC_CONTRACT_TEXT
    sanitize_fn: Callable[[str], Callable[..., float]] = sanitize_and_compile_heuristic
    lm: Optional[dspy.LM] = None
    max_sanitize_attempts: int = 5
    last_attempt_history: list[_AttemptRecord] = field(default_factory=list, init=False)
    last_env_description: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.lm is None:
            self.lm = configure_gemini_lm()
            dspy.configure(lm=self.lm)
        if self.max_sanitize_attempts < 1:
            raise ValueError("max_sanitize_attempts must be >= 1")

    def _log_event(self, event: str, **fields: Any) -> None:
        """Emit compact structured logs for synthesis progress.

        This helper keeps long-running heuristic generation visible in stdout. It
        is needed because GEPA evaluations may block on model calls or sanitizer
        retries, and it differs from plain prints by emitting one JSON object
        per line with stable metadata keys.
        """

        payload = {
            "component": "heuristic_generator",
            "event": event,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        payload.update(fields)
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)

    def generate(
        self,
        *,
        env_description: str,
        heuristic_contract: Optional[str] = None,
    ) -> tuple[Callable[..., float], str]:
        """Generate, sanitize, and compile one heuristic implementation.

        This method is the main synthesis entrypoint used by the batch runner.
        It is needed because one candidate prompt must be turned into a callable
        heuristic exactly once per job evaluation, and it differs from the raw
        DSPy synthesizer by retrying with sanitizer feedback before failing.
        """

        contract_text = heuristic_contract or self.heuristic_contract
        self.last_env_description = env_description
        self.last_attempt_history = []
        total_start = time.time()
        for attempt_idx in range(1, self.max_sanitize_attempts + 1):
            feedback_block = self._build_feedback_block(self.last_attempt_history)
            constraints = self.constraints_text
            if feedback_block:
                constraints = f"{constraints}\n\n{feedback_block}"
            self._log_event("llm_call_start", attempt_idx=attempt_idx)
            with dspy.settings.context(lm=self.lm):
                code = self.synthesizer(
                    env_description=env_description,
                    heuristic_contract=contract_text,
                    constraints=constraints,
                )
            code_sha16 = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
            self._log_event(
                "llm_call_end",
                attempt_idx=attempt_idx,
                code_len=len(code),
                code_sha16=code_sha16,
            )
            try:
                heuristic_fn = self.sanitize_fn(code)
            except (SyntaxError, ValueError) as exc:
                error_text = f"{exc.__class__.__name__}: {exc}"
                self.last_attempt_history.append(
                    _AttemptRecord(
                        code=code,
                        error_text=error_text,
                        timestamp=dt.datetime.now(dt.timezone.utc).strftime(
                            "%Y%m%d-%H%M%S"
                        ),
                    )
                )
                self._log_event(
                    "sanitize_failure",
                    attempt_idx=attempt_idx,
                    error=error_text,
                )
                if attempt_idx >= self.max_sanitize_attempts:
                    raise RuntimeError(self._format_retry_failure()) from exc
                continue
            self._log_event(
                "heuristic_generate_end",
                attempt_idx=attempt_idx,
                elapsed_sec=round(time.time() - total_start, 4),
                code_sha16=code_sha16,
            )
            return heuristic_fn, code
        raise RuntimeError("heuristic generation exited unexpectedly")

    def _build_feedback_block(self, attempts: list[_AttemptRecord]) -> str:
        """Build sanitizer retry guidance for the next synthesis attempt.

        This helper turns prior sanitizer failures into deterministic prompt
        guidance. It is needed because GEPA may discover prompt revisions that
        still occasionally emit invalid code, and it differs from runtime
        exceptions by giving the model structured information it can act on.
        """

        if not attempts:
            return ""
        lines = [
            "Sanitizer retry guidance:",
            "You already emitted invalid heuristic code. Fix every issue below.",
        ]
        for idx, attempt in enumerate(attempts, start=1):
            lines.append(f"- Attempt {idx}: {attempt.error_text}")
        lines.append(
            "Re-emit exactly one valid `heuristic_cost_to_go(ts, env_params, ctx)` function."
        )
        return "\n".join(lines)

    def _format_retry_failure(self) -> str:
        """Format the terminal retry failure message.

        This helper summarizes every sanitizer failure after retries are
        exhausted. It is needed because the batch runner stores the failure text
        in feedback artifacts, and it differs from a raw exception chain by
        presenting one compact deterministic summary.
        """

        lines = [
            f"Failed to sanitize heuristic_cost_to_go after {len(self.last_attempt_history)} attempt(s)."
        ]
        lines.extend(
            f"Attempt {idx}: {attempt.error_text}"
            for idx, attempt in enumerate(self.last_attempt_history, start=1)
        )
        return "\n".join(lines)


__all__ = ["HeuristicGenerator", "HeuristicSynthesizer"]
