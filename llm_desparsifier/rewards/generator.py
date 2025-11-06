"""High-level reward generation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import dspy

from llm_desparsifier.rewards.llm_client import configure_portkey_lm
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT, describe_ruleset
from llm_desparsifier.rewards.sanitizer import sanitize_and_compile


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

    def __post_init__(self):
        if self.lm is None:
            self.lm = configure_portkey_lm()
        else:
            dspy.configure(lm=self.lm)

    def generate(self, env, env_params) -> Tuple[Callable, str]:
        """Return `(dense_fn, emitted_code)` for the given environment setup."""
        env_text = self.describe_fn(env, env_params)
        code = self.synthesizer(env_text, self.constraints_text)

        if self.verbose:
            print("\n==== Generated dense_reward candidate (pre-sanitize) ====\n")
            print(code)
            print("\nEnvironment Description: \n", env_text)
            print("\n=========================================================\n")

        dense_fn = self.sanitize_fn(code)

        if self.verbose:
            print("\n\n----\n")
            print("Dense Function: \n", dense_fn)
            print("\n----\n\n")

        return dense_fn, code


def create_reward_generator(**kwargs) -> RewardGenerator:
    """Helper to instantiate a RewardGenerator with optional overrides."""
    return RewardGenerator(**kwargs)


__all__ = ["RewardGenerator", "RewardSynthesizer", "create_reward_generator"]
