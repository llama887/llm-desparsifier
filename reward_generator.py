"""Compatibility wrapper around the refactored reward generation modules."""

from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

from llm_desparsifier.rewards.generator import RewardGenerator
from llm_desparsifier.rewards.llm_client import configure_portkey_lm as configure_dspy_with_portkey
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT, describe_ruleset
from llm_desparsifier.rewards.sanitizer import sanitize_and_compile

__all__ = [
    "configure_dspy_with_portkey",
    "sanitize_and_compile",
    "describe_ruleset",
    "CONSTRAINTS_TEXT",
    "make_dense_reward",
]

_GENERATOR_LOCK = threading.Lock()
_DEFAULT_GENERATOR: Optional[RewardGenerator] = None


def _get_default_generator() -> RewardGenerator:
    global _DEFAULT_GENERATOR
    with _GENERATOR_LOCK:
        if _DEFAULT_GENERATOR is None:
            _DEFAULT_GENERATOR = RewardGenerator()
        return _DEFAULT_GENERATOR


def make_dense_reward(env, env_params, dspy_model=None) -> Tuple[Callable, str]:
    """Backward compatible entry-point returning `(dense_fn, emitted_code)`."""
    if dspy_model is not None:
        generator = RewardGenerator(synthesizer=dspy_model)
        return generator.generate(env, env_params)
    generator = _get_default_generator()
    return generator.generate(env, env_params)
