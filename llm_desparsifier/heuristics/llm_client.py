"""LLM client helpers reused by heuristic synthesis.

This module re-exports the shared DeepSeek configuration helper so the
heuristic-only pipeline can use the same transport without duplicating setup
code. It is needed because the refactor changes the artifact being synthesized
rather than the provider integration itself, and it differs from importing the
reward package directly by giving the heuristic pipeline a heuristic-native
import surface.
"""

from llm_desparsifier.rewards.llm_client import (  # noqa: F401
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GEMINI_MODEL,
    configure_deepseek_lm,
    configure_gemini_lm,
)

__all__ = [
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "configure_deepseek_lm",
    "configure_gemini_lm",
]
