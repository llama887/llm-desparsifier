"""DSPy LLM client configuration utilities."""

from __future__ import annotations

import json
import os
from typing import Optional

import dspy
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
# Backward-compatible import name for older call sites and tests.
DEFAULT_GEMINI_MODEL = DEFAULT_DEEPSEEK_MODEL


def configure_deepseek_lm(
    *,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_DEEPSEEK_MODEL,
    temperature: float = 1.0,
    max_completion_tokens: int = 32_000,
) -> dspy.LM:
    """Create and configure a DSPy LM that calls DeepSeek directly.

    This helper is the single supported LLM entrypoint for the repository. It
    is needed because reward synthesis, heuristic synthesis, and GEPA
    reflection all require the same DeepSeek-backed DSPy LM setup, and it
    differs from the removed Portkey configuration by talking directly to
    DeepSeek with `DEEPSEEK_API_KEY` only.
    """

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    normalized_model = model_name.strip() or DEFAULT_DEEPSEEK_MODEL
    if not normalized_model.startswith("deepseek/"):
        normalized_model = f"deepseek/{normalized_model}"

    print(
        json.dumps(
            {
                "event": "deepseek_lm_config",
                "model": normalized_model,
                "temperature": temperature,
                "max_tokens": max_completion_tokens,
                "has_api_key": bool(api_key),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    lm = dspy.LM(
        model=normalized_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_completion_tokens,
    )
    dspy.configure(lm=lm)
    return lm


def configure_gemini_lm(**kwargs: object) -> dspy.LM:
    """Backward-compatible alias for the repository's DeepSeek LM setup."""

    return configure_deepseek_lm(**kwargs)


__all__ = [
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "configure_deepseek_lm",
    "configure_gemini_lm",
]
