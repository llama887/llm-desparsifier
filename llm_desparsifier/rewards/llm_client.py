"""DSPy LLM client configuration utilities."""

from __future__ import annotations

import json
import os
from typing import Optional

import dspy
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"


def configure_gemini_lm(
    *,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 1.0,
    max_completion_tokens: int = 32_000,
) -> dspy.LM:
    """Create and configure a DSPy LM that calls Gemini directly.

    This helper is the single supported LLM entrypoint for the repository. It
    is needed because reward synthesis, heuristic synthesis, and GEPA
    reflection all require the same Gemini-backed DSPy LM setup, and it differs
    from the removed Portkey configuration by talking directly to Gemini with
    `GEMINI_API_KEY` only.
    """

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    normalized_model = model_name.strip() or DEFAULT_GEMINI_MODEL
    if not normalized_model.startswith("gemini/"):
        normalized_model = f"gemini/{normalized_model}"

    print(
        json.dumps(
            {
                "event": "gemini_lm_config",
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


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "configure_gemini_lm",
]
