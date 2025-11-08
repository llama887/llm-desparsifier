"""DSPy/Portkey client configuration utilities."""

from __future__ import annotations

import os
from typing import Optional

import dspy
from dotenv import load_dotenv

load_dotenv()


def configure_portkey_lm(
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
    model_alias: str = "@o3-mini-5791cb/o3-mini",
    temperature: float = 1.0,
    max_completion_tokens: int = 16_000,
) -> dspy.LM:
    """Configure DSPy to route requests through a Portkey gateway."""
    api_key = api_key or os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PORTKEY_API_KEY")

    print("Temperature | Max tokens", temperature, max_completion_tokens)
    lm = dspy.LM(
        model=f"openai/{model_alias}",
        api_base=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_completion_tokens,
    )
    dspy.configure(lm=lm)
    return lm


__all__ = ["configure_portkey_lm"]
