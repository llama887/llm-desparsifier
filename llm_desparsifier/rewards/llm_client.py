"""DSPy LLM client configuration utilities."""

from __future__ import annotations

import json
import os
from typing import Optional

import dspy
from dotenv import load_dotenv

load_dotenv()

# Default Portkey model alias used across the project.
DEFAULT_MODEL_ALIAS = "@vertex-ai-3e806d/gemini-2.5-pro"

# Default Gemini model name used when calling Gemini directly.
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"


def configure_portkey_lm(
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
    model_alias: str = DEFAULT_MODEL_ALIAS,
    temperature: float = 1.0,
    max_completion_tokens: int = 32_000,
) -> dspy.LM:
    """Create and configure a DSPy LM that routes through the Portkey gateway.

    This helper centralizes Portkey-specific configuration (API base URL,
    model alias, temperature, and token budget) so all reward synthesis and
    reflection calls share the same connection settings. It is needed because
    multiple entrypoints initialize LMs, and it differs from ad-hoc instantiation
    by validating credentials and emitting a structured config log for
    debugging rate limits and routing issues.
    """
    api_key = api_key or os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PORTKEY_API_KEY")

    print(
        json.dumps(
            {
                "event": "portkey_lm_config",
                "model_alias": model_alias,
                "api_base": base_url,
                "temperature": temperature,
                "max_tokens": max_completion_tokens,
                "has_api_key": bool(api_key),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    lm = dspy.LM(
        model=f"openai/{model_alias}",
        api_base=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_completion_tokens,
    )
    dspy.configure(lm=lm)
    return lm


def configure_gemini_lm(
    *,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 1.0,
    max_completion_tokens: int = 32_000,
) -> dspy.LM:
    """Create and configure a DSPy LM that calls Gemini directly.

    This helper standardizes Gemini-specific setup by selecting the model name,
    validating credentials, emitting a structured configuration log, and
    returning a ready-to-use `dspy.LM`. It is needed because the project
    supports multiple LLM backends (Portkey vs. direct Gemini) and the
    application code expects a single, preconfigured LM object regardless of
    provider. It differs from `configure_portkey_lm` by bypassing the Portkey
    gateway, requiring `GEMINI_API_KEY`, and normalizing model identifiers to
    the Gemini provider namespace (e.g., `gemini/<model>`), without using a
    gateway base URL or alias routing.
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
    "DEFAULT_MODEL_ALIAS",
    "configure_gemini_lm",
    "configure_portkey_lm",
]
