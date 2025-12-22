"""Helpers for reading/writing shared GEPA state artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

ACTIVE_PROMPT_FILENAME = "active_prompt.json"

def _ensure_state_root(state_root: Path) -> Path:
    state_root = state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    return state_root


def get_active_prompt_path(state_root: Path) -> Path:
    """Return the resolved path to the active prompt JSON."""
    root = _ensure_state_root(state_root)
    return root / ACTIVE_PROMPT_FILENAME


def write_active_prompt(state_root: Path, prompt_payload: Mapping[str, Any]) -> Path:
    """Atomically write the active prompt payload and return the final path."""
    target_path = get_active_prompt_path(state_root)
    temp_path = target_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as temp_file:
        json.dump(prompt_payload, temp_file, indent=2, sort_keys=True)
        temp_file.flush()
    temp_path.replace(target_path)
    return target_path


__all__ = ["get_active_prompt_path", "write_active_prompt"]
