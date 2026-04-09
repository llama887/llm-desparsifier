"""Artifact writers for search evaluation and replay.

This module owns the simplified heuristic-only artifact tree. It is needed
because the batch runner, replay tooling, and tests must agree on file names and
payload schemas, and it differs from ad hoc JSON writes by centralizing the new
artifact contract under `heuristic_runs/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON payload with stable formatting and return the path.

    This helper centralizes artifact serialization for the search-only pipeline.
    It is needed because each candidate job writes several replay-related JSON
    files, and it differs from raw `write_text` calls by always creating parent
    directories and using stable formatting.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    """Write a UTF-8 text artifact and return the final path.

    This helper keeps feedback and synthesized code writes consistent with the
    JSON artifact helpers. It is needed because the new artifact tree stores
    source code and feedback alongside JSON metadata, and it differs from raw
    file writes by ensuring parent directories exist first.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["write_json", "write_text"]
