"""Pytest bootstrap for libraries that create caches at import time."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault(
    "DSPY_CACHEDIR",
    tempfile.mkdtemp(prefix="llm-desparsifier-test-dspy-cache-"),
)
os.environ.setdefault("DSPY_DISABLE_DISK_CACHE", "1")
