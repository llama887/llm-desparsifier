"""DSPy cache setup for long-running batch entrypoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


_FALSE_VALUES = {"0", "false", "no", "off"}


def dspy_disk_cache_disabled() -> bool:
    """Return whether batch scripts should disable DSPy's SQLite disk cache."""

    return os.environ.get("DSPY_DISABLE_DISK_CACHE", "1").strip().lower() not in _FALSE_VALUES


def prepare_dspy_import(script_name: str) -> None:
    """Prepare a safe import-time DSPy cache directory.

    DSPy constructs its disk cache while importing `dspy`. Even when we disable
    the disk cache immediately after import, the import-time constructor still
    needs a directory that will not collide with another job or stale SQLite
    database. A private temp directory keeps that constructor harmless.
    """

    if not dspy_disk_cache_disabled():
        return

    base_dir = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or tempfile.gettempdir())
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in script_name)
    os.environ["DSPY_CACHEDIR"] = tempfile.mkdtemp(
        prefix=f"{safe_name}-dspy-import-",
        dir=str(base_dir),
    )


def configure_dspy_cache(dspy_module: Any, script_name: str) -> None:
    """Disable DSPy's disk cache after import while keeping in-memory caching."""

    if not dspy_disk_cache_disabled():
        return

    dspy_module.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    print(
        f"[dspy-cache] {script_name}: disk cache disabled; memory cache enabled",
        flush=True,
    )
