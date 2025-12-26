"""Launch and manage the GPU occupier process."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_truthy(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class GpuOccupier:
    process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if not _env_truthy("GEPA_OCCUPY_GPU", default=True):
            return
        if self.process is not None and self.process.poll() is None:
            return
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "occupy_gpu.py"
        if not script_path.exists():
            print(f"[gpu_occupier] missing script at {script_path}")
            return
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=os.environ.copy(),
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
        except Exception as exc:
            print(f"[gpu_occupier] failed to start: {exc}")
            self.process = None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            self.process = None
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None


_OCCUPIER = GpuOccupier()


def start_gpu_occupier() -> None:
    _OCCUPIER.start()


def stop_gpu_occupier() -> None:
    _OCCUPIER.stop()
