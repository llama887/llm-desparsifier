"""Generate low-priority background GPU work that tracks target utilization.

This helper is intended for Slurm jobs whose real workload alternates between
bursty LLM inference and CPU-array waits. It runs as a separate low-priority
process and adds small CUDA matrix-multiply bursts only when device utilization
falls below a configured target.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import time


@dataclass(frozen=True)
class HeartbeatConfig:
    target_utilization: int
    utilization_tolerance: int
    check_interval: float
    matrix_size: int
    min_compute_seconds: float
    max_compute_seconds: float
    compute_gain_seconds: float
    matmuls_per_chunk: int
    dtype_name: str


DEFAULT_CONFIG = HeartbeatConfig(
    target_utilization=int(
        os.getenv("GPU_HEARTBEAT_TARGET_UTILIZATION", os.getenv("GPU_HEARTBEAT_THRESHOLD", "70"))
    ),
    utilization_tolerance=int(os.getenv("GPU_HEARTBEAT_UTILIZATION_TOLERANCE", "3")),
    check_interval=float(os.getenv("GPU_HEARTBEAT_CHECK_INTERVAL", "0.2")),
    matrix_size=int(os.getenv("GPU_HEARTBEAT_MATRIX_SIZE", "6144")),
    min_compute_seconds=float(os.getenv("GPU_HEARTBEAT_MIN_COMPUTE_SECONDS", "0.10")),
    max_compute_seconds=float(os.getenv("GPU_HEARTBEAT_MAX_COMPUTE_SECONDS", "1.20")),
    compute_gain_seconds=float(os.getenv("GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS", "0.03")),
    matmuls_per_chunk=int(os.getenv("GPU_HEARTBEAT_MATMULS_PER_CHUNK", "8")),
    dtype_name=os.getenv("GPU_HEARTBEAT_DTYPE", "bfloat16"),
)


def resolve_matmul_dtype(torch_module, dtype_name: str):
    dtype_aliases = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    normalized_name = dtype_name.strip().lower()
    if normalized_name not in dtype_aliases:
        raise ValueError(f"unsupported GPU_HEARTBEAT_DTYPE: {dtype_name}")
    return dtype_aliases[normalized_name]


def get_gpu_utilization() -> int:
    """Return current GPU utilization, or 100 on observability failure.

    Returning 100 is the conservative fallback: if `nvidia-smi` is unavailable
    or returns malformed output, the heartbeat sleeps instead of adding load.
    """

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return 100

    values: list[int] = []
    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            return 100
    return max(values) if values else 100


def compute_burst_seconds(
    current_utilization: int,
    *,
    config: HeartbeatConfig = DEFAULT_CONFIG,
) -> float:
    """Choose how long to keep the GPU busy for one controller cycle."""

    utilization_gap = config.target_utilization - current_utilization
    if utilization_gap <= config.utilization_tolerance:
        return 0.0

    unclamped_seconds = (
        config.min_compute_seconds
        + utilization_gap * config.compute_gain_seconds
    )
    return max(
        config.min_compute_seconds,
        min(unclamped_seconds, config.max_compute_seconds),
    )


def main() -> None:
    import torch  # Imported lazily so policy tests do not require CUDA.

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_heartbeat.py requires CUDA but no CUDA device is visible")

    device = torch.device("cuda")
    print(f"Starting GPU heartbeat on {torch.cuda.get_device_name(0)}")
    print(f"PID: {os.getpid()}")
    print(
        "Settings: "
        f"target_utilization={DEFAULT_CONFIG.target_utilization}, "
        f"utilization_tolerance={DEFAULT_CONFIG.utilization_tolerance}, "
        f"check_interval={DEFAULT_CONFIG.check_interval}, "
        f"matrix_size={DEFAULT_CONFIG.matrix_size}, "
        f"min_compute_seconds={DEFAULT_CONFIG.min_compute_seconds}, "
        f"max_compute_seconds={DEFAULT_CONFIG.max_compute_seconds}, "
        f"compute_gain_seconds={DEFAULT_CONFIG.compute_gain_seconds}, "
        f"matmuls_per_chunk={DEFAULT_CONFIG.matmuls_per_chunk}, "
        f"dtype={DEFAULT_CONFIG.dtype_name}",
        flush=True,
    )

    matmul_dtype = resolve_matmul_dtype(torch, DEFAULT_CONFIG.dtype_name)
    x = torch.randn(
        DEFAULT_CONFIG.matrix_size,
        DEFAULT_CONFIG.matrix_size,
        device=device,
        dtype=matmul_dtype,
    )
    y = torch.randn(
        DEFAULT_CONFIG.matrix_size,
        DEFAULT_CONFIG.matrix_size,
        device=device,
        dtype=matmul_dtype,
    )

    while True:
        current_util = get_gpu_utilization()
        burst_seconds = compute_burst_seconds(current_util)
        if burst_seconds == 0.0:
            time.sleep(DEFAULT_CONFIG.check_interval)
            continue

        deadline = time.monotonic() + burst_seconds
        while time.monotonic() < deadline:
            for _ in range(DEFAULT_CONFIG.matmuls_per_chunk):
                torch.mm(x, y)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
