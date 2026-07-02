"""Generate low-priority background GPU work that tracks target utilization.

This helper is intended for Slurm jobs whose real workload alternates between
bursty LLM inference and CPU-array waits. It runs as a separate low-priority
process and adds small CUDA matrix-multiply bursts only when device utilization
falls below a configured target.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class HeartbeatConfig:
    """Runtime policy for the low-priority CUDA utilization controller."""

    target_utilization: int
    utilization_tolerance: int
    check_interval: float
    matrix_size: int
    min_compute_seconds: float
    max_compute_seconds: float
    compute_gain_seconds: float
    matmuls_per_chunk: int
    dtype_name: str
    log_interval_s: float = 60.0


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
    log_interval_s=float(os.getenv("GPU_HEARTBEAT_LOG_INTERVAL_S", "60.0")),
)


def resolve_matmul_dtype(torch_module, dtype_name: str):
    """Resolve a configured dtype alias to a `torch` matrix-multiply dtype.

    The heartbeat is intended to run on tensor-core GPUs from a shell script, so
    the environment accepts common short aliases while still rejecting unknown
    values before any CUDA buffers are allocated.
    """

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


def _visible_cuda_device_indices() -> list[str] | None:
    """Return physical device indices from `CUDA_VISIBLE_DEVICES` when explicit.

    Slurm commonly exposes allocated GPUs through `CUDA_VISIBLE_DEVICES`.
    `nvidia-smi` still reports physical device indices, so we filter its output
    when this environment variable contains numeric ids. UUID and MIG forms are
    left unfiltered because they do not map directly to the simple index query.
    """

    raw_value = os.getenv("CUDA_VISIBLE_DEVICES")
    if not raw_value:
        return None
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not parts:
        return None
    if all(part.isdigit() for part in parts):
        return parts
    return None


def get_gpu_utilizations() -> list[int]:
    """Return visible GPU utilizations, or `[100]` on observability failure.

    Returning 100 is the conservative fallback: if `nvidia-smi` is unavailable,
    returns malformed output, or cannot be matched to visible devices, the
    heartbeat sleeps instead of adding load.
    """

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return [100]

    indexed_values: dict[str, int] = {}
    ordered_values: list[int] = []
    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            index_text, utilization_text = [part.strip() for part in line.split(",", 1)]
        else:
            index_text = str(len(ordered_values))
            utilization_text = line
        try:
            utilization = int(utilization_text)
        except ValueError:
            return [100]
        indexed_values[index_text] = utilization
        ordered_values.append(utilization)

    if not ordered_values:
        return [100]

    visible_indices = _visible_cuda_device_indices()
    if visible_indices is None:
        return ordered_values

    try:
        return [indexed_values[index] for index in visible_indices]
    except KeyError:
        return [100]


def get_gpu_utilization() -> int:
    """Return the maximum visible GPU utilization for legacy callers."""

    return max(get_gpu_utilizations())


def _normalize_gpu_utilizations(current_utils: list[int], device_count: int) -> list[int]:
    """Return one utilization value for each visible CUDA device.

    `nvidia-smi` observability is outside the CUDA process and can briefly return
    fewer devices than `torch.cuda.device_count()`, especially around Slurm
    startup. In that ambiguous case the controller fails closed by treating all
    devices as fully busy instead of accidentally adding load to the wrong GPU.
    Extra values are trimmed to the devices visible to this process.
    """

    if len(current_utils) < device_count:
        return [100] * device_count
    return current_utils[:device_count]


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
    import torch  # pylint: disable=import-error  # Lazy import for sbatch CUDA runtime.

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_heartbeat.py requires CUDA but no CUDA device is visible")

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("gpu_heartbeat.py requires at least one visible CUDA device")

    devices = [torch.device(f"cuda:{index}") for index in range(device_count)]
    device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    print(f"Starting GPU heartbeat on {device_count} CUDA device(s): {device_names}")
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
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
        f"dtype={DEFAULT_CONFIG.dtype_name}, "
        f"log_interval_s={DEFAULT_CONFIG.log_interval_s}",
        flush=True,
    )

    matmul_dtype = resolve_matmul_dtype(torch, DEFAULT_CONFIG.dtype_name)
    matrices = []
    for device in devices:
        with torch.cuda.device(device):
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
            output = torch.empty(
                DEFAULT_CONFIG.matrix_size,
                DEFAULT_CONFIG.matrix_size,
                device=device,
                dtype=matmul_dtype,
            )
        matrices.append((device, x, y, output))

    next_log_at = time.monotonic()
    while True:
        current_utils = _normalize_gpu_utilizations(get_gpu_utilizations(), device_count)
        bursts = [
            compute_burst_seconds(current_utilization)
            for current_utilization in current_utils
        ]
        now = time.monotonic()
        if DEFAULT_CONFIG.log_interval_s > 0 and now >= next_log_at:
            print(
                "[gpu-heartbeat] utilizations="
                f"{current_utils} bursts={[round(value, 3) for value in bursts]}",
                flush=True,
            )
            next_log_at = now + DEFAULT_CONFIG.log_interval_s

        if not any(burst_seconds > 0.0 for burst_seconds in bursts):
            time.sleep(DEFAULT_CONFIG.check_interval)
            continue

        deadlines = [time.monotonic() + burst_seconds for burst_seconds in bursts]
        while True:
            active_indices = [
                index
                for index, deadline in enumerate(deadlines)
                if bursts[index] > 0.0 and time.monotonic() < deadline
            ]
            if not active_indices:
                break
            for index in active_indices:
                device, x, y, output = matrices[index]
                with torch.cuda.device(device):
                    for _ in range(DEFAULT_CONFIG.matmuls_per_chunk):
                        torch.mm(x, y, out=output)
            for index in active_indices:
                torch.cuda.synchronize(matrices[index][0])


if __name__ == "__main__":
    main()
