#!/usr/bin/env python3
"""Keep GPU busy with repeated matmul operations."""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[occupy_gpu] invalid int for {name}={value!r}, using {default}", flush=True)
        return default


def _get_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _resolve_dtype(name: str) -> Any:
    mapping = {
        "float16": jnp.float16,
        "fp16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
        "float32": jnp.float32,
        "fp32": jnp.float32,
    }
    dtype = mapping.get(name.lower())
    if dtype is None:
        print(f"[occupy_gpu] unknown dtype {name!r}, defaulting to float16", flush=True)
        return jnp.float16
    return dtype


def _backend_is_gpu() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def main() -> int:
    if not _backend_is_gpu():
        print("[occupy_gpu] No GPU backend detected; exiting.", flush=True)
        return 0

    matrix_size = _get_int("GEPA_OCCUPY_GPU_M", 4096)
    duty_ms = _get_int("GEPA_OCCUPY_GPU_DUTY_MS", 0)
    dtype_name = _get_str("GEPA_OCCUPY_GPU_DTYPE", "float16")
    dtype = _resolve_dtype(dtype_name)

    print(
        f"[occupy_gpu] starting pid={os.getpid()} size={matrix_size} dtype={dtype_name} duty_ms={duty_ms}",
        flush=True,
    )

    running = True

    def _handle_stop(_: int, __: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    a = jnp.ones((matrix_size, matrix_size), dtype=dtype)
    b = jnp.ones((matrix_size, matrix_size), dtype=dtype)

    @jax.jit
    def _step(x: jax.Array, y: jax.Array) -> jax.Array:
        return x @ y

    _step(a, b).block_until_ready()

    while running:
        _step(a, b).block_until_ready()
        if duty_ms > 0:
            time.sleep(duty_ms / 1000.0)

    print("[occupy_gpu] stopping", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
