"""Tests for the GPU heartbeat control policy."""

from __future__ import annotations

from sbatch.gpu_heartbeat import (
    HeartbeatConfig,
    _normalize_gpu_utilizations,
    compute_burst_seconds,
    get_gpu_utilization,
    get_gpu_utilizations,
    resolve_matmul_dtype,
)


def _config() -> HeartbeatConfig:
    return HeartbeatConfig(
        target_utilization=70,
        utilization_tolerance=3,
        check_interval=0.5,
        matrix_size=4096,
        min_compute_seconds=0.05,
        max_compute_seconds=0.60,
        compute_gain_seconds=0.015,
        matmuls_per_chunk=4,
        dtype_name="bfloat16",
    )


def test_compute_burst_seconds_returns_zero_inside_target_deadband() -> None:
    config = _config()

    assert compute_burst_seconds(70, config=config) == 0.0
    assert compute_burst_seconds(67, config=config) == 0.0


def test_compute_burst_seconds_scales_up_with_larger_utilization_gap() -> None:
    config = _config()
    smaller_gap = compute_burst_seconds(60, config=config)
    larger_gap = compute_burst_seconds(30, config=config)

    assert smaller_gap > 0.0
    assert larger_gap > smaller_gap


def test_compute_burst_seconds_respects_configured_caps() -> None:
    config = HeartbeatConfig(
        target_utilization=70,
        utilization_tolerance=3,
        check_interval=0.5,
        matrix_size=4096,
        min_compute_seconds=0.10,
        max_compute_seconds=0.50,
        compute_gain_seconds=0.05,
        matmuls_per_chunk=4,
        dtype_name="bfloat16",
    )

    assert compute_burst_seconds(0, config=config) == 0.50


def test_resolve_matmul_dtype_supports_tensor_core_friendly_aliases() -> None:
    class _FakeTorch:
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

    fake_torch = _FakeTorch()

    assert resolve_matmul_dtype(fake_torch, "bf16") == "bfloat16"
    assert resolve_matmul_dtype(fake_torch, "fp16") == "float16"
    assert resolve_matmul_dtype(fake_torch, "fp32") == "float32"


def test_get_gpu_utilization_uses_conservative_fallback(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr("subprocess.check_output", _raise)

    assert get_gpu_utilization() == 100


def test_get_gpu_utilization_parses_multiple_visible_devices(monkeypatch) -> None:
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "4\n71\n")

    assert get_gpu_utilization() == 71


def test_get_gpu_utilizations_filters_cuda_visible_device_indices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *args, **kwargs: "0, 4\n1, 71\n2, 15\n3, 9\n",
    )

    assert get_gpu_utilizations() == [15, 9]


def test_normalize_gpu_utilizations_returns_one_value_per_device() -> None:
    assert _normalize_gpu_utilizations([12, 34, 56], 2) == [12, 34]
    assert _normalize_gpu_utilizations([12], 2) == [100, 100]
