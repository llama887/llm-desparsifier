"""EUREKA-style reward reflection builder."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import dspy
import numpy as np

from llm_desparsifier.rewards.llm_client import configure_portkey_lm

EUREKA_GUIDANCE = (
    "You are the Reward Reflection module, your job is to reflect on the current generated reward function."
    "Your job is NOT to fix the reward function. We trained a dense-reward RL policy, "
    "logged sparse success curves plus per-component checkpoints, and now need actionable "
    "feedback. Carefully study the policy feedback and propose a revised reward strategy. "
    "IMPORTANT: Do NOT output code, pseudo-code, diffs, or function definitions. "
    "Do NOT include any fenced code blocks. Use concise natural-language guidance only. "
    "Describe changes at a conceptual level (what to change and why), not how to implement them. "
    "Follow these tips: (1) If success/sparse returns stay "
    "near zero, propose a new reward structure in words. (2) When a component stays nearly "
    "constant, RL could not optimize it—suggest scaling its magnitude, changing its "
    "temperature/shape, rewriting it, or discarding it. (3) If a component's magnitude "
    "dominates the rest, rescale it into a reasonable range. Reference specific component "
    "names and explain how to adjust each one before describing how to revise the overall "
    "reward function, without writing any code."
)


class RewardReflectionSignature(dspy.Signature):
    """DSPy signature describing the reflection task."""

    env_summary: str = dspy.InputField(desc="Environment description and goal context.")
    reward_code: str = dspy.InputField(desc="Current dense_reward implementation.")
    sparse_curve_summary: str = dspy.InputField(desc="Sparse reward checkpoints over training.")
    component_curve_summary: str = dspy.InputField(desc="Per-component reward snapshots.")
    metrics_summary: str = dspy.InputField(desc="Aggregate evaluation metrics and gaps.")
    guidance: str = dspy.InputField(desc="Instructions describing the desired reflection style.")
    reflection: str = dspy.OutputField(
        desc="Actionable EUREKA-style reward reflection (natural language only, no code blocks)."
    )


def create_reward_reflection_module(lm: Optional[dspy.LM] = None) -> dspy.Module:
    """Create a DSPy module that will emit EUREKA-style reflections."""

    if lm is None:
        lm = configure_portkey_lm()
    dspy.configure(lm=lm)
    return dspy.ChainOfThought(RewardReflectionSignature)


def build_reward_reflection(
    run_record: Mapping[str, Any],
    *,
    reflection_module: Optional[dspy.Module] = None,
    guidance_text: str = EUREKA_GUIDANCE,
) -> str:
    """Generate a EUREKA-style reward reflection for a single RL run record.

    The guidance_text may already include run-specific context (env description, budgets, sanitizer notes).
    """

    env_summary = _compose_env_summary(run_record)
    reward_code = str(run_record.get("reward_code", "")).strip()
    sparse_summary = _format_sparse_curve(run_record.get("sparse_return_curve"))
    component_summary = _format_component_curves(run_record.get("component_curves"))
    component_stats_summary = _format_component_stats(run_record.get("component_curves"))
    metrics_summary = _format_metrics(run_record.get("final_metrics"))

    module = reflection_module or create_reward_reflection_module()

    try:
        prediction = module(
            env_summary=env_summary,
            reward_code=reward_code,
            sparse_curve_summary=sparse_summary,
            component_curve_summary="\n".join(
                [text for text in [component_summary, component_stats_summary] if text]
            ),
            metrics_summary=metrics_summary,
            guidance=guidance_text,
        )
        reflection_text = str(getattr(prediction, "reflection", "")).strip()
        if reflection_text:
            return _compose_feedback_with_raw_inputs(
                reflection_text,
                env_summary=env_summary,
                reward_code=reward_code,
                sparse_summary=sparse_summary,
                component_summary=component_summary,
                component_stats_summary=component_stats_summary,
                metrics_summary=metrics_summary,
            )
        raise ValueError("Empty reflection output from LM")
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        return _compose_fallback_text(
            env_summary=env_summary,
            sparse_summary=sparse_summary,
            component_summary=component_summary,
            metrics_summary=metrics_summary,
            error_message=str(exc),
        )


def _compose_env_summary(run_record: Mapping[str, Any]) -> str:
    env_description = run_record.get("env_description") or run_record.get("env_id") or "Unknown env"
    benchmark_id = run_record.get("benchmark_id")
    if benchmark_id:
        return f"{env_description} (benchmark={benchmark_id})"
    return str(env_description)


def _format_sparse_curve(values: Any) -> str:
    samples = _sample_series(values)
    if not samples:
        return "Sparse reward checkpoints unavailable (no evaluation logs)."
    formatted = ", ".join(f"{value:.3f}" for value in samples)
    final_value = float(samples[-1])
    return f"Sparse reward checkpoints: [{formatted}] → final={final_value:.3f}"


def _format_component_curves(curves: Any) -> str:
    if not isinstance(curves, Mapping) or not curves:
        return "Reward component checkpoints unavailable."
    lines: list[str] = []
    for component_name in sorted(curves.keys()):
        samples = _sample_series(curves[component_name])
        if samples:
            sample_text = ", ".join(f"{value:.3f}" for value in samples)
            line = f"{component_name}: [{sample_text}]"
        else:
            line = f"{component_name}: (no data)"
        lines.append(line)
    return "\n".join(lines)


def _format_component_stats(curves: Any) -> str:
    if not isinstance(curves, Mapping) or not curves:
        return ""
    lines: list[str] = ["Component magnitude stats (min/mean/max):"]
    for name in sorted(curves.keys()):
        arr = np.asarray(curves[name], dtype=float).flatten()
        if arr.size == 0:
            lines.append(f"- {name}: no data")
            continue
        lines.append(
            f"- {name}: min={float(arr.min()):.3f}, mean={float(arr.mean()):.3f}, max={float(arr.max()):.3f}"
        )
    return "\n".join(lines)


def _format_metrics(metrics: Any) -> str:
    if not isinstance(metrics, Mapping) or not metrics:
        return "No aggregate metrics reported."
    parts: list[str] = []
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, (int, float)):
            parts.append(f"{key}={float(value):.3f}")
        else:
            parts.append(f"{key}={value}")
    return "Metrics: " + ", ".join(parts)


def _sample_series(values: Any, num_points: int = 6) -> list[float]:
    if values is None:
        return []
    array = np.asarray(values, dtype=float).flatten()
    if array.size == 0:
        return []
    if array.size <= num_points:
        return array.tolist()
    indices = np.linspace(0, array.size - 1, num_points, dtype=int)
    return [float(array[index]) for index in indices]


def _compose_feedback_with_raw_inputs(
    reflection_text: str,
    *,
    env_summary: str,
    reward_code: str,
    sparse_summary: str,
    component_summary: str,
    component_stats_summary: str,
    metrics_summary: str,
) -> str:
    raw_sections = [
        "Env summary:",
        env_summary or "(empty)",
        "",
        "Sparse curve summary:",
        sparse_summary or "(empty)",
        "",
        "Component curve summary:",
        component_summary or "(empty)",
        "",
        "Component stats summary:",
        component_stats_summary or "(empty)",
        "",
        "Metrics summary:",
        metrics_summary or "(empty)",
        "",
        "Reward code (raw):",
        reward_code or "(empty)",
    ]
    return "\n".join(
        [
            "[Reward reflection]",
            reflection_text.strip(),
            "",
            "[Raw EUREKA inputs]",
            "\n".join(raw_sections).strip(),
        ]
    ).strip()


def _compose_fallback_text(
    *,
    env_summary: str,
    sparse_summary: str,
    component_summary: str,
    metrics_summary: str,
    error_message: str,
) -> str:
    return (
        "[Fallback reward reflection due to LLM error: "
        + error_message
        + "]\n"
        + env_summary
        + "\n"
        + sparse_summary
        + "\n"
        + component_summary
        + "\n"
        + metrics_summary
    )


__all__ = ["build_reward_reflection", "create_reward_reflection_module", "RewardReflectionSignature"]
