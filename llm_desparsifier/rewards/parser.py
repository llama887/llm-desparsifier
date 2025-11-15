"""Prompt construction helpers for reward synthesis."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from typing import List, Optional, Tuple

try:
    from xminigrid.rendering.text_render import print_ruleset as _print_ruleset
except Exception:  # pragma: no cover - optional dependency guard
    _print_ruleset = None


__all__ = ["describe_ruleset", "CONSTRAINTS_TEXT"]

_LAYOUT_HINTS = {
    "R1": "a single rectangular room (no interior walls)",
    "R2": "two rooms separated by an interior wall with one doorway",
    "R4": "four rooms separated by interior walls (the classic Four Rooms layout)",
    "R6": "six rooms separated by interior walls and doors",
    "R9": "nine rooms in a 3×3 arrangement with interior doors",
}

_ACTIONS_LINE = (
    "Actions: move_forward, turn_left, turn_right, pick_up, put_down, toggle (one object carried at a time)."
)

_GOAL_TILE_NEAR_RIGHT_RE = re.compile(
    r"TileNearRightGoal\s*\(\s*([^) ,]+(?:\s+[^\),]+)?)\s*,\s*([^) ,]+(?:\s+[^\),]+)?)\s*\)",
    re.IGNORECASE,
)


def _safe_getattr(obj, name: str, default: str) -> str:
    try:
        value = getattr(obj, name, default)
        return str(value if value is not None else default)
    except Exception:
        return default


def _parse_ruleset_text(text: str) -> tuple[Optional[str], List[str], List[str]]:
    goal_line = None
    rule_lines: List[str] = []
    init_lines: List[str] = []

    lines = [ln.strip() for ln in text.splitlines()]
    section = None
    for ln in lines:
        if not ln:
            continue
        key = ln.upper().rstrip(":")
        if key == "GOAL":
            section = "GOAL"
            continue
        if key == "RULES":
            section = "RULES"
            continue
        if key.startswith("INIT TILES"):
            section = "INIT"
            continue

        if section == "GOAL" and goal_line is None:
            goal_line = ln
        elif section == "RULES":
            rule_lines.append(ln)
        elif section == "INIT":
            init_lines.append(ln)

    return goal_line, rule_lines, init_lines


def _explain_goal(goal_line: Optional[str]) -> Optional[str]:
    if not goal_line:
        return None
    match = _GOAL_TILE_NEAR_RIGHT_RE.search(goal_line)
    if match:
        left_obj, right_obj = match.group(2).strip(), match.group(1).strip()
        return (
            f"SUCCESS when **{right_obj}** is immediately to the **left** of **{left_obj}** "
            f"(i.e., {left_obj} is exactly one cell to the right of {right_obj}, same row, adjacent columns)."
        )
    return f"SUCCESS when condition holds: {goal_line}"


def describe_ruleset(env, env_params) -> str:
    """Produce an LLM-friendly description of the current task."""
    height = _safe_getattr(env_params, "height", "?")
    width = _safe_getattr(env_params, "width", "?")
    view = _safe_getattr(env_params, "view_size", "?")
    max_steps = _safe_getattr(env_params, "max_steps", "?")
    grid_type = _safe_getattr(env_params, "grid_type", "unknown")

    layout_hint = _LAYOUT_HINTS.get(str(grid_type), "a grid-world layout with interior walls and doors")

    goal_line = None
    rule_lines: List[str] = []
    init_lines: List[str] = []
    if _print_ruleset is not None:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                _print_ruleset(getattr(env_params, "ruleset", None))
            summary = buf.getvalue().strip()
            if summary:
                goal_line, rule_lines, init_lines = _parse_ruleset_text(summary)
        except Exception:
            pass

    goal_expl = _explain_goal(goal_line)

    init_obj_list = ", ".join(init_lines[:10]) if init_lines else "unknown (randomized at reset)"
    if init_lines and len(init_lines) > 10:
        init_obj_list += f", ... (+{len(init_lines) - 10} more)"

    rules_summary = "\n".join(f"- {r}" for r in rule_lines[:8]) if rule_lines else "No explicit transformation rules provided."
    if rule_lines and len(rule_lines) > 8:
        rules_summary += f"\n- ... (+{len(rule_lines) - 8} more)"

    lines = [
        f"grid_type={grid_type} → {layout_hint}",
        f"size={height}x{width}, view={view} (agent-centered egocentric  {view}×{view}  symbolic grid), max_steps={max_steps}.",
        _ACTIONS_LINE,
        "",
    ]

    if goal_line:
        lines.append("GOAL:")
        lines.append(goal_line)
        if goal_expl:
            lines.append(goal_expl)
        lines.append("")

    lines.append("RULES:")
    lines.append(rules_summary)
    lines.append("")

    lines.append("INITIAL OBJECTS:")
    lines.append(init_obj_list)
    lines.append("")

    lines.append("Observations are partially observable and symbolic (not pixels). Use distances and spatial relations; avoid Python-side branching.")

    return "\n".join(lines)


CONSTRAINTS_TEXT = """
You are designing a dense reward function for the Xland-Minigrid environment
You must output exactly ONE function:
  def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    # returns (total_reward: jnp.float32, reward_components: dict[str, jnp.float32])

Context:
- The function will be installed as `dense_fn` inside `DesparsifyRewardWrapper` (see snippet below) and invoked either with three args `(ts_prev, action, ts_next)` or the five-arg signature shown above. Always implement the five-arg form; the wrapper detects it via `inspect.signature`.
- `ctx` is produced (when configured) by a pure `ctx_fn(env_params, ts_prev, ts_next)` that runs right after `env.step`. It returns a dictionary mapping strings to JAX arrays (you may only access fields of ctx by using ctx.get(...) instead of ctx[...]). Each entry is derived from the `xminigrid.types.TimeStep` objects:
    - Base scalar/pose keys per timestep: `agent_pos`, `agent_direction`, `step_num`, `is_carrying`, `carried_item`, `yellow_square_pos`, `green_ball_pos`, plus `_prev` copies.
    - Nested `object_positions`: a dict-like map whose keys follow `"{color}_{tile}"` snake_case (e.g., `"yellow_square"`, `"green_ball"`). Each entry is a `[row, col]` array or `[-1, -1]` if the object is absent. Always read it via `obj_pos = ctx.get("object_positions", {})` and `obj_pos.get("yellow_square", jnp.array([-1, -1], dtype=jnp.int32))`.
    - Additional helpers (distances, flags, etc.) may appear for specific ctx functions, but you must guard them with `ctx.get` because they are optional.
- When the wrapper runs, it replaces the environment's sparse reward with your dense value: `ts_next = env.step(...)`, `dense_reward` executes, and the wrapper expects a tuple `(total_reward, reward_components)`.
- `reward_components` MUST be a Python dict literal whose keys are descriptive strings (`"progress"`, `"penalty"`, `"shaping"`, etc.) and whose values are scalar jnp arrays produced in the same function. These components will be logged individually for EUREKA reward reflection.
- Existing placeholder reward (for reference only):

```python
def dummy_dense_reward(env_params, ts_prev, action, ts_next, ctx):
    zeros = jnp.full_like(ts_next.reward, 0.0)
    reward_components = {
        "progress": zeros,
        "penalty": zeros,
    }
    return zeros, reward_components
```

- Use ONLY jax.numpy as jnp (import not needed) and jax.lax if necessary.
- Do NOT add import statements; jnp and jax are already available.
- Use ONLY values that arrive via `ctx`. Every key is optional—*always* pull them with `.get` and provide explicit fallbacks (e.g., `ctx.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))`). Accessing `ctx[...]` directly is invalid and will be rejected.
- When dealing with nested structures such as `object_positions`, guard both the parent map and each child lookup. Example:
  ```python
  obj_pos = ctx.get("object_positions", {}) # you must use ctx.get(...) instead of ctx[...]
  yellow_square = obj_pos.get("yellow_square", jnp.array([-1, -1], dtype=jnp.int32))
  green_ball = obj_pos.get("green_ball", jnp.array([-1, -1], dtype=jnp.int32))
  ```
- Call **only** JAX primitives (`jnp.*`, `jax.lax.*`) or helper functions you define inside `dense_reward`. Do **not** invoke Python `math.*`, `numpy.*`, or arbitrary library functions; the sanitizer will reject them.
- If you define helper functions inside dense_reward, ensure they are pure, side-effect free, and only call jnp/jax operations.
- Do NOT access Python globals, files, network, randomness, or environment internals.
- The function must be pure and JIT-friendly: no Python branching on array values; use jnp.where / lax.cond.
- Reward should be shaped dense potential: make partial progress yield **positive** rewards (e.g., `potential - potential_prev`), and penalize regress/idle steps; small per-step penalty ok.
- Must gracefully handle episode termination: set to 0 after terminal or add a success bonus that is consistent with sparse=1.
- Build `reward_components = {"name": component_value, ...}` (string keys only) and return `(total_reward, reward_components)`.

YOU MUST WRITE VALID JITTABLE JAX CODE
"""
