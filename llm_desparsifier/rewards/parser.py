"""Prompt construction helpers for reward synthesis."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from typing import List, Optional

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
    "Actions (ids): 0=move_forward, 1=turn_right (clockwise), "
    "2=turn_left (counterclockwise), 3=pick_up, 4=put_down, 5=toggle "
    "(one object carried at a time)."
)

_GOAL_AGENT_HOLD_RE = re.compile(r"AgentHold\s*\(\s*([^)]+)\s*\)", re.IGNORECASE)
_GOAL_AGENT_NEAR_RE = re.compile(r"AgentNear\s*\(\s*([^)]+)\s*\)", re.IGNORECASE)
_GOAL_AGENT_NEAR_DIR_RE = re.compile(
    r"AgentNear(Up|Right|Down|Left)Goal\s*\(\s*([^)]+)\s*\)",
    re.IGNORECASE,
)
_GOAL_TILE_NEAR_RE = re.compile(
    r"TileNear\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
    re.IGNORECASE,
)
_GOAL_TILE_NEAR_DIR_RE = re.compile(
    r"TileNear(Up|Right|Down|Left)Goal\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
    re.IGNORECASE,
)

_OPPOSITE_DIRECTION = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}


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


def _format_object_name(name: str) -> str:
    cleaned = " ".join(name.split())
    return cleaned.lower()


def _subject_relation_direction(goal_direction: str) -> str:
    """Map XMiniGrid directional-goal token to subject-relative world direction.

    XMiniGrid directional goal identifiers encode direction from the perspective
    of the *second operand* (or target tile) rather than from the subject phrase
    used in this project's natural-language objective text. For example,
    ``TileNearLeftGoal(tile_a, tile_b)`` is satisfied when ``tile_a`` is one cell
    to the right of ``tile_b``. This helper is needed to prevent left/right and
    up/down inversions in generated goal descriptions, and it differs from a
    plain lowercase normalizer by applying the semantics-preserving opposite
    mapping required by XMiniGrid's goal-check implementation.

    Args:
        goal_direction: Direction token extracted from goal names such as
            ``Up``, ``Right``, ``Down``, or ``Left``.

    Returns:
        Lowercase world-relative direction describing where the subject should be
        relative to the referenced object.
    """
    normalized = goal_direction.lower().strip()
    return _OPPOSITE_DIRECTION.get(normalized, normalized)


def _relation_alignment_text(direction: str) -> str:
    """Return an explicit coordinate relation phrase for a direction word.

    This helper emits deterministic coordinate-language constraints that can be
    copied into reward shaping logic (for example, equal-row plus +/-1 column).
    It is needed because concise words like "left" can still be interpreted
    ambiguously by language models, and it differs from generic alignment labels
    by encoding exact row/column offset expectations.

    Args:
        direction: Lowercase relation direction of the subject relative to target.

    Returns:
        Human-readable clause describing exact grid-coordinate constraints.
    """
    if direction == "left":
        return "same row, and the first column is exactly one less than the second"
    if direction == "right":
        return "same row, and the first column is exactly one greater than the second"
    if direction == "up":
        return "same column, and the first row is exactly one less than the second"
    if direction == "down":
        return "same column, and the first row is exactly one greater than the second"
    return "adjacent in the expected direction"


def _goal_sentences(
    goal_line: Optional[str],
    *,
    agent_pos_example: str,
    obj_positions_example: str,
) -> List[str]:
    """Translate one ruleset goal line into two plain-language objective clauses.

    This helper converts the symbolic goal expression emitted by
    ``xminigrid.rendering.text_render.print_ruleset`` into a concise pair of
    sentences: a task instruction and a success criterion. It is needed because
    reward-synthesis prompts must expose goal semantics in natural language while
    still naming concrete ``ctx.get(...)`` lookup patterns for implementation.
    It differs from raw ruleset text by resolving parser-specific naming quirks
    (including directional inversion semantics in directional goal variants) and
    by producing model-friendly phrasing that can be copied into dense reward
    logic.

    Args:
        goal_line: Optional raw goal line such as
            ``TileNearLeftGoal(blue key, red star)``.
        agent_pos_example: Ready-to-paste code snippet for retrieving agent
            position from context.
        obj_positions_example: Ready-to-paste code snippet for retrieving object
            positions map from context.

    Returns:
        Two-element list containing task and success sentences.
    """
    if not goal_line:
        return [
            "Your task is to satisfy the level goal condition.",
            "Success is determined by the hidden ruleset goal.",
        ]

    goal_line = goal_line.strip()
    if not goal_line:
        return [
            "Your task is to satisfy the level goal condition.",
            "Success is determined by the hidden ruleset goal.",
        ]

    match = _GOAL_AGENT_HOLD_RE.search(goal_line)
    if match:
        obj = _format_object_name(match.group(1))
        return [
            f"Your task is to pick up and hold the {obj}.",
            (
                f"Success when the agent is carrying the {obj}; "
                f'use {obj_positions_example}.get("{obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                'to locate the object, and ctx.get("is_carrying", jnp.array(False)) '
                'plus ctx.get("carried_item", jnp.array([-1, -1], dtype=jnp.int32)) to check inventory.'
            ),
        ]

    match = _GOAL_AGENT_NEAR_DIR_RE.search(goal_line)
    if match:
        direction = _subject_relation_direction(match.group(1))
        obj = _format_object_name(match.group(2))
        relation = _relation_alignment_text(direction)
        return [
            f"Your task is to move the agent immediately {direction} of the {obj}.",
            (
                f"Success when the agent is exactly one cell {direction} of the {obj}; "
                f"that means {relation}; "
                f"use {agent_pos_example} for the agent and "
                f'{obj_positions_example}.get("{obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                "for the object."
            ),
        ]

    match = _GOAL_AGENT_NEAR_RE.search(goal_line)
    if match:
        obj = _format_object_name(match.group(1))
        return [
            f"Your task is to move next to the {obj}.",
            (
                f"Success when the agent is adjacent to the {obj}; "
                f"use {agent_pos_example} for the agent and "
                f'{obj_positions_example}.get("{obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                "for the object."
            ),
        ]

    match = _GOAL_TILE_NEAR_DIR_RE.search(goal_line)
    if match:
        direction = _subject_relation_direction(match.group(1))
        first_obj = _format_object_name(match.group(2))
        second_obj = _format_object_name(match.group(3))
        alignment = _relation_alignment_text(direction)
        return [
            f"Your task is to place the {first_obj} immediately {direction} of the {second_obj}.",
            (
                f"Success when the {first_obj} is exactly one cell {direction} of the {second_obj} "
                f"({alignment}); use "
                f'{obj_positions_example}.get("{first_obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                "and "
                f'{obj_positions_example}.get("{second_obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                "to locate the tiles."
            ),
        ]

    match = _GOAL_TILE_NEAR_RE.search(goal_line)
    if match:
        first_obj = _format_object_name(match.group(1))
        second_obj = _format_object_name(match.group(2))
        return [
            f"Your task is to bring the {first_obj} next to the {second_obj}.",
            (
                f"Success when the {first_obj} is adjacent to the {second_obj}; use "
                f'{obj_positions_example}.get("{first_obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                "and "
                f'{obj_positions_example}.get("{second_obj.replace(" ", "_")}", jnp.array([-1, -1], dtype=jnp.int32)) '
                "to locate the tiles."
            ),
        ]

    return [
        "Your task is to satisfy the level goal condition.",
        f"Success when this condition holds: {goal_line}",
    ]


def describe_ruleset(env, env_params) -> str:
    """Produce the environment text shown to the reward-synthesis LLM.

    This function builds a compact but concrete task description that combines
    layout metadata, action semantics, object initialization hints, and explicit
    code-level access patterns for both global context and egocentric
    observations. It is needed because reward synthesis quality depends heavily
    on the model understanding *how* to access task state in sanitized code, not
    just *what* the task goal is. It differs from raw ruleset dumps by
    translating goal predicates into plain language and by embedding stable
    `ctx.get(...)` snippets that match the sanitizer and wrapper contracts.
    """
    height = _safe_getattr(env_params, "height", "?")
    width = _safe_getattr(env_params, "width", "?")
    view = _safe_getattr(env_params, "view_size", "?")
    max_steps = _safe_getattr(env_params, "max_steps", "?")
    grid_type = _safe_getattr(env_params, "grid_type", "unknown")

    layout_hint = _LAYOUT_HINTS.get(
        str(grid_type), "a grid-world layout with interior walls and doors"
    )

    goal_line = None
    init_lines: List[str] = []
    if _print_ruleset is not None:
        try:
            ruleset = env_params.ruleset
            buf = io.StringIO()
            with redirect_stdout(buf):
                _print_ruleset(ruleset)
            summary = buf.getvalue().strip()
            if summary:
                goal_line, _, init_lines = _parse_ruleset_text(summary)
        except Exception:
            pass

    init_obj_list = ", ".join(init_lines[:10]) if init_lines else "unknown (randomized at reset)"
    if init_lines and len(init_lines) > 10:
        init_obj_list += f", ... (+{len(init_lines) - 10} more)"
    init_obj_keys = [obj.lower().replace(" ", "_") for obj in init_lines]
    example_obj_key = init_obj_keys[0] if init_obj_keys else "red_square"
    swap_obj_keys = init_obj_keys[1:] if len(init_obj_keys) > 1 else []

    ctx_prefix = "ctx"
    agent_pos_example = f'{ctx_prefix}.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))'
    obj_positions_example = f'{ctx_prefix}.get("object_positions", {{}})'
    goal_sentences = _goal_sentences(
        goal_line,
        agent_pos_example=agent_pos_example,
        obj_positions_example=obj_positions_example,
    )
    lines = [
        "You are in the XLand MiniGrid world, a grid-based puzzle level.",
        f"This level uses layout {grid_type}: {layout_hint}. The map is {height}x{width}, and you have up to {max_steps} steps.",
        f"The agent sees an egocentric {view}x{view} symbolic window (partially observable, not pixels).",
        (f"The agent position comes from {agent_pos_example}."),
        (
            "Observation interface (symbolic-first): use "
            f'{ctx_prefix}.get("visible_object_positions", {{}}) for currently visible objects and '
            f'{ctx_prefix}.get("visible_object_positions_prev", {{}}) for the previous step. '
            "Each lookup returns local [row, col] view coordinates, and [-1, -1] means not visible."
        ),
        (
            "Example visible-object lookup: "
            f'{ctx_prefix}.get("visible_object_positions", {{}}).get("{example_obj_key}", jnp.array([-1, -1], dtype=jnp.int32)).'
        ),
        (
            "Raw symbolic observation fallback: "
            f'obs = {ctx_prefix}.get("observation", ts_next.observation).astype(jnp.int32) and '
            f'obs_prev = {ctx_prefix}.get("observation_prev", ts_prev.observation).astype(jnp.int32). '
            "obs[..., 0] is tile id and obs[..., 1] is color id."
        ),
        (
            "Useful ids for observation-based checks: "
            "tiles FLOOR=1, WALL=2, BALL=3, SQUARE=4, PYRAMID=5, GOAL=6, KEY=7, HEX=11, STAR=12; "
            "colors RED=1, GREEN=2, BLUE=3, PURPLE=4, YELLOW=5, GREY=6, BLACK=7."
        ),
        (
            "Available actions are 0=move_forward, 1=turn_right (clockwise), "
            "2=turn_left (counterclockwise), 3=pick_up, 4=put_down, and 5=toggle; "
            "the agent can carry only one object at a time "
            '(check ctx.get("is_carrying", jnp.array(False)) and '
            'ctx.get("carried_item", jnp.array([-1, -1], dtype=jnp.int32))).'
        ),
        (
            f"Initial objects include: {init_obj_list}. "
            "Object locations come from "
            f'{obj_positions_example}.get("{example_obj_key}", jnp.array([-1, -1], dtype=jnp.int32)).'
        ),
    ]
    if swap_obj_keys:
        lines.append(
            "To get positions for the other initial objects, swap the key to one of: "
            + ", ".join(f'"{key}"' for key in swap_obj_keys)
            + "."
        )
    lines.extend(goal_sentences)
    lines.append("Use distances and spatial relations; avoid Python-side branching.")

    return " ".join(lines)


CONSTRAINTS_TEXT = """
You are designing a dense reward function for the Xland-Minigrid environment
You must output exactly ONE function:
  def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    # returns (total_reward: jnp.float32, reward_components: dict[str, jnp.float32])

Hard requirements the sanitizer enforces:
- The return statement must be `return total_reward, reward_components` where `reward_components` is a Python dict with string keys and scalar jnp arrays as values, **or** `return total_reward, { ... }` using a dict literal with string keys.
- Access every ctx field with `.get(key, fallback)`; using `ctx[...]` is invalid.
- Guard nested maps like `object_positions` with `.get` at each level.
- Do NOT add any import statements; the sanitizer only tolerates the tiny allowlist (`import jax`, `import jax.numpy as jnp`, `import jax.lax`) and everything else is rejected.

Minimal valid pattern:
```python
def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    agent_pos = ctx.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))
    step_num = ctx.get("step_num", jnp.array(0, dtype=jnp.int32))
    progress = jnp.where(agent_pos[0] >= 0, 0.01 * step_num.astype(jnp.float32), jnp.array(0.0))
    reward_components = {"progress": progress, "penalty": jnp.array(0.0)}
    total_reward = reward_components["progress"] + reward_components["penalty"]
    return total_reward, reward_components
```

Context:
- The function will be installed as `dense_fn` inside `DesparsifyRewardWrapper` (see snippet below) and invoked either with three args `(ts_prev, action, ts_next)` or the five-arg signature shown above. Always implement the five-arg form; the wrapper detects it via `inspect.signature`.
- `ctx` is produced (when configured) by a pure `ctx_fn(env_params, ts_prev, ts_next)` that runs right after `env.step`. It returns a dictionary mapping strings to JAX arrays (you may only access fields of ctx by using ctx.get(...) instead of ctx[...]). Each entry is derived from the `xminigrid.types.TimeStep` objects:
    - Base scalar/pose keys per timestep: `agent_pos`, `agent_direction`, `step_num`, `is_carrying`, `carried_item`, `yellow_square_pos`, `green_ball_pos`, plus `_prev` copies.
    - Observation keys per timestep: `observation`, `observation_tile_ids`, `observation_color_ids`, and `visible_object_positions`, plus `_prev` copies.
    - Nested maps:
        - `object_positions`: full-grid object coordinates keyed by `"{color}_{tile}"` snake_case (for example, `"yellow_square"`, `"green_ball"`).
        - `visible_object_positions`: egocentric-view object coordinates keyed by the same naming scheme. Coordinates are local `[row, col]` in the current view; `[-1, -1]` means not currently visible.
      Always guard parent and child lookups, for example:
      `obj_pos = ctx.get("object_positions", {})` then `obj_pos.get("yellow_square", jnp.array([-1, -1], dtype=jnp.int32))`.
      `visible_pos = ctx.get("visible_object_positions", {})` then `visible_pos.get("yellow_square", jnp.array([-1, -1], dtype=jnp.int32))`.
    - Raw observation channel contract: `obs = ctx.get("observation", ts_next.observation).astype(jnp.int32)` has shape `(view_size, view_size, 2)` where `obs[..., 0]` is tile id and `obs[..., 1]` is color id.
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
- When dealing with nested structures such as `object_positions`, guard both the parent map and each child lookup.
- Call **only** JAX primitives (`jnp.*`, `jax.lax.*`) or helper functions you define inside `dense_reward`. Method calls are restricted to `.astype(...)`. Do **not** invoke Python `math.*`, `numpy.*`, or arbitrary library functions; the sanitizer will reject them.
- If you define helper functions inside dense_reward, ensure they are pure, side-effect free, and only call jnp/jax operations.
- Do NOT access Python globals, files, network, randomness, or environment internals.
- The function must be pure and JIT-friendly: no Python branching on array values; use jnp.where / lax.cond.
- Reward should be shaped dense potential: make partial progress yield **positive** rewards (e.g., `potential - potential_prev`), and penalize regress/idle steps; small per-step penalty ok.
- Must gracefully handle episode termination: set to 0 after terminal or add a success bonus that is consistent with sparse=1.
- Build `reward_components = {"name": component_value, ...}` (string keys only) and return `(total_reward, reward_components)`, or return a dict literal directly as the second tuple element.

YOU MUST WRITE VALID JITTABLE JAX CODE
"""
