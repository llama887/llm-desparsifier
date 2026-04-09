"""Prompting helpers for heuristic synthesis.

This module owns the stable prompt contract for the heuristic-only GEPA path.
It is needed because prompt rewriting, example construction, and artifact
generation must agree on the exact heuristic interface and task description, and
it differs from the legacy reward parser by describing full-state search
semantics instead of dense-reward shaping guidance.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

try:
    from xminigrid.rendering.text_render import print_ruleset as _print_ruleset
except Exception:  # pragma: no cover - optional dependency guard
    _print_ruleset = None

_CTX_ACCESS_RULES_TEXT = """
Ctx access rules required for valid code:
- Access top-level context values only with `ctx.get("key")` or `ctx.get("key", default)`.
- Do not use `ctx["key"]`.
- Do not call methods on values loaded from `ctx` such as `agent_state.get(...)`,
  `object_positions.get(...)`, or `ctx.keys()`.
- After reading a mapping from `ctx`, inspect it with `is None`, `'name' in mapping`,
  and `mapping['name']`.
""".strip()

BASE_HEURISTIC_PROMPT = """
You are writing Python code for an admissible A* heuristic.

Write exactly one function named `heuristic_cost_to_go(ts, env_params, ctx)`.
The function must return a finite non-negative float lower bound on the
remaining path cost to solve the current XLand-MiniGrid task.

Requirements:
- Prefer a weaker but safe lower bound over a strong but risky estimate.
- Use only the provided `ctx` mapping plus basic Python control flow and math.
- Do not import modules, mutate state, read global variables, or depend on
  rollout history.
- Return `0.0` on solved states.
- Keep the heuristic simple and interpretable.

{ctx_access_rules}

Common safe patterns:
- Manhattan distance to a mandatory object or goal tile.
- Lower bounds on required pickup/dropoff steps.
- `max(...)` over independent lower bounds when summation would risk
  overestimation.

Output only Python code. Do not use markdown fences.
""".strip()

BASE_HEURISTIC_PROMPT = BASE_HEURISTIC_PROMPT.format(
    ctx_access_rules=_CTX_ACCESS_RULES_TEXT
)

HEURISTIC_CONTRACT_TEXT = """
Heuristic contract:
- Signature: `def heuristic_cost_to_go(ts, env_params, ctx) -> float`
- Return a finite scalar float.
- Return a non-negative lower bound on remaining sparse path cost.
- Return `0.0` on solved states.
- Treat `ctx` as the primary interface.
- `ctx` contains:
  - `env_id: str`
  - `benchmark_id: str`
  - `ruleset_text: str`
  - `grid_shape: tuple[int, int]`
  - `action_names: tuple[str, ...]`
  - `step_cost: int`
  - `goal_description: str`
  - `agent_state: dict` with `position` and legacy alias `pos`, plus `direction` and `carrying`
  - `object_positions: dict[str, tuple[int, int]]`
  - `object_metadata: dict[str, dict]`
  - `static_walls: tuple[tuple[int, int], ...]`
  - `task_features: dict`

{ctx_access_rules}
""".strip()

HEURISTIC_CONTRACT_TEXT = HEURISTIC_CONTRACT_TEXT.format(
    ctx_access_rules=_CTX_ACCESS_RULES_TEXT
)

_GOAL_LINE_RE = re.compile(r"^GOAL\s*:?\s*$", re.IGNORECASE)


def _safe_getattr(obj: object, name: str, default: str) -> str:
    """Return a stringified attribute value with a robust fallback.

    This helper normalizes optional environment parameters into printable text.
    It is needed because prompt construction should remain resilient across
    XLand wrappers and partial mocks used in tests, and it differs from raw
    `getattr` by always returning a string rather than propagating arbitrary
    attribute types or exceptions.
    """

    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    if value is None:
        return default
    return str(value)


def _render_ruleset_text(env_params: object) -> str:
    """Render one XLand ruleset into deterministic plain text.

    This helper captures `xminigrid`'s human-readable ruleset printer output so
    prompt examples and replay artifacts can share the same task wording. It is
    needed because benchmarks expose symbolic rulesets rather than stable text,
    and it differs from the legacy reward prompt builder by returning the raw
    ruleset text without observation-centric guidance.
    """

    if _print_ruleset is None:
        return "Ruleset text unavailable."
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_ruleset(getattr(env_params, "ruleset"))
        rendered = buf.getvalue().strip()
    except Exception:
        rendered = ""
    return rendered or "Ruleset text unavailable."


def extract_goal_description_from_ruleset_text(ruleset_text: str) -> str:
    """Extract the primary benchmark objective from rendered ruleset text.

    This helper is the canonical parser for one-line task objectives shared by
    prompt construction, runtime task serialization, and replay overlays. It is
    needed in this repository because the heuristic-only pipeline writes the
    same ruleset text into prompt examples, `task_instance.json`, and replay
    metadata, and those surfaces must agree on the exact goal string. It
    differs from ad hoc line scanning in nearby helpers by specifically
    recognizing the `GOAL:` section emitted by XLand's text renderer instead of
    falling back to unrelated header lines such as `Grid shape: ...`.
    """

    lines = [line.strip() for line in ruleset_text.splitlines() if line.strip()]
    goal_section = False
    for line in lines:
        if _GOAL_LINE_RE.match(line):
            goal_section = True
            continue
        if goal_section:
            return line
    return lines[0] if lines else "Satisfy the benchmark goal."


def describe_ruleset_for_heuristic(env: object, env_params: object) -> str:
    """Build the full task description shown to the heuristic synthesis LLM.

    This function assembles a compact but search-oriented environment summary
    from one concrete XLand ruleset. It is needed because heuristic synthesis
    should reason from the full symbolic task description rather than agent
    observations, and it differs from the old reward prompt text by emphasizing
    admissible lower bounds, state semantics, and the `ctx` contract.
    """

    del env
    height = _safe_getattr(env_params, "height", "?")
    width = _safe_getattr(env_params, "width", "?")
    max_steps = _safe_getattr(env_params, "max_steps", "?")
    grid_type = _safe_getattr(env_params, "grid_type", "unknown")
    ruleset_text = _render_ruleset_text(env_params)
    goal_description = extract_goal_description_from_ruleset_text(ruleset_text)
    return "\n".join(
        [
            f"Grid shape: {height} x {width}",
            f"Grid type: {grid_type}",
            f"Max steps: {max_steps}",
            "Action names: move_forward, turn_right, turn_left, pick_up, put_down, toggle",
            f"Goal description: {goal_description}",
            "Full ruleset:",
            ruleset_text,
            "",
            HEURISTIC_CONTRACT_TEXT,
        ]
    ).strip()


__all__ = [
    "BASE_HEURISTIC_PROMPT",
    "HEURISTIC_CONTRACT_TEXT",
    "describe_ruleset_for_heuristic",
    "extract_goal_description_from_ruleset_text",
]
