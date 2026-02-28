"""Diagnostics for object-key alignment between reward code and task context.

This module extracts object keys referenced in synthesized reward code and
compares them against object keys present in the environment description text.
It is used to detect a common failure mode where dense rewards query object
names that do not exist in the current task, which causes shaping terms to
collapse to zero while time penalties still fire.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

_TRACKED_CTX_MAP_KEYS = frozenset(
    {"object_positions", "visible_object_positions", "visible_object_positions_prev"}
)
_TASK_OBJECT_TOKEN_RE = re.compile(r'"([a-z]+(?:_[a-z]+)+)"')


@dataclass(frozen=True)
class RewardObjectKeyDiagnostics:
    """Structured object-key alignment diagnostics for one reward/environment pair.

    This payload captures which object keys the synthesized reward function
    references via nested `ctx` maps, which object-like keys are described in
    the environment text shown to the synthesis model, and which referenced keys
    are absent from that task description. It is needed so reflection feedback
    can explicitly explain why shaping components may remain near zero even when
    policy behavior appears goal-directed.
    """

    referenced_object_keys: tuple[str, ...]
    task_object_keys: tuple[str, ...]
    missing_from_task: tuple[str, ...]


def build_reward_object_key_diagnostics(
    reward_code: str,
    env_description: Any,
) -> RewardObjectKeyDiagnostics:
    """Compute deterministic object-key mismatch diagnostics for a reward.

    This function parses generated reward code using AST inspection, extracts
    all nested `.get("<key>", ...)` calls made on maps retrieved from
    `ctx.get("object_positions", {})`,
    `ctx.get("visible_object_positions", {})`, and
    `ctx.get("visible_object_positions_prev", {})`, then compares those keys to
    object-like quoted tokens in the environment description text. It is needed
    because LLM-generated rewards often look valid syntactically while silently
    targeting impossible objects, and it differs from sanitizer checks by
    validating semantic key alignment instead of code structure.

    Args:
        reward_code: Synthesized dense reward source, optionally wrapped in
            Markdown code fences.
        env_description: Environment text presented to reward synthesis.

    Returns:
        A stable, sorted diagnostics payload suitable for logging and reflection
        feedback.
    """
    referenced_keys = _extract_reward_object_lookup_keys(reward_code)
    task_keys = _extract_task_object_keys(env_description)
    missing = tuple(sorted(set(referenced_keys) - set(task_keys)))
    return RewardObjectKeyDiagnostics(
        referenced_object_keys=tuple(sorted(set(referenced_keys))),
        task_object_keys=tuple(sorted(set(task_keys))),
        missing_from_task=missing,
    )


def _extract_reward_object_lookup_keys(reward_code: str) -> tuple[str, ...]:
    """Extract nested object-map lookup keys from synthesized reward code.

    This parser tracks local variables assigned from relevant `ctx.get(...)`
    maps and then captures string-literal keys passed to `.get(...)` on those
    locals (or directly on chained `ctx.get(...).get(...)` expressions). It is
    needed to make mismatch detection robust to common coding styles in emitted
    rewards, and it differs from regex matching by relying on AST shape so
    comments and unrelated literals do not create false positives.
    """
    source = _strip_markdown_fences(reward_code)
    tree = ast.parse(source)

    alias_to_map_name: dict[str, str] = {}
    referenced: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            map_name = _extract_ctx_map_key_from_expr(node.value)
            if map_name is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        alias_to_map_name[target.id] = map_name
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                map_name = _resolve_map_name_from_get_owner(node.func.value, alias_to_map_name)
                if map_name is not None:
                    referenced.add(str(node.args[0].value))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return tuple(sorted(referenced))


def _resolve_map_name_from_get_owner(
    owner_expr: ast.AST,
    alias_to_map_name: dict[str, str],
) -> str | None:
    """Resolve whether a `.get(...)` owner expression is one of tracked ctx maps.

    The owner may be either a local alias (`object_positions`) or a chained
    expression like `ctx.get("object_positions", {})`. This helper normalizes
    both forms into one tracked map identifier.
    """
    if isinstance(owner_expr, ast.Name):
        return alias_to_map_name.get(owner_expr.id)
    return _extract_ctx_map_key_from_expr(owner_expr)


def _extract_ctx_map_key_from_expr(expr: ast.AST) -> str | None:
    """Return tracked map key if expression is `ctx.get("<map>", ...)`.

    This helper isolates recognition of top-level map retrieval expressions so
    both assignment and chained-call paths share identical map-selection logic.
    """
    if not isinstance(expr, ast.Call):
        return None
    if not isinstance(expr.func, ast.Attribute):
        return None
    if expr.func.attr != "get" or not isinstance(expr.func.value, ast.Name):
        return None
    if expr.func.value.id != "ctx" or not expr.args:
        return None
    first_arg = expr.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return None
    map_key = str(first_arg.value)
    if map_key in _TRACKED_CTX_MAP_KEYS:
        return map_key
    return None


def _extract_task_object_keys(env_description: Any) -> tuple[str, ...]:
    """Extract quoted snake_case tokens from environment description text.

    This uses the same style as existing task-context tests so diagnostics align
    with the prompt text contract used during reward synthesis.
    """
    if not isinstance(env_description, str):
        return ()
    tokens = {
        token for token in _TASK_OBJECT_TOKEN_RE.findall(env_description) if "_" in token
    }
    return tuple(sorted(tokens))


def _strip_markdown_fences(code: str) -> str:
    """Strip top-level Markdown fences so AST parsing can succeed reliably."""
    stripped = code.strip()
    if not stripped.startswith("```"):
        return code
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return code


__all__ = ["RewardObjectKeyDiagnostics", "build_reward_object_key_diagnostics"]
