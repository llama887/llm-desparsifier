"""Lightweight heuristic sanitizer for PuzzleScript.

The XLand sanitizer enforces ctx.get()-only access, which is too strict
for PuzzleScript where ctx is a plain dict and the LLM needs to access
nested dicts (object_positions, etc.). This sanitizer validates:
  - Exactly one function named heuristic_cost_to_go(ts, env_params, ctx)
  - No imports, exec, eval, open, __builtins__, compile
  - No direct non-finite returns such as return float("inf") or return math.nan
  - Only math is available in the namespace
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable

_BANNED_NAMES = frozenset({
    "exec", "eval", "compile", "open", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
    "__builtins__", "os", "sys", "subprocess", "shutil",
})
_NONFINITE_FLOAT_STRINGS = frozenset({
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
    "nan",
    "+nan",
    "-nan",
})


def _is_direct_nonfinite_expr(node: ast.AST) -> bool:
    """Return True when an expression is an obvious non-finite literal value.

    PuzzleScript heuristics may use ``float("inf")`` as an internal matching
    sentinel and then replace it with a finite value. Runtime search already
    rejects actual non-finite heuristic returns, so static validation only
    rejects direct return expressions that are unambiguously non-finite.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_direct_nonfinite_expr(node.operand)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float) and not math.isfinite(node.value)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "math"
        and node.attr in {"inf", "nan"}
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.strip().lower() in _NONFINITE_FLOAT_STRINGS
    )


def _strip_markdown_fences(code: str) -> str:
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


class _PuzzleScriptHeuristicValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found_function = False

    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError("imports are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError("imports are not allowed")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "heuristic_cost_to_go":
            if self.found_function:
                raise ValueError("duplicate heuristic_cost_to_go definition")
            self.found_function = True
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names != ["ts", "env_params", "ctx"]:
                raise ValueError(
                    "heuristic_cost_to_go must accept (ts, env_params, ctx)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_NAMES:
            raise ValueError(f"'{node.func.id}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BANNED_NAMES:
            raise ValueError(f"'{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and _is_direct_nonfinite_expr(node.value):
            raise ValueError("direct non-finite heuristic returns are not allowed")
        self.generic_visit(node)


def sanitize_and_compile_puzzlescript_heuristic(code: str) -> Callable[..., float]:
    """Validate and compile a PuzzleScript heuristic function.

    Less strict than the XLand sanitizer: allows normal dict access,
    list comprehensions, etc. Still bans imports and dangerous builtins.
    """
    cleaned = _strip_markdown_fences(code)
    tree = ast.parse(cleaned)
    validator = _PuzzleScriptHeuristicValidator()
    validator.visit(tree)
    if not validator.found_function:
        raise ValueError("expected a heuristic_cost_to_go definition")
    namespace: dict[str, Any] = {"math": math, "abs": abs, "min": min,
                                  "max": max, "sum": sum, "len": len,
                                  "range": range, "enumerate": enumerate,
                                  "zip": zip, "sorted": sorted,
                                  "float": float, "int": int,
                                  "tuple": tuple, "list": list,
                                  "dict": dict, "set": set, "bool": bool,
                                  "str": str, "isinstance": isinstance,
                                  "any": any, "all": all}
    exec(compile(tree, "<heuristic>", "exec"), namespace)  # noqa: S102
    func = namespace.get("heuristic_cost_to_go")
    if not callable(func):
        raise ValueError("compiled heuristic_cost_to_go is not callable")
    return func
