"""AST sanitizer for LLM-generated heuristic code.

This module validates the narrow heuristic function contract used during search
evaluation. It is needed because synthesized code executes inside the evaluator
loop, and it differs from the legacy reward sanitizer by targeting a single
scalar-returning heuristic function instead of dense reward component dicts.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable

__all__ = ["sanitize_and_compile_heuristic"]

_ALLOWED_CALLS = {
    "abs",
    "all",
    "any",
    "float",
    "int",
    "len",
    "max",
    "min",
    "round",
    "sum",
}
_ALLOWED_TOP_LEVEL_ASSIGNS = {"SOLVED_REWARD"}
_ALLOWED_CTX_METHODS = {"get"}
_BANNED_NAMES = {"eval", "exec", "open", "__import__", "compile", "globals", "locals"}


def _strip_markdown_fences(code: str) -> str:
    """Remove optional Markdown fences around generated code.

    This helper keeps synthesis robust when the model ignores prompt
    instructions and wraps code in triple backticks. It is needed because the
    sanitizer should validate the underlying code rather than the presentation
    wrapper, and it differs from a blind `replace` by trimming only the outer
    fence pair.
    """

    stripped = code.strip()
    if not stripped.startswith("```"):
        return code
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return code


class _HeuristicSanitizer(ast.NodeVisitor):
    """Validate that synthesized heuristic code stays within the contract.

    This visitor rejects imports, mutation-oriented constructs, and calls to
    unsafe builtins while allowing the simple control flow needed for handcrafted
    lower-bound heuristics. It is needed because runtime safety is more
    important than expressiveness for generated evaluation code, and it differs
    from the reward sanitizer by focusing on one scalar function and a
    Python-native `ctx` mapping.
    """

    def __init__(self) -> None:
        self.found_function = False
        self.function_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> Any:  # type: ignore[override]
        raise ValueError("imports are not allowed in synthesized heuristics")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:  # type: ignore[override]
        raise ValueError("imports are not allowed in synthesized heuristics")

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # type: ignore[override]
        raise ValueError("classes are not allowed in synthesized heuristics")

    def visit_With(self, node: ast.With) -> Any:  # type: ignore[override]
        raise ValueError("with statements are not allowed in synthesized heuristics")

    def visit_Try(self, node: ast.Try) -> Any:  # type: ignore[override]
        raise ValueError("try/except is not allowed in synthesized heuristics")

    def visit_Delete(self, node: ast.Delete) -> Any:  # type: ignore[override]
        raise ValueError("deletes are not allowed in synthesized heuristics")

    def visit_Global(self, node: ast.Global) -> Any:  # type: ignore[override]
        raise ValueError("global declarations are not allowed in synthesized heuristics")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:  # type: ignore[override]
        raise ValueError("nonlocal declarations are not allowed in synthesized heuristics")

    def visit_Lambda(self, node: ast.Lambda) -> Any:  # type: ignore[override]
        raise ValueError("lambda expressions are not allowed in synthesized heuristics")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # type: ignore[override]
        self.function_names.add(node.name)
        if node.name != "heuristic_cost_to_go":
            raise ValueError("only heuristic_cost_to_go may be defined")
        if self.found_function:
            raise ValueError("exactly one heuristic_cost_to_go function is required")
        self.found_function = True
        arg_names = [arg.arg for arg in node.args.args]
        if arg_names != ["ts", "env_params", "ctx"]:
            raise ValueError(
                "heuristic_cost_to_go must accept exactly (ts, env_params, ctx)"
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:  # type: ignore[override]
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in _ALLOWED_TOP_LEVEL_ASSIGNS:
                continue
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:  # type: ignore[override]
        if isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_NAMES:
                raise ValueError(f"{node.func.id} is not allowed in synthesized heuristics")
            if node.func.id not in _ALLOWED_CALLS and node.func.id not in self.function_names:
                raise ValueError(f"call to unsupported function '{node.func.id}'")
        elif isinstance(node.func, ast.Attribute):
            if not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ctx"
                and node.func.attr in _ALLOWED_CTX_METHODS
            ):
                raise ValueError("only ctx.get(...) attribute calls are allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # type: ignore[override]
        if isinstance(node.value, ast.Name) and node.value.id == "ctx":
            if node.attr not in _ALLOWED_CTX_METHODS:
                raise ValueError("ctx may only be accessed through ctx.get(...)")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:  # type: ignore[override]
        if isinstance(node.value, ast.Name) and node.value.id == "ctx":
            raise ValueError("use ctx.get(...) instead of ctx[...]")
        self.generic_visit(node)


def sanitize_and_compile_heuristic(code: str) -> Callable[..., float]:
    """Validate and compile synthesized heuristic code.

    This function strips presentation wrappers, enforces the AST contract, and
    returns the compiled `heuristic_cost_to_go` callable. It is needed because
    the evaluator executes generated code inside search loops, and it differs
    from raw `exec` by refusing unsupported syntax before the code is loaded.
    """

    cleaned = _strip_markdown_fences(code)
    tree = ast.parse(cleaned)
    sanitizer = _HeuristicSanitizer()
    sanitizer.visit(tree)
    if not sanitizer.found_function:
        raise ValueError("expected a heuristic_cost_to_go definition")
    namespace: dict[str, Any] = {"math": math}
    exec(compile(tree, "<heuristic>", "exec"), namespace)  # noqa: S102
    func = namespace.get("heuristic_cost_to_go")
    if not callable(func):
        raise ValueError("compiled heuristic_cost_to_go is not callable")
    return func
