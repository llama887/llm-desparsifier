"""AST sanitizer for LLM-generated reward code."""

from __future__ import annotations

import ast
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["sanitize_and_compile"]

_ALLOWED_FUNCS = {
    "jnp": {
        "where",
        "clip",
        "maximum",
        "minimum",
        "abs",
        "sqrt",
        "exp",
        "log",
        "tanh",
        "sign",
        "linalg",
    },
    "jax": {"lax"},
}
_ALLOWED_TOP = {"jnp", "jax"}
_ALLOWED_IMPORTS = {("jax.numpy", "jnp")}
_ALLOWED_VALUE_ATTRS = {"dtype", "shape", "ndim", "size"}
_ALLOWED_ATTRS = {
    "ctx": {"get"},
    "ts_prev": {"reward", "discount", "observation", "step_type", "last", "shape", "dtype", "size"},
    "ts_next": {"reward", "discount", "observation", "step_type", "last", "shape", "dtype", "size"},
    "ts": {"reward", "discount", "observation", "step_type", "last", "shape", "dtype", "size"},
    "action": {"shape", "dtype", "size"},
    "env_params": {"height", "width", "view_size", "max_steps", "grid_type", "ruleset"},
}
_ALLOWED_CALL_ATTRS = {
    "ctx": {"get"},
    "ts_prev": {"last"},
    "ts_next": {"last"},
    "ts": {"last"},
}


def _decompose_attribute(node: ast.AST) -> tuple[ast.AST, list[str]]:
    chain: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    return cur, chain


class _Sanitizer(ast.NodeVisitor):
    def __init__(self) -> None:
        self._scope_stack: list[set[str]] = [set()]

    def _is_name_allowed(self, name: str) -> bool:
        return any(name in scope for scope in self._scope_stack)

    def visit_Import(self, node: ast.Import) -> Any:  # type: ignore[override]
        for alias in node.names:
            if (alias.name, alias.asname) in _ALLOWED_IMPORTS:
                continue
            raise ValueError("imports not allowed")

    def visit_ImportFrom(self, node):  # noqa: D401
        raise ValueError("imports not allowed")

    def visit_Global(self, node):  # noqa: D401
        raise ValueError("global not allowed")

    def visit_Nonlocal(self, node):  # noqa: D401
        raise ValueError("nonlocal not allowed")

    def visit_With(self, node):  # noqa: D401
        raise ValueError("with not allowed")

    def visit_Lambda(self, node):  # noqa: D401
        raise ValueError("lambda not allowed")

    def visit_ClassDef(self, node):  # noqa: D401
        raise ValueError("class not allowed")

    def visit_AugAssign(self, node):  # noqa: D401
        raise ValueError("augassign not allowed")

    def visit_Delete(self, node):  # noqa: D401
        raise ValueError("delete not allowed")

    def visit_Try(self, node):  # noqa: D401
        raise ValueError("try/except not allowed")

    def visit_Exec(self, node):  # noqa: D401
        raise ValueError("exec not allowed")

    def visit_FunctionDef(self, node):  # type: ignore[override]
        if self._scope_stack:
            self._scope_stack[-1].add(node.name)
        self._scope_stack.append(set())
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Attribute(self, node):  # type: ignore[override]
        root, chain = _decompose_attribute(node)
        if isinstance(root, ast.Name) and root.id in _ALLOWED_TOP:
            return self.generic_visit(node)
        if isinstance(root, ast.Name):
            allowed = _ALLOWED_ATTRS.get(root.id)
            if allowed is not None and set(chain).issubset(allowed):
                return self.generic_visit(node)
            if set(chain).issubset(_ALLOWED_VALUE_ATTRS):
                return self.generic_visit(node)
        raise ValueError("attribute access not allowed except jnp/jax.*")

    def visit_Call(self, node):  # type: ignore[override]
        if isinstance(node.func, ast.Attribute):
            root, chain = _decompose_attribute(node.func)
            if isinstance(root, ast.Name) and root.id in _ALLOWED_TOP:
                return self.generic_visit(node)
            if isinstance(root, ast.Name):
                allowed_attrs = _ALLOWED_ATTRS.get(root.id)
                allowed_calls = _ALLOWED_CALL_ATTRS.get(root.id, set())
                if allowed_attrs is not None and set(chain).issubset(allowed_attrs):
                    if chain and chain[0] in allowed_calls:
                        return self.generic_visit(node)
        elif isinstance(node.func, ast.Name):
            if node.func.id in {"float", "int"}:
                return self.generic_visit(node)
            if self._is_name_allowed(node.func.id):
                return self.generic_visit(node)
        raise ValueError("function calls must be to jnp/jax.* or basic casts")


def sanitize_and_compile(code: str):
    """Validate the generated code and return the compiled dense reward function."""
    tree = ast.parse(code)
    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fdefs) != 1 or fdefs[0].name != "dense_reward":
        raise ValueError("output must define dense_reward(...) as the first top-level def")
    _Sanitizer().visit(tree)

    bytecode = compile(tree, filename="<dense_reward>", mode="exec")
    safe_globals = {"jnp": jnp, "jax": jax}
    safe_locals: dict[str, Any] = {}
    exec(bytecode, safe_globals, safe_locals)
    dense_reward = safe_locals.get("dense_reward") or safe_globals.get("dense_reward")
    if not callable(dense_reward):
        raise ValueError("dense_reward not found after exec")
    return dense_reward
