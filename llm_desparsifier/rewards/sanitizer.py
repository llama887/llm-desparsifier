"""AST sanitizer for LLM-generated reward code."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ["SanitizedRewardResult", "sanitize_and_compile", "sanitize_reward_code"]

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
_ALLOWED_IMPORTS = {
    ("jax", None),
    ("jax.numpy", "jnp"),
    ("jax.lax", None),
}
_ALLOWED_VALUE_ATTRS = {"dtype", "shape", "ndim", "size"}
_ALLOWED_ATTRS = {
    "ctx": {"get"},
    "ts_prev": {
        "reward",
        "discount",
        "observation",
        "step_type",
        "last",
        "shape",
        "dtype",
        "size",
    },
    "ts_next": {
        "reward",
        "discount",
        "observation",
        "step_type",
        "last",
        "shape",
        "dtype",
        "size",
    },
    "ts": {
        "reward",
        "discount",
        "observation",
        "step_type",
        "last",
        "shape",
        "dtype",
        "size",
    },
    "action": {"shape", "dtype", "size"},
    "env_params": {"height", "width", "view_size", "max_steps", "grid_type", "ruleset"},
}
_ALLOWED_CALL_ATTRS = {
    "ctx": {"get"},
    "ts_prev": {"last"},
    "ts_next": {"last"},
    "ts": {"last"},
}


def _strip_markdown_fences(code: str) -> str:
    stripped = code.strip()
    if not stripped.startswith("```"):
        return code
    lines = stripped.splitlines()
    if not lines:
        return code
    if not lines[0].strip().startswith("```"):
        return code
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return code


@dataclass(frozen=True)
class SanitizedRewardResult:
    """Canonical sanitized reward payload returned by AST validation.

    This record packages the compiled dense reward callable together with the
    exact sanitized Python source and the stable reward-component key set
    extracted during validation. It is needed because downstream callers now
    persist canonical reward artifacts and validation metadata in addition to
    executing the compiled function, and it differs from the legacy
    `sanitize_and_compile` API by exposing the post-sanitization source text
    rather than only the callable.
    """

    dense_reward: Callable[..., Any]
    sanitized_code: str
    component_keys: tuple[str, ...]


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
        if getattr(node, "attr", None) == "astype":
            return self.generic_visit(node)
        if getattr(node, "attr", None) == "get" and isinstance(
            node.value, (ast.Name, ast.Attribute)
        ):
            return self.generic_visit(node)
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
            if node.func.attr == "astype":
                return self.generic_visit(node)
            if node.func.attr == "get" and isinstance(
                node.func.value, (ast.Name, ast.Attribute)
            ):
                return self.generic_visit(node)
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

    def visit_Subscript(self, node):  # type: ignore[override]
        # Allow subscripts on intermediate dicts that authors obtain via ctx.get,
        # but block direct `ctx[...]` accesses so prompts enforcing `.get` remain effective.
        if isinstance(node.value, ast.Name) and node.value.id == "ctx":
            raise ValueError("use ctx.get(...) instead of ctx[...]")
        return self.generic_visit(node)


def _validate_reward_structure(func_node: ast.FunctionDef) -> list[str]:
    class _RewardVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.component_keys: list[str] = []
            self._func_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
            # Only validate returns/assignments in the top-level dense_reward body.
            self._func_depth += 1
            if self._func_depth == 1:
                self.generic_visit(node)
            self._func_depth -= 1

        @staticmethod
        def _extract_keys(node: ast.Dict) -> list[str]:
            if not node.keys:
                raise ValueError(
                    "reward_components dict must contain at least one entry"
                )
            keys: list[str] = []
            for key in node.keys:
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise ValueError("reward_components keys must be string literals")
                keys.append(str(key.value))
            return keys

        def _register_keys(self, keys: list[str]) -> None:
            if not self.component_keys:
                self.component_keys = keys
                return
            if set(keys) != set(self.component_keys):
                raise ValueError("reward_components keys must remain constant")

        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "reward_components":
                    if not isinstance(node.value, ast.Dict):
                        raise ValueError(
                            "reward_components must be defined as a dict literal"
                        )
                    keys = self._extract_keys(node.value)
                    self._register_keys(keys)
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:  # type: ignore[override]
            if self._func_depth != 1:
                return
            if node.value is None:
                raise ValueError(
                    "dense_reward must return (total_reward, reward_components)"
                )
            if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2:
                raise ValueError(
                    "dense_reward must return (total_reward, reward_components)"
                )
            components_node = node.value.elts[1]
            if isinstance(components_node, ast.Name):
                if components_node.id != "reward_components":
                    raise ValueError(
                        "second element of return must be reward_components"
                    )
            elif isinstance(components_node, ast.Dict):
                keys = self._extract_keys(components_node)
                self._register_keys(keys)
            else:
                raise ValueError("second element of return must be reward_components")
            self.generic_visit(node)

    visitor = _RewardVisitor()
    visitor.visit(func_node)
    if not visitor.component_keys:
        raise ValueError("reward_components dict literal is required before returning")
    return visitor.component_keys


def sanitize_reward_code(code: str) -> SanitizedRewardResult:
    """Validate reward code and return its canonical sanitized representation.

    This helper strips top-level Markdown fences, validates the AST against the
    repo's restricted reward contract, compiles the resulting function, and
    returns all canonicalized outputs needed by artifact writers. It is needed
    because the pipeline now distinguishes between raw LM text and the exact
    executable source that passed sanitizer checks, and it differs from
    `sanitize_and_compile` by preserving the sanitized source and component-key
    metadata for downstream persistence and validation.
    """
    sanitized_code = _strip_markdown_fences(code).strip()
    tree = ast.parse(sanitized_code)
    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fdefs) != 1 or fdefs[0].name != "dense_reward":
        raise ValueError(
            "output must define dense_reward(...) as the first top-level def"
        )
    component_keys = _validate_reward_structure(fdefs[0])
    _Sanitizer().visit(tree)

    bytecode = compile(tree, filename="<dense_reward>", mode="exec")
    safe_globals = {"jnp": jnp, "jax": jax}
    safe_locals: dict[str, Any] = {}
    exec(bytecode, safe_globals, safe_locals)
    dense_reward = safe_locals.get("dense_reward") or safe_globals.get("dense_reward")
    if not callable(dense_reward):
        raise ValueError("dense_reward not found after exec")
    if component_keys:
        setattr(dense_reward, "__reward_component_keys__", tuple(component_keys))
    return SanitizedRewardResult(
        dense_reward=dense_reward,
        sanitized_code=sanitized_code,
        component_keys=tuple(component_keys),
    )


def sanitize_and_compile(code: str):
    """Validate the generated code and return the compiled dense reward function.

    This compatibility wrapper keeps the historical sanitizer API available for
    callers and tests that only need an executable function. It is needed so
    the new canonical artifact workflow can be adopted incrementally, and it
    differs from `sanitize_reward_code` by discarding the sanitized source and
    component metadata after validation.
    """
    return sanitize_reward_code(code).dense_reward
