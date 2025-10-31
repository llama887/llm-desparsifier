from __future__ import annotations
import ast, types
import io
import os
from contextlib import redirect_stdout
import jax
import jax.numpy as jnp
import dspy
from dotenv import load_dotenv

try:
    from xminigrid.rendering.text_render import print_ruleset as _print_ruleset
except Exception:  # pragma: no cover - optional dependency guard
    _print_ruleset = None

load_dotenv()  # pulls PORTKEY_API_KEY from .env if present

def configure_dspy_with_portkey(
    api_key: str | None = None,
    base_url: str = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
    model_alias: str = "@o3-mini-5791cb/o3-mini",
    temperature: float = 1.0,
    max_completion_tokens: int = 16000,  # NOTE: o3-mini uses this name
):
    """
    Point DSPy (via LiteLLM) at your Portkey gateway and the o3-mini route.
    """
    api_key = api_key or os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PORTKEY_API_KEY")

    # DSPy’s LM forwards kwargs to LiteLLM → OpenAI-compatible gateway (Portkey).
    print("Temperature | Max tokens", temperature, max_completion_tokens)
    lm = dspy.LM(
        model=f"openai/{model_alias}",   # provider/model style; RHS is the OpenAI 'model' field
        api_base=base_url,               # Portkey base URL
        api_key=api_key,
        temperature=temperature,
        # Important for o3-mini:
        max_tokens=max_completion_tokens,
        # If your Portkey route needs any extra headers, pass them here:
        # additional_headers={"x-portkey-...": "..."}
    )

    dspy.configure(lm=lm)
    return lm


lm = configure_dspy_with_portkey()

class RewardSynthesis(dspy.Signature):
    """Create a single, JAX-friendly dense reward from the environment description."""
    env_description: str = dspy.InputField()
    constraints: str = dspy.InputField()
    reward_code: str = dspy.OutputField(desc="Only one Python function named dense_reward(...)")

class RewardSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(RewardSynthesis)

    def forward(self, env_description: str, constraints: str) -> str:
        out = self.gen(env_description=env_description, constraints=constraints)
        return out.reward_code

_ALLOWED_FUNCS = {
    "jnp": {"where", "clip", "maximum", "minimum", "abs", "sqrt", "exp", "log", "tanh",
            "sign", "linalg"},
    "jax": {"lax"},
}
_ALLOWED_TOP = {"jnp", "jax"}
_ALLOWED_IMPORTS = {("jax.numpy", "jnp")}
_ALLOWED_VALUE_ATTRS = {"dtype", "shape", "ndim", "size"}

# Attribute/call whitelist for non-jnp/jax objects exposed to the reward.
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
    chain = []
    cur = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    return cur, chain

class _Sanitizer(ast.NodeVisitor):
    def __init__(self):
        self._scope_stack: list[set[str]] = [set()]

    def _is_name_allowed(self, name: str) -> bool:
        return any(name in scope for scope in self._scope_stack)

    def visit_Import(self, node):
        for alias in node.names:
            if (alias.name, alias.asname) in _ALLOWED_IMPORTS:
                continue
            raise ValueError("imports not allowed")

    def visit_ImportFrom(self, node): raise ValueError("imports not allowed")
    def visit_Global(self, node): raise ValueError("global not allowed")
    def visit_Nonlocal(self, node): raise ValueError("nonlocal not allowed")
    def visit_With(self, node): raise ValueError("with not allowed")
    def visit_Lambda(self, node): raise ValueError("lambda not allowed")
    def visit_ClassDef(self, node): raise ValueError("class not allowed")
    def visit_AugAssign(self, node): raise ValueError("augassign not allowed")
    def visit_Delete(self, node): raise ValueError("delete not allowed")
    def visit_Try(self, node): raise ValueError("try/except not allowed")
    def visit_Exec(self, node): raise ValueError("exec not allowed")  # py2 relic

    def visit_FunctionDef(self, node):
        # Record function name in current scope so inner code can call it.
        if self._scope_stack:
            self._scope_stack[-1].add(node.name)
        # Push a new scope for the function body.
        self._scope_stack.append(set())
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Attribute(self, node):
        # allow chained attributes only for whitelisted roots
        root, chain = _decompose_attribute(node)
        if isinstance(root, ast.Name) and root.id in _ALLOWED_TOP:
            # e.g. jnp.linalg.norm or jax.lax.cond
            return self.generic_visit(node)
        if isinstance(root, ast.Name):
            allowed = _ALLOWED_ATTRS.get(root.id)
            if allowed is not None and set(chain).issubset(allowed):
                return self.generic_visit(node)
            if set(chain).issubset(_ALLOWED_VALUE_ATTRS):
                return self.generic_visit(node)
        raise ValueError("attribute access not allowed except jnp/jax.*")

    def visit_Call(self, node):
        # allow calls only if function is Name/Attribute on allowed roots
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
            if node.func.id in ("float", "int"):  # harmless casts
                return self.generic_visit(node)
            if self._is_name_allowed(node.func.id):
                return self.generic_visit(node)
        raise ValueError("function calls must be to jnp/jax.* or basic casts")

def sanitize_and_compile(code: str):
    tree = ast.parse(code)
    # must contain exactly one function named dense_reward
    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fdefs) != 1 or fdefs[0].name != "dense_reward":
        raise ValueError("output must define dense_reward(...) as the first top-level def")
    _Sanitizer().visit(tree)

    bytecode = compile(tree, filename="<dense_reward>", mode="exec")
    safe_globals = {"jnp": jnp, "jax": jax}
    safe_locals = {}
    exec(bytecode, safe_globals, safe_locals)
    dense_reward = safe_locals.get("dense_reward") or safe_globals.get("dense_reward")
    if not callable(dense_reward):
        raise ValueError("dense_reward not found after exec")
    return dense_reward

# ---------- Env description builder (text) [REPLACEMENT] ----------
import re
from typing import Optional, Tuple, List

# Map common grid_type codes to human-readable room-layout descriptions.
# R1/R2/R4/R6/R9 refer to number of rooms in XLand-MiniGrid presets.
_LAYOUT_HINTS = {
    "R1": "a single rectangular room (no interior walls)",
    "R2": "two rooms separated by an interior wall with one doorway",
    "R4": "four rooms separated by interior walls (the classic Four Rooms layout)",
    "R6": "six rooms separated by interior walls and doors",
    "R9": "nine rooms in a 3×3 arrangement with interior doors",
}

# Standard MiniGrid-like discrete action set (kept short; adjust if your env differs).
_ACTIONS_LINE = (
    "Actions: move_forward, turn_left, turn_right, pick_up, put_down, toggle (one object carried at a time)."
)

_GOAL_TILE_NEAR_RIGHT_RE = re.compile(
    r"TileNearRightGoal\s*\(\s*([^) ,]+(?:\s+[^\),]+)?)\s*,\s*([^) ,]+(?:\s+[^\),]+)?)\s*\)",
    re.IGNORECASE,
)

def _safe_getattr(obj, name: str, default: str) -> str:
    try:
        v = getattr(obj, name, default)
        return str(v if v is not None else default)
    except Exception:
        return default

def _parse_ruleset_text(text: str) -> tuple[Optional[str], List[str], List[str]]:
    """
    Returns (goal_line, rule_lines, init_tile_lines).
    Accepts the textual block emitted by _print_ruleset, which often looks like:

        GOAL:
        TileNearRightGoal(yellow square, green ball)

        RULES:
        <ruleA>
        <ruleB>

        INIT TILES:
        yellow square
        green ball
        ...

    Robust to missing sections.
    """
    goal_line = None
    rule_lines: List[str] = []
    init_lines: List[str] = []

    # Normalize newlines and strip extra spaces
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

        if section == "GOAL":
            # first non-empty line after GOAL
            if goal_line is None:
                goal_line = ln
        elif section == "RULES":
            rule_lines.append(ln)
        elif section == "INIT":
            init_lines.append(ln)

    return goal_line, rule_lines, init_lines

def _explain_goal(goal_line: Optional[str]) -> Optional[str]:
    """
    Produce a plain-English explanation of the goal if recognized.
    Currently recognizes TileNearRightGoal(objectA, objectB).
    """
    if not goal_line:
        return None
    m = _GOAL_TILE_NEAR_RIGHT_RE.search(goal_line)
    if m:
        left_obj, right_obj = m.group(2).strip(), m.group(1).strip()
        # In TileNearRightGoal(A,B), A must be to the right of B → success when B at (x,y) and A at (x,y+1).
        return (
            f"SUCCESS when **{right_obj}** is immediately to the **left** of **{left_obj}** "
            f"(i.e., {left_obj} is exactly one cell to the right of {right_obj}, same row, adjacent columns)."
        )
    # Fallback: just echo the line if we don't recognize a pattern
    return f"SUCCESS when condition holds: {goal_line}"

def describe_ruleset(env, env_params) -> str:
    """
    Return a human-readable, LLM-friendly summary of the environment and current task,
    enriched with layout semantics, action hints, and a plain-English goal explanation.
    Falls back gracefully if xminigrid debug utilities are unavailable.
    """
    height = _safe_getattr(env_params, "height", "?")
    width = _safe_getattr(env_params, "width", "?")
    view = _safe_getattr(env_params, "view_size", "?")
    max_steps = _safe_getattr(env_params, "max_steps", "?")
    grid_type = _safe_getattr(env_params, "grid_type", "unknown")

    # Layout blurb
    layout_hint = _LAYOUT_HINTS.get(str(grid_type), "a grid-world layout with interior walls and doors")

    # Try to obtain the ruleset printout (if helper is available), then parse it.
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

    # Explain the goal (if we found one).
    goal_expl = _explain_goal(goal_line)

    # Compose object list (if provided); we’ll show at most 10 to avoid walls of text.
    init_obj_list = ", ".join(init_lines[:10]) if init_lines else "unknown (randomized at reset)"
    if init_lines and len(init_lines) > 10:
        init_obj_list += f", ... (+{len(init_lines) - 10} more)"

    # Rules summary (short; they’re usually production rules that fire on events).
    rules_summary = "\n".join(f"- {r}" for r in rule_lines[:8]) if rule_lines else "No explicit transformation rules provided."
    if rule_lines and len(rule_lines) > 8:
        rules_summary += f"\n- ... (+{len(rule_lines) - 8} more)"

    # Final narrative (kept concise but specific for program synthesis / shaping).
    lines = []
    lines.append(f"grid_type={grid_type} → {layout_hint}")
    lines.append(f"size={height}x{width}, view={view} (agent-centered egocentric  {view}×{view}  symbolic grid), max_steps={max_steps}.")
    lines.append(_ACTIONS_LINE)
    lines.append("")  # spacer

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


# ---------- Constraints prompt (what the LLM can assume/use) ----------
CONSTRAINTS_TEXT = """
You are designing a dense reward function for the Xland-Minigrid environment
You must output exactly ONE function:
  def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    # returns a scalar jnp.float32

Context:
- The function will be installed as `dense_fn` inside `DesparsifyRewardWrapper` (see snippet below) and invoked either with three args `(ts_prev, action, ts_next)` or the five-arg signature shown above. Always implement the five-arg form; the wrapper detects it via `inspect.signature`.
- `ctx` is produced (when configured) by a pure `ctx_fn(env_params, ts_prev, ts_next)` that runs right after `env.step`. It returns a dictionary mapping strings to JAX arrays. Each entry is derived from the `xminigrid.types.TimeStep` objects:
    - `ts_prev.state` / `ts_next.state` expose the full grid (`grid` shaped `[height, width, 2]` with `(tile_id, color_id)`), the agent (`agent.position` as zero-based `[row, col]`, `agent.direction` in {0: up, 1: right, 2: down, 3: left}, and `agent.pocket`), plus goal and rule encodings.
    - We commonly precompute arrays such as `ctx["agent_pos_prev"] = ts_prev.state.agent.position`, `ctx["agent_pos"] = ts_next.state.agent.position`, object positions extracted from the grid (e.g., `ctx["yellow_square_pos"]`), boolean masks (`ctx["has_key"]`), or scalar distances between entities. Every value is a `jnp` array (often `jnp.int32` positions or `jnp.float32` distances).
    - When no `ctx_fn` is provided the wrapper hands you `{}`; your code must therefore tolerate missing keys and supply reasonable defaults via `ctx.get("name", fallback)`.
- When the wrapper runs, it replaces the environment's sparse reward with your dense value: `ts_next = env.step(...)`, then `dense_reward` is called and its output stored in `ts_next.reward`.
- Existing placeholder reward (for reference only):

```python
def dummy_dense_reward(ts_prev, action, ts_next):
    ones = jnp.ones_like(ts_next.reward)
    zeros = jnp.full_like(ts_next.reward, 0.0)
    return jnp.where(ts_next.last() > 0, zeros, zeros)
```

Rules:
- Use ONLY jax.numpy as jnp (import not needed) and jax.lax if necessary.
- Do NOT add import statements; jnp and jax are already available.
- Use ONLY values in 'ctx' (e.g., ctx['agent_pos'], ctx['goal_pos'], ctx['has_key'], distances).
- Access ctx via ctx.get("key", default) and timestep helpers like ts_next.last(); avoid other Python-only methods (e.g., .item(), .tolist()).
- If you define helper functions inside dense_reward, ensure they are pure, side-effect free, and only call jnp/jax operations.
- Do NOT access Python globals, files, network, randomness, or environment internals.
- The function must be pure and JIT-friendly: no Python branching on array values; use jnp.where / lax.cond.
- Reward should be shaped dense potential: positive when closer to achieving the goal; small step penalty ok.
- Must gracefully handle episode termination: set to 0 after terminal or add a success bonus that is consistent with sparse=1.
- Return jnp.asarray(<scalar>, dtype=jnp.float32).

YOU MUST WRITE VALID JITTABLE JAX CODE
"""

# ---------- Public API ----------
def make_dense_reward(env, env_params, dspy_model=None):
    """
    Returns a callable dense_fn(env_params, ts_prev, action, ts_next, ctx) -> jax.Array
    """
    # Use default DSPy model if not injected; configure externally (OpenAI/Azure/etc.)
    _prog = RewardSynthesizer() if dspy_model is None else dspy_model
    env_text = describe_ruleset(env, env_params)
    code = _prog(env_text, CONSTRAINTS_TEXT)


    print("\n==== Generated dense_reward candidate (pre-sanitize) ====\n")
    print(code)
    print("\nEnvironment Description: \n", env_text)
    print("\n=========================================================\n")

    dense_fn = sanitize_and_compile(code)
    
    print("\n\n----\n")
    print("Dense Function: \n", dense_fn)
    print("\n----\n\n")

    return dense_fn, code

if __name__ == "__main__":
    dspy.configure(lm=lm)  # make sure it's set
    class EchoSig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    p = dspy.Predict(EchoSig)
    out = p(question="Say 'ok' once.")
    print("LM check:", out.answer[:120])
