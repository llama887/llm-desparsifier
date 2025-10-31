from __future__ import annotations
import ast, types
import os
import jax
import jax.numpy as jnp
import dspy
from dotenv import load_dotenv

load_dotenv()  # pulls PORTKEY_API_KEY from .env if present

# reward_generator.py (or dspy_portkey.py)
import os
import dspy

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

class _Sanitizer(ast.NodeVisitor):
    def visit_Import(self, node): raise ValueError("imports not allowed")
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

    def visit_Attribute(self, node):
        # allow chained attributes only for whitelisted roots
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in _ALLOWED_TOP:
            # e.g. jnp.linalg.norm or jax.lax.cond
            return self.generic_visit(node)
        raise ValueError("attribute access not allowed except jnp/jax.*")

    def visit_Call(self, node):
        # allow calls only if function is Name/Attribute on allowed roots
        if isinstance(node.func, ast.Attribute):
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _ALLOWED_TOP:
                return self.generic_visit(node)
        elif isinstance(node.func, ast.Name):
            if node.func.id in ("float", "int"):  # harmless casts
                return self.generic_visit(node)
        raise ValueError("function calls must be to jnp/jax.* or basic casts")

def sanitize_and_compile(code: str):
    tree = ast.parse(code)
    # must contain exactly one function named dense_reward
    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fdefs or fdefs[0].name != "dense_reward":
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

# ---------- Env description builder (text) ----------
def describe_ruleset(env, env_params) -> str:
    """
    Robust textual description. Uses getattr checks so it won't crash if fields differ.
    Keep it short; we just need the essentials for the LLM.
    """
    parts = []
    rs = getattr(env_params, "ruleset", None)
    if rs is not None:
        # Very generic fields often present in XLand-like minigrid rulesets:
        goal = getattr(rs, "goal", None)
        if goal is not None:
            parts.append(f"goal_type={getattr(goal, 'type', 'unknown')}")
            gp = getattr(goal, "position", None)
            if gp is not None:
                parts.append(f"goal_position={gp}")
        objs = getattr(rs, "objects", None)
        if objs is not None:
            parts.append(f"objects={objs}")
        size = getattr(rs, "grid_size", None)
        if size is not None:
            parts.append(f"grid_size={size}")
        actions = getattr(rs, "actions", None)
        if actions is not None:
            parts.append(f"actions={actions}")
        term = getattr(rs, "termination", None)
        if term is not None:
            parts.append(f"termination={term}")
    else:
        parts.append("ruleset=unknown")
    parts.append("sparse_reward=1 on success else 0")
    return " | ".join(parts)

# ---------- Constraints prompt (what the LLM can assume/use) ----------
CONSTRAINTS_TEXT = """
You must output exactly ONE function:
  def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    # returns a scalar jnp.float32

Rules:
- Use ONLY jax.numpy as jnp (import not needed) and jax.lax if necessary.
- Use ONLY values in 'ctx' (e.g., ctx['agent_pos'], ctx['goal_pos'], ctx['has_key'], distances).
- Do NOT access Python globals, files, network, randomness, or environment internals.
- The function must be pure and JIT-friendly: no Python branching on array values; use jnp.where / lax.cond.
- Reward should be shaped dense potential: positive when closer to achieving the goal; small step penalty ok.
- Must gracefully handle episode termination: set to 0 after terminal or add a success bonus that is consistent with sparse=1.
- Return jnp.asarray(<scalar>, dtype=jnp.float32).
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
    dense_fn = sanitize_and_compile(code)
    return dense_fn, code

if __name__ == "__main__":
    dspy.configure(lm=lm)  # make sure it's set
    class EchoSig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    p = dspy.Predict(EchoSig)
    out = p(question="Say 'ok' once.")
    print("LM check:", out.answer[:120])
