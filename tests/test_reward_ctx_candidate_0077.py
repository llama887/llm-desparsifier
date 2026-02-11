from __future__ import annotations

import ast
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
xminigrid = pytest.importorskip("xminigrid")
from xminigrid.wrappers import GymAutoResetWrapper

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "candidate_0077_trivial_r1_15x15_seed2"
_MISSING_POS = jnp.array([-1, -1], dtype=jnp.int32)
_MAX_REPLAY_STEPS = 32


def _load_context_module() -> Any:
    """Load the context extractor module directly from its source file path.

    Importing `llm_desparsifier.utils.context` through package imports triggers
    top-level `llm_desparsifier.__init__` side effects, which include DSPy cache
    initialization that can fail in read-only environments. This helper is
    needed to keep the test hermetic and focused on context extraction logic,
    and it differs from normal imports by constructing a module spec from the
    concrete file location under the repository root.

    Returns:
        Imported `context.py` module object.
    """

    module_path = (
        Path(__file__).resolve().parents[1]
        / "llm_desparsifier"
        / "utils"
        / "context.py"
    )
    spec = importlib.util.spec_from_file_location("ctx_module_for_candidate_0077", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load context module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract_xland_ctx = _load_context_module().extract_xland_ctx


@dataclass(frozen=True)
class _CandidateFixture:
    """Container for all artifact copies required by this regression test.

    This dataclass groups the two fixture payloads that would otherwise be read
    directly from the gitignored `artifacts/` directory: the synthesized reward
    source code and the saved replay trajectory metadata. The grouping is needed
    so the test can pass around one immutable value instead of repeatedly reading
    files, and it differs from ad-hoc dict usage by giving explicit typed fields
    that document exactly which persisted artifacts are required for deterministic
    replay-based context validation.
    """

    reward_code: str
    trajectory: dict[str, Any]


def _load_candidate_fixture() -> _CandidateFixture:
    """Load candidate-0077 reward and trajectory from committed test fixtures.

    The test intentionally avoids reading from `artifacts/` because those files
    are gitignored and can be deleted between runs. This helper is needed to
    guarantee the regression remains reproducible in CI and fresh checkouts, and
    it differs from direct file reads in the test body by centralizing existence
    checks and decoding errors into one clear failure point.

    Returns:
        `_CandidateFixture` with synthesized reward source and trajectory payload.

    Raises:
        AssertionError: If required fixture files are missing.
        ValueError: If the trajectory JSON cannot be decoded.
    """

    reward_path = _FIXTURE_DIR / "dense_reward_synthesized.py"
    trajectory_path = _FIXTURE_DIR / "eval_trajectory.json"

    assert reward_path.exists(), f"missing reward fixture: {reward_path}"
    assert trajectory_path.exists(), f"missing trajectory fixture: {trajectory_path}"

    return _CandidateFixture(
        reward_code=reward_path.read_text(encoding="utf-8"),
        trajectory=json.loads(trajectory_path.read_text(encoding="utf-8")),
    )


def _strip_markdown_fences(code: str) -> str:
    """Remove surrounding triple-backtick fences from synthesized reward code.

    Generated reward artifacts are frequently saved as fenced Markdown blocks,
    which are invalid for direct AST parsing. This helper is needed to parse the
    exact emitted reward source without mutating the fixture content itself, and
    it differs from broad text cleanup by only removing a top-level fenced block
    while leaving interior code untouched.

    Args:
        code: Raw reward file content that may include Markdown fences.

    Returns:
        Plain Python source suitable for `ast.parse`.
    """

    stripped = code.strip()
    if not stripped.startswith("```"):
        return code
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return code


@dataclass(frozen=True)
class _CtxLookupKeys:
    """Reward-referenced nested object keys grouped by ctx map source.

    Dense rewards can read object coordinates from multiple nested context maps
    (`object_positions`, `visible_object_positions`, and previous-view variants).
    This structure is needed so the test can assert map-specific expectations,
    and it differs from a single flat key set by preserving which ctx namespace
    each lookup came from.
    """

    object_positions: frozenset[str]
    visible_object_positions: frozenset[str]
    visible_object_positions_prev: frozenset[str]


def _extract_ctx_object_lookup_keys(reward_code: str) -> _CtxLookupKeys:
    """Parse synthesized reward code and collect nested object lookup keys.

    This parser identifies keys passed to `.get("<key>", ...)` when called on
    locals that were assigned from `ctx.get("object_positions", {})` or one of
    the visible-object maps. It is needed to ensure assertions are driven by the
    reward's actual lookup behavior rather than manually duplicated expectations,
    and it differs from string matching by using AST structure to avoid false
    positives in comments or unrelated literals.

    Args:
        reward_code: Raw synthesized reward source.

    Returns:
        `_CtxLookupKeys` containing distinct looked-up keys per ctx map source.
    """

    source = _strip_markdown_fences(reward_code)
    tree = ast.parse(source)

    alias_to_source: dict[str, str] = {}
    map_to_keys: dict[str, set[str]] = {
        "object_positions": set(),
        "visible_object_positions": set(),
        "visible_object_positions_prev": set(),
    }

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "ctx"
                and node.value.func.attr == "get"
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)
            ):
                key = str(node.value.args[0].value)
                if key in map_to_keys:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            alias_to_source[target.id] = key
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                key = str(node.args[0].value)
                owner_name: str | None = None
                if isinstance(node.func.value, ast.Name):
                    owner_name = node.func.value.id
                if owner_name in alias_to_source:
                    source_name = alias_to_source[owner_name]
                    map_to_keys[source_name].add(key)
            self.generic_visit(node)

    _Visitor().visit(tree)

    return _CtxLookupKeys(
        object_positions=frozenset(map_to_keys["object_positions"]),
        visible_object_positions=frozenset(map_to_keys["visible_object_positions"]),
        visible_object_positions_prev=frozenset(
            map_to_keys["visible_object_positions_prev"]
        ),
    )


def _coerce_key(words: list[int], *, name: str) -> Any:
    """Convert two stored uint32 words into a replay-ready JAX PRNG key.

    The saved trajectory stores PRNG keys as two integers so it can be serialized
    to JSON. This helper is needed to reconstruct the same key format expected by
    XLand replay APIs, and it differs from sampling a new key by preserving exact
    captured key bits for deterministic reconstruction.

    Args:
        words: Serialized key data from trajectory JSON.
        name: Field name used in validation errors.

    Returns:
        A typed JAX key when supported, or a uint32 fallback array.

    Raises:
        ValueError: If the serialized key does not contain exactly two words.
    """

    arr = jnp.asarray(words, dtype=jnp.uint32)
    if arr.shape != (2,):
        raise ValueError(f"{name} must contain exactly two uint32 values")
    try:
        return jax.random.wrap_key_data(arr)
    except Exception:
        return arr


def _is_valid_pos(value: Any) -> bool:
    """Return whether an object position is a concrete grid coordinate.

    Context extraction uses `[-1, -1]` as a sentinel for absent objects. This
    helper is needed to keep position validity checks consistent and explicit in
    assertions, and it differs from raw truthiness checks by matching the same
    sentinel semantics used by synthesized reward logic.

    Args:
        value: Any array-like position value from ctx maps.

    Returns:
        True when both coordinates are greater than -1, otherwise False.
    """

    arr = jnp.asarray(value)
    return bool(jax.device_get(jnp.all(arr > -1)))


def _env_text_object_keys(env_text: str) -> set[str]:
    """Extract quoted object keys mentioned in environment text guidance.

    The reward generator embeds object lookup examples such as `"blue_key"` in
    the textual prompt. This helper is needed to compare reward lookup keys
    against that prompt-level contract, and it differs from strict parser-level
    ruleset introspection by operating directly on the saved `env_text` payload
    used for this candidate.

    Args:
        env_text: Serialized environment description from trajectory metadata.

    Returns:
        Set of snake_case object-like tokens found inside double quotes.
    """

    return {
        token
        for token in re.findall(r'"([a-z]+(?:_[a-z]+)+)"', env_text)
        if "_" in token
    }


def test_candidate_0077_ctx_object_lookups_resolve_non_default_positions() -> None:
    """Ensure candidate-0077 reward ctx object lookups resolve for replayed seed.

    This regression test replays the exact saved trajectory for candidate-0077
    and validates that every object key referenced through
    `ctx.get("object_positions", {}).get(...)` in the synthesized reward resolves
    to at least one non-sentinel coordinate during replay. The test is needed to
    catch failures where reward code targets objects not present in the task
    prompt (for example, querying `red_key` when the task uses `blue_key` and
    `red_star`), which causes shaping terms to remain zero while only time
    penalty changes. It differs from generic context-shape tests by asserting
    alignment between reward lookup keys, environment text guidance, and concrete
    replayed ctx values for this specific candidate/seed.
    """

    fixture = _load_candidate_fixture()
    trajectory = fixture.trajectory
    lookup_keys = _extract_ctx_object_lookup_keys(fixture.reward_code)

    assert lookup_keys.object_positions, (
        "Reward references no object_positions lookups; this regression expects at least "
        "one nested object_positions key lookup in dense_reward_synthesized.py"
    )

    env_text = str(trajectory.get("env_text", ""))
    env_text_keys = _env_text_object_keys(env_text)

    env, env_params = xminigrid.make(str(trajectory["env_id"]))
    env = GymAutoResetWrapper(env)
    benchmark = xminigrid.load_benchmark(str(trajectory["benchmark_id"]))

    if bool(trajectory.get("deterministic_rulesets")):
        ruleset_index = trajectory.get("ruleset_index")
        if ruleset_index is None:
            ruleset_index = 0
        ruleset = benchmark.get_ruleset(int(ruleset_index))
    else:
        ruleset = benchmark.sample_ruleset(
            _coerce_key(list(trajectory["ruleset_key"]), name="ruleset_key")
        )
    env_params = env_params.replace(ruleset=ruleset)

    ts_prev = env.reset(
        env_params,
        _coerce_key(list(trajectory["reset_key"]), name="reset_key"),
    )

    seen_valid_object_positions = {
        key: False for key in sorted(lookup_keys.object_positions)
    }

    for action in trajectory["actions"][:_MAX_REPLAY_STEPS]:
        ts_next = env.step(env_params, ts_prev, jnp.int32(action))
        ctx = extract_xland_ctx(env_params, ts_prev, ts_next)
        object_positions = dict(ctx["object_positions"])

        for key in seen_valid_object_positions:
            value = object_positions.get(key, _MISSING_POS)
            if _is_valid_pos(value):
                seen_valid_object_positions[key] = True

        ts_prev = ts_next

    unresolved = [
        key for key, was_valid in seen_valid_object_positions.items() if not was_valid
    ]
    missing_from_env_text = sorted(
        key for key in lookup_keys.object_positions if key not in env_text_keys
    )

    assert not unresolved and not missing_from_env_text, (
        "Reward ctx object-key validation failed for candidate-0077 replay. "
        f"unresolved_ctx_keys={unresolved}; "
        f"missing_from_env_text={missing_from_env_text}; "
        f"seen_valid={seen_valid_object_positions}; "
        f"reward_object_keys={sorted(lookup_keys.object_positions)}; "
        f"reward_visible_keys={sorted(lookup_keys.visible_object_positions)}; "
        f"env_text_keys={sorted(env_text_keys)}. "
        "This indicates the reward is querying object keys that do not map to concrete "
        "task objects in ctx, which can leave shaping components at zero."
    )
