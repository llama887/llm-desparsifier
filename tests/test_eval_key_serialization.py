from __future__ import annotations

import jax

from llm_desparsifier.rl.eval import _key_to_list


def test_key_to_list_supports_typed_prng_keys() -> None:
    """Ensure typed JAX keys serialize without crashing trajectory capture.

    This test guards against regressions where trajectory serialization attempts
    to cast typed keys (`key<fry>`) directly to `uint32`, which raises
    `Cannot convert_element_type from key<fry> to uint32` on modern JAX. It is
    needed because GEPA candidate runs can spend minutes training before hitting
    eval, and a key-serialization crash at that point silently prevents
    `eval_trajectory.json` from being written. It differs from broad integration
    tests by validating the exact low-level conversion helper that replay
    tooling depends on.
    """

    key = jax.random.key(123)
    expected = [int(v) for v in jax.random.key_data(key).reshape(-1).tolist()]
    assert _key_to_list(key) == expected


def test_key_to_list_supports_legacy_uint32_prng_keys() -> None:
    """Verify legacy uint32 PRNG keys remain compatible with serialization.

    This test ensures `_key_to_list` keeps working for call sites that still use
    `jax.random.PRNGKey`, which returns classic `uint32[2]` arrays. It is needed
    because replay payloads may be produced by mixed key styles during library
    transitions, and it differs from the typed-key test by validating backward
    compatibility rather than typed-key correctness.
    """

    key = jax.random.PRNGKey(456)
    expected = [int(v) for v in jax.random.key_data(key).reshape(-1).tolist()]
    assert _key_to_list(key) == expected
