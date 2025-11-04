# Plan: Diagnose and Fix Empty `ctx` Driving Dense Reward Degeneration

## Current Context Flow (What `ctx` Should Be)
- `xland_meta_learning_baseline.make_states` wraps the raw XLand MiniGrid env with `GymAutoResetWrapper` and then `DesparsifyRewardWrapper` (`reward_wrapper.py`).
- `DesparsifyRewardWrapper.step` (`reward_wrapper.py:84-113`) calls `env.step`, captures the original reward, and prepares `ctx`:
  * If a `ctx_fn` is provided, it is invoked as `ctx_fn(env_params, ts_prev, ts_next)`.
  * Otherwise it forwards `ts_next.extras` (falling back to `{}` if `extras` is `None`).
- `reward_generator.make_dense_reward` emits a five-argument `dense_reward` (see `dense_reward_synthesized.py`) that expects keys like `"yellow_square_pos"` and `"green_ball_pos"` to be present inside `ctx`.
- Normalized dense reward perfectly matching the sparse curve implies that either (a) the dense reward fell back to the sparse reward path, or (b) the dense reward collapsed to a constant/degenerate signal that normalizes identically to ground-truth returns. Both are consistent with an empty or incomplete `ctx`.

## A) Potential Root Causes to Investigate
- **Underlying env never populates extras**: if `xminigrid`’s `TimeStep.extras` is empty unless a specific wrapper (e.g. a "state tracking" helper) is enabled, `ctx` will always be `{}` without a custom `ctx_fn`.
- **`GymAutoResetWrapper` strips extras**: this wrapper might drop or reset the `extras` field when auto-resetting environments, leaving `ts_next.extras` empty even if the base env had context.
- **`DesparsifyRewardWrapper` retunes extras improperly**: `_augment_extras` copies the incoming mapping before freezing it; if the original extras are stored under a different attribute or require a deep structure (e.g. `ts_next.info`), they may be lost.
- **Missing bespoke `ctx_fn`**: the pipeline never supplies `ctx_fn`, so any contextual values must already live in `extras`. If XLand MiniGrid does not expose block positions there by default, the dense reward cannot see them.
- **Signature mismatch fallback**: if `inspect.signature` misdetects the synthesized function (e.g. due to decorators or partials) the wrapper falls back to `ts_next.reward`, yielding an exact copy of the sparse reward.
- **Key name mismatch**: `dense_reward` requests `"yellow_square_pos"` / `"green_ball_pos"`, but the env (or future `ctx_fn`) may emit different naming (`"yellow_square_position"`). Defaults would then fire, returning constant arrays.
- **FrozenDict semantics**: `ctx` may be a `flax.core.FrozenDict`; if `ctx.get` behaves differently under JIT (returning defaults every time), the dense reward effectively uses the fallback path.

## Required Deep Dives
- **Document what `ctx` is in XLand MiniGrid**: inspect `xminigrid` source (likely `xminigrid/types.py` or environment implementation) to confirm whether `TimeStep` exposes an `extras` mapping, what keys it contains (agent position, block positions, mission text, etc.), and under which wrappers it is populated.
- **Trace the full `ctx` path**: follow the object from the base env → `GymAutoResetWrapper` → `DesparsifyRewardWrapper`. Confirm whether `extras` survives each layer and whether any step replaces the object with `{}`.
- **Quantify dense reward fallback behaviour**: evaluate the generated `dense_reward` under `ctx = {}` to verify it returns a constant signal (e.g. `-1.01`) that could normalize to the same curve as the sparse reward.

## Focused Action Plan (Current Workstream)
1. **Locate XLand MiniGrid Source**
   - Check whether `xminigrid` is installed; if not, identify installation steps (e.g., `uv pip install xminigrid[baselines]`).
   - Once available, record its filesystem path via `python -c "import xminigrid, inspect; print(xminigrid.__file__)"`.
2. **Catalogue Available Extras**
   - Review the package’s environment implementation (`environment.py`, `types.py`, or wrappers) to document how `TimeStep.extras` is populated for XLand tasks.
   - Note any required wrappers or benchmark flags to enable richer extras (positions, mission metadata, etc.).
3. **Prototype `debug_ctx.py`**
   - Recreate the training env stack: `xminigrid.make` → `GymAutoResetWrapper` → `DesparsifyRewardWrapper`.
   - Use a dummy dense reward that logs incoming `ctx`, and capture `ts_next.extras` before and after wrapping.
   - Run a few deterministic `step`s to confirm whether context survives the pipeline and identify missing keys.
   - Summarize observed outputs (key presence, reward divergence) for future debugging.

## B) Step-by-Step Resolution Plan
1. **Inspect XLand MiniGrid extras**  
   Locate the installed `xminigrid` package (`python -c "import xminigrid, inspect; print(xminigrid.__file__)"`) and review its `TimeStep`/wrapper code to catalogue what `extras` contains for XLand tasks. Note any conditions (benchmarks, wrappers) that enable richer context.
2. **Instrument a thin probe**  
   Write a short script (e.g. `debug_ctx.py`) that instantiates the same env stack as training, runs a few `reset`/`step` calls, and prints/logs both `ts_next.extras` and the `ctx` observed inside a dummy dense reward. This validates whether extras are empty before modifying core code.
3. **Decide on `ctx_fn` vs. env extras**  
   - Probe results (`logs/ctx_probe-1556111.out`) show `TimeStep.extras` is always `None`; the wrapper only injects `ground_truth_reward`/`dense_reward`, and the dense callback sees an empty `ctx`.
   - Therefore, we must implement a dedicated `ctx_fn` that inspects `ts_next.state.grid` (and optionally `ts_prev.state.grid`) to surface task objects.
   - Target keys for the current dense reward: `ctx["yellow_square_pos"]`, `ctx["green_ball_pos"]`. Compute them via `jnp.argwhere` over the grid; default to `jnp.array([-1, -1], dtype=jnp.int32)` if missing.
   - Additional context worth exposing (future-proof): agent position (`ts_next.state.agent.position`), inventory flags (`state.agent.pocket`), and step counters. Keep all values as JAX arrays (`int32`/`bool`).
   - Implementation sketch:
     1. Add a new module (e.g., `ctx_extractors.py`) with `def make_ctx(env_params, ts_prev, ts_next) -> dict[str, jax.Array]`.
     2. Inside, import `TILES_REGISTRY`, `Tiles`, and `Colors` to build tile IDs for objects relevant to the LLM reward.
     3. Use `jnp.argwhere(ts_next.state.grid == tile_id)` to locate objects; apply `jax.lax.cond` to handle missing tiles without dynamic Python branches.
     4. Return a plain `dict` mapping strings to arrays (the wrapper will freeze it).
   - Wire the extractor into `DesparsifyRewardWrapper` construction within `xland_meta_learning_baseline.make_states`, and rerun `debug_ctx.py` to confirm the new keys appear in `ctx`.
4. **Thread the `ctx_fn` through configuration**  
   Update `make_states` (or wherever `DesparsifyRewardWrapper` is constructed) to pass the new `ctx_fn`. Ensure the function is JAX-compatible (pure, array outputs). Re-run the probe script to confirm the dense reward now sees real context data.
5. **Validate dense reward behaviour**  
   Using the probe, compare dense vs sparse rewards over a few steps to confirm they diverge once context flows. Optionally add assertions/logs for key presence to catch regressions early.
6. **Retune/regen dense reward if keys differ**  
   If context provides different key names or formats, either adjust the synthesizer prompt/constraints so the generated code uses the right keys, or add a translation layer in `ctx_fn`.

## Lightweight Test/Debug Hook (No Full Training Required)
- Add a developer-only script (e.g. `scripts/ctx_smoke_test.py`) that:
  1. Creates the wrapped env.
  2. Performs `reset` + a few deterministic `step`s.
  3. Captures the `ctx` dict handed to a stub dense reward and prints/asserts the presence of key contextual entries (agent/block positions).
  4. Reports the dense reward output alongside the sparse reward for comparison.
- This script can be JIT-free (pure Python + JAX eager) so it runs quickly and confirms the data path before launching expensive training runs.

By executing the above investigation and instrumentation steps we can isolate why `ctx` is empty, reintroduce meaningful context, and verify the dense reward diverges from the sparse baseline.
