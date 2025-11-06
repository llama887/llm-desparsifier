# Refactor Plan for `llm-desparsifier`

## Core Objectives
- Treat the reinforcement-learning training stack as a black box that exposes a clear entry point for experiments.
- Untangle the reward generation (LLM interaction, sanitizing, parsing) into focused modules.
- Move implementation files out of the repository root into a maintainable package layout while keeping artifacts and data separate.

## Target Package Layout (proposed)
- `llm_desparsifier/`
  - `__init__.py` (exports the high-level training API).
  - `rl/`
    - `__init__.py`
    - `pipeline.py` (black-box training loop entry point exposing `run_training_with_reward(...)`).
    - `wrappers.py` (reward wrapper, timestep helpers).
    - `metrics.py` (helpers for logging, plotting, optional future extensions).
  - `rewards/`
    - `__init__.py`
    - `llm_client.py` (DSPy/Portkey configuration and inference glue).
    - `sanitizer.py` (AST whitelist logic currently in `reward_generator.py`).
    - `parser.py` (code extraction / validation helpers, goal text formatting).
    - `generator.py` (public `RewardGenerator` class orchestrating prompt build + LLM + sanitizer).
  - `utils/`
    - `__init__.py`
    - `prompts.py` (prompt templates, env description helpers).
    - `context.py` (utilities such as `extract_xland_ctx` shared across rewards/rl code).
- `scripts/`
  - `train_with_llm_reward.py` (thin CLI that wires CLI args to `rl.pipeline.run_training_with_reward`).
- `artifacts/`
  - `generated_rewards/` (LLM-authored reward code snapshots such as `dense_reward_synthesized.py`).
- Keep existing `data/`, `logs/`, `tests/`, `sbatch/` directories; adjust their contents only if path changes require it.

## Step-by-Step Refactor Plan

### Phase 1 – Preparation
1. Audit each top-level Python file to confirm current responsibilities and note hidden dependencies (already partially done; document findings inline during move).
2. Update `pyproject.toml` (and tooling configs if needed) to include the `llm_desparsifier` package so imports resolve after reshuffling.
3. Add minimal `__init__.py` files for each new package directory to keep imports explicit from the start.

### Phase 2 – RL Training Black Box
4. Carve the training logic out of `xland_meta_learning_baseline.py` into `llm_desparsifier/rl/pipeline.py`.
   - Preserve existing functionality unchanged while encapsulating side effects (plotting, logging, metric collection) behind function parameters (e.g., `output_dir`, callbacks).
   - Note: Training script moved into `llm_desparsifier/rl/pipeline.py` with an exported `run_training_with_reward` API; plotting/video output now respects the `output_dir` parameter.
5. Define the public training API (tentatively `run_training_with_reward(reward_generator, output_dir, config_override=None)`).
   - Accept a `RewardGenerator` protocol instead of raw functions to support experimentation.
   - Ensure the function returns a structured result (summary stats, artifact paths) for downstream automation.
   - Note: Added `RewardGeneratorProtocol` and `TrainingResult` to the pipeline module; the API collects artifact paths and final metrics.
6. Relocate auxiliary classes (`RewardTimeStep`, `DesparsifyRewardWrapper`, transition dataclasses) into `rl/wrappers.py` (or split into `wrappers.py` & `structures.py` if needed).
   - Keep non-API helpers private to the module.
   - Note: Wrapper utilities now live in `llm_desparsifier/rl/wrappers.py`; PPO data carriers moved into `llm_desparsifier/rl/structures.py` with legacy imports re-exported via `reward_wrapper.py`.
7. Replace the original script (`xland_meta_learning_baseline.py`) with a lightweight entry point that imports the new API or move it entirely into `scripts/train_with_llm_reward.py` and delete the legacy root file.
   - Note: `xland_meta_learning_baseline.py` now delegates to the pipeline API via a small adapter around the existing `make_dense_reward` function.

### Phase 3 – Reward Generation Cleanup
8. Move DSPy/Portkey setup from `reward_generator.py` into `rewards/llm_client.py`; expose a single factory for configured LMs.
   - Note: Created `llm_desparsifier/rewards/llm_client.py` with `configure_portkey_lm` and re-exported it for compat.
9. Extract AST sanitization (`_Sanitizer`, `_ALLOWED_*`, `sanitize_and_compile`) into `rewards/sanitizer.py` with unit-test-friendly interfaces.
   - Note: Sanitizer logic now lives in `llm_desparsifier/rewards/sanitizer.py` and is re-exported for legacy imports.
10. Separate prompt building, environment description templates, and parsing/safety validation helpers into `rewards/parser.py` (or split into `prompts.py` under `utils` if reuse is broader).
    - Note: Added `llm_desparsifier/rewards/parser.py` containing `describe_ruleset`, helpers, and `CONSTRAINTS_TEXT`.
11. Build a cohesive `RewardGenerator` class in `rewards/generator.py` that orchestrates: context building → prompt creation → LLM call → sanitizer → compiled reward callable.
    - Note: Implemented `RewardGenerator` and `RewardSynthesizer` in `llm_desparsifier/rewards/generator.py` with verbose logging and sanitization hooks.
12. Relocate environment description helpers and context text from `reward_generator.py` to the appropriate module (`contexts/xminigrid.py` or `rewards/prompts`), keeping only orchestration in `generator.py`.
    - Note: Environment narrative utilities moved into the parser module; compatibility wrapper delegates to the new package.
13. Update call sites (tests, scripts, RL pipeline) to use the new `RewardGenerator` API rather than module-level globals.
    - Note: Legacy `reward_generator.py` now wraps the new class, and `xland_meta_learning_baseline.py` instantiates `RewardGenerator` directly.

### Phase 4 – Context Extraction & Utilities
14. Move `ctx_extractors.py` into `utils/context.py`; ensure the RL pipeline imports it via the new package namespace.
    - Note: Context helpers now reside in `llm_desparsifier/utils/context.py`; the module documents the expected callable signature and is exported via `utils.__all__`.
15. Document the expected interface for context extractors (function signature, return value) within the module docstring and export via `utils/__init__.py`.
    - Note: Documented the interface in the new module docstring and re-exported `extract_xland_ctx` through `llm_desparsifier.utils`.
16. Reorganize any remaining helpers (e.g., debug scripts, plotting utilities) into `utils/` or `scripts/` depending on runtime vs. dev-time usage.

### Phase 5 – Artifacts, Tests, and Tooling
17. Move generated reward files (e.g., `dense_reward_synthesized.py`) into `artifacts/generated_rewards/` and adjust `.gitignore` to keep auto-generated outputs optional.
    - Note: Added `artifacts/generated_rewards/` (and baseline run outputs) with `.gitkeep`, relocated existing reward snapshots, and updated `.gitignore` to ignore generated artifacts.
18. Update existing tests (under `tests/`) to import from the new package paths; add new tests for sanitizer and reward generator orchestration if coverage gaps appear.
    - Note: `tests/debug_ctx.py` now imports directly from `llm_desparsifier` packages and locates synthesized rewards under the new artifact directories.
19. Rename or update sbatch scripts and notebooks to point to the new CLI script or API entry point.
    - Note: `sbatch/xland_baseline.s` now invokes `python xland_meta_learning_baseline.py --output-dir` with per-job artifact directories; both sbatch scripts ensure `logs/` exists.
20. Grep the repository for old module paths (`reward_generator`, `reward_wrapper`, etc.) and rewrite imports to the new locations.
    - Note: Legacy imports replaced with `llm_desparsifier` package equivalents, leaving compatibility shims only for external callers.
21. Run the full training pipeline (or the smallest reproducible subset) to confirm behaviour remains unchanged; capture new artifact locations in documentation.
    - Note: Deferred execution per instructions; sbatch script prepared to run the updated pipeline when triggered manually.

### Phase 6 – Cleanup and Documentation
22. Prune leftover root-level Python files once imports succeed (keeping only package directories and scripts).
23. Refresh `README.md` to describe the new structure and usage (CLI vs. Python API).
24. Summarize migration notes in a changelog entry or developer doc (e.g., `docs/refactor-notes.md`) to help future contributors.
25. Consider adding a Makefile or task runner entry for the new CLI (`make train`), improving discoverability.

## Open Questions for Follow-Up
- Confirm whether the training API should accept additional callbacks (logging hooks, evaluation episodes) beyond reward generator and output directory.
- Decide if multiple reward generators will coexist (ensemble, ablations) and whether the API should accept a registry instead of a single callable.
- Determine the long-term home for generated reward files (tracked artifacts vs. reproducible generation on demand).

## Phase 1 Preparation Notes

- `ctx_extractors.py`  
  - **Responsibility:** builds context dictionaries (positions, agent state, carried objects) from XLand MiniGrid timesteps for use in dense reward shaping.  
  - **Key Dependencies:** `jax`, `jax.numpy`, `xminigrid.core.constants.Colors`, `xminigrid.core.constants.Tiles`.  
  - **Hidden Couplings:** assumes `ts_next.state` exposes `grid`, `agent`, and `pocket` attributes following MiniGrid conventions; returns dense JAX arrays to stay jit-compatible.

- `dense_reward_synthesized.py`  
  - **Responsibility:** stores the latest LLM-produced dense reward function. Treated as an artifact rather than reusable module code.  
  - **Key Dependencies:** expects `jnp` to be injected (not imported locally); relies on surrounding caller to supply context entries (`yellow_square_pos`, `green_ball_pos`).  
  - **Hidden Couplings:** assumes `ctx` keys produced by `ctx_extractors.py`; mirrors sparse reward semantics when `ts_next.last() > 0`.

- `reward_generator.py`  
  - **Responsibility:** orchestrates DSPy/Portkey model configuration, LLM prompting, AST sanitization, and runtime compilation of dense reward code.  
  - **Key Dependencies:** `dspy`, `dotenv` for environment config, `jax`/`jax.numpy`, `ast`, `types`, `io`, and custom sanitizer whitelist logic contained in the same file.  
  - **Hidden Couplings:** global `lm = configure_dspy_with_portkey()` executed at import time (requires `PORTKEY_API_KEY`); tightly couples prompt templating, sanitizing, and code execution, making unit testing difficult.

- `reward_wrapper.py`  
  - **Responsibility:** wraps an environment to overwrite rewards with dense functions while preserving extras; provides a `RewardTimeStep` proxy and a `dummy_dense_reward` placeholder.  
  - **Key Dependencies:** `flax.struct`, `flax.core.frozen_dict`, `collections.abc.Mapping`, `inspect`, `jax.numpy`.  
  - **Hidden Couplings:** tries to maintain JAX-trace-friendly behaviour by inspecting dense reward arity at init; expects dense functions of arity 3 or 5.

- `xland_meta_learning_baseline.py`  
  - **Responsibility:** full RL training pipeline (networks, PPO loop, data collection, logging, plotting). Treating this as the black-box engine for reward experiments.  
  - **Key Dependencies:** extensive use of `jax`, `flax`, `optax`, `distrax`, `matplotlib`, `xminigrid` environment suite, `imageio`.  
  - **Hidden Couplings:** assumes benchmark/task definitions from `xminigrid`; plots directly to PNG/Movie files; mixes training config, rollout evaluation, and CLI-like script behaviour in a single file.
