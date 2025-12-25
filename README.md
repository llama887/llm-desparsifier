# llm-desparsifier - GEPA On-Policy Loop

This project runs DSPy GEPA end-to-end on-policy to synthesize dense rewards for XLand-MiniGrid. The main loop proposes reward prompts, runs the RL pipeline for each proposal, and feeds solve-rate scores plus EUREKA-style reflection text back into GEPA. LLM calls are routed through a Portkey gateway and require `PORTKEY_API_KEY`.

## DSPy GEPA contract (top level)
- **Program under optimization**: `PromptOnlyProgram` in `scripts/run_reward_batch.py`. Its `forward(env_description, constraints=None)`:
  - Uses `PromptGenerator` (`dspy.Predict(PromptSearch)`) to rewrite the current `constraints_text` unless GEPA passes `constraints` explicitly.
  - Returns `dspy.Prediction(prompt_text=...)` only (no reward-code predictor in the program graph).
- **Optimizer**: `dspy.GEPA(metric=on_policy_metric, max_metric_calls, reflection_lm, track_stats=True, num_threads=1)` created in `run_batch()`.
- **Metric contract**: `metric(gold: Example, pred: Prediction, trace=None, pred_name=None, pred_trace=None) -> float | ScoreWithFeedback`. `ScoreWithFeedback` contains a numeric `score` and free-form `feedback` text. Higher scores are better and are expected to be in `[0, 1]`.
- **How GEPA uses it here**: `on_policy_metric` runs a full RL training + evaluation loop per *example* and returns `ScoreWithFeedback(score=solve_rate, feedback=reflection_text)`. Feedback requests for predictors reuse the cached `solve_rate` for that example+prompt so the feedback score matches the module score, per the DSPy GEPA contract.

## Inputs and state
- **Env grid -> DSPy examples**: `configs/gepa_envs.yaml` is parsed into `EnvJob` objects (`env_id`, `benchmark_id`, `total_timesteps`, `train_seed`, `eval_seed`). `build_examples()` converts each job into a `dspy.Example(env_description="<env_id> | benchmark=<id>")`, attaches `example.job_config`, and sets `example.job_name`. Edit the YAML to change which environments GEPA optimizes over.
- **Prompt state**: `STATE_ROOT/active_prompt.json` (default `artifacts/gepa_state/`) stores:
  - `constraints_text` (the current prompt block),
  - `prompt_state` (DSPy weights for `PromptGenerator`),
  - `updated_at`, `source` metadata.
  The runner loads `active_prompt.json` if present; otherwise it falls back to `llm_desparsifier/rewards/prompts/base_reward_prompt.txt`, else the hard-coded `CONSTRAINTS_TEXT`. After GEPA completes it overwrites `active_prompt.json` atomically via `write_active_prompt`.
- **Run artifacts**:
  - `STATE_ROOT/gepa_runs/candidate-####-<job>`: per-candidate training outputs, reward code, metrics, W&B run dir.
  - `STATE_ROOT/gepa_runs/sparse_baseline/<job>`: sparse baseline runs (cached).
  - `sparse_baseline.json`: cached sparse baseline summary (repo root).
  - `STATE_ROOT/gepa_runs/gepa_stats.json`: GEPA optimizer stats plus sparse baselines.

## Candidate evaluation flow (per GEPA proposal)
- **Budget clamp**: `clamp_job_budget()` caps per-candidate cost (`total_timesteps <= 20M`, `num_envs <= 1024`, `eval_num_envs <= 128`, `eval_num_episodes <= 20`) and records any reductions in `budget_notes`. Missing keys are defaulted to the caps.
- **Seeds**: if the env grid does not specify `train_seed`/`eval_seed`, they are derived deterministically from `example_id` + `prompt_text` (train seed) and `train_seed + 1` (eval seed). If the grid provides seeds, they are respected. Reward-code hashes/logs are derived from the emitted code used in training.
- **Reward synthesis**: `RewardGenerator.generate()` builds the env description via `describe_ruleset`, calls the DSPy reward LLM using the GEPA-optimized `prompt_text`, sanitizes/compiles with `sanitize_and_compile`, and retries with detailed sanitizer feedback on failure. In `run_reward_batch.py` the max sanitize attempts is set to 5.
- **Training**: `run_training_with_reward()` launches PPO with an RNN policy (`TrainConfig` defaults) and installs the dense reward via `DesparsifyRewardWrapper`. Dense rewards are used only for training; evaluation uses sparse rewards.
- **Ground-truth evaluation**: `run_ground_truth_eval()` executes `eval_num_episodes` sparse episodes with the trained policy and returns per-episode returns/lengths plus success counts. These feed the GEPA score and reflection text.

## Metric used by GEPA
- **Environment source**: envs come from XLand-MiniGrid benchmarks (`env_id` / `benchmark_id` in the grid). Each job is deterministic given seeds and the generated dense reward.
- **Success criterion**: an eval episode is "solved" iff the ground-truth sparse return is `> 0` (goal satisfied before timeout).
- **Score fed to GEPA**: `solve_rate = successes / eval_num_episodes`. If eval returns are missing, the metric falls back to the last sparse curve point from training; if everything fails, score defaults to `0.0`.
- **Storage**: each GEPA candidate run logs per-env artifacts under `STATE_ROOT/gepa_runs/`. W&B (if enabled) logs `gepa/solve_rate` and candidate tables. The score is always in `[0, 1]`, so dense reward magnitude does not affect the metric.

## Feedback channel to GEPA
- The metric returns `ScoreWithFeedback(score=solve_rate, feedback=<reflection text>)` so GEPA can optimize with both numbers and natural-language guidance.
- If GEPA requests predictor-specific feedback (`pred_name`/`pred_trace`), the feedback text is prefixed with a short predictor header and the score is reused from the cached per-example evaluation.
- `build_reward_reflection()` composes the reflection input from:
  - **Env summary**: full `describe_ruleset` text when available, else `env_id | benchmark=<id>`.
  - **Reward code**: sanitized `dense_reward` source string.
  - **Sparse curve**: 6 checkpoints sampled from `eval/ground_truth_returns_mean` plus the final value.
  - **Per-component curves + stats**: checkpoints for each `reward_components` series plus min/mean/max for each component.
  - **Final metrics**: sorted dump of `TrainingResult.final_metrics` (includes `solve_rate`, `eval_successes`, etc.).
  - **Guidance**: `EUREKA_GUIDANCE` plus a reminder to give environment-aware but non-hard-coded advice.
- Failure path: on training/sanitization errors, feedback becomes `Training failed: <error>` plus sanitizer retry history (if available), and `score = 0.0`.

## What the synthesis LLM sees
- `describe_ruleset(env, env_params)` assembles the environment prompt: grid type -> layout hint, size/view/max_steps, action set, GOAL line plus a natural-language restatement (when ruleset text can be printed), truncated RULES list, INITIAL OBJECTS summary, and a reminder about partial observability.
- Reward synthesis uses a **deterministic** ruleset snapshot (`benchmark.get_ruleset(0)`) so the LLM sees a stable task description, while training/evaluation sample rulesets from the benchmark.
- The full constraints block provided to the LLM comes from `constraints_text` (loaded from `active_prompt.json`, then `base_reward_prompt.txt`, then `CONSTRAINTS_TEXT`). GEPA rewrites this full block; there is no immutable suffix automatically appended after rewrite.

## Dense reward constraints (sanitizer enforced)
- Output must define **exactly one** `dense_reward(env_params, ts_prev, action, ts_next, ctx)` function.
- Only JAX primitives are allowed (`jnp.*`, `jax.lax.*`), with method calls limited to `.astype(...)`.
- `reward_components` must be a **dict literal** with constant string keys, and the return must be `(total_reward, reward_components)`.
- `ctx` access must use `ctx.get(key, fallback)`. Direct `ctx[...]` access is rejected.
- Helper functions inside `dense_reward` must be pure and JIT-friendly; no Python-side control flow on array values.

## Environment wrapper behavior
- `DesparsifyRewardWrapper` replaces the env reward with the dense reward and preserves the original reward in `extras["ground_truth_reward"]`.
- It also logs `extras["dense_reward"]` and `extras["reward_components"]` (keys must remain constant; violations raise errors).
- Dense reward functions can accept either `(ts_prev, action, ts_next)` or `(env_params, ts_prev, action, ts_next, ctx)`; the wrapper detects the signature at runtime.

## Artifacts and layout
- `STATE_ROOT/gepa_runs/candidate-####-<job>`: per-candidate outputs, emitted reward code, metrics, W&B run dir.
- `STATE_ROOT/gepa_runs/sparse_baseline/<job>`: sparse baseline outputs.
- `sparse_baseline.json`: cached sparse baseline summary (repo root).
- `STATE_ROOT/active_prompt.json`: latest prompt state after GEPA completes.
- `STATE_ROOT/gepa_runs/gepa_stats.json`: GEPA stats plus sparse baseline summary.

## Running the loop
- Cluster: `sbatch sbatch/train_dense_batch.s` (sets caches, state root, syncs deps, runs `scripts/run_reward_batch.py`).
- Local: `uv run scripts/run_reward_batch.py --state-root artifacts/gepa_state`

Required env vars:
- `PORTKEY_API_KEY` (loaded via `.env` if present). Without it, reward synthesis and reflection will error.

Optional env vars:
- `WANDB_DISABLED=1` to skip W&B logging.
- `WANDB_PROJECT` to set the W&B project name.
- `XLAND_MINIGRID_DATA` to override the XLand data cache location.

## Behavioral notes
- The GEPA score is scale-free; multiplying a bad dense reward by 100 cannot improve solve rate.
- GEPA uses on-policy evaluation only; there is no offline dataset cache.
- Per-env and per-episode details remain in logs/W&B for debugging, but GEPA optimizes only the mean solve rate over the env grid.

## Hyperparameters and where to change them
- **Per-env budgets**: `configs/gepa_envs.yaml` (`total_timesteps`, `train_seed`, `eval_seed`, etc.). Any value above the caps will be clamped in `clamp_job_budget`; lower values are respected.
- **Global caps**: `MAX_TOTAL_TIMESTEPS`, `MAX_NUM_ENVS`, `MAX_EVAL_ENVS`, `MAX_EVAL_EPISODES` near the top of `scripts/run_reward_batch.py`.
- **RL training defaults**: `llm_desparsifier/rl/pipeline.py:TrainConfig` (policy sizes, PPO knobs, eval counts, seeds). Override by adding keys to a job entry in the env grid; GEPA passes them through to `TrainConfig(**config_override)`.
- **GEPA search budget**: `--max-metric-calls` (default 80) when running `scripts/run_reward_batch.py`.
- **LLM routing**: `llm_desparsifier/rewards/llm_client.py` (`base_url`, `model_alias`, `temperature`, `max_completion_tokens`).
- **Reward sanitizer**: `llm_desparsifier/rewards/sanitizer.py` (allowed ops and structure checks).

## Logging (minimal, high-signal)
- `WANDB_DISABLED=1` skips W&B; otherwise project/name are set in `scripts/run_reward_batch.py`.
- Each run logs:
  - `gepa/solve_rate` and `gepa/solve_rate_mean`: per-example solve rate for the current metric call (both set to the same value).
  - `gepa/sparse_baseline_solve_rate_mean`: global mean sparse baseline solve rate across the env grid (logged once at step 0).
  - `gepa/sparse_baseline_solve_rate`: per-env sparse baseline solve rate (logged once at step 0 alongside `gepa/example_id` and `gepa/env_id`).
  - `gepa/candidates` table: `step`, `env_id`, `solve_rate`, `sparse_baseline` (per-env baseline for that example), `reward_code_sha16`, `prompt_text`, `feedback`, `run_dir`.
  - `gepa/io_table` (per metric call): `metric_call`, `score`, `feedback`, `prompt_text`.
  - Artifacts under `STATE_ROOT/gepa_runs/` plus `active_prompt.json` and `gepa_stats.json`.
