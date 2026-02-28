# llm-desparsifier - GEPA On-Policy Loop

This project runs DSPy GEPA end-to-end on-policy to synthesize dense rewards for XLand-MiniGrid. The main loop proposes reward prompts, runs the RL pipeline for each proposal, and feeds solve-rate scores plus EUREKA-style reflection text back into GEPA. LLM calls can be routed through the Portkey gateway or sent directly to Gemini depending on `--llm-provider`.

## DSPy GEPA contract (top level)
- **Program under optimization**: `PromptOnlyProgram` in `scripts/run_reward_batch.py`. Its `forward(env_description, constraints=None)`:
  - Uses `PromptGenerator` (`dspy.Predict(PromptSearch)`) to rewrite the current constraints block unless GEPA passes `constraints` explicitly.
  - Returns `dspy.Prediction(prompt_text=...)` only (no reward-code predictor in the program graph).
- **Optimizer**: `dspy.GEPA(metric=on_policy_metric, max_metric_calls, reflection_lm, reflection_minibatch_size=1, track_stats=True, num_threads=1)` created in `run_batch()` where `max_metric_calls = max_gepa_iterations * num_jobs`.
- **Metric contract**: `metric(gold: Example, pred: Prediction, trace=None, pred_name=None, pred_trace=None) -> float | ScoreWithFeedback`. `ScoreWithFeedback` contains a numeric `score` and free-form `feedback` text. Higher scores are better and are expected to be in `[0, 1]`.
- **How GEPA uses it here**: `on_policy_metric` runs a full RL training + evaluation loop per *example* and returns `ScoreWithFeedback(score=solve_rate, feedback=reflection_text)`. Feedback requests for predictors reuse the cached `solve_rate` for that example+prompt so the feedback score matches the module score.

## Inputs and state
- **Env grid -> DSPy examples**: `configs/gepa_envs.yaml` is parsed into training `jobs` plus holdout `eval_jobs` (`env_id`, `benchmark_id`, `total_timesteps`, `train_seed`, `eval_seed`). `build_examples()` converts each training job into a `dspy.Example(env_description="<env_id> | benchmark=<id>")`, attaches `example.job_config`, and sets `example.job_name`.
- **Holdout envs**: `eval_jobs` from `configs/gepa_envs.yaml` drive the post-GEPA holdout evaluation (skipped when `--test-single-env` is enabled). If `eval_jobs` is missing, the runner falls back to the default list in `scripts/run_reward_batch.py`. Each holdout environment is evaluated with multiple reward generations (default 5) at the reward LLM temperature, and the reported solve rates aggregate mean and standard deviation across tries.
- **Prompt state**: `STATE_ROOT/active_prompt.json` (default `artifacts/gepa_state/`) stores:
  - `constraints_text` (the base constraints block used as the rewrite starting point),
  - `prompt_state` (DSPy weights for `PromptGenerator`),
  - `updated_at`, `source` metadata.
  The runner loads `active_prompt.json` if present; otherwise it falls back to `llm_desparsifier/rewards/prompts/base_reward_prompt.txt`, else `CONSTRAINTS_TEXT`. After GEPA completes it overwrites `active_prompt.json` with the updated `prompt_state` but keeps the base `constraints_text`.
- **Best prompt text**: the optimized prompt text is saved to `STATE_ROOT/<model_alias>.txt` via `save_best_prompt_text`.
- **Run artifacts**:
  - `STATE_ROOT/gepa_runs/candidate-####-<job>`: per-candidate training outputs, reward code, metrics, W&B run dir.
  - `STATE_ROOT/gepa_runs/candidate-####-<job>/eval_trajectory.json`: compact eval trajectory (ruleset/reset keys + action sequence, plus `env_seed` and human-readable `env_text`) for deterministic replay and goal inspection; if the main eval produces no trajectory, a single-episode fallback capture is run.
  - `STATE_ROOT/gepa_runs/candidate-####-<job>/training_curve_dense_vs_sparse.png`: dense return curve normalized to [-1, 1] alongside the ground-truth sparse return curve scaled by `eval_num_episodes` to a [0, 1] success rate over training.
  - `STATE_ROOT/gepa_runs/sparse_baseline/<job>`: sparse baseline runs (cached).
  - `STATE_ROOT/gepa_runs/holdout-dense/<job>/try-##`: holdout dense evaluations (one subdir per reward generation).
  - `STATE_ROOT/gepa_runs/holdout_reward_functions.jsonl`: per-try reward code and solve rates emitted for holdout runs.
- `STATE_ROOT/gepa_runs/training_curve.png`: solve-rate series across GEPA iterations.
- `STATE_ROOT/gepa_runs/training_curve_noise.png`: GEPA solve-rate trend with a rolling mean/std noise band (helps detect when task randomization variance dominates optimization progress).
- `STATE_ROOT/gepa_runs/holdout_solve_rates_by_env.png`, `STATE_ROOT/gepa_runs/holdout_solve_rate_aggregate.png`: holdout plots with mean +/- std error bars and the try count (if matplotlib available).
- `STATE_ROOT/gepa_runs/last_stage.json`: heartbeat snapshot of the most recent GEPA stage/event (useful for diagnosing stalls).
- `sparse_baseline.json`: cached sparse baseline summary (repo root).
- `STATE_ROOT/active_prompt.json`: latest prompt state after GEPA completes.
  - `STATE_ROOT/gepa_runs/gepa_stats.json`: GEPA optimizer stats plus sparse baseline and holdout summaries.

## Candidate evaluation flow (per GEPA proposal)
- **Budget clamp**: `clamp_job_budget()` caps per-candidate cost (`total_timesteps <= 170M`, `num_envs <= 2048`, `eval_num_envs <= 128`, `eval_num_episodes <= 20`) and records any reductions in `budget_notes`. Missing keys are defaulted to the caps.
- **Seeds**: by default the env grid provides `train_seed` and `eval_seed`. If missing, seeds are derived deterministically from `job.name` + `prompt_text` (train seed) and `train_seed + 1` (eval seed). `--test-single-env` uses a fixed single-env configuration (`XLand-MiniGrid-R1-11x11`) with locked train/eval seeds (`1000/1000`). Reward-code hashes/logs are derived from the emitted code used in training.
- **Reward synthesis**: `RewardGenerator.generate()` builds the env description via `describe_ruleset`, calls the DSPy reward LLM using the GEPA-optimized `prompt_text`, sanitizes/compiles with `sanitize_and_compile`, and retries with sanitizer feedback (plus sanitizer source on retries). In `run_reward_batch.py` the max sanitize attempts is set to 5.
- **Training**: `run_training_with_reward()` launches PPO with an RNN policy (`TrainConfig` defaults) and installs the dense reward via `DesparsifyRewardWrapper`. Dense rewards are used only for training; evaluation uses sparse rewards.
- **Ground-truth evaluation**: `run_ground_truth_eval()` executes `eval_num_episodes` sparse episodes with the trained policy and returns per-episode returns/lengths. Success counts are computed in `run_training_with_reward()`.

## Metric used by GEPA
- **Environment source**: envs come from XLand-MiniGrid benchmarks (`env_id` / `benchmark_id` in the grid). GEPA candidate evaluation now samples one benchmark ruleset per `(candidate prompt, env job)` and reuses it for reward synthesis, PPO training, sparse eval, and replay capture so task randomization is reintroduced without reward/task mismatch.
- **Success criterion**: an eval episode is "solved" iff the ground-truth sparse return is `> 0`.
- **Score fed to GEPA**: `solve_rate = successes / eval_num_episodes`. If eval returns are missing, the metric falls back to `success_rate` when available, else the last sparse curve point from training; if everything fails, score defaults to `0.0`.

## Feedback channel to GEPA
- The metric returns `ScoreWithFeedback(score=solve_rate, feedback=<reflection text>)` so GEPA can optimize with both numbers and natural-language guidance.
- If GEPA requests predictor-specific feedback (`pred_name`/`pred_trace`), the feedback text is prefixed with a short predictor header and the score is reused from the cached per-example evaluation.
- `build_reward_reflection()` composes the reflection input from:
  - **Env summary**: full `describe_ruleset` text when available, else `env_id | benchmark=<id>`.
  - **Reward code**: sanitized `dense_reward` source string.
  - **Sparse curve**: 6 checkpoints sampled from `eval/ground_truth_returns_mean` (the final point is the last sample).
  - **Per-component curves + stats**: checkpoints for each `reward_components` series plus min/mean/max for each component.
  - **Object-key alignment diagnostics**: compares object keys referenced by reward code lookups (`object_positions` / `visible_object_positions` / `visible_object_positions_prev`) against quoted task-object keys in the captured eval trajectory `env_text` when available (fallback: reward-generation env description); mismatches are called out explicitly because they can force shaping terms to stay near zero.
  - **Trajectory behavior summary**: compact action diagnostics from `eval_trajectory.json` (action histogram, manipulation/churn rates, turn oscillation, repeat loops, and short suspicious event sketches). This is additive and does not replace existing sparse/component/metric feedback.
  - **Final metrics**: sorted dump of `TrainingResult.final_metrics` (includes `solve_rate`, `eval_successes`, etc.).
  - **Guidance**: `EUREKA_GUIDANCE` plus a reminder to give environment-aware but non-hard-coded advice.
- Failure path: on training/sanitization errors, feedback becomes `Training failed: <error>` plus sanitizer retry history (if available), and `score = 0.0`.

## What the synthesis LLM sees
- `describe_ruleset(env, env_params)` assembles the environment prompt: layout hint, size/view/max_steps, action set, goal sentences, initial objects summary, and an explicit observation interface. The prompt now documents both symbolic-first visibility lookups (`ctx.get("visible_object_positions", {})`) and raw observation fallback (`ctx.get("observation", ts_next.observation)` with `obs[..., 0]=tile_id`, `obs[..., 1]=color_id`). It does not include a RULES list; if a ruleset summary can be printed, it is used only to extract goal/object names.
- Reward synthesis now uses the same benchmark ruleset that PPO/eval will use for that candidate/job. GEPA metric evaluation samples one ruleset seed per `(candidate prompt, env job)`, freezes it within the metric call, and passes it through reward synthesis, PPO training, sparse eval, and replay metadata.
- The full constraints block provided to the LLM comes from `constraints_text` (loaded from `active_prompt.json`, then `base_reward_prompt.txt`, then `CONSTRAINTS_TEXT`). GEPA rewrites this full block; there is no immutable suffix automatically appended after rewrite.

## Dense reward constraints (sanitizer enforced)
- Output must define **exactly one** `dense_reward` function.
- Allowed operations: JAX primitives (`jnp.*`, `jax.lax.*`), `float`/`int` casts, and helper functions defined inside `dense_reward`; method calls are restricted to `.astype(...)` and `ctx.get(...)`.
- `reward_components` must be a **dict literal** with constant string keys, and the return must be `(total_reward, reward_components)`.
- `ctx` access must use `ctx.get(key, fallback)` at the top level. Direct `ctx[...]` access is rejected; for nested maps, retrieve the map via `ctx.get("object_positions", default_dict)` / `ctx.get("visible_object_positions", default_dict)` and then use bracket or `.get(...)` access on the returned dict.
- The sanitizer strips markdown fences and rejects non-JAX imports (only `import jax`, `import jax.numpy as jnp`, `import jax.lax` are tolerated) or additional top-level definitions.

## Environment wrapper behavior
- `DesparsifyRewardWrapper` replaces the env reward with the dense reward and preserves the original reward in `extras["ground_truth_reward"]`.
- It also logs `extras["dense_reward"]` and `extras["reward_components"]` (keys must remain constant; violations raise errors). If component keys are known but missing, it fills a zero template.
- Dense reward functions can accept either `(ts_prev, action, ts_next)` or `(env_params, ts_prev, action, ts_next, ctx)`; the wrapper detects the signature at runtime. `ctx` comes from the configured `ctx_fn` when available.

## Artifacts and layout
- `STATE_ROOT/gepa_runs/candidate-####-<job>`: per-candidate outputs, emitted reward code, metrics, W&B run dir.
- `STATE_ROOT/gepa_runs/candidate-####-<job>/eval_trajectory.json`: compact eval trajectory (ruleset/reset keys + action sequence, plus `env_seed` and `env_text`) for deterministic replay and human-readable task context; if the main eval produces no trajectory, a single-episode fallback capture is run.
- `STATE_ROOT/gepa_runs/candidate-####-<job>/training_curve_dense_vs_sparse.png`: dense return curve normalized to [-1, 1] alongside the ground-truth sparse return curve scaled by `eval_num_episodes` to a [0, 1] success rate over training.
- `STATE_ROOT/gepa_runs/sparse_baseline/<job>`: sparse baseline outputs.
- `STATE_ROOT/gepa_runs/holdout-dense/<job>/try-##`: holdout dense outputs (one subdir per reward generation).
- `STATE_ROOT/gepa_runs/holdout_reward_functions.jsonl`: per-try reward code and solve rates emitted for holdouts.
- `STATE_ROOT/gepa_runs/training_curve.png`: solve-rate time series across GEPA iterations.
- `STATE_ROOT/gepa_runs/training_curve_noise.png`: solve-rate trend with a rolling mean/std noise band for diagnosing GEPA variance under candidate-scoped task randomization.
- `STATE_ROOT/gepa_runs/holdout_solve_rates_by_env.png`, `STATE_ROOT/gepa_runs/holdout_solve_rate_aggregate.png`: holdout plots with mean +/- std error bars and the try count.
- `STATE_ROOT/gepa_runs/last_stage.json`: heartbeat snapshot of the most recent GEPA stage/event.
- `sparse_baseline.json`: cached sparse baseline summary (repo root).
- `STATE_ROOT/active_prompt.json`: latest prompt state after GEPA completes.
- `STATE_ROOT/<model_alias>.txt`: best prompt text for the selected LLM.
- `STATE_ROOT/gepa_runs/gepa_stats.json`: GEPA stats plus sparse baseline and holdout summary.

## Running the loop
- Cluster: `sbatch sbatch/train_dense_batch.s` (sets caches, state root, syncs deps, runs `scripts/run_reward_batch.py`).
- Local (Portkey): `uv run scripts/run_reward_batch.py --state-root artifacts/gepa_state`
- Local (Gemini): `uv run scripts/run_reward_batch.py --state-root artifacts/gepa_state --llm-provider gemini --llm gemini-3-pro-preview`
- Room-filtered train+holdout (R1 only): `uv run scripts/run_reward_batch.py --state-root artifacts/gepa_state --room-count 1`
- Room-filtered train+holdout (R1 and R4): `uv run scripts/run_reward_batch.py --state-root artifacts/gepa_state --room-count 1 --room-count 4`
- Useful flags:
  - `--max-gepa-iterations` (default 50)
  - `--llm` (Portkey model alias or Gemini model name)
  - `--llm-provider` (`portkey` or `gemini`)
  - `--reward-llm-temp` (default 0.1)
  - `--reflection-llm-temp` (default 0.5)
  - `--room-count` (repeatable; filter both training and holdout envs by room count parsed from `env_id`, e.g. `--room-count 1 --room-count 4`; runner fails fast if filtering empties either set)
  - `--test-single-env` (single env overfit run)
  - `--deterministic-envs` (use fixed benchmark rulesets instead of the default candidate-scoped random sampled rulesets)

Required env vars:
- `PORTKEY_API_KEY` (loaded via `.env` if present). Required when `--llm-provider=portkey`.
- `GEMINI_API_KEY` (loaded via `.env` if present). Required when `--llm-provider=gemini`.

Optional env vars:
- `WANDB_DISABLED=1` to skip W&B logging.
- `WANDB_PROJECT` to set the W&B project name.
- `XLAND_MINIGRID_DATA` to override the XLand data cache location.

## Generate a training video
- After a GEPA run finishes, use `scripts/generate_training_video.py` to replay the saved eval trajectory, render an enlarged map viewport, place dense-reward diagnostics in a right-side panel, add a per-component instantaneous-reward line plot over timesteps, and write an MP4.
- Default: `uv run scripts/generate_training_video.py --state-root artifacts/gepa_state`
- Specific run directory: `uv run scripts/generate_training_video.py --run-dir artifacts/gepa_state/gepa_runs/candidate-0001-job-0`
- If CUDA backend initialization fails (for example GPU OOM), the script automatically retries once on CPU by re-executing with `JAX_PLATFORMS=cpu`.
- Outputs: `training_video.mp4` and `training_video_trace.json` in the run directory; the trace JSON mirrors the trajectory, includes `env_seed`/`env_text`, records `reward_object_key_diagnostics` (to flag reward/task object mismatches), and records `replay_complete` plus `replay_error` so partial diagnostics are preserved even if replay fails.
- Replay metadata also records deterministic task provenance (`fixed_ruleset_seed` when candidate-scoped randomization is used) so the exact sampled benchmark task can be reconstructed.

## Play a level manually for reward debugging
- Use `scripts/play_level.py` to load the same replay artifacts as video generation and step the environment with keyboard input while keeping the same right-side diagnostics panel (goal text, dense/sparse totals, per-component totals, and per-component reward trend plot).
- Default launch (latest eligible run): `uv run scripts/play_level.py --state-root artifacts/gepa_state`
- Specific run directory: `uv run scripts/play_level.py --run-dir artifacts/gepa_state/gepa_runs/candidate-0001-job-0`
- Deterministic initialization: run-dir mode starts from the exact saved `eval_trajectory.json` initial state (`ruleset_index`/`ruleset_key` + `reset_key`) so human and policy behavior can be compared on the same level instance.
- Human-play overlay now prioritizes objective clarity with labeled `OBJECTIVE` and `WIN CONDITION` lines derived from the reconstructed live ruleset (with trajectory `env_text` as fallback), plus an on-screen `CONTROLS` legend. This display is human-play-only and is not added to policy rollout videos.
- Key bindings:
  - `Up` or `W`: move_forward
  - `Left` or `A`: turn_left
  - `Right` or `D`: turn_right
  - `Tab` or `E`: pick_up
  - `Left Shift` or `Q`: put_down
  - `Space`: toggle
  - `R`: reset back to the same saved initial state
  - `Esc` or closing the window: quit
- Trace output: `play_level_trace.json` is always written (including failure paths) and stores full per-step action/reward diagnostics so you can inspect whether intended dense-reward components fired during human play.

## Behavioral notes
- The GEPA score is scale-free; multiplying a bad dense reward by 100 cannot improve solve rate.
- GEPA uses on-policy evaluation only; there is no offline dataset cache.
- Per-env and per-episode details remain in logs/W&B for debugging, but GEPA optimizes only the mean solve rate over the env grid.

## Hyperparameters and where to change them
- **Per-env budgets**: `configs/gepa_envs.yaml` `jobs` list (`total_timesteps`, `train_seed`, `eval_seed`, etc.) controls the training set, while `eval_jobs` controls the holdout evaluation set. Any value above the caps will be clamped in `clamp_job_budget`; lower values are respected.
- **Global caps**: `MAX_TOTAL_TIMESTEPS` (170M), `MAX_NUM_ENVS`, `MAX_EVAL_ENVS`, `MAX_EVAL_EPISODES` near the top of `scripts/run_reward_batch.py`.
- **RL training defaults**: `llm_desparsifier/rl/pipeline.py:TrainConfig` (policy sizes, PPO knobs, eval counts, seeds). Override by adding keys to a job entry in the env grid; GEPA passes them through to `TrainConfig(**config_override)`.
- **GEPA search budget**: `--max-gepa-iterations` (default 50); each iteration evaluates one prompt across the full training job list.
- **LLM routing**: `llm_desparsifier/rewards/llm_client.py` (`configure_portkey_lm` vs `configure_gemini_lm`, plus `model_alias` or `model_name`).
- **Holdout reward tries**: `HOLDOUT_REWARD_TRIES` in `scripts/run_reward_batch.py` controls how many reward generations are evaluated per holdout environment (used for mean +/- std plots).
- **Reward generator retries**: `RewardGenerator(max_sanitize_attempts=5)` in `scripts/run_reward_batch.py`.
- **Reward sanitizer**: `llm_desparsifier/rewards/sanitizer.py` (allowed ops and structure checks).

## Logging (minimal, high-signal)
- `WANDB_DISABLED=1` skips W&B; otherwise project/name are set in `scripts/run_reward_batch.py`.
- Each run logs:
  - `gepa/solve_rate` once per GEPA iteration (mean over the training jobs).
  - `gepa/sparse_baseline_solve_rate` once per GEPA iteration (mean sparse baseline).
  - `gepa/iteration` as the iteration index used for the W&B step.
  - `gepa/sparse_baseline_solve_rate` per-env at step 0 (logged alongside `gepa/example_id` and `gepa/env_id`).
  - `gepa/sparse_baseline_solve_rate_mean` at step 0 (overall baseline mean).
  - `gepa/rl_runs` table with columns: `rl_run_id`, `env_id`, `env_text`, `prompt_text`, `reward_code_sha16`, `reward_code`, `solve_rate`, `sparse_baseline`, `ruleset_seed`, `env_text_sha16`, `goal_signature`, `reward_task_mismatch`, `reward_missing_keys`, `feedback`, `sanitizer_feedback`, `run_dir`.
  - `compare/*` metrics in `--test-single-env` mode.
  - Artifacts under `STATE_ROOT/gepa_runs/` plus `active_prompt.json`, `gepa_stats.json`, always-on plot PNGs (including `training_curve_noise.png`), and the best prompt `.txt` file.
