# llm-desparsifier – GEPA On-Policy Loop

This project runs DSPy GEPA end-to-end on-policy to synthesize dense rewards for XLand-MiniGrid. The README explains the GEPA contract, what goes into the metric, and how candidate prompts are trained and evaluated.

## DSPy GEPA contract (top level)
- **Program under optimization**: `RewardPromptProgram(constraints_text, synthesizer_state, prompt_state)` in `scripts/run_reward_batch.py`. Its `forward(env_description, constraints=None)`:
  - Picks or rewrites a prompt (`prompt_text`) via the nested `PromptGenerator` (`dspy.Predict(PromptSearch)`) unless `constraints` is passed through from GEPA.
  - Calls `RewardSynthesizer` (`dspy.Predict(RewardSynthesis)`) to emit `prediction.reward_code`.
  - Returns `dspy.Prediction(reward_code=..., prompt_text=...)`. GEPA mutates only `prompt_text`; the base constraints appended by `parser.CONSTRAINTS_TEXT` stay fixed.
- **Optimizer**: `dspy.GEPA(metric=on_policy_metric, auto | max_full_evals | max_metric_calls, reflection_minibatch_size, use_merge, num_threads, …)` instantiated in `run_reward_batch()`.
- **Metric contract**: DSPy expects a callable `metric(gold: Example, pred: Prediction, trace=None, pred_name=None, pred_trace=None) -> float | ScoreWithFeedback`. A `ScoreWithFeedback` must contain a numeric `score` and free-form `feedback`; GEPA consumes the text to steer mutations. Higher scores are better and should be bounded in `[0,1]`. citeturn0search0
- **How GEPA uses it here**: GEPA passes the env example as `gold` and the synthesized reward/prompt as `pred`. `on_policy_metric` executes a full RL run, then returns `ScoreWithFeedback(score=solve_rate, feedback=reflection_text)`. GEPA averages scores across the current env grid batch to decide which prompt rewrites survive. citeturn0search0

## Inputs and state
- **Env grid → DSPy examples**: `configs/gepa_envs.yaml` is parsed into `EnvJob` objects (`env_id`, `benchmark_id`, `total_timesteps`, `train_seed`, `eval_seed`). `build_examples()` turns each job into `dspy.Example(env_description="<env_id> | benchmark=<id>")` and attaches `example.job_config` (used by the metric) plus `example.job_name` (used in fingerprints and logging). Changing this YAML changes which environments GEPA optimizes over.
- **Prompt state**: `STATE_ROOT/active_prompt.json` (default `artifacts/gepa_state/`) holds:
  - `constraints_text` (the latest combined prompt),
  - `synthesizer_state` (DSPy weights for `RewardSynthesizer`),
  - `prompt_state` (DSPy weights for `PromptGenerator`).
  The runner loads this file if present; otherwise it falls back to `llm_desparsifier/rewards/prompts/base_reward_prompt.txt`, else the hard-coded `CONSTRAINTS_TEXT`. After a GEPA session completes it overwrites `active_prompt.json` atomically via `write_active_prompt`.
- **Run artifacts & caches**:
  - SBATCH sets `XLAND_MINIGRID_DATA`, `XDG_CACHE_HOME`, `WANDB_DATA_DIR`, `WANDB_DIR` (see `sbatch/train_dense_batch.s`).
  - Metric cache (`metric_cache` in-memory) deduplicates by `(example_id, candidate_fingerprint)` so retries reuse score/feedback.
  - W&B run (if `WANDB_DISABLED` not set) logs candidate table and metrics.
- **Entrypoints**: `scripts/run_reward_batch.py` (local: `uv run ... --state-root <dir>`), cluster launcher `sbatch/train_dense_batch.s` (creates caches, syncs deps, runs the script).

## Candidate evaluation flow (per GEPA proposal)
- **Budget clamp**: `clamp_job_budget()` caps per-candidate cost (`total_timesteps ≤ 20M`, `num_envs ≤ 1024`, `eval_num_envs ≤ 128`, `eval_num_episodes ≤ 20`) and records any reductions in `budget_notes`.
- **Deterministic seeds**: `derive_seed(example_id, candidate_fingerprint)` → `train_seed` and `eval_seed=train_seed+1`. The fingerprint hashes reward code + env id + eval budget, so changing reward text or eval episodes forces a fresh run.
- **Reward synthesis**: `RewardGenerator.generate()` builds the env description via `describe_ruleset`, runs the DSPy reward LLM, sanitizes/compiles, retries with detailed sanitizer guidance on failure, and returns `(dense_fn, emitted_code)`.
- **Training**: `run_training_with_reward()` launches PPO with RNN policy (`TrainConfig` defaults) using the dense reward for training but **never** replaces sparse reward in evaluation logging.
- **Ground-truth evaluation**: `run_ground_truth_eval()` executes `eval_num_episodes` sparse episodes with the trained policy; logs per-episode return/length, success counts, videos, and CSV summaries. These feed both the metric score and the reflection text.

## Metric used by GEPA
- **Environment source**: envs come from XLand-MiniGrid benchmarks (`env_id` / `benchmark_id` in the grid). Each job is deterministic given seeds and the generated dense reward.
- **Success criterion**: an eval episode is “solved” iff the ground-truth sparse return `> 0` (goal satisfied before timeout). Returns are computed by the environment’s built-in sparse reward.
- **Score fed to GEPA**: `solve_rate = successes / eval_num_episodes` (clamped episodes). If eval returns are missing, the metric falls back to the last sparse curve point from training; if everything fails, score defaults to `0.0`.
- **Seeds & randomness**: `train_seed`/`eval_seed` are either loaded from the grid or deterministically derived; Python `random` and NumPy are also seeded for reproducibility in reward gen/training wrappers.
- **Storage**: `MetricCacheEntry.score` plus `feedback` and `sparse_curve` are cached and reused; W&B logs `gepa/solve_rate` and the candidate table row. The score is always in `[0,1]`, making it scale-free with respect to dense reward magnitude.

## Feedback channel to GEPA
- The metric returns `ScoreWithFeedback(score=solve_rate, feedback=<reflection text>)` so GEPA can optimize with both numbers and natural-language guidance. citeturn0search0
- `build_reward_reflection()` composes the feedback input to the reflection LLM from:
  - **Env summary**: full `describe_ruleset` text when available, else `env_id | benchmark=<id>`.
  - **Reward code**: sanitized `dense_reward` source string.
  - **Sparse curve**: 6 checkpoints sampled from `eval/ground_truth_returns_mean` plus the final value.
  - **Per-component curves + stats**: checkpoints for each `reward_components` series and min/mean/max per component.
  - **Final metrics**: sorted dump of `TrainingResult.final_metrics` (includes `solve_rate`, `eval_successes`, etc.).
  - **Run context**: truncated candidate prompt, budgets (`total_timesteps`, `num_envs`, `eval_num_envs`, `eval_num_episodes`, `max_steps`, `gt_success_threshold`), eval successes, sanitizer retry note if present.
  - **Guidance**: `EUREKA_GUIDANCE` plus a reminder to give environment-aware but non-hard-coded advice.
- Failure path: on training/sanitization errors, feedback becomes `Training failed: <error>\n\nSanitizer feedback:\n<retry table>` (when available) and `score = 0.0`; the run is still cached.
- **Example feedback snippet** (typical shape):
  ```
  Environment: XLand-MiniGrid-R4-13x13 (benchmark=trivial-1m)
  Sparse reward checkpoints: [0.00, 0.10, 0.35, 0.55, 0.60, 0.62] → final=0.620
  progress: [0.00, 0.08, 0.30, 0.44, 0.50, 0.52]
  penalty: [0.00, -0.12, -0.20, -0.18, -0.15, -0.14]
  Metrics: dense_return=0.730, ground_truth_return=0.620, eval_successes=12 / 20
  Suggestions: reward magnitude is dominated by penalty; rescale to ≤ |0.1|, add shaping for approaching goal object, and give completion bonus when GOAL predicate is met.
  ```

## What the synthesis LLM sees
- `describe_ruleset(env, env_params)` assembles the environment prompt: grid type → layout hint, size/view/max_steps, action set, GOAL line plus natural-language restatement (when ruleset text can be printed), truncated RULES list, INITIAL OBJECTS summary, and a reminder about partial observability and the `ctx` dict keys available to dense_reward.
- The constraints block appended after any GEPA mutations is `llm_desparsifier/rewards/parser.CONSTRAINTS_TEXT` (hard-coded JAX/ctx safety rules, expected return signature, allowed ops). GEPA only mutates the prefix (`prompt_text`) that precedes these constraints; constraints themselves stay immutable.
- Example synthesis prompt skeleton:
  ```
  XLand-MiniGrid-R2-11x11 | benchmark=trivial-1m
  grid_type=R2 → two rooms separated by an interior wall with one doorway
  size=11x11, view=5 (agent-centered egocentric 5×5 symbolic grid), max_steps=128.
  Actions: move_forward, turn_left, turn_right, pick_up, put_down, toggle (one object carried at a time).
  GOAL:
  SUCCESS when TileNearRightGoal(yellow_square, green_ball)
  SUCCESS when **green_ball** is immediately to the **left** of **yellow_square** …
  RULES:
  - … (truncated)
  INITIAL OBJECTS:
  yellow_square at ?, green_ball at ?, …
  Observations are partially observable…
  <prompt_text produced by GEPA mutations>
  <CONSTRAINTS_TEXT (immutable safety and API contract)>
  ```

## Artifacts and layout
- `logs/candidate-####-{example}`: per-candidate training outputs, emitted reward code, metrics, W&B run dir.
- `gepa_runs/gepa_stats.json`: optimizer stats from the last session.
- `STATE_ROOT/active_prompt.json`: updated prompt/synthesizer state after GEPA completes.

## Running the loop
- Cluster: `sbatch sbatch/train_dense_batch.s` (sets caches, state root, syncs deps, runs `scripts/run_reward_batch.py`).
- Local: `uv run scripts/run_reward_batch.py --state-root artifacts/gepa_state` (ensure `XLAND_MINIGRID_DATA` exists or let the script create it).

## Behavioral notes
- Metric is now scale-free; multiplying a bad dense reward by 100 cannot improve the score.
- Caching includes eval seed and episode budget; changing either forces a fresh evaluation.
- Per-env and per-episode details stay in logs/W&B for debugging, but GEPA optimizes only the pooled solve rate over the env grid.

## Hyperparameters and where to change them
- **Per-env budgets**: `configs/gepa_envs.yaml` (`total_timesteps`, `train_seed`, `eval_seed`, etc.). Any value above the caps will be clamped in `clamp_job_budget`; lower values are respected.
- **Global caps**: edit `MAX_TOTAL_TIMESTEPS`, `MAX_NUM_ENVS`, `MAX_EVAL_ENVS`, `MAX_EVAL_EPISODES` near the top of `scripts/run_reward_batch.py`.
- **RL training defaults**: `llm_desparsifier/rl/pipeline.py:TrainConfig` (policy sizes, PPO knobs, eval counts, seeds). Override by adding keys to a job entry in the env grid; GEPA passes them through to `TrainConfig(**config_override)`.
- **GEPA search budget**: choose exactly one of `--gepa-auto {light,medium,heavy}`, `--max-full-evals`, or `--max-metric-calls` when running `scripts/run_reward_batch.py`. `--reflection-minibatch-size` controls how many envs are batched per feedback call; `--disable-merge` turns off GEPA’s candidate merging.
- **State location**: set `STATE_ROOT=/your/path` (env var for sbatch or flag locally) to isolate prompt/checkpoint state.
- **Logging**: `WANDB_DISABLED=1` skips W&B; otherwise project/name are set in `run_reward_batch.py`. Candidate artifacts live under `STATE_ROOT/gepa_runs/`.
