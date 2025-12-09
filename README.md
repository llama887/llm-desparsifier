# llm-desparsifier – GEPA On-Policy Loop

This project runs DSPy GEPA end-to-end on-policy to synthesize dense rewards for XLand-MiniGrid. The README explains the GEPA contract, what goes into the metric, and how candidate prompts are trained and evaluated.

## DSPy GEPA contract (top level)
- Program: `RewardPromptProgram(constraints_text, synthesizer_state, prompt_state)`
- Optimizer: `dspy.GEPA(metric=on_policy_metric, auto|budget args…)`
- Metric signature: `on_policy_metric(example, prediction, …) -> ScoreWithFeedback(score: float, feedback: str)`
- GEPA supplies a `prediction.reward_code` (dense reward) and optional `prediction.prompt_text`; the metric must return a scalar score plus textual feedback. The optimizer averages the returned scores across examples (the env grid) and drives prompt mutations.

## Inputs and state
- **Env grid**: `configs/gepa_envs.yaml` lists jobs (env_id, benchmark_id, train/eval seeds, timesteps). These become DSPy examples.
- **Prompt state**: `STATE_ROOT/active_prompt.json` (defaults to `artifacts/gepa_state/`) holds `constraints_text`, synthesizer dump, and prompt-generator state. The job always loads the latest copy and overwrites it when GEPA finishes.
- **Caches / data dirs** (auto-set in sbatch): `XLAND_MINIGRID_DATA`, `XDG_CACHE_HOME`, W&B dirs.
- **Entrypoints**: `scripts/run_reward_batch.py` (GEPA loop), `sbatch/train_dense_batch.s` (cluster launcher).

## Candidate evaluation flow (per GEPA proposal)
- Budget clamp: limit total_timesteps/num_envs/eval episodes to configurable caps.
- Deterministic seeding: `derive_seed(example_id, candidate_fp)`; eval seed and budget are part of the fingerprint to avoid cache collisions.
- Reward synthesis: `RewardGenerator.generate` emits a dense reward function; wrapped via `DesparsifyRewardWrapper`.
- Training: `run_training_with_reward` (PPO/RNN) on the specified env/rulesets; uses the synthesized dense reward for training, but always keeps ground-truth sparse reward for evaluation.
- Ground-truth evaluation: `run_ground_truth_eval` runs `eval_num_episodes` sparse episodes; records per-episode returns/lengths.

## Metric used by GEPA
- **Success criterion**: an eval episode is successful if its ground-truth return > 0 (goal predicate satisfied before timeout).
- **Solve rate**: `successes / eval_num_episodes` for that env. This is the scalar fed to GEPA as `score` (bounded in [0,1], insensitive to reward scaling).
- Stored in `MetricCacheEntry.score`, logged to W&B (`gepa/solve_rate`), and printed as the candidate score.

## Feedback channel to GEPA
- `build_reward_reflection` consumes the training result (including sparse curve and ground-truth eval stats) and produces textual feedback returned alongside the score. GEPA uses this to guide prompt mutations.

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
