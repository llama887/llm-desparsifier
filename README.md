# llm-desparsifier - GEPA Heuristic Search

This repository uses DSPy GEPA to optimize prompts that cause an LLM to emit admissible-leaning A* heuristics for XLand-MiniGrid tasks. Each candidate prompt is evaluated by synthesizing heuristic code, validating it, running deterministic A* over a grid of environments and seeds, and feeding a scalar search-quality score plus heuristic-specific feedback back into GEPA.

## Main entrypoint
- Run the search-only pipeline with `uv run scripts/run_heuristic_batch.py --state-root artifacts/gepa_state`
- Before running GEPA, calibrate every job budget with `uv run scripts/calibrate_astar_budgets.py --env-grid configs/gepa_envs.yaml --write`
- Core switches:
  - `--env-grid configs/gepa_envs.yaml`
  - `--max-phase-iterations <int>` (default `10`)
  - `--llm <gemini-model-name>`
  - `--astar-max-nodes <int>`
  - `--astar-max-expansions <int>`
  - `--deterministic-envs`
  - `--room-count <int>` (repeatable)
- W&B logging defaults to project `llm-astar` unless `WANDB_DISABLED=1` is set

## Calibrating A* Budgets
- Budget calibration is a required methods step for this repo's supported heuristic-search workflow.
- Run `uv run scripts/calibrate_astar_budgets.py --env-grid configs/gepa_envs.yaml --write` before starting GEPA on a new grid or after changing its jobs.
- The calibration script runs blind A* (`h=0`) on each job's explicit `holdout_seeds`, not on GEPA's sampled training seeds.
- When a job sets `deterministic_rulesets: false`, those holdout seeds also select distinct benchmark rulesets, so calibration covers both reset randomness and task-semantic variation.
- Each calibration seed gets up to 5 minutes of wall-clock time.
- If blind A* solves within that window, the script sets:
  - `astar_max_nodes = floor(generated_states * 0.95)`
  - `astar_max_expansions = floor(expanded_states * 0.95)`
- If blind A* does not solve within that window, the script keeps the exact observed `generated_states` and `expanded_states`.
- The final job-level `astar_max_nodes` and `astar_max_expansions` values are the worst calibrated counts across that job's holdout seeds.
- This is intentional: the budgets should sit just outside the range of blind or weak heuristics, so the optimizer must synthesize genuinely strong heuristics to solve the levels.

## Curriculum schedule
- GEPA no longer trains on all 11 training environments at once.
- The default curriculum uses three coarse cumulative phases over the ordered training jobs: `3`, `7`, and `11`.
- The curriculum order is exactly the training-job order from the YAML after any runtime filtering such as `--room-count`.
- Each phase now owns one persistent GEPA `log_dir` under `STATE_ROOT/heuristic_runs/phase-##-gepa/`.
- Repeated outer-loop iterations within the same phase resume that phase-local GEPA run with a larger cumulative `max_metric_calls` budget instead of resetting the optimizer archive.
- Advancing to the next phase intentionally starts a fresh GEPA archive seeded by the best prompt from the previous phase, because the active task set has changed.
- A non-final phase advances only when the active-phase mean per-job solve rate reaches `>= 0.80`.
- A phase baseline is defined by the first GEPA iteration in that phase.
- Each later phase iteration must set a strictly higher phase best GEPA score to reset patience.
- A non-final phase stops the whole run if it fails to improve the active-phase best score for 3 consecutive iterations before reaching the `0.80` solve-rate gate.
- Once the final phase is reached, the runner ignores patience and keeps iterating on the full training set until `--max-phase-iterations` is reached for that phase.

## End-Of-Run Holdout Report
- After training completes, the runner evaluates three holdout policies on the YAML `eval_jobs`:
  - the best optimized prompt from GEPA
  - the original base heuristic prompt
  - blind A* with no heuristic (`h=0`)
- These comparisons are written into `STATE_ROOT/heuristic_runs/gepa_stats.json` under `holdout_comparisons`.
- The runner also writes two matplotlib bar charts under `STATE_ROOT/heuristic_runs/`:
  - `holdout_comparison_aggregate.png` for the aggregate holdout solve-rate comparison
  - `holdout_comparison_by_env.png` for per-environment holdout solve-rate comparisons

## GEPA contract
- Program under optimization: `PromptOnlyProgram` in [scripts/run_heuristic_batch.py](/home/jupyter-franklinyiu/llm-desparsifier/scripts/run_heuristic_batch.py)
- Optimized artifact: prompt text consumed by `HeuristicGenerator`
- Metric output: `ScoreWithFeedback(score=<job_score>, feedback=<deterministic heuristic feedback>)`
- Training metric: one synthesized heuristic is reused across fresh sampled seeds for each active-phase job
- With the default grid, those sampled seeds randomize both reset state and benchmark ruleset selection, so the curriculum now broadens task semantics as well as map family and size.

Per-seed score:

```text
N = astar_max_expansions
S = expanded_states if solved else (N + 1)
seed_score = ((N + 1) - S) / (N + 1)
```

Job score is the mean seed score over sampled seeds. GEPA averages job scores over the training jobs.

## Synthesized heuristic contract
- Function signature: `def heuristic_cost_to_go(ts, env_params, ctx) -> float`
- Return a finite non-negative lower bound on remaining sparse path cost
- Return `0.0` on solved states
- Use the full-information `ctx` mapping, not cropped observations or rollout history
- `ctx` includes:
  - `env_id`
  - `benchmark_id`
  - `ruleset_text`
  - `grid_shape`
  - `action_names`
  - `step_cost`
  - `goal_description`
  - `agent_state` with `position` plus legacy alias `pos`, and also `direction` / `carrying`
  - `object_positions`
  - `object_metadata`
  - `static_walls`
  - `task_features`

## Environment grid schema
- `configs/gepa_envs.yaml` now uses search-native jobs:
  - `env_id`
  - `benchmark_id`
  - `num_gepa_eval_seeds`
  - `holdout_seeds`
  - `deterministic_rulesets`
  - Omit the field or set it to `false` to sample a fresh benchmark ruleset from each evaluation seed. Set it to `true` only when you intentionally want all seeds for that job to share one fixed canonical ruleset.
  - `fixed_ruleset_seed` (optional)
  - `astar_max_nodes` as a calibrated blind-A* node cap, not an arbitrary constant
  - `astar_max_expansions` as a calibrated blind-A* expansion cap, not an arbitrary constant

## Artifacts
- Prompt state remains in `STATE_ROOT/active_prompt.json`
- Best prompt text is written to `STATE_ROOT/<model>.txt`
- Candidate outputs live under `STATE_ROOT/heuristic_runs/candidate-####-<job>/`
- Phase-local GEPA checkpoints live under `STATE_ROOT/heuristic_runs/phase-##-gepa/`
- `STATE_ROOT/active_prompt.json` now also stores curriculum checkpoint data:
  - current phase
  - cumulative phase job counts
  - completed phases
  - per-phase baselines and best scores
  - phase stop reasons
  - per-phase GEPA checkpoint directories
  - `max_phase_iterations`
- `STATE_ROOT/heuristic_runs/gepa_stats.json` now records the full curriculum history:
  - ordered training jobs
  - phase-by-phase iteration summaries
  - final stop reason
  - final holdout summary
- Key files per candidate:
  - `heuristic_synthesized.py`
  - `heuristic_validation.json`
  - `astar_search_stats.json`
  - `astar_plan.json`
  - `astar_trace.json`
  - `task_instance.json`
  - `feedback.txt`
- Holdout outputs live under `STATE_ROOT/heuristic_runs/holdout-heuristic/<job>/`

## Replay
- The search pipeline writes replay-first artifacts directly.
- `task_instance.json` stores the deterministic task materialization metadata.
- `astar_plan.json` stores the chosen action sequence.
- `astar_trace.json` stores compact search diagnostics for overlays and debugging.

## JAxtar note
- The repository records the upstream JAxtar source revision in [pyproject.toml](/home/jupyter-franklinyiu/llm-desparsifier/pyproject.toml).
- The current XLand integration uses a repo-local compatibility backend with the same search-oriented boundary while the repo-specific XLand adapter remains internal.
