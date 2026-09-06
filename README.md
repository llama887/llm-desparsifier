# llm-desparsifier - GEPA Search Synthesis

This repository uses GEPA to optimize one **global prompt** that makes an LLM write the search
artifact for a grid puzzle it has never seen. A candidate prompt is evaluated by synthesizing code
per level, sanitizing and compiling it, running it under a bounded deterministic search boundary,
and feeding a scalar score plus textual feedback back into GEPA.

Two experiments share that machinery. They use the same runner, evaluator, sandbox, Slurm topology,
and artifact layout, and differ in **what the synthesized artifact is allowed to be** and **what the
objective rewards**:

| | Experiment 1: heuristic prompt | Experiment 2: search code |
| --- | --- | --- |
| synthesized artifact | `heuristic_cost_to_go(ts, env_params, ctx)` | that, **or** `search_plan(api, seed)` |
| search loop | fixed, validated A* | whatever the artifact implements, inside a bounded API |
| seed contract | `--seed-contract astar-heuristic` | `--seed-contract dual-route` (default) |
| objective | `--objective adjusted` (base-relative) | `--objective blind-relative-time` |
| domain | 44-game random PuzzleScript gallery | Sokoban-family PuzzleScript games |
| launcher | `sbatch/train_puzzlescript_gallery_gepa_codex_cpu.s` | `sbatch/train_sokoban_search_code_gepa_cpu.s` |
| holdout | `sbatch/compare_puzzlescript_holdout_gpu.s` | `CONFIG=full`, or `sbatch/resume_sokoban_search_code_holdout_cpu.s` |

The earlier XLand-MiniGrid pipeline is still present and is documented at the end of this file.

## Shared machinery

### Topology

`scripts/run_puzzlescript_batched_gepa.py` is the entrypoint for both experiments. It calls
standalone `gepa.optimize` through a custom adapter and does not use DSPy.

1. A controller job holds the synthesis backend and generates one artifact per active game level for
   the current candidate prompt, batched across levels with `--llm-concurrency`.
2. The controller writes an evaluation manifest and hands search to CPU Slurm workers.
3. Merged per-level scores and feedback return to GEPA, which reflects and proposes the next prompt.

Games and levels come from the sibling [script-doctor](https://github.com/smearle/script-doctor)
checkout (`data/scraped_games` and `custom_games`) and run on its C++ PuzzleScript engine. Point at
it with `--script-doctor ../script-doctor`.

Search is farmed out two ways:

- `--submit-search-array` submits `sbatch/evaluate_puzzlescript_search_array.s` once per evaluation
  batch. Each task reads `STATE_ROOT/candidate_evals/<eval>/search_manifest.json`, evaluates its
  level slice, and writes a shard next to it.
- `--search-pool-dir` reuses a persistent pool from `sbatch/evaluate_puzzlescript_search_pool.s`.
  Pool workers poll `current_manifest` and run the same array script without paying Slurm queue
  latency per batch. Both current experiments use the pool.

`--synthesis-replicates N` runs N independent generations plus searches per logical level and
averages them, which is what makes stochastic agentic synthesis comparable across candidates.
`--synthesis-cache-dir` reuses generations for an unchanged (prompt, level) pair across runs.

### Synthesis and reflection backends

- `--synthesis-backend openai` calls a local OpenAI-compatible endpoint.
  `sbatch/train_puzzlescript_batched_gepa_gpu.s` starts vLLM on an allocated H100 and waits for
  `/health` before launching the controller.
- `--synthesis-backend codex-cli` calls the Codex CLI. With `--synthesis-agentic` the model drafts
  and tests `candidate.py` in an isolated temporary workspace before returning it; the runtime API
  contract lives in that workspace rather than in the prompt, which keeps the seed prompt minimal
  and keeps GEPA optimizing strategy instead of restating the API.
- `--reflection-backend {same,codex-cli}` selects the model that writes the next prompt.
  `--reflection-artifact-tools` lets reflection browse run artifacts; the search-code experiment
  pins `--no-reflection-artifact-tools` so reflection cannot read holdout material.

By default GEPA rewrites a short addendum appended to a stable base prompt.
`--optimize-full-prompt` lets it replace the whole synthesis prompt; both experiments use that.

### The two synthesis routes

`sanitize_and_compile_puzzlescript_search` accepts exactly one entrypoint and reports which route it
was. Both routes are compiled into a namespace with no `__builtins__`, no imports, no private or
dunder attribute access, and no `exec`/`eval`/`open`/network, and both run under the same expansion,
wall-clock, memory, and replay boundaries.

**Heuristic route** - `def heuristic_cost_to_go(ts, env_params, ctx) -> float`, run inside the
repository's validated A* loop. Returns a finite non-negative estimate, `0.0` on solved states, and
reads only the full-information `ctx` built by `build_puzzlescript_ctx`:

- `game_title`, `grid_width`, `grid_height`
- `object_names`, `object_positions`
- `win_conditions`, `win_conditions_text`
- `ascii_state`
- `score` (engine heuristic, lower is closer) and `score_normalized` (`[0,1]`, higher is closer)
- `is_winning`, `action_names`, `n_rules`

**Custom-search route** - `def search_plan(api, seed)` plus a `SEARCH_STRATEGY` label, returning an
action sequence. The artifact drives the search itself through a bounded API: `api.initial()`,
`api.successors(state)`, `api.key(state)`, `api.ctx(state)`, `api.is_winning(state)`, and
`api.expansion_budget()`. Exceeding the budget stops the search as one scored task rather than
taking down the worker, and `LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB` caps per-search heap the same way.

The prompt also receives an environment description from `build_env_description`: objects, actions,
win conditions, PuzzleScript legend, collision layers, rules, and raw game source.

### Objectives

Per-level raw search quality, with `N` the expansion cap for that evaluation:

```text
S = expanded if solved else (N + 1)
level_score = ((N + 1) - S) / (N + 1)
```

GEPA never sees that number directly. `--objective` picks the transform:

- **`adjusted`** (default, experiment 1) - **base-relative**: each instance is scored against the
  stored base-prompt result for the same level, so the objective is "improve on the base prompt".
  Clipped score delta (`--score-delta-weight`, `--score-delta-clip`), `+--new-solve-bonus` for a
  level only the candidate solves, `---lost-solve-penalty` for a base solve the candidate loses, a
  capped log expansion-ratio term on levels both solve
  (`--common-solve-efficiency-weight/-clip`), `---candidate-error-penalty` for synthesis or
  execution errors, and `--partial-progress-weight` times engine progress on unsolved levels.
  Defaults: lost `8.0`, new `4.0`, error `2.0`, delta weight `1.0` clip `0.5`, partial `0.05`,
  efficiency `0.75` clip `1.0`. `--global-lost-solve-gate-penalty` and
  `--global-net-solve-loss-gate-penalty` add optional evaluation-wide no-regression gates.
- **`speedup-constrained`** - maximize mean `log2(blind / candidate)` expansions subject to not
  losing solves, instead of folding solve events and efficiency into one weighted sum.
- **`base-relative-time`** - the same shape measured in wall-clock time against the base prompt.
- **`blind-relative-time`** (experiment 2) - measured against a **blind reference** table, so the
  question is "how much faster than plain uninformed search", not "better than a previous prompt".
  `--unsolved-log2` prices an unsolved level, `--speedup-clip` and `--slow-solve-clip` bound the
  per-level ratio, and `--min-reference-seconds` drops levels the blind reference finishes so fast
  that a ratio would measure cluster noise. `--include-frontier-levels` keeps levels blind search
  never solved, which are exactly the ones a smarter search has to win.

Build the blind reference with `scripts/calibrate_puzzlescript_budgets.py` (or
`sbatch/calibrate_puzzlescript_budgets_cpu.s`); `--require-blind-reference` refuses to start without
full coverage, and `--blind-budget-multiplier` sets the candidate budget relative to it.

### Splits, holdout hygiene, and final selection

- Train/dev is split at **game** granularity (`--dev-fraction`), so a validation level cannot leak
  mechanics from the same game into training.
- `eval_jobs` games stay outside the optimization metric entirely; `--guard-levels` regression guards
  resolve only from training jobs. Each search-code run writes `holdout_boundary_audit.json`
  recording that the holdout was untouched during optimization.
- `--sibling-level-holdout` shows the synthesis agent one level and scores it on a different level of
  the same game, so an artifact cannot bank a precomputed plan for the instance it is graded on.
- The final prompt is chosen by `select_generalizing_candidate`, not by raw GEPA aggregate. Per game
  it averages `20 * solve_rate_delta + clipped efficiency delta`, then a non-base candidate is
  eligible only if its mean per-game solve-rate delta is positive, or the solve-rate delta is zero
  while at least two games improve, more games improve than regress, and mean per-game score is
  positive. Eligible candidates rank by `(mean solve delta, median game score, mean game score)`.
- The base prompt is always eligible, so a run can legitimately finish by keeping it.
  `gepa_result.json` records `best_idx`, `selected_best_idx`, and `selection_diagnostics`, so a
  disagreement between GEPA's aggregate and the generalization gate is visible after the fact.

### Environment grids

`configs/*.yaml` list `jobs` (train/dev) and `eval_jobs` (holdout) as `{name, levels}` entries where
`levels` is the game's level count; `--levels-per-game` decides how many are sampled.

- `configs/gepa_puzzlescript_envs.yaml` - Sokoban-family curriculum: 19 train/dev games, 22 holdout
  games, 254 holdout tasks at `--levels-per-game 0`.
- `configs/gepa_puzzlescript_gallery_random_20260723.yaml` - fixed-seed random gallery from
  `script-doctor/data/scraped_games`: 44 compileable games with at least 3 levels (32 train/dev, 12
  holdout), explicit Sokoban names capped at 4, 3 levels sampled per game.
- `configs/gepa_puzzlescript_envs_smoke3.yaml` - three-game smoke grid.
- `configs/puzzlescript_blind_reference.json` - calibrated blind reference for 440 Sokoban levels.

### Artifacts

Under `STATE_ROOT`:

- `input_split.json`, `train_tasks.json`, `val_tasks.json`, `optimization_split.json` - the resolved
  game/level split
- `baseline_outputs.json`, `baseline_summary.json` - base-prompt results used as the scoring
  baseline, reusable via `--scoring-baseline-outputs-file`
- `candidate_evals/eval-#####-<candidate>-<taskset>/` - `candidate.json`, `prompt_label.txt`,
  `synthesis_manifest.json`, `heuristics/`, `search_manifest.json`, `search_shards/`,
  `merged_results.json`, `scored_results.json`
- `gepa_run/` - GEPA's own log directory
- `gepa_result.json` - candidates, parents, val aggregates and subscores, `best_idx`,
  `selected_best_idx`, `selection_diagnostics`
- `best_prompt.txt`, `run_git_state.json`, `holdout_boundary_audit.json`
- `holdout/`, `holdout_compare/`, or `untouched_holdout/` - the held-out comparison

### Holdout comparison and replicates

`scripts/compare_puzzlescript_batched_prompts.py` re-synthesizes for the base prompt and the
optimized prompt on `eval_jobs` and writes `comparison_summary.json`,
`per_game_comparison.{csv,json}`, `per_level_comparison.{csv,json}`, and three PNGs. Pass the same
`--seed-contract` the optimized prompt was trained under, or the base arm is not a paired control.

Because agentic synthesis is stochastic, single comparisons are noisy.
`sbatch/compare_puzzlescript_holdout_gpu.s` with `REPLICATES=N` runs N paired comparisons into
`replicate-01/ ... replicate-NN/` and then runs
`scripts/summarize_puzzlescript_holdout_replicates.py`, writing `replicate_summary.json` with
bootstrap 95% CIs over paired deltas plus PNG/PDF figures for solve rate, paired deltas, efficiency
difference, per-game solve-rate delta, and the budget profile.

```bash
uv run python scripts/plot_puzzlescript_paper_results.py --state-root artifacts/<run> --output-dir <dir>
uv run python scripts/monitor_puzzlescript_gepa_runs.py
uv run python scripts/summarize_puzzlescript_gepa_artifacts.py artifacts/<run> [--limit 20]
```

`docs/puzzlescript_slurm_arrays.md` documents the array, replica, and pool launchers and their knobs.

## Experiment 1: heuristic-prompt optimization

One global prompt, optimized so the LLM writes better A* heuristics for unseen games. Search stays
fixed; only the heuristic changes.

```bash
sbatch sbatch/train_puzzlescript_gallery_gepa_codex_cpu.s          # gallery, Codex synthesis, CPU pool
REPLICATES=10 OPTIMIZED_PROMPT=artifacts/<run>/best_prompt.txt \
  sbatch sbatch/compare_puzzlescript_holdout_gpu.s                 # paired holdout replicates
```

`sbatch/train_puzzlescript_batched_gepa_gpu.s` is the vLLM variant on one H100.

### Results

Single run `11913680`, `Qwen/Qwen3-Coder-30B-A3B-Instruct`, 254 holdout tasks
(`docs/puzzlescript_gepa_11913680_holdout.md`):

| prompt | score mean | solved | solve rate | expanded mean |
| --- | ---: | ---: | ---: | ---: |
| base | 0.4367 | 118/254 | 46.46% | 9603.2 |
| optimized | 0.4627 | 125/254 | 49.21% | 9330.4 |

12 new solves against 5 lost solves. The selected prompt was code-shaped rather than instruction
prose, which explains the tradeoff: it encodes target/crate/flag structure well and regresses on
games whose movement rules deviate from ordinary Sokoban.

Ten paired holdout replicates
(`artifacts/gepa_luna_prompt_holdout_replicates10_20260721/replicate_summary.json`, 254 tasks):

| metric | base | optimized | paired delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| score mean | 0.4829 | 0.4923 | +0.0094 | [+0.0008, +0.0166] |
| solve rate | 50.67% | 51.26% | +0.59 pp | [-0.35, +1.50] pp |
| expansions | 7013.0 | 6565.9 | -5.93% | [-9.76%, -2.33%] |

The score gain and the expansion reduction clear zero across replicates; the solve-rate gain does
not. The honest reading is "same solves, meaningfully cheaper search".

On the 44-game random gallery (`gepa_gallery_luna_agentic_sol_robust50k_5rep_20iter_restart_20260728`)
no proposed candidate passed the generalization gate: the best validation candidate scored `+0.016`
mean per-game but improved 4 games while regressing 5 at zero net solve-rate delta, so the base
prompt was kept and the holdout comparison is identically zero across all 34 tasks. Generalizing
from the Sokoban family to arbitrary scraped games is the open problem for this experiment.

## Experiment 2: search-code synthesis

The same optimizer, but the artifact may replace the search algorithm itself, and the objective is
speed against a blind reference rather than improvement over a previous prompt. The point is to test
whether an LLM-written search strategy beats plain A* on levels plain A* does badly on - which is
why `--include-frontier-levels` keeps levels blind search never solved.

```bash
# one-time, per environment grid
sbatch sbatch/calibrate_puzzlescript_budgets_cpu.s

CONFIG=train STATE_ROOT=$PWD/artifacts/<run> sbatch sbatch/train_sokoban_search_code_gepa_cpu.s
# CONFIG=smoke for a two-iteration single-level check, CONFIG=full to append the untouched holdout
```

The two Codex roles are pinned to different models, because they do different jobs. Synthesis
writes the code that is actually executed and scored, so it gets `gpt-5.6-luna`
(`SYNTHESIS_MODEL`); reflection only rewrites the prompt, and gets `gpt-6-astra` (`GEPA_MODEL`).
Note that `--codex-reasoning-effort` is shared by both clients, so `SYNTHESIS_EFFORT` (default
`high`) currently sets the effort for reflection too.

The launcher pins the objective and its guards rather than leaving them to defaults:
`--objective blind-relative-time --blind-reference configs/puzzlescript_blind_reference.json
--require-blind-reference --sibling-level-holdout --include-frontier-levels
--min-reference-seconds 1.0 --unsolved-log2 -3.0 --speedup-clip 14.0 --slow-solve-clip 2.0`.

Reflection additionally receives feedback-only **route probes**: on some generations the prompt
forces the custom-search route (`CUSTOM_SEARCH_PROBE`) or the legacy A* route
(`LEGACY_ASTAR_PROBE`). Those probes never contribute to the candidate's scalar score, so the
optimizer gets evidence about which route works where without being able to farm the probe for
points.

### Results

Latest run `blindrel_v13_20260901_1445` (Codex `gpt-5.6-sol` agentic synthesis, 5 replicates per
level, 70 train and 24 dev tasks over Sokoban-family games, 20 iterations, 31 h wall clock,
`COMPLETED`):

Two scales matter here and they disagree. GEPA's own validation aggregate is the mean
`log2(blind / candidate)` speedup; the generalization gate scores `20 * solve_rate_delta + clipped
efficiency` per game, so it weights solve coverage far above speed.

| dev candidate | GEPA val aggregate (log2) | implied speedup vs blind | dev solved | gate score |
| --- | ---: | ---: | ---: | ---: |
| 0, seed prompt | 1.83 | 3.6x | - | 17.75 |
| 1 | 2.21 | 4.6x | - | 16.79 |
| 2, **selected** | 1.48 | 2.8x | 20/24 | **18.20** |
| 3, GEPA best | **3.43** | **10.8x** | 18/24 | 13.26 |

Every candidate beat the blind reference with zero lost solves, so synthesized search code is
clearly faster than uninformed search on these levels. The interesting result is the disagreement:
candidate 3 is 10.8x faster than blind on average but solves 18/24, while candidate 2 is only 2.8x
faster and solves 20/24. The gate takes coverage, so the shipped prompt is the slower, broader one.
Which of those is the right artifact to select is an open design question, not a settled one -
candidate 3's speedup is concentrated on fewer games.


**These are training and dev numbers only.** The v13 run was launched with `CONFIG=train`, so no
untouched holdout has been evaluated for the search-code experiment yet; that is the next step
(`CONFIG=full`, or `sbatch/resume_sokoban_search_code_holdout_cpu.s` against the finished run).
Until that lands, experiment 2 shows that synthesized search code beats blind search on the levels
it was optimized over, not that the prompt transfers to unseen games.

Objective iterations leading here: `v8-v9` speedup-constrained, `v10-v11` base-relative time,
`v12-v13` blind-relative time with the frontier levels and the timing floor.

## XLand-MiniGrid heuristic search (earlier pipeline)

`scripts/run_heuristic_batch.py` optimizes the same heuristic contract for XLand-MiniGrid with a
DSPy GEPA loop and a phase curriculum.

```bash
uv run scripts/calibrate_astar_budgets.py --env-grid configs/gepa_envs.yaml --write
uv run scripts/run_heuristic_batch.py --state-root artifacts/gepa_state
```

Switches: `--env-grid`, `--max-phase-iterations`, `--llm`, `--astar-max-nodes`,
`--astar-max-expansions`, `--deterministic-envs`, `--room-count`. W&B logging defaults to project
`llm-astar` unless `WANDB_DISABLED=1`.

**Budget calibration** is a required methods step. The script runs blind A* (`h=0`) on each job's
explicit `holdout_seeds` with a 5-minute per-seed budget. If blind A* solves, it sets
`astar_max_nodes = floor(generated_states * 0.95)` and
`astar_max_expansions = floor(expanded_states * 0.95)`; otherwise it keeps the observed counts. Job
budgets take the worst calibrated counts across that job's holdout seeds, so budgets sit just
outside the reach of blind or weak heuristics.

**Curriculum**: three coarse cumulative phases over the ordered training jobs (`3`, `7`, `11`), each
with its own persistent GEPA `log_dir` under `STATE_ROOT/heuristic_runs/phase-##-gepa/`. Repeated
iterations in a phase resume that phase's archive with a larger `max_metric_calls`; advancing starts
a fresh archive seeded by the previous phase's best prompt. A non-final phase advances at mean
per-job solve rate `>= 0.80` and stops the run after 3 iterations without a strictly higher phase
best. The final phase ignores patience until `--max-phase-iterations`.

**Holdout report**: after training, the runner evaluates the best optimized prompt, the base prompt,
and blind A* on the YAML `eval_jobs`, writing `holdout_comparisons` into
`STATE_ROOT/heuristic_runs/gepa_stats.json` plus `holdout_comparison_aggregate.png` and
`holdout_comparison_by_env.png`.

`ctx` for XLand includes `env_id`, `benchmark_id`, `ruleset_text`, `grid_shape`, `action_names`,
`step_cost`, `goal_description`, `agent_state` (`position` with legacy alias `pos`, `direction`,
`carrying`), `object_positions`, `object_metadata`, `static_walls`, and `task_features`.

Grid schema in `configs/gepa_envs.yaml`: `env_id`, `benchmark_id`, `num_gepa_eval_seeds`,
`holdout_seeds`, `deterministic_rulesets` (omit or `false` to sample a fresh ruleset per evaluation
seed), optional `fixed_ruleset_seed`, and the calibrated `astar_max_nodes` / `astar_max_expansions`.

Artifacts: `STATE_ROOT/active_prompt.json` (prompt state plus curriculum checkpoint data),
`STATE_ROOT/<model>.txt`, `STATE_ROOT/heuristic_runs/candidate-####-<job>/` with
`heuristic_synthesized.py`, `heuristic_validation.json`, `astar_search_stats.json`,
`astar_plan.json`, `astar_trace.json`, `task_instance.json`, and `feedback.txt`, and
`STATE_ROOT/heuristic_runs/holdout-heuristic/<job>/`.

## Replay

The search pipeline writes replay-first artifacts directly. `task_instance.json` stores the
deterministic task materialization metadata, `astar_plan.json` the chosen action sequence, and
`astar_trace.json` compact search diagnostics for overlays and debugging.

## JAxtar note

The repository records the upstream JAxtar source revision in [pyproject.toml](pyproject.toml). The
current XLand integration uses a repo-local compatibility backend with the same search-oriented
boundary while the repo-specific XLand adapter remains internal.
