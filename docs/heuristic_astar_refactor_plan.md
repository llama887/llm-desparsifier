# Heuristic Synthesis for A* Search: Target Refactor Plan

## Purpose

This document describes the intended end state of this repository after removing the RL reward-learning pipeline and refactoring the project into a heuristic-generation system for deterministic A* search. The goal is to optimize an LLM prompt with DSPy GEPA so that the LLM reliably emits admissible heuristics that solve XLand-MiniGrid tasks while expanding as few states as possible.

The emphasis here is not on the exact migration sequence. The emphasis is on the final algorithm, the final code shape, and the exact quantities that should be computed at each stage so the implementation can be judged against a crisp target.

## High-Level Goal

The final system should do one thing:

1. Take an environment/task description and a prompt under optimization.
2. Ask an LLM to synthesize a heuristic function for A*.
3. Run A* with that heuristic on many seeds for each training environment.
4. Score the heuristic by combining solve rate and search efficiency.
5. Produce textual feedback about admissibility, failure modes, and search behavior.
6. Feed the scalar score plus feedback back into GEPA so GEPA improves the prompt.

There should be no PPO training loop, no dense reward wrapper as the primary optimization object, no policy-learning curves, and no reward-hacking machinery intended only to make RL training work.

## External Search Backend

The target refactor should use `JAxtar` as the intended A* backend rather than continuing to treat the current repo-local Python planner as the long-term solution.

Reference:

- JAxtar GitHub: https://github.com/tinker495/JAxtar

Why this matters:

- JAxtar presents itself as a pure-JAX, batched, parallelizable A* and Q* solver;
- it is designed around JAX-native search data structures instead of Python `heapq` and Python dictionaries;
- that is much closer to the end state we want for multi-seed heuristic evaluation, where evaluating 10 seeds should ideally be one batched search call rather than 10 separate host-side searches;
- it gives us a plausible path to real batched A* execution, which the current planner does not provide.

This means the long-term target is not merely "improve `llm_desparsifier/search/astar.py`." The long-term target is to build an XLand-facing integration layer around JAxtar and make that the default search engine.

## Core Design Principles

### 1. The optimized artifact is a heuristic, not a reward

The LLM should emit code for a heuristic function `heuristic(...) -> float`, not a dense reward function. The entire prompt language, sanitizer language, reflection language, artifact naming, and metrics should be rewritten around heuristic quality.

### 2. Admissibility is a first-class objective

The prompt should explicitly ask for an admissible heuristic. The evaluator should check admissibility on sampled transitions and solved rollouts, and feedback should explicitly report when the heuristic appears to overestimate cost-to-go. From the planner's point of view, however, the generated function is always treated as the heuristic used by A*. We should not carry a separate runtime branch, sign-flip hack, or fallback behavior for "non-admissible mode." Pressure toward admissibility should come from GEPA feedback and heuristic-quality metrics, not from changing the planner semantics.

### 3. GEPA should optimize one scalar objective, but the scalar should reflect two real desiderata

The user-level objective is:

- solve the task often;
- among successful heuristics, search fewer states.

DSPy GEPA only needs a higher-is-better scalar score plus optional text feedback. Because of that, we should not feed GEPA a raw tuple. We should convert the tuple into a scalar in a way that is numerically stable across environments of different sizes.

### 4. Ignore partial observability

The heuristic should receive full task/state information, not an agent observation crop. This removes the current observation-oriented prompt content and simplifies both admissibility reasoning and heuristic debugging.

### 5. Video generation should replay recorded A* output, not rerun a separate planner

The evaluation pipeline should already produce the chosen A* plan, expansion statistics, and optional frontier snapshots. Video generation should consume those artifacts directly and render the search/solution. It should not carry an independent planning implementation unless there is a narrow debugging mode that is clearly separate from normal usage.

## End-State Pipeline

## Stage 0: Environment Grid and Example Construction

The input dataset to GEPA remains a list of environment jobs, but each job is now defined entirely in search terms.

Each job should include at least:

- `env_id`
- `benchmark_id`
- `num_gepa_eval_seeds`
- `holdout_seeds`
- `deterministic_rulesets`
- optional `fixed_ruleset_seed`
- `astar_max_nodes`
- `astar_max_expansions`

The GEPA training set should still be a list of `dspy.Example` objects, but the example payload should become heuristic-oriented. The minimal input string should not just be `"env_id | benchmark=..."`. It should contain or make available:

- the environment identifier;
- the benchmark identifier;
- the full natural-language task description from the sampled ruleset;
- the state/action semantics relevant to A*;
- the heuristic code contract.

Conceptually:

```python
example = dspy.Example(
    env_description=full_ruleset_text,
    heuristic_contract=contract_text,
    env_id=job.env_id,
    benchmark_id=job.benchmark_id,
)
```

The `job_config` attachment pattern from the current runner can stay, but the config should stop carrying PPO-specific keys.

### Seed configuration semantics

The seed fields need to have different meanings for GEPA training and for holdout evaluation.

- `num_gepa_eval_seeds`
  - number of fresh seeds sampled for that job on each GEPA metric call.
- `holdout_seeds`
  - fixed explicit seeds used only for post-training evaluation and reporting.

For GEPA training, we should not keep one fixed list of seeds forever. Instead, each metric call should sample a fresh batch of seeds from a reproducible job-specific RNG stream so GEPA cannot overfit to a tiny static seed set.

One concrete rule is:

- define a job-level seed sampler keyed by `(global_experiment_seed, metric_call_idx, job_name)`;
- on each GEPA metric call, sample `num_gepa_eval_seeds` fresh seeds for that job;
- use those sampled seeds for the entire evaluation of that candidate on that job.

The exact sampling rule should be:

- sample integer seeds uniformly from `[0, 2**31 - 1]`;
- sample without replacement within one `(metric_call_idx, job_name)` batch;
- allow the same seed to reappear across different metric calls;
- keep the sampled seed batch fixed for the entire evaluation of one candidate prompt on one job;
- do not resample separately for different predictors inside the same GEPA metric call.

This keeps the sampling reproducible while still changing the actual task instances that GEPA sees over time.

## Stage 1: Program Under Optimization

The current optimized program is effectively a prompt rewriter that produces the prompt used by `RewardGenerator`. The same broad GEPA pattern can remain, but the optimized prompt should now target a heuristic generator module.

The conceptual program is:

```python
class PromptOnlyProgram(dspy.Module):
    def forward(self, env_description, constraints=None):
        ...
        return dspy.Prediction(prompt_text=optimized_heuristic_prompt)
```

The important change is downstream:

- `RewardGenerator` should be replaced by something like `HeuristicGenerator`.
- emitted code should define a heuristic;
- sanitizer/validator should validate heuristic code, not reward code.

### Heuristic code contract

The generated module should ideally export one function with a narrow interface, for example:

```python
def heuristic(ts, env_params, ctx) -> float:
    ...
```

or, if we want to make the contract more explicit:

```python
def heuristic_cost_to_go(ts, env_params, ctx) -> float:
    """Return a non-negative lower bound on the remaining path cost."""
```

The contract should require:

- return type is a finite scalar float;
- output is non-negative;
- output is zero on solved states;
- output should be a lower bound on true remaining path cost;
- heuristic may use full state information in `ts` and structured task metadata in `ctx`;
- heuristic must not depend on hidden rollout history unless that history is part of the Markov state representation we deliberately expose.

### Exact `ctx` contract

The heuristic interface needs an exact, implementation-level contract so generated code and runtime validation agree on the same fields.

`ctx` should be a read-only mapping with these keys:

- `env_id: str`
- `benchmark_id: str`
- `ruleset_text: str`
- `grid_shape: tuple[int, int]`
- `action_names: tuple[str, ...]`
- `step_cost: int`
- `goal_description: str`
- `agent_state: Mapping[str, Any]`
  - at minimum: `position`, `direction`, `carrying`
- `object_positions: Mapping[str, tuple[int, int]]`
- `object_metadata: Mapping[str, Mapping[str, Any]]`
  - optional static details such as object type/color when available
- `static_walls: tuple[tuple[int, int], ...]`
- `task_features: Mapping[str, Any]`
  - normalized task-level facts extracted from the ruleset

The contract should also define what is intentionally absent:

- no cropped agent observation;
- no partial-observability-only helpers;
- no hidden planner statistics;
- no previous-state history unless we later decide to expose a Markov-safe version explicitly.

The runtime should reject any generated heuristic that depends on fields outside this contract.

## Stage 2: Heuristic Synthesis

The current prompt and parser stack are reward-centric and include observation-facing advice. Those should become heuristic-centric.

The new base prompt should tell the LLM:

- it is writing an admissible A* heuristic;
- the quantity being estimated is remaining sparse path cost to a goal state;
- underestimation is acceptable, overestimation is not;
- if uncertain, prefer a weaker but safer lower bound;
- use full-state symbolic information;
- do not reason from partial observability or cropped observations;
- return simple code with a small number of interpretable terms.

### Information provided to the LLM

For now, the LLM should be given more structured state/task information than in the current reward pipeline. The prompt context should include at least:

- full ruleset description;
- map size and static layout when available;
- object identities and canonical keys;
- agent position, orientation, carried object;
- full object positions, not just visible positions;
- goal predicate or win condition in explicit text;
- action set and step-cost convention;
- a note that the heuristic must estimate remaining path length or another equivalent lower bound under unit-cost A*.

This is intentionally more information than an agent would have under partial observability. That is acceptable because the new target is heuristic synthesis for deterministic search, not policy learning under observation constraints.

### Sanitization and validation

The sanitizer should reject or flag code that:

- returns negative values;
- returns non-scalars;
- references banned APIs;
- mutates state;
- depends on observation-only fields that are being removed;
- contains obvious overestimation claims like adding arbitrary bonuses;
- fails compilation.

The validator should also support heuristic-specific checks:

- `h(goal) == 0` on solved sampled states;
- `h(s) >= 0` on sampled states;
- optional consistency check on sampled edges:
  `h(s) <= 1 + h(s')` for unit-cost transitions.

Consistency is stronger than admissibility. We should report it, but not require it if that prunes too many useful heuristics early.

## Stage 3: Search Evaluation

Each candidate prompt is evaluated by generating one heuristic per environment job and running A* on multiple seeds.

The search backend should be the center of the system. The current `llm_desparsifier/search` package is the correct starting seam for the integration layer, but the evaluator should stop pretending it is a sibling of PPO training. It should become the primary evaluation pathway.

### Search engine choice

The intended final implementation should use JAxtar for the actual A* engine.

Concretely:

- `llm_desparsifier/search/` should evolve into a thin adapter layer around JAxtar;
- repo-local search code should focus on XLand-specific state encoding, heuristic plumbing, batching setup, artifact capture, and metric computation;
- we should avoid investing heavily in a bespoke Python `heapq` planner if JAxtar can provide the batched solver core we need.

One reasonable abstraction boundary is:

```python
class SearchBackend(Protocol):
    def solve_many(task_batch, heuristic_fn, search_config) -> SearchBatchResult:
        ...
```

with a JAxtar-backed implementation as the default and the current local planner retained, if needed, only as a temporary migration fallback or debugging path.

### Exact JAxtar adapter boundary

JAxtar should not receive raw Python objects or repo-specific wrapper classes. The adapter boundary should be an explicit encoded search state.

The recommended split is:

- `xland_adapter.py` owns conversion between XLand timesteps and a compact immutable state encoding;
- `jaxtar_backend.py` operates only on:
  - encoded states,
  - discrete action ids,
  - a pure transition function,
  - a goal predicate,
  - a heuristic callback.

Concretely, the backend-facing state representation should be a fixed-shape pytree of JAX arrays containing only:

- agent position/direction/carry state;
- object positions and relevant object-state flags;
- any door/lock/toggle state needed for exact transition dynamics;
- any task-instance identifiers needed to interpret the state.

This boundary matters because it prevents the JAxtar layer from depending on ad hoc Python-side environment objects and makes batching practical.

### Per-job, multi-seed execution

For each environment job `j`:

1. Sample `num_gepa_eval_seeds_j` fresh evaluation seeds for the current GEPA metric call.
2. Build one heuristic-generation context for the job itself, not one separate prompt-generation request per seed.
3. Generate heuristic code once for the job.
4. Compile and validate that heuristic once.
5. For each sampled seed in the job's seed batch:
   - build or sample the deterministic task instance for that seed;
   - run A* with:
     - unit step cost `c(s, a, s') = 1`;
     - `f(s) = g(s) + h(s)`;
     - the same generated heuristic value `h(s)`;
     - configured node and expansion budgets.
6. Record, for each seed:
   - `solved_seed` in `{0, 1}`;
   - `expanded_states_seed`;
   - `generated_states_seed`;
   - `solution_length_seed` if solved;
   - `termination_reason_seed`;
   - admissibility diagnostics gathered during or after the run.

This document assumes one synthesized heuristic is reused across all sampled seeds for a job evaluation. That choice is deliberate: it tests whether the heuristic generalizes across multiple task instances instead of letting the LLM specialize the code to each specific seed.

Here, "job" should mean one `(env_id, benchmark_id)` evaluation unit plus its search-budget settings. It should not mean an entire heterogeneous environment family spanning multiple unrelated benchmarks.

In the final JAxtar-oriented design, this should be executed as a batched solve over many seeds whenever possible rather than as one host-side planner call per seed. The conceptual unit of scoring remains "per seed", but the implementation should prefer one batched search invocation for the whole seed set.

### Seed randomization during GEPA

GEPA evaluation should intentionally randomize the environment seeds across candidate evaluations so the optimizer cannot overfit to a fixed small set of task instances.

That requirement has an important metric consequence:

- the primary GEPA metric should not depend on a cached no-heuristic baseline computed on fixed seeds;
- otherwise the normalization target would be mismatched to the actually sampled seeds or would require rerunning the baseline on every metric call, which would be too expensive.

Because of this, the primary GEPA score should be computed only from quantities produced during the candidate's own evaluation on the freshly sampled seeds. No-heuristic baselines can still be useful for offline analysis and optional paper tables, but they should not define the main optimization score inside GEPA.

### Which search count should define "states searched"?

The primary count should be `expanded_states`.

Reason:

- `expanded_states` is the cleanest measure of actual search work in A* because it counts the number of states removed from the open set and evaluated;
- `generated_states` is useful but more sensitive to duplicate-generation behavior and bookkeeping details;
- the current planner already records both, so we should log both but optimize mainly on `expanded_states`.

If we want the paper language to say "average number of states searched", that quantity should map to:

`average_number_of_states_searched = mean(expanded_states_seed over evaluation seeds)`

## Stage 4: Metric Fed Into GEPA

## GEPA constraint from the docs

DSPy GEPA expects a metric function that returns either a float or `ScoreWithFeedback`, with higher scores treated as better. That means we should produce one scalar, not a tuple. The feedback channel is where we can preserve the richer explanation of why a heuristic failed.

Source checked:

- DSPy docs page for `GEPA`, which documents `metric` as returning `float | ScoreWithFeedback`.

## Desired semantic objective

The user-level objective is:

`win_rate / average_number_of_states_searched`

That is directionally correct, but the final implementation should not use the raw unnormalized ratio directly. A raw ratio would unfairly penalize larger maps, be hard to compare across environments, and produce tiny values that are awkward for GEPA to optimize.

Because GEPA should randomize seeds across candidate evaluations, the main score also should not depend on a no-heuristic baseline cache. The score needs to be self-contained and computable from the candidate's own search results.

### Recommended scalarization: solved-path-normalized per-seed score

The cleanest formulation is to score each seed independently and then average.

For each environment job `j` and evaluation seed `s`, define:

- `solved_{j,s} in {0, 1}`
- `expanded_{j,s}` = number of expanded states for the candidate heuristic on that seed
- `solution_length_{j,s}` = length of the returned solution if solved
- `cap_j = astar_max_expansions_j`

To make failure handling explicit, define capped search effort:

- `candidate_cost_{j,s} = expanded_{j,s}` if solved
- `candidate_cost_{j,s} = cap_j` if not solved

Then define solved-path-normalized efficiency:

`efficiency_{j,s} = min(1.0, solution_length_{j,s} / candidate_cost_{j,s})` if solved, else `0`

### Metric edge-case rules

To avoid ambiguity, the metric should also define:

- `expanded_states` means the number of states popped from the open set and expanded by the search backend;
- the solved goal state counts as expanded if and only if the backend pops it from the frontier before terminating, and the implementation should document whichever convention JAxtar uses;
- duplicate generated states do not matter for the primary metric because the primary metric uses `expanded_states`, not `generated_states`;
- if a solved run somehow reports `solution_length = 0`, then treat it as:
  - `1` if the start state already satisfies the goal;
  - otherwise mark the run invalid and assign `seed_score = 0` with a validation error;
- if a solved run reports `expanded_states < solution_length`, treat that as a backend accounting bug and set `seed_score = 0` while logging the inconsistency;
- clamp the final `seed_score` into `[0.0, 1.0]` after all arithmetic, even though it should already lie in that range.

Then define the per-seed optimization score:

`seed_score_{j,s} = solved_{j,s} * efficiency_{j,s}`

This is the key formula. It gives the behavior we want:

- unsolved seed -> `seed_score_{j,s} = 0`
- solved seed with search effort close to the final path length -> `seed_score_{j,s}` close to `1`
- solved seed with search effort much larger than the final path length -> `seed_score_{j,s}` close to `0`

Then aggregate:

- `job_score_j = mean(seed_score_{j,s} over all seeds s for job j)`
- `gepa_score = mean(job_score_j over all training jobs j)`

This gives GEPA a scalar in `[0, 1]` with an exact interpretation:

- `0` means the heuristic never solved anything;
- `1` means it solved every seed while expanding almost no more states than the final solution length itself;
- intermediate values jointly reflect solve reliability and search efficiency.

### Why this formulation is sharper than budget normalization

This formulation gives a much stronger efficiency signal than normalizing by the raw expansion budget.

If we used:

`seed_score = solved * (1 - expanded / max_expansions)`

then almost every solved run would score very close to `1` whenever `max_expansions` is large. That would make the score behave almost like pure solve rate.

By normalizing with `solution_length` instead, we ask a more meaningful question:

"How much extra search did A* perform relative to the length of the plan it ultimately needed?"

That makes differences among solved heuristics much more visible.

### Why scoring per seed before averaging is still important

If we instead compute:

- `win_rate_j = mean(solved_{j,s})`
- `avg_expanded_j = mean(candidate_cost_{j,s})`
- `job_score_j = win_rate_j * f(avg_expanded_j)`

then one catastrophic seed can distort the denominator in a way that is harder to interpret. The per-seed score is more transparent:

- each seed contributes one bounded number in `[0, 1]`;
- failed seeds contribute exactly `0`;
- solved but inefficient seeds contribute a partial score.

This is also compatible with seed randomization across GEPA iterations because the score depends only on the candidate's own sampled runs.

### Equivalent verbal description

The scalar GEPA sees is:

"For each sampled seed, give zero credit if the heuristic fails. If it solves, compare the length of the final solution path to the total number of expanded states and give more credit when those two numbers are closer. Then average across seeds and environments."

### Why this is preferable to the current A* score

The current search score in `llm_desparsifier/search/evaluator.py` is lexicographic:

- all solved runs score above unsolved runs;
- within each bucket, fewer expanded states is better.

That was a good transitional metric, but the final heuristic-only system should average over many seeds and should reflect the user’s desired quantity directly. The per-seed bounded score above is closer to the stated objective and easier to interpret.

### Alternative if strict solve-first ordering is desired

If we decide that any solved seed should always dominate any amount of search-efficiency improvement on unsolved seeds, then the current recommendation already does that locally because unsolved seeds score `0` and solved seeds score `> 0`.

The only stronger version would be a two-stage comparison outside the scalar metric:

1. compare job-level solve counts first;
2. break ties using average efficiency among solved seeds.

I do not recommend encoding that stronger lexicographic rule into the primary GEPA metric unless we observe GEPA over-optimizing efficiency while solve counts remain low.

## Stage 5: Admissibility Evaluation

Admissibility should not be treated as a vague aspiration. It should be measured and logged explicitly.

### Exact definition

For unit-cost A*:

`h(s) <= h*(s)`

where `h*(s)` is the true optimal remaining path cost from state `s` to any goal state.

### Practical admissibility checks

We will usually not compute `h*(s)` for every generated state in large runs. Instead, the evaluator should combine three checks:

1. Goal-state check:
   - whenever a solved terminal state is reached, require `h(goal) == 0` up to tolerance `eps_goal`.

2. Edge consistency check:
   - for sampled transitions `(s, s')`, require:
     `h(s) <= 1 + h(s') + eps_consistency`
   - because the planner uses unit costs.

3. Solved-path exact check:
   - once A* finds a solution of length `L`, for states on the chosen solution path at depth `d`, require:
     `h(s_d) <= L - d + eps_path`
   - because the remaining cost on that realized optimal-or-found path is an upper bound we can cheaply compute.

These checks do not prove global admissibility, but they provide strong concrete signals.

### Logged admissibility quantities

For each seed:

- `admissibility_goal_violation_count_seed`
- `consistency_violation_count_seed`
- `consistency_violation_rate_seed`
- `path_overestimate_count_seed`
- `max_path_overestimate_seed`
- `admissibility_pass_seed`

For each job:

- `admissibility_pass_rate_j = mean(admissibility_pass_seed)`

### How admissibility interacts with the search loop

The search loop should always assume it has been given a heuristic and should run standard A* with that heuristic.

Admissibility diagnostics are still important, but their role is:

- to inform GEPA feedback;
- to explain poor search behavior;
- to help us debug prompt and heuristic design.

They should not introduce an alternate planner mode or a special-case execution path for heuristics that appear non-admissible.

## Stage 6: Text Feedback to GEPA

The current reflection stack is still written as a reward-reflection system with RL training curves and behavior summaries. The final feedback stack should be heuristic-specific.

The new feedback text should answer four questions:

1. Did the heuristic solve the task?
2. Did it solve the task with low search overhead relative to the final solution length?
3. Did it appear admissible/consistent?
4. What structural mistake in the heuristic likely caused the failure?

### Inputs to the feedback builder

For each job, the feedback builder should receive:

- environment summary;
- emitted heuristic code;
- solve rate over seeds;
- average expanded/generated states;
- average solution length when solved;
- average `solution_length / expanded_states` over solved seeds;
- termination-reason histogram;
- admissibility diagnostics;
- object-key alignment diagnostics;
- optional frontier snapshots or state examples where `h` behaved badly.

### Recommended feedback structure

The text should have deterministic sections, even if an LM later rewrites them:

1. Task summary
2. Search outcome summary
3. Admissibility summary
4. Failure diagnosis
5. Revision guidance

An example skeleton:

```text
Task: ...
Search outcome: solved 7/10 seeds; average expanded states 842; average solution length 31; average solved-seed efficiency 0.037.
Admissibility: passed goal-state checks, but violated consistency on 12/500 sampled edges and overestimated remaining cost by up to 3.0 on solved-path states.
Diagnosis: the heuristic appears to add extra penalties for object arrangement that are not guaranteed lower bounds on remaining path length.
Revision guidance: replace speculative penalties with distances or obstacle-aware lower bounds that remain valid under all completions of the plan.
```

### When to use an LM for feedback vs deterministic text

I recommend a hybrid:

- build a deterministic summary first;
- optionally pass that summary and the heuristic code through an LLM reflection module to produce cleaner revision guidance.

That preserves debuggability. If the reflection LM fails, GEPA still receives valid deterministic feedback.

## Stage 7: Holdout Evaluation

Holdout evaluation should mirror training evaluation, but it should never influence GEPA updates.

Unlike GEPA training, holdout evaluation should use fixed explicit `holdout_seeds` so results are comparable across runs, prompt revisions, and paper figures.

For each holdout job:

1. Load the fixed `holdout_seeds` for that job.
2. Generate one or more heuristics from the final optimized prompt.
3. Run the same multi-seed A* evaluation.
4. Report:
   - holdout `win_rate`;
   - holdout `avg_expanded_states`;
   - holdout GEPA-style solved-path-normalized score;
   - holdout admissibility pass rate.

If multiple heuristic generations are evaluated per holdout job, aggregate:

- mean and standard deviation of win rate;
- mean and standard deviation of average expanded states;
- mean and standard deviation of the GEPA-style solved-path-normalized score.

The holdout generation policy should be:

- default: generate exactly one heuristic per holdout job with deterministic decoding;
- optional robustness mode: generate `N > 1` heuristics at nonzero temperature and report mean/std across generations.

The default should be deterministic so holdout numbers remain easy to compare unless we explicitly opt into generation-noise analysis.

## Stage 8: Artifact Model

The artifact tree should be renamed and simplified around heuristics.

Suggested structure:

```text
STATE_ROOT/
  active_prompt.json
  <model_alias>.txt
  heuristic_runs/
    candidate-####-<job>/
      heuristic_synthesized.py
      heuristic_validation.json
      astar_search_stats.json
      astar_plan.json
      astar_trace.json
      task_instance.json
      feedback.txt
    offline_analysis/
      astar_no_heuristic_baseline/
        <job>/
          astar_search_stats.json
          astar_plan.json
    holdout-heuristic/
      <job>/
        try-##/
          heuristic_synthesized.py
          astar_search_stats.json
          astar_plan.json
  gepa_stats.json
```

### Meaning of the key files

- `heuristic_synthesized.py`
  - the compiled heuristic code emitted by the LLM.

- `heuristic_validation.json`
  - compile success, sanitizer issues, non-negativity checks, goal-zero checks, consistency checks, admissibility warnings.

  Required fields:
  - `compile_ok: bool`
  - `sanitizer_errors: list[str]`
  - `sanitizer_warnings: list[str]`
  - `contract_violations: list[str]`
  - `goal_zero_pass: bool | null`
  - `nonnegative_pass: bool | null`
  - `consistency_pass: bool | null`
  - `admissibility_summary: Mapping[str, Any]`

- `astar_search_stats.json`
  - aggregated statistics for that run:
    - solved;
    - expanded/generated states;
    - solution length;
    - solved-path-normalized score;
    - termination reason;
    - heuristic timing if we measure it.

  Required fields:
  - `env_id: str`
  - `benchmark_id: str`
  - `seed: int`
  - `solved: bool`
  - `expanded_states: int`
  - `generated_states: int`
  - `solution_length: int | null`
  - `seed_score: float`
  - `termination_reason: str`
  - `max_nodes: int`
  - `max_expansions: int`
  - `heuristic_eval_count: int | null`
  - `wallclock_sec: float | null`

- `astar_plan.json`
  - the actual chosen action sequence, enough to replay the solved or best-found rollout.

  Required fields:
  - `seed: int`
  - `actions: list[int]`
  - `action_names: list[str]`
  - `replay_complete: bool`
  - `final_state_summary: Mapping[str, Any]`

- `astar_trace.json`
  - optional richer trace:
    - frontier size over time;
    - sampled expanded states;
    - `g`, `h`, and `f` values;
    - states where admissibility checks failed.

  Required fields when present:
  - `seed: int`
  - `expanded_trace: list[Mapping[str, Any]]`
  - `frontier_sizes: list[int]`
  - `admissibility_events: list[Mapping[str, Any]]`
  - `terminated_reason: str`

- `task_instance.json`
  - enough information to recreate the exact deterministic task instance for replay.

  Required fields:
  - `env_id: str`
  - `benchmark_id: str`
  - `seed: int`
  - `ruleset_seed: int | null`
  - `ruleset_text: str`
  - `reset_payload: Mapping[str, Any]`

The `offline_analysis/astar_no_heuristic_baseline/` subtree is optional. It exists only if we decide to keep no-heuristic baselines for analysis and reporting outside the GEPA critical path.

## Stage 9: Video Generation

Video generation should become a pure renderer over recorded A* outputs.

### What the evaluator should save for video

The A* evaluator should save:

- the task instance seed/materialization;
- the selected action sequence;
- per-step state snapshots or enough data to replay from the saved initial state;
- optional per-step overlay stats:
  - step index,
  - `g`,
  - `h`,
  - `f`,
  - cumulative expanded states so far.

### What the video script should do

The video script should:

1. Load the saved task instance.
2. Replay the recorded A* solution or best-found path.
3. Render the map and overlay.
4. Optionally render frontier/admissibility annotations from `astar_trace.json`.

The video script should not run its own A* loop in the standard path. That logic should already have happened during evaluation.

### Optional comparison video

If we still want heuristic-vs-no-heuristic comparison videos, the comparison should come from two stored evaluator outputs:

- one baseline run with no heuristic;
- one heuristic run.

The renderer can place them side by side or render them separately, but it should not be the thing deciding the plan.

## Target Codebase Shape

## Packages to remove or gut

The following areas should stop being central:

- `llm_desparsifier/rl/`
- reward-training-specific logic in `scripts/run_reward_batch.py`
- reward reflection text tied to PPO curves
- observation-oriented prompt content
- negative reward hacks intended only to make reward-as-heuristic work

This does not mean every file must literally disappear immediately, but the final architecture should not depend on them.

## Packages to keep and repurpose

- `llm_desparsifier/search/`
  - this should become the core evaluation and JAxtar-integration layer.

- current prompt/state persistence utilities
  - these are still useful.

- current GEPA orchestration pattern
  - keep the broad shape, replace the backend.

## Packages to rename or replace

I recommend introducing:

```text
llm_desparsifier/heuristics/
  generator.py
  sanitizer.py
  prompting.py
  reflection.py
  validation.py
```

and likely a search integration layout along the lines of:

```text
llm_desparsifier/search/
  jaxtar_backend.py
  xland_adapter.py
  metrics.py
  replay.py
```

### Responsibilities

- `generator.py`
  - turns prompt text plus task context into heuristic code.

- `sanitizer.py`
  - compilation and static safety validation.

- `validation.py`
  - heuristic-specific runtime validation and admissibility diagnostics.

- `reflection.py`
  - builds deterministic or LM-assisted heuristic feedback for GEPA.

- `prompting.py`
  - contains the base heuristic prompt, contract text, and any prompt templates.

- `jaxtar_backend.py`
  - owns the actual JAxtar solve invocation and batching-oriented search configuration.

- `xland_adapter.py`
  - converts XLand task/state/action semantics into the representation expected by JAxtar and converts JAxtar outputs back into repo-local artifacts and metrics.

## Main runner

The current `scripts/run_reward_batch.py` should likely become something like:

`scripts/run_heuristic_batch.py`

This file should own:

- CLI parsing;
- environment-grid loading;
- randomized seed sampling for GEPA evaluation;
- GEPA setup;
- the `on_policy_metric` replacement, which is now purely search-based;
- holdout evaluation;
- artifact writing.

The internal control flow can stay similar, but all code/comments/docstrings should describe search evaluation rather than on-policy RL.

## Detailed Metric Calculation Example

Suppose one training job uses 5 sampled seeds.

Candidate heuristic A* gives:

- solved indicators: `[1, 1, 0, 1, 1]`
- expanded states: `[410, 390, 100000, 470, 450]`
- solution lengths on solved seeds: `[28, 30, -, 29, 31]`

Then:

- `candidate_costs = [410, 390, 100000, 470, 450]`
- `efficiencies = [28/410, 30/390, 0, 29/470, 31/450]`
- `seed_scores = [`
  `1 * (28 / 410),`
  `1 * (30 / 390),`
  `0,`
  `1 * (29 / 470),`
  `1 * (31 / 450)`
  `]`

Numerically:

- `seed_scores = [0.06829, 0.07692, 0.0, 0.06170, 0.06889]`
- `job_score_j = mean(seed_scores) = 0.05516`

That score is intentionally far from `1`:

- one failed seed contributes `0`;
- the solved seeds still get modest scores because A* expanded many more states than the final plan length.

This is exactly the intended behavior under randomized seeds: success is required for positive credit, and among successful runs, heuristics that make A* search closer to the eventual solution length get more credit.

## Recommendation on unsolved seeds

For unsolved seeds:

- set `expanded_states_seed = astar_max_expansions_j`;
- set `generated_states_seed` to the actual recorded count;
- set `solved_seed = 0`.

That makes the metric easy to interpret and ensures failures are penalized heavily.

## Feedback Examples for Common Failure Modes

### Case 1: Solves often but violates admissibility

Feedback should say:

- search efficiency is good;
- the heuristic likely overestimates because it adds penalties not tied to guaranteed remaining path cost;
- revise toward safer lower bounds such as shortest-path distances to mandatory subgoals.

### Case 2: Admissible but too weak

Feedback should say:

- admissibility checks passed;
- solve rate is acceptable or high;
- state expansions remain high relative to the final solution lengths;
- add stronger but still safe lower-bound structure, such as decomposing the goal into mandatory subgoals and summing only clearly unavoidable costs.

### Case 3: References wrong task objects

Feedback should say:

- the heuristic references object keys absent from the task;
- those terms likely collapse to default values or irrelevant distances;
- rewrite using only objects named in the ruleset/task description.

### Case 4: Uses observation-only signals

Feedback should say:

- the heuristic is using observation-oriented fields that are no longer part of the contract;
- rewrite against full-state symbolic fields only.

## README-Level User Story for the Final Repo

Once the refactor is complete, the top-level README should describe the project approximately as:

"This repository uses DSPy GEPA to optimize prompts that cause an LLM to emit admissible A* heuristics for XLand-MiniGrid tasks. Each candidate prompt is evaluated by synthesizing heuristic code, validating it, running deterministic A* over a grid of environments and seeds, and feeding a scalar search-quality score plus heuristic-specific feedback back into GEPA."

That is the correct one-sentence identity of the final codebase.

## Concrete Implementation Consequences

The following specific changes are implied by this vision:

1. Replace reward terminology with heuristic terminology in prompts, filenames, variable names, metrics, and artifact names.
2. Delete or isolate PPO training code from the main experiment path.
3. Remove reliance on observation fields from the prompt contract and sanitizer allowlist.
4. Replace RL reflection summaries with heuristic/admissibility summaries.
5. Make A* evaluation multi-seed by default.
6. Randomize GEPA evaluation seeds across metric calls rather than training on one fixed seed list.
7. Save evaluator-owned A* traces so video generation can replay instead of replanning.
8. Make admissibility diagnostics visible in both logs and GEPA feedback.
9. Integrate JAxtar as the default search backend so multi-seed evaluation can be batched inside the solver rather than only parallelized outside it.

## Open Questions Worth Settling Before Implementation

These are the places where I think we should explicitly agree before doing the full refactor:

1. Should heuristic generation be per ruleset instance or per environment family?
   - Per-instance gives the LLM more information and should work better immediately.
   - Per-family is a stronger generalization test.

2. How should admissibility diagnostics influence the GEPA objective, if at all?
   - feedback and reporting only;
   - or an explicit metric penalty in addition to the feedback channel.

3. Should the heuristic contract allow learned constants and object-specific coefficients?
   - probably yes, as long as they preserve lower-bound semantics.

4. Do we want the primary score to use `expanded_states` only, or a combination of expanded and generated states?
   - I recommend optimizing on expanded states and logging both.

5. Do we want to generate one heuristic per job and reuse it across the sampled seeds, or one heuristic per broader environment family?
   - This document assumes one heuristic per job evaluation, reused across that job’s sampled seeds, because that better measures robustness without forcing a single heuristic to span unrelated benchmark families.

6. Should we still run no-heuristic A* baselines for offline analysis and paper tables even though they are not part of the GEPA metric?
   - I think yes, but only outside the critical GEPA optimization path.

## Current Repo Seams This Plan Builds On

This plan is intentionally anchored to the existing codebase:

- the current GEPA orchestration in `scripts/run_reward_batch.py` already supports `ScoreWithFeedback`;
- the current search evaluator in `llm_desparsifier/search/evaluator.py` already records `expanded_states` and `generated_states`;
- the current video tooling already understands replay artifacts, but it should stop owning a separate planner loop in the normal path.

This plan is also intentionally anchored to an external target:

- JAxtar explicitly targets pure-JAX, batched, parallelizable A* search with JAX-native queue/hash-table support, which matches the desired end state of multi-seed heuristic evaluation better than the current Python-hosted planner.

Those seams should be preserved where they help. The RL-specific middle of the stack should not be preserved just because it already exists.
