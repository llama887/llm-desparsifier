# GEPA PuzzleScript Generalization Experiments

Date: 2026-05-20

Goal: identify the smallest runner/evaluator change that improves poor
generalization in the previous GEPA MCTS/PuzzleScript heuristic loop.

Initial diagnosis:
- The latest/current PuzzleScript GEPA artifacts have 0 solved candidates and
  mean score 0.0.
- Feedback traces show many generated heuristics return `0.0` everywhere or
  crash and are coerced to `0.0`, making A* behave like blind BFS.
- The runner evaluates a game on level 0 only, despite the config listing many
  levels. This encourages layout-specific heuristics rather than mechanics.

Hypotheses:
1. Strict heuristic exception reporting will expose broken generated code and
   give GEPA more useful feedback than silently converting exceptions to zero.
2. Multi-level candidate evaluation will make train performance better predict
   holdout/generalization performance.
3. The current best heuristic may look good on level 0 but degrade across other
   levels, confirming that level-0-only scoring is too weak.

Experiment log:

## E1: Saved Best Heuristic on First 5 Training Games, First 3 Levels

Command:

```bash
../script-doctor/.venv/bin/python scripts/experiment_puzzlescript_generalization.py \
  --max-games 5 --levels-per-game 3 --budget 5000 \
  --output artifacts/experiments/puzzlescript_generalization/train5_levels3_best.json
```

Result:
- blind: 12/14 solved, mean score 0.6657, mean expanded 1671.9
- builtin: 12/14 solved, mean score 0.7024, mean expanded 1487.9
- saved best heuristic: 14/14 solved, mean score 0.9801, mean expanded 99.5

Interpretation:
- The saved best heuristic is not merely memorizing level 0 for the easy phase.
- It transfers strongly across nearby levels for the first five curriculum games.
- The poor run conclusion is therefore more about the GEPA search/evaluation loop
  producing and selecting dead heuristics than this final heuristic lacking any
  local generalization.

## E2: Saved Best Heuristic on All 10 Training Games, First 3 Levels

Command:

```bash
../script-doctor/.venv/bin/python scripts/experiment_puzzlescript_generalization.py \
  --max-games 10 --levels-per-game 3 --budget 5000 \
  --output artifacts/experiments/puzzlescript_generalization/train10_levels3_best.json
```

Result:
- blind: 18/29 solved, mean score 0.4841, mean expanded 2579.9
- builtin: 18/29 solved, mean score 0.5081, mean expanded 2459.4
- saved best heuristic: 22/29 solved, mean score 0.6997, mean expanded 1501.5

Important failures:
- `Algorithm-Generated_Sokoban_Levels`: heuristic solves level 0 at 1707
  expansions, but fails levels 1 and 2 under the 5000 cap.
- `Muddy_Sokoban_Level_Set_I`: heuristic fails levels 0 and 1 under cap, but
  solves level 2 at 3528 expansions.
- `Ultimate_Sokoban_Supreme`: heuristic fails levels 0, 1, and 2 under cap;
  blind/builtin solve level 2 easily.

Interpretation:
- Multi-level evaluation does reveal a real generalization gap on the harder
  phase-2 games.
- The saved heuristic is still better than baselines on average, but it is not
  robust enough for mechanics-heavy games.
- A level-0-only GEPA metric can select a heuristic that looks strong on the
  nominal train set while missing other levels in the same game family.

## E3: Hard Games at 50k Budget, Saved Best Heuristic Only

Command:

```bash
../script-doctor/.venv/bin/python scripts/experiment_puzzlescript_generalization.py \
  --games Algorithm-Generated_Sokoban_Levels,Muddy_Sokoban_Level_Set_I,Ultimate_Sokoban_Supreme \
  --max-games 10 --levels-per-game 3 --budget 50000 --methods heuristic \
  --output artifacts/experiments/puzzlescript_generalization/hard3_levels3_best_budget50000_heuristic.json
```

Result:
- `Algorithm-Generated_Sokoban_Levels`: solved levels 0, 1, 2 at 1707, 9305,
  and 5347 expansions. The 5k failures were mostly budget-limited.
- `Muddy_Sokoban_Level_Set_I`: failed levels 0 and 1 even at 50k; solved level
  2 at 3528.
- `Ultimate_Sokoban_Supreme`: failed levels 0, 1, 2 even at 50k.
- Aggregate hard-subset heuristic solve rate: 4/9, mean score 0.4003.

Interpretation:
- Raising the cap alone is insufficient.
- Multi-level evaluation should be paired with per-game mechanic feedback,
  especially for `Muddy` and `Ultimate`, because these are not just near-misses
  under a tight budget.

## E4: Contract Mismatch Reproduction

Observation:
- Sanitized PuzzleScript heuristics implement the documented contract:
  `heuristic_cost_to_go(ts, env_params, ctx)`.
- `puzzlescript_astar` and `evaluate_one_game` expect a one-argument callable:
  `heuristic_fn(ctx)`.
- `synthesize_heuristic_from_prompt` returned the raw three-argument function.
- Every heuristic call raised `TypeError`, and the search loop caught that and
  substituted `0.0`.

Minimal reproduction on `sokoban_basic`, level 0, budget 531:
- unwrapped saved heuristic: solved false, expanded 531, score 0.0,
  root_h 0.0, successor range 0.0.
- wrapped saved heuristic: solved true, expanded 132, score 0.7519,
  root_h 42.0, successor range 2.0.

Patch:
- `scripts/run_puzzlescript_batch.py::synthesize_heuristic_from_prompt` now
  wraps sanitized `(ts, env_params, ctx)` functions as `heuristic_from_ctx(ctx)`.
- Final holdout loading wraps `best_code` the same way.
- Added `tests/test_run_puzzlescript_batch.py` to lock this down.

Conclusion:
- This arity mismatch is the primary reason the latest/current GEPA run had
  all-zero scores and terrible generalization. The run was not evaluating the
  generated heuristics at all; it was evaluating blind search after swallowing
  the `TypeError`.
- Multi-level evaluation is still useful for stronger generalization, but the
  first required edit is the contract wrapper fix.

