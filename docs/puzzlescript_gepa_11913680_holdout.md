# PuzzleScript Batched GEPA Holdout Results

Run: `11913680`
Branch commit: `f82f4b2`
Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
Hardware: 1 H100
Holdout tasks: 254

## Summary

The optimized global prompt improved holdout performance over the base prompt:

| prompt | score mean | solved | solve rate | expanded mean | result errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.4367 | 118/254 | 46.46% | 9603.2 | 22 |
| optimized | 0.4627 | 125/254 | 49.21% | 9330.4 | 22 |

Net deltas:

- Score: `+0.0260` absolute, `+5.96%` relative.
- Solves: `+7` net.
- New solves: `12`.
- Lost solves: `5`.
- Per-level score movement: `55` better, `60` worse, `139` equal.

The result is a real improvement, but not a clean dominance result. The optimized prompt found more high-value solves than it lost, while still regressing a small number of base-solved levels.

## Where Gains Came From

Largest score gains by game:

| game | levels | score delta | solved delta | new solves | lost solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Sokoban_Dungeon` | 3 | +0.2925 | 0 | 0 | 0 |
| `Sokocross` | 22 | +0.2415 | +5 | 5 | 0 |
| `Beam_Islands` | 8 | +0.1184 | +1 | 1 | 0 |
| `Cold_Feet_Sokoban` | 3 | +0.0884 | +1 | 1 | 0 |
| `Memory_Push` | 9 | +0.0782 | +1 | 1 | 0 |
| `Ice_Cubes` | 29 | +0.0540 | +2 | 3 | 1 |
| `Inswaption` | 21 | +0.0444 | +1 | 1 | 0 |

Most of the net solve improvement came from `Sokocross`, where optimized went from `9/22` to `14/22`. The broader score gain was also helped by `Sokoban_Dungeon`, where all levels were solved by both prompts but optimized solved them with much better search efficiency.

## Regressions

Largest score losses by game:

| game | levels | score delta | solved delta | new solves | lost solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Soko-bine` | 8 | -0.1059 | -1 | 0 | 1 |
| `Not_Normal_Crates` | 18 | -0.0865 | -2 | 0 | 2 |
| `Gravity_Sokoban` | 10 | -0.0458 | 0 | 0 | 0 |
| `Crates_and_Portals` | 30 | -0.0287 | -1 | 0 | 1 |
| `No_Right_Turn_Sokoban` | 4 | -0.0238 | 0 | 0 | 0 |

Lost solve levels:

| task | game | level | base score | optimized score |
| ---: | --- | ---: | ---: | ---: |
| 15 | `Soko-bine` | 6 | 0.9154 | 0.0000 |
| 69 | `Crates_and_Portals` | 15 | 0.8472 | 0.0000 |
| 108 | `Ice_Cubes` | 14 | 0.9256 | 0.0000 |
| 157 | `Not_Normal_Crates` | 4 | 0.7814 | 0.0000 |
| 164 | `Not_Normal_Crates` | 11 | 0.7752 | 0.0000 |

The losses are concentrated in crate/portal and non-standard crate mechanics. This is consistent with the selected prompt being useful for target/crate/flag-style structure, but less robust for mechanics where object movement rules differ from ordinary Sokoban assumptions.

## New Solves

New solve levels:

| task | game | level | optimized score |
| ---: | --- | ---: | ---: |
| 7 | `Cold_Feet_Sokoban` | 1 | 0.2658 |
| 48 | `Beam_Islands` | 2 | 0.9491 |
| 99 | `Ice_Cubes` | 5 | 0.5891 |
| 101 | `Ice_Cubes` | 7 | 0.8952 |
| 122 | `Ice_Cubes` | 28 | 0.9819 |
| 142 | `Inswaption` | 19 | 0.9742 |
| 147 | `Memory_Push` | 3 | 0.6876 |
| 192 | `Sokocross` | 0 | 0.9813 |
| 194 | `Sokocross` | 2 | 0.9981 |
| 200 | `Sokocross` | 8 | 0.9975 |
| 203 | `Sokocross` | 11 | 0.7898 |
| 206 | `Sokocross` | 14 | 0.9971 |

## Interpretation

The revised metric and feedback did what they were meant to do better than the previous run: the selected candidate was not the base prompt, and it improved held-out score and solved count. The result is strongest evidence that the base-relative objective is directionally useful.

The main caveat is the form of the selected prompt. The best prompt is only 3250 characters, but it begins with:

```python
def heuristic_cost_to_go(ts, env_params, ctx) -> float:
```

In other words, GEPA selected a code-shaped global heuristic template rather than ordinary instruction prose. That may be acceptable if the research question allows one global reusable heuristic prior, but it is different from optimizing natural-language prompting. It also explains the tradeoff pattern: the prompt strongly encodes target/crate/flag/block heuristics, which helps many Sokoban-like levels and hurts a few mechanics that deviate from that template.

## Operational Notes

The SLURM job completed successfully with exit code `0:0`.

During optimized holdout evaluation, the CPU array job `11924480` produced no shards for 600 seconds. The stall guard cancelled it and fell back to local shard evaluation inside the parent GPU job. The fallback completed all 101 shards and wrote the final comparison artifacts.

The final vLLM `EngineDeadError` in the log happened during API server shutdown after all holdout artifacts were written. SLURM still marked the job `COMPLETED`.

## Artifact Paths

- Summary: `artifacts/gepa_puzzlescript_batched_11913680/holdout_compare/comparison_summary.json`
- Per-game CSV: `artifacts/gepa_puzzlescript_batched_11913680/holdout_compare/per_game_comparison.csv`
- Per-level CSV: `artifacts/gepa_puzzlescript_batched_11913680/holdout_compare/per_level_comparison.csv`
- Best prompt: `artifacts/gepa_puzzlescript_batched_11913680/best_prompt.txt`
- Score delta by game plot: `artifacts/gepa_puzzlescript_batched_11913680/holdout_compare/holdout_score_delta_by_game.png`
- Solve delta by game plot: `artifacts/gepa_puzzlescript_batched_11913680/holdout_compare/holdout_solve_delta_by_game.png`
- Base vs optimized scatter: `artifacts/gepa_puzzlescript_batched_11913680/holdout_compare/holdout_score_base_vs_optimized.png`
