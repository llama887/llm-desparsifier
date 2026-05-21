#!/usr/bin/env python3
"""Plot fixed-cap PuzzleScript search comparisons against baselines.

This script supports two evaluation splits:
1. `train`: the 10 training games with per-game calibrated caps.
2. `holdout`: the 6 holdout games with the global search cap used by the batch run.

For `train`, it compares:
1. Blind A* at the same per-game expansion cap.
2. The engine builtin heuristic.
3. The first-pass "base prompt" heuristic proxy, taken as the earliest saved
   candidate for each environment in the original GEPA run directory.
4. The saved optimized heuristic, re-evaluated locally under the same cap.

For `holdout`, it compares:
1. Blind A* at the shared global expansion cap.
2. The engine builtin heuristic.
3. The saved optimized heuristic at that same cap.

The holdout split omits a base-prompt curve because the checked-in artifacts do
not include a saved base-prompt holdout evaluation.

Outputs:
  - comparison_under_cap.json
  - comparison_under_cap.png
  - comparison_under_cap.svg
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DOCTOR_ROOT = REPO_ROOT.parent / "script-doctor"
SCRIPT_DOCTOR_SITE_PACKAGES = (
    SCRIPT_DOCTOR_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
)

if str(SCRIPT_DOCTOR_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DOCTOR_SITE_PACKAGES))
if str(REPO_ROOT / "llm_desparsifier" / "search") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "llm_desparsifier" / "search"))

from puzzle_evaluator import PuzzleScriptEvaluator
from puzzlescript_astar import builtin_heuristic, puzzlescript_astar
from puzzlescript_sanitizer import sanitize_and_compile_puzzlescript_heuristic
from puzzlescript_adapter import build_env_description


TRAINING_JOBS: list[tuple[str, int]] = [
    ("sokoban_basic", 531),
    ("Broken_Leg_Sokoban", 152),
    ("Collapsable_Sokoban", 350),
    ("Pulling_Box_Sokoban", 221),
    ("Swap_Sokoban", 781),
    ("Sokoban_Flipped", 946),
    ("Algorithm-Generated_Sokoban_Levels", 9220),
    ("Tractor_Beam_Sokoban9", 1118),
    ("Muddy_Sokoban_Level_Set_I", 5554),
    ("Ultimate_Sokoban_Supreme", 10108),
]

HOLDOUT_JOBS: list[tuple[str, int]] = [
    ("sokoban_sanity", 50_000),
    ("No_Right_Turn_Sokoban", 50_000),
    ("Cold_Feet_Sokoban", 50_000),
    ("Soko-bine", 50_000),
    ("Remote_Control_Sokoban", 50_000),
    ("Darkness_Sokoban", 50_000),
]

# Blind-search expansions as logged in sbatch/logs/llm-desparsifier-6308463.out.
BLIND_EXPANDED: dict[str, int] = {
    "sokoban_basic": 559,
    "Broken_Leg_Sokoban": 161,
    "Collapsable_Sokoban": 369,
    "Pulling_Box_Sokoban": 233,
    "Swap_Sokoban": 823,
    "Sokoban_Flipped": 996,
    "Algorithm-Generated_Sokoban_Levels": 9706,
    "Tractor_Beam_Sokoban9": 1177,
    "Muddy_Sokoban_Level_Set_I": 5847,
    "Ultimate_Sokoban_Supreme": 10640,
}

BUILTIN_EXPANDED: dict[str, int] = {
    "sokoban_basic": 541,
    "Broken_Leg_Sokoban": 99,
    "Collapsable_Sokoban": 341,
    "Pulling_Box_Sokoban": 200,
    "Swap_Sokoban": 646,
    "Sokoban_Flipped": 897,
    "Algorithm-Generated_Sokoban_Levels": 8362,
    "Tractor_Beam_Sokoban9": 620,
    "Muddy_Sokoban_Level_Set_I": 8205,
    "Ultimate_Sokoban_Supreme": 8434,
}


@dataclass
class ComparisonRow:
    env: str
    cap: int
    blind_expanded: int
    blind_solved_under_cap: bool
    builtin_expanded: int
    builtin_solved_under_cap: bool
    base_candidate_id: int
    base_expanded: int
    base_solved: bool
    optimized_expanded: int
    optimized_solved: bool


@dataclass
class HoldoutComparisonRow:
    env: str
    cap: int
    blind_expanded: int
    blind_solved: bool
    builtin_expanded: int
    builtin_solved: bool
    base_expanded: int | None
    base_solved: bool | None
    optimized_expanded: int
    optimized_solved: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-state-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gepa_puzzlescript_state",
        help="State root whose earliest saved candidates proxy the unoptimized base prompt.",
    )
    parser.add_argument(
        "--optimized-state-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gepa_puzzlescript_state_llm_feedback_20260415_1239",
        help="State root containing best_heuristic.py for optimized evaluation.",
    )
    parser.add_argument(
        "--script-doctor",
        type=Path,
        default=SCRIPT_DOCTOR_ROOT,
        help="Path to the local script-doctor checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "plots" / "puzzlescript_cap_comparison",
        help="Directory for the plot and the derived comparison JSON.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Wall-clock timeout per optimized search evaluation.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "holdout"),
        default="train",
        help="Which PuzzleScript split to plot.",
    )
    parser.add_argument(
        "--include-base-holdout",
        action="store_true",
        help="Synthesize one base-prompt heuristic per holdout game and include it in the holdout plot.",
    )
    parser.add_argument(
        "--llm",
        default="gemini/gemini-3-pro-preview",
        help="LLM used when synthesizing the base prompt on holdout.",
    )
    return parser.parse_args()


def load_base_prompt_proxy(base_state_root: Path) -> dict[str, dict[str, int | bool]]:
    records: dict[str, dict[str, int | bool]] = {}
    run_dir = base_state_root / "runs"
    for path in sorted(run_dir.glob("candidate-*-*/search_stats.json")):
        match = re.match(r"candidate-(\d+)-(.+)", path.parent.name)
        if match is None:
            continue
        candidate_id = int(match.group(1))
        env = match.group(2)
        payload = json.loads(path.read_text())
        current = records.get(env)
        if current is None or candidate_id < int(current["candidate_id"]):
            records[env] = {
                "candidate_id": candidate_id,
                "expanded": int(payload["expanded"]),
                "solved": bool(payload["solved"]),
            }
    return records


def evaluate_optimized(
    optimized_state_root: Path,
    script_doctor: Path,
    timeout_s: float,
    jobs: list[tuple[str, int]],
) -> dict[str, dict[str, int | bool]]:
    evaluator = PuzzleScriptEvaluator(script_doctor)
    heuristic_code = (optimized_state_root / "best_heuristic.py").read_text()
    heuristic_fn = sanitize_and_compile_puzzlescript_heuristic(heuristic_code)

    rows: dict[str, dict[str, int | bool]] = {}
    for env, cap in jobs:
        compiled = json.loads(evaluator.compile_game_file(env))
        engine = evaluator.load_engine(json.dumps(compiled))
        engine.load_level(0)
        result = puzzlescript_astar(
            engine,
            compiled,
            lambda ctx, h=heuristic_fn: h(None, None, ctx),
            max_expansions=cap,
            timeout_s=timeout_s,
        )
        rows[env] = {
            "expanded": int(result.expanded_states),
            "solved": bool(result.solved),
        }
    return rows


def build_rows(
    base_prompt_proxy: dict[str, dict[str, int | bool]],
    optimized_results: dict[str, dict[str, int | bool]],
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for env, cap in TRAINING_JOBS:
        base_row = base_prompt_proxy[env]
        optimized_row = optimized_results[env]
        blind_expanded = BLIND_EXPANDED[env]
        rows.append(
            ComparisonRow(
                env=env,
                cap=cap,
                blind_expanded=blind_expanded,
                blind_solved_under_cap=blind_expanded <= cap,
                builtin_expanded=BUILTIN_EXPANDED[env],
                builtin_solved_under_cap=BUILTIN_EXPANDED[env] <= cap,
                base_candidate_id=int(base_row["candidate_id"]),
                base_expanded=int(base_row["expanded"]),
                base_solved=bool(base_row["solved"]),
                optimized_expanded=int(optimized_row["expanded"]),
                optimized_solved=bool(optimized_row["solved"]),
            )
        )
    rows.sort(
        key=lambda row: (
            row.optimized_expanded / row.cap,
            row.env.lower(),
        )
    )
    return rows


def plot_rows(rows: list[ComparisonRow], output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Blind A*", "Builtin Heuristic", "Base Prompt", "Optimized Prompt"]
    win_rates = [
        sum(row.blind_solved_under_cap for row in rows) / len(rows),
        sum(row.builtin_solved_under_cap for row in rows) / len(rows),
        sum(row.base_solved for row in rows) / len(rows),
        sum(row.optimized_solved for row in rows) / len(rows),
    ]

    colors = {
        "Blind A*": "#7f8c8d",
        "Builtin Heuristic": "#4c78a8",
        "Base Prompt": "#e67e22",
        "Optimized Prompt": "#1b9e77",
    }

    fig, (ax_rate, ax_eff) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.0, 2.4]},
    )

    ax_rate.bar(labels, win_rates, color=[colors[label] for label in labels], width=0.65)
    ax_rate.set_ylim(0.0, 1.05)
    ax_rate.set_ylabel("Win Rate Under Fixed Cap")
    ax_rate.set_title("Aggregate Success")
    for idx, rate in enumerate(win_rates):
        wins = int(round(rate * len(rows)))
        ax_rate.text(idx, rate + 0.03, f"{wins}/{len(rows)}", ha="center", va="bottom")
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)

    y_positions = list(range(len(rows)))
    offsets = {
        "Blind A*": -0.30,
        "Builtin Heuristic": -0.10,
        "Base Prompt": 0.10,
        "Optimized Prompt": 0.30,
    }
    method_points = {
        "Blind A*": [
            (row.blind_expanded / row.cap, row.blind_solved_under_cap) for row in rows
        ],
        "Builtin Heuristic": [
            (row.builtin_expanded / row.cap, row.builtin_solved_under_cap) for row in rows
        ],
        "Base Prompt": [(row.base_expanded / row.cap, row.base_solved) for row in rows],
        "Optimized Prompt": [
            (row.optimized_expanded / row.cap, row.optimized_solved) for row in rows
        ],
    }

    for method, points in method_points.items():
        xs = [point[0] for point in points]
        solved_mask = [point[1] for point in points]
        ys = [y + offsets[method] for y in y_positions]
        solved_x = [x for x, solved in zip(xs, solved_mask) if solved]
        solved_y = [y for y, solved in zip(ys, solved_mask) if solved]
        failed_x = [x for x, solved in zip(xs, solved_mask) if not solved]
        failed_y = [y for y, solved in zip(ys, solved_mask) if not solved]
        if solved_x:
            ax_eff.scatter(
                solved_x,
                solved_y,
                color=colors[method],
                marker="o",
                s=72,
                linewidths=0.8,
                edgecolors="black",
                zorder=3,
            )
        if failed_x:
            ax_eff.scatter(
                failed_x,
                failed_y,
                color=colors[method],
                marker="X",
                s=88,
                linewidths=0.8,
                edgecolors="black",
                zorder=3,
            )

    ax_eff.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_eff.set_yticks(y_positions)
    ax_eff.set_yticklabels([row.env.replace("_", " ") for row in rows])
    ax_eff.invert_yaxis()
    ax_eff.set_xlabel("Expanded States / Search Cap")
    ax_eff.set_title("Per-Game Search Efficiency")
    ax_eff.set_xlim(0.0, max(max(x for x, _ in pts) for pts in method_points.values()) * 1.05)
    ax_eff.grid(axis="x", alpha=0.25)
    ax_eff.spines["top"].set_visible(False)
    ax_eff.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=colors[label],
            markeredgecolor="black",
            markersize=8,
        )
        for label in labels
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                label="solved",
                markerfacecolor="white",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="black",
                linestyle="None",
                label="failed under cap",
                markerfacecolor="white",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="cap boundary",
            ),
        ]
    )
    ax_eff.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.suptitle(
        "PuzzleScript Heuristic Search Under Matched Per-Game Caps",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = output_dir / "comparison_under_cap.png"
    svg_path = output_dir / "comparison_under_cap.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def evaluate_blind(
    script_doctor: Path,
    timeout_s: float,
    jobs: list[tuple[str, int]],
) -> dict[str, dict[str, int | bool]]:
    evaluator = PuzzleScriptEvaluator(script_doctor)
    rows: dict[str, dict[str, int | bool]] = {}
    for env, cap in jobs:
        compiled = json.loads(evaluator.compile_game_file(env))
        engine = evaluator.load_engine(json.dumps(compiled))
        engine.load_level(0)
        result = puzzlescript_astar(
            engine,
            compiled,
            lambda _ctx: 0.0,
            max_expansions=cap,
            timeout_s=timeout_s,
        )
        rows[env] = {
            "expanded": int(result.expanded_states),
            "solved": bool(result.solved),
        }
    return rows


def evaluate_builtin(
    script_doctor: Path,
    timeout_s: float,
    jobs: list[tuple[str, int]],
) -> dict[str, dict[str, int | bool]]:
    evaluator = PuzzleScriptEvaluator(script_doctor)
    rows: dict[str, dict[str, int | bool]] = {}
    for env, cap in jobs:
        compiled = json.loads(evaluator.compile_game_file(env))
        engine = evaluator.load_engine(json.dumps(compiled))
        engine.load_level(0)
        result = puzzlescript_astar(
            engine,
            compiled,
            builtin_heuristic,
            max_expansions=cap,
            timeout_s=timeout_s,
        )
        rows[env] = {
            "expanded": int(result.expanded_states),
            "solved": bool(result.solved),
        }
    return rows


def _load_run_puzzlescript_batch_module():
    module_path = REPO_ROOT / "scripts" / "run_puzzlescript_batch.py"
    spec = importlib.util.spec_from_file_location("run_puzzlescript_batch", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_base_prompt_holdout(
    script_doctor: Path,
    timeout_s: float,
    jobs: list[tuple[str, int]],
    llm_name: str,
) -> dict[str, dict[str, int | bool]]:
    import dspy

    runner = _load_run_puzzlescript_batch_module()
    evaluator = PuzzleScriptEvaluator(script_doctor)
    lm = dspy.LM(llm_name)
    dspy.configure(lm=lm)

    rows: dict[str, dict[str, int | bool]] = {}
    for env, cap in jobs:
        print(f"[base-holdout] synthesizing {env}", flush=True)
        game_text = runner.load_game_text(env, script_doctor)
        if not game_text:
            raise FileNotFoundError(f"Could not load holdout game text for {env}")
        json_str = evaluator.compile_game(game_text)
        compiled = json.loads(json_str)
        engine = evaluator.load_engine(json_str)
        engine.load_level(0)
        env_description = build_env_description(compiled, engine.get_id_dict(), game_text)
        heuristic_fn, _code, error = runner.synthesize_heuristic_from_prompt(
            runner.PUZZLESCRIPT_HEURISTIC_CONTRACT,
            env_description,
            lm,
        )
        if error:
            raise RuntimeError(f"Base prompt synthesis failed for {env}: {error}")
        engine.load_level(0)
        print(f"[base-holdout] evaluating {env}", flush=True)
        result = puzzlescript_astar(
            engine,
            compiled,
            lambda ctx, h=heuristic_fn: h(None, None, ctx),
            max_expansions=cap,
            timeout_s=timeout_s,
        )
        print(
            f"[base-holdout] {env} solved={result.solved} expanded={result.expanded_states}",
            flush=True,
        )
        rows[env] = {
            "expanded": int(result.expanded_states),
            "solved": bool(result.solved),
        }
    return rows


def build_holdout_rows(
    blind_results: dict[str, dict[str, int | bool]],
    builtin_results: dict[str, dict[str, int | bool]],
    base_results: dict[str, dict[str, int | bool]] | None,
    optimized_results: dict[str, dict[str, int | bool]],
) -> list[HoldoutComparisonRow]:
    rows: list[HoldoutComparisonRow] = []
    for env, cap in HOLDOUT_JOBS:
        blind_row = blind_results[env]
        builtin_row = builtin_results[env]
        base_row = base_results.get(env) if base_results is not None else None
        optimized_row = optimized_results[env]
        rows.append(
            HoldoutComparisonRow(
                env=env,
                cap=cap,
                blind_expanded=int(blind_row["expanded"]),
                blind_solved=bool(blind_row["solved"]),
                builtin_expanded=int(builtin_row["expanded"]),
                builtin_solved=bool(builtin_row["solved"]),
                base_expanded=(
                    int(base_row["expanded"]) if base_row is not None else None
                ),
                base_solved=(
                    bool(base_row["solved"]) if base_row is not None else None
                ),
                optimized_expanded=int(optimized_row["expanded"]),
                optimized_solved=bool(optimized_row["solved"]),
            )
        )
    rows.sort(
        key=lambda row: (
            row.optimized_expanded / row.cap,
            row.env.lower(),
        )
    )
    return rows


def plot_holdout_rows(rows: list[HoldoutComparisonRow], output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)

    include_base = any(row.base_expanded is not None for row in rows)
    labels = ["Blind A*", "Builtin Heuristic"]
    win_rates = [
        sum(row.blind_solved for row in rows) / len(rows),
        sum(row.builtin_solved for row in rows) / len(rows),
    ]
    if include_base:
        labels.append("Base Prompt")
        win_rates.append(
            sum(bool(row.base_solved) for row in rows) / len(rows)
        )
    labels.append("Optimized Prompt")
    win_rates.append(sum(row.optimized_solved for row in rows) / len(rows))

    colors = {
        "Blind A*": "#7f8c8d",
        "Builtin Heuristic": "#4c78a8",
        "Base Prompt": "#e67e22",
        "Optimized Prompt": "#1b9e77",
    }

    fig, (ax_rate, ax_eff) = plt.subplots(
        1,
        2,
        figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1.0, 2.2]},
    )

    ax_rate.bar(labels, win_rates, color=[colors[label] for label in labels], width=0.65)
    ax_rate.set_ylim(0.0, 1.05)
    ax_rate.set_ylabel("Win Rate Under Fixed Cap")
    ax_rate.set_title("Holdout Success")
    for idx, rate in enumerate(win_rates):
        wins = int(round(rate * len(rows)))
        ax_rate.text(idx, rate + 0.03, f"{wins}/{len(rows)}", ha="center", va="bottom")
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)

    y_positions = list(range(len(rows)))
    offsets = {
        "Blind A*": -0.27,
        "Builtin Heuristic": -0.09,
        "Base Prompt": 0.09,
        "Optimized Prompt": 0.27,
    }
    method_points = {
        "Blind A*": [(row.blind_expanded / row.cap, row.blind_solved) for row in rows],
        "Builtin Heuristic": [
            (row.builtin_expanded / row.cap, row.builtin_solved) for row in rows
        ],
        "Optimized Prompt": [(row.optimized_expanded / row.cap, row.optimized_solved) for row in rows],
    }
    if include_base:
        method_points["Base Prompt"] = [
            ((row.base_expanded or row.cap) / row.cap, bool(row.base_solved))
            for row in rows
        ]

    for method, points in method_points.items():
        xs = [point[0] for point in points]
        solved_mask = [point[1] for point in points]
        ys = [y + offsets[method] for y in y_positions]
        solved_x = [x for x, solved in zip(xs, solved_mask) if solved]
        solved_y = [y for y, solved in zip(ys, solved_mask) if solved]
        failed_x = [x for x, solved in zip(xs, solved_mask) if not solved]
        failed_y = [y for y, solved in zip(ys, solved_mask) if not solved]
        if solved_x:
            ax_eff.scatter(
                solved_x,
                solved_y,
                color=colors[method],
                marker="o",
                s=72,
                linewidths=0.8,
                edgecolors="black",
                zorder=3,
            )
        if failed_x:
            ax_eff.scatter(
                failed_x,
                failed_y,
                color=colors[method],
                marker="X",
                s=88,
                linewidths=0.8,
                edgecolors="black",
                zorder=3,
            )

    ax_eff.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_eff.set_yticks(y_positions)
    ax_eff.set_yticklabels([row.env.replace("_", " ") for row in rows])
    ax_eff.invert_yaxis()
    ax_eff.set_xlabel("Expanded States / Search Cap")
    ax_eff.set_title("Holdout Search Efficiency")
    ax_eff.set_xlim(0.0, max(max(x for x, _ in pts) for pts in method_points.values()) * 1.05)
    ax_eff.grid(axis="x", alpha=0.25)
    ax_eff.spines["top"].set_visible(False)
    ax_eff.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=colors[label],
            markeredgecolor="black",
            markersize=8,
        )
        for label in labels
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                label="solved",
                markerfacecolor="white",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="black",
                linestyle="None",
                label="failed under cap",
                markerfacecolor="white",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="cap boundary",
            ),
        ]
    )
    ax_eff.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.suptitle(
        "PuzzleScript Holdout Search Under Matched Global Cap",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = output_dir / "holdout_comparison_under_cap.png"
    svg_path = output_dir / "holdout_comparison_under_cap.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        base_prompt_proxy = load_base_prompt_proxy(args.base_state_root)
        optimized_results = evaluate_optimized(
            args.optimized_state_root,
            args.script_doctor,
            args.timeout_s,
            TRAINING_JOBS,
        )
        rows = build_rows(base_prompt_proxy, optimized_results)
        json_path = args.output_dir / "comparison_under_cap.json"
        json_path.write_text(
            json.dumps([asdict(row) for row in rows], indent=2) + "\n",
            encoding="utf-8",
        )
        plot_paths = plot_rows(rows, args.output_dir)
    else:
        blind_results = evaluate_blind(args.script_doctor, args.timeout_s, HOLDOUT_JOBS)
        builtin_results = evaluate_builtin(args.script_doctor, args.timeout_s, HOLDOUT_JOBS)
        base_results = None
        if args.include_base_holdout:
            base_results = evaluate_base_prompt_holdout(
                args.script_doctor,
                args.timeout_s,
                HOLDOUT_JOBS,
                args.llm,
            )
        optimized_results = evaluate_optimized(
            args.optimized_state_root,
            args.script_doctor,
            args.timeout_s,
            HOLDOUT_JOBS,
        )
        rows = build_holdout_rows(
            blind_results,
            builtin_results,
            base_results,
            optimized_results,
        )
        json_path = args.output_dir / "holdout_comparison_under_cap.json"
        json_path.write_text(
            json.dumps([asdict(row) for row in rows], indent=2) + "\n",
            encoding="utf-8",
        )
        plot_paths = plot_holdout_rows(rows, args.output_dir)

    print(f"wrote {json_path}")
    for path in plot_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
