"""PuzzleScript environment evaluator for GEPA.

Wraps the script-doctor C++ PuzzleScript backend to evaluate the quality
of candidate PuzzleScript environments (games). Used by GEPA to score
how good a generated environment is — solvability, difficulty, etc.

Requires:
  - script-doctor repo cloned with C++ extension built
  - Node.js available on PATH (for the JS game compiler)
  - PuzzleScript submodule cloned inside script-doctor
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SolvabilityResult:
    """Result of running a solver on one level."""
    solved: bool
    iterations: int
    time_s: float
    solution_length: int  # -1 if unsolved
    fps: float
    score: float
    timeout: bool


@dataclass
class RolloutResult:
    """Result of parallel random rollouts on one game."""
    any_win: bool
    win_count: int
    n_rollouts: int
    n_steps: int
    total_time_s: float
    steps_per_sec: float


@dataclass
class QualityScore:
    """Aggregated quality score for a candidate environment."""
    game_name: str
    compiled: bool
    n_levels: int
    n_objects: int
    grid_w: int
    grid_h: int
    heuristic_score: float  # 0-1, higher = closer to win
    solvability: Optional[SolvabilityResult]
    fitness: float  # final GEPA fitness score


class PuzzleScriptEvaluator:
    """Evaluates PuzzleScript environments for GEPA.

    Uses the script-doctor C++ backend for fast solving and the JS
    compiler for parsing PuzzleScript game text into the engine format.
    """

    def __init__(self, script_doctor_path: str | Path):
        self._root = Path(script_doctor_path).resolve()
        self._cpp = self._load_cpp_module()
        self._js_engine = None

    def _load_cpp_module(self):
        """Load the compiled C++ PuzzleScript module directly."""
        so_files = list(self._root.glob(
            "puzzlescript_cpp/_puzzlescript_cpp*.so"))
        if not so_files:
            raise FileNotFoundError(
                f"C++ extension not found in {self._root}/puzzlescript_cpp/. "
                "Run: cd script-doctor && python setup_cpp.py build_ext --inplace"
            )
        spec = importlib.util.spec_from_file_location(
            "puzzlescript_cpp._puzzlescript_cpp", str(so_files[0]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _get_js_engine(self):
        """Lazy-load the JS PuzzleScript compiler."""
        if self._js_engine is None:
            from javascript import require
            engine_js = str(self._root / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")
            self._js_engine = require(engine_js)
        return self._js_engine

    def compile_game(self, game_text: str) -> str:
        """Compile PuzzleScript game text to JSON via the JS compiler.

        Args:
            game_text: Raw PuzzleScript source code.

        Returns:
            JSON string that can be loaded into the C++ engine.

        Raises:
            RuntimeError: If compilation fails.
        """
        js = self._get_js_engine()
        try:
            js.compile(["restart"], game_text)
            return str(js.serializeCompiledStateJSON())
        except Exception as e:
            raise RuntimeError(f"Game compilation failed: {e}") from e

    def compile_game_file(self, game_name: str) -> str:
        """Compile a named game from the script-doctor data directories.

        Searches data/scraped_games/ and custom_games/ for {game_name}.txt.
        """
        for subdir in ("data/scraped_games", "custom_games"):
            path = self._root / subdir / f"{game_name}.txt"
            if path.is_file():
                return self.compile_game(path.read_text())
        raise FileNotFoundError(
            f"Game '{game_name}' not found in scraped_games or custom_games")

    def load_engine(self, json_str: str):
        """Create a C++ Engine loaded with a compiled game."""
        engine = self._cpp.Engine()
        if not engine.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled game into C++ engine")
        return engine

    def get_game_info(self, json_str: str) -> dict:
        """Get basic game metadata from compiled JSON."""
        engine = self.load_engine(json_str)
        engine.load_level(0)
        return {
            "n_levels": engine.get_num_levels(),
            "n_objects": engine.get_object_count(),
            "grid_w": engine.get_width(),
            "grid_h": engine.get_height(),
        }

    def evaluate_solvability(
        self,
        json_str: str,
        level_i: int = 0,
        algo: str = "astar",
        max_iters: int = 50_000,
        timeout_ms: int = 5_000,
    ) -> SolvabilityResult:
        """Run a solver on one level and return the result.

        Args:
            json_str: Compiled game JSON.
            level_i: Which level to solve.
            algo: One of 'bfs', 'astar', 'gbfs'.
            max_iters: Maximum solver iterations.
            timeout_ms: Timeout in milliseconds (-1 for unlimited).
        """
        engine = self.load_engine(json_str)
        engine.load_level(level_i)

        solve_fn = {
            "bfs": self._cpp.solve_bfs,
            "astar": self._cpp.solve_astar,
            "gbfs": self._cpp.solve_gbfs,
        }[algo]

        t0 = time.perf_counter()
        raw = solve_fn(engine, max_iters, timeout_ms)
        elapsed = time.perf_counter() - t0
        fps = raw.iterations / elapsed if elapsed > 0 else 0

        return SolvabilityResult(
            solved=raw.won,
            iterations=raw.iterations,
            time_s=elapsed,
            solution_length=len(raw.actions) if raw.won else -1,
            fps=fps,
            score=raw.score,
            timeout=raw.timeout,
        )

    def evaluate_batch(
        self,
        json_strs: list[str],
        level_indices: Optional[list[int]] = None,
        algo: str = "astar",
        max_iters: int = 50_000,
        timeout_ms: int = 5_000,
        max_workers: int = 8,
    ) -> list[SolvabilityResult]:
        """Evaluate multiple games in parallel using threads.

        The C++ solver releases the GIL, so thread-based parallelism
        provides real speedup for cross-game evaluation.
        """
        if level_indices is None:
            level_indices = [0] * len(json_strs)

        def _eval_one(args):
            json_str, level_i = args
            return self.evaluate_solvability(
                json_str, level_i, algo, max_iters, timeout_ms)

        workers = min(max_workers, len(json_strs))
        if workers <= 1:
            return [_eval_one((js, li))
                    for js, li in zip(json_strs, level_indices)]

        results = [None] * len(json_strs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_eval_one, (json_strs[i], level_indices[i])): i
                for i in range(len(json_strs))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def evaluate_batch_random_rollout(
        self,
        json_str: str,
        level_i: int = 0,
        n_rollouts: int = 32,
        n_steps: int = 300,
    ) -> RolloutResult:
        """Run parallel random rollouts on one game using BatchedEngine.

        All rollouts execute the same level in parallel. Useful for
        estimating solvability when A* is too expensive.
        """
        be = self._cpp.BatchedEngine(n_rollouts)
        if not be.load_from_json(json_str):
            raise RuntimeError("Failed to load game into BatchedEngine")
        be.set_levels([level_i] * n_rollouts)
        be.reset_all()

        t0 = time.perf_counter()
        any_win = False
        win_count = 0
        for step in range(n_steps):
            actions = np.random.randint(0, 5, size=n_rollouts, dtype=np.int32)
            be.step(actions)
            wins = np.array(be.get_wins())
            step_wins = int(wins.sum())
            if step_wins > 0:
                any_win = True
                win_count += step_wins
            dones = np.array(be.get_dones())
            done_indices = np.where(dones)[0].tolist()
            if done_indices:
                be.reset(done_indices)
        elapsed = time.perf_counter() - t0
        total_steps = n_rollouts * n_steps
        sps = total_steps / elapsed if elapsed > 0 else 0

        return RolloutResult(
            any_win=any_win,
            win_count=win_count,
            n_rollouts=n_rollouts,
            n_steps=n_steps,
            total_time_s=elapsed,
            steps_per_sec=sps,
        )

    def compute_heuristic(self, json_str: str, level_i: int = 0) -> float:
        """Compute a fast heuristic quality score for a game level.

        Returns a value in [0, 1] where 1 means "already at win state"
        and 0 means "maximally far from winning". Based on the engine's
        built-in normalized score.
        """
        engine = self.load_engine(json_str)
        engine.load_level(level_i)
        return float(engine.get_score_normalized())

    def score_candidate(
        self,
        game_text: str,
        game_name: str = "unnamed",
        heuristic_threshold: float = 0.0,
        solver_algo: str = "astar",
        solver_max_iters: int = 50_000,
        solver_timeout_ms: int = 5_000,
    ) -> QualityScore:
        """Full GEPA scoring pipeline for one candidate environment.

        1. Compile game text
        2. Get game metadata
        3. Compute heuristic (fast)
        4. If heuristic > threshold: run A* solver
        5. Compute fitness score

        Fitness formula:
          - If solvable: 1.0 - (solution_length / max_solution_length_cap)
            clamped to [0.1, 1.0]
          - If unsolvable: heuristic_score * 0.1
          - If compilation fails: 0.0
        """
        # Step 1: Compile
        try:
            json_str = self.compile_game(game_text)
        except (RuntimeError, Exception):
            return QualityScore(
                game_name=game_name, compiled=False, n_levels=0,
                n_objects=0, grid_w=0, grid_h=0,
                heuristic_score=0.0, solvability=None, fitness=0.0,
            )

        # Step 2: Metadata
        info = self.get_game_info(json_str)

        # Step 3: Heuristic
        heuristic = self.compute_heuristic(json_str, level_i=0)

        # Step 4: Solver (if heuristic passes threshold)
        solvability = None
        if heuristic >= heuristic_threshold:
            try:
                solvability = self.evaluate_solvability(
                    json_str, level_i=0, algo=solver_algo,
                    max_iters=solver_max_iters, timeout_ms=solver_timeout_ms,
                )
            except Exception:
                pass

        # Step 5: Fitness
        max_solution_cap = 200
        if solvability and solvability.solved:
            raw = 1.0 - (solvability.solution_length / max_solution_cap)
            fitness = max(0.1, min(1.0, raw))
        elif solvability and not solvability.solved:
            fitness = heuristic * 0.1
        else:
            fitness = heuristic * 0.05

        return QualityScore(
            game_name=game_name,
            compiled=True,
            n_levels=info["n_levels"],
            n_objects=info["n_objects"],
            grid_w=info["grid_w"],
            grid_h=info["grid_h"],
            heuristic_score=heuristic,
            solvability=solvability,
            fitness=fitness,
        )
