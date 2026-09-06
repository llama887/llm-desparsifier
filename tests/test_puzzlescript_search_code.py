from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "llm_desparsifier" / "search"
sys.path[:0] = [str(ROOT), str(SEARCH)]

import puzzlescript_astar as astar_module
from puzzlescript_astar import puzzlescript_search
from puzzlescript_sanitizer import sanitize_and_compile_puzzlescript_search

import scripts.run_puzzlescript_batched_gepa as runner
from scripts.run_puzzlescript_batched_gepa import evaluate_search_task


class TinyEngine:
    def __init__(self) -> None:
        self.state = 0

    def backup_level(self):
        return self.state

    def load_level(self, _level):
        self.state = 0

    def restore_level(self, state):
        self.state = state

    def process_input(self, action):
        if action == 0 and self.state < 2:
            self.state += 1
            return True
        return False

    def is_againing(self):
        return False

    def is_winning(self):
        return self.state == 2

    def get_objects(self):
        return [self.state]

    def has_metadata(self, _name):
        return True


def _ctx(engine, _compiled):
    return {
        "score_normalized": engine.state / 2,
        "is_winning": engine.is_winning(),
        "object_positions": {},
        "ascii_state": str(engine.state),
    }


def test_legacy_heuristic_and_custom_search_share_result_schema(monkeypatch) -> None:
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    kind, heuristic = sanitize_and_compile_puzzlescript_search(
        "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0\n"
    )
    legacy = puzzlescript_search(TinyEngine(), {}, kind, heuristic, max_expansions=10)

    kind, custom = sanitize_and_compile_puzzlescript_search(
        "def search_plan(api, seed):\n"
        "    todo = [(api.initial(), [])]\n"
        "    seen = set()\n"
        "    while todo:\n"
        "        state, path = todo.pop(0)\n"
        "        if api.key(state) in seen:\n"
        "            continue\n"
        "        seen.add(api.key(state))\n"
        "        if api.is_winning(state):\n"
        "            return path\n"
        "        for action, child in api.successors(state):\n"
        "            todo.append((child, path + [action]))\n"
        "    return []\n"
    )
    custom_result = puzzlescript_search(
        TinyEngine(), {}, kind, custom, max_expansions=10, seed=7
    )

    assert legacy.solved and custom_result.solved
    assert custom_result.actions == [0, 0]
    assert set(vars(legacy)) == set(vars(custom_result))
    assert legacy.trace_summary["search_strategy"] == "legacy_astar"
    assert custom_result.trace_summary["search_strategy"] == "custom_search"


def test_custom_search_budget_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    kind, custom = sanitize_and_compile_puzzlescript_search(
        "def search_plan(api, seed):\n"
        "    state = api.initial()\n"
        "    while True:\n"
        "        api.successors(state)\n"
    )
    result = puzzlescript_search(TinyEngine(), {}, kind, custom, max_expansions=2)
    assert not result.solved
    assert result.expanded_states == 2
    assert result.trace_summary["terminated_reason"] == "expansion_budget"


def test_custom_search_can_read_budget_and_use_repr(monkeypatch) -> None:
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    kind, custom = sanitize_and_compile_puzzlescript_search(
        "def search_plan(api, seed):\n"
        "    assert repr(api.key(api.initial()))\n"
        "    assert api.expansion_budget() == 3\n"
        "    return []\n"
    )
    result = puzzlescript_search(TinyEngine(), {}, kind, custom, max_expansions=3)
    assert result.trace_summary["terminated_reason"] == "search_exhausted"


def test_custom_search_reports_validated_algorithm_label(monkeypatch) -> None:
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    kind, custom = sanitize_and_compile_puzzlescript_search(
        'SEARCH_STRATEGY = "novelty_hybrid"\n'
        "def search_plan(api, seed):\n"
        "    return []\n"
    )
    result = puzzlescript_search(TinyEngine(), {}, kind, custom, max_expansions=2)

    assert result.trace_summary["search_strategy"] == "custom_search"
    assert result.trace_summary["search_algorithm"] == "novelty_hybrid"


@pytest.mark.parametrize("label", ["", "contains spaces", "x" * 49])
def test_custom_search_rejects_invalid_algorithm_label(label: str) -> None:
    with pytest.raises(ValueError, match="SEARCH_STRATEGY"):
        sanitize_and_compile_puzzlescript_search(
            f"SEARCH_STRATEGY = {label!r}\n"
            "def search_plan(api, seed):\n"
            "    return []\n"
        )


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef search_plan(api, seed):\n    return []\n",
        "def search_plan(api, seed):\n    return api.__class__\n",
        "def search_plan(api, seed):\n    return open('/tmp/x')\n",
        "def search_plan(api, seed):\n    return vars(api)\n",
    ],
)
def test_custom_search_rejects_unsafe_code(code: str) -> None:
    with pytest.raises(ValueError):
        sanitize_and_compile_puzzlescript_search(code)


def test_artifact_must_define_exactly_one_entrypoint() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        sanitize_and_compile_puzzlescript_search(
            "def heuristic_cost_to_go(ts, env_params, ctx):\n    return 0.0\n"
            "def search_plan(api, seed):\n    return []\n"
        )


def test_custom_search_runs_through_shared_task_evaluator(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    code = tmp_path / "search.py"
    game = tmp_path / "game.txt"
    code.write_text(
        "def search_plan(api, seed):\n"
        "    frontier = [(api.initial(), [])]\n"
        "    while frontier:\n"
        "        state, path = frontier.pop(0)\n"
        "        if api.is_winning(state): return path\n"
        "        for action, child in api.successors(state):\n"
        "            frontier.append((child, path + [action]))\n"
        "    return []\n"
    )
    game.write_text("fake")

    class Evaluator:
        def compile_game(self, _text):
            return "{}"

        def load_engine(self, _json):
            return TinyEngine()

    result = evaluate_search_task(
        evaluator=Evaluator(),
        task={
            "task_id": 1,
            "game": "tiny",
            "level": 0,
            "budget": 10,
            "game_text_path": str(game),
            "heuristic_code_path": str(code),
            "replicate": 3,
        },
        astar_timeout_s=1,
    )
    assert result["solved"] is True
    assert result["search_strategy"] == "custom_search"
    assert result["trace_summary"]["seed"] == 3


def test_search_sandbox_exposes_documented_safe_builtins() -> None:
    strategy, search = sanitize_and_compile_puzzlescript_search(
        "SEARCH_STRATEGY = 'builtin_probe'\n"
        "def search_plan(api, seed):\n"
        "    try:\n"
        "        ord(None)\n"
        "    except TypeError:\n"
        "        return [ord('a') - 97]\n"
        "    except Exception:\n"
        "        return []\n"
    )

    assert strategy == "custom_search"
    assert search(None, 0) == [0]


def test_cpu_launcher_keeps_holdout_out_of_optimization() -> None:
    launcher = (ROOT / "sbatch" / "train_sokoban_search_code_gepa_cpu.s").read_text()
    gepa = launcher.index("scripts/run_puzzlescript_batched_gepa.py")
    holdout = launcher.index("scripts/compare_puzzlescript_batched_prompts.py")
    assert gepa < holdout
    assert "--no-reflection-artifact-tools" in launcher
    assert "smoke|train|full" in launcher
    assert 'if [ "$MODE" = full ]' in launcher
    # Both experiments now live in one repository, so the search-code jobs
    # need their own log namespace. Sharing sbatch/logs/ with the
    # heuristic-prompt experiment made concurrent runs overwrite each
    # other's controller logs, which is how a dead run was mistaken for a
    # live one.
    assert "sbatch/logs/search_code/" in launcher
    assert "sbatch/logs/%x-%j" not in launcher
    # The model that writes the code and the model that rewrites the prompt
    # are pinned separately: synthesis is the one whose output is executed
    # and scored, so it gets the stronger coding model at high effort.
    assert '${SYNTHESIS_MODEL:-gpt-5.6-luna}' in launcher
    assert '${SYNTHESIS_EFFORT:-high}' in launcher
    assert '${GEPA_MODEL:-gpt-6-astra}' in launcher
    assert "#SBATCH --mem=64G" in launcher
    assert '--llm-concurrency "${LLM_CONCURRENCY:-32}"' in launcher
    assert "--synthesis-cache-dir" in launcher
    # The objective, its reference table, and the level holdout must all be
    # pinned in the launcher, not left to defaults.
    assert '--objective "${OBJECTIVE:-blind-relative-time}"' in launcher
    assert "--blind-reference" in launcher
    assert "--sibling-level-holdout" in launcher
    assert "--require-blind-reference" in launcher
    # Frontier levels and the timing floor are part of what the objective
    # means, so they are pinned here too. Without the frontier the run only
    # ever measures levels plain A* already solves; without the floor a
    # wall-time ratio on a millisecond solve measures cluster noise.
    assert "--include-frontier-levels" in launcher
    assert '--min-reference-seconds "${MIN_REFERENCE_SECONDS:-1.0}"' in launcher
    # The weighted-sum knobs belong to the `adjusted` objective only. Leaving
    # them on the command line implied they still shaped the score.
    assert "--lost-solve-penalty" not in launcher
    assert "--new-solve-bonus" not in launcher
    assert "--common-solve-efficiency-weight" not in launcher
    # SLURM_JOB_PARTITION can be an alias that is not a valid submit target.
    assert 'POOL_PARTITION="${POOL_PARTITION:-' in launcher
    assert '--time="$POOL_TIME"' in launcher
    assert 'deadline=$((SECONDS + 21600))' in launcher


def test_search_api_does_not_retain_decoded_state(monkeypatch) -> None:
    """Only backups are kept per state; ctx and key live in a bounded cache."""
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    api = astar_module.PuzzleScriptSearchAPI(
        TinyEngine(), {}, max_expansions=10, timeout_s=5.0, ctx_cache_size=1
    )
    api.successors(api.initial())
    api.successors(1)

    assert all(len(entry) == 2 for entry in api._states)
    assert len(api._decoded) == 1
    # An evicted state is rebuilt from its backup rather than lost.
    assert api.key(api.initial()) == (0,)
    assert api.ctx(api.initial())["ascii_state"] == "0"
    assert api.is_winning(2) is True


def test_search_api_rebuilds_evicted_ctx_consistently(monkeypatch) -> None:
    monkeypatch.setattr(astar_module, "build_puzzlescript_ctx", _ctx)
    roomy = astar_module.PuzzleScriptSearchAPI(
        TinyEngine(), {}, max_expansions=10, timeout_s=5.0, ctx_cache_size=64
    )
    tight = astar_module.PuzzleScriptSearchAPI(
        TinyEngine(), {}, max_expansions=10, timeout_s=5.0, ctx_cache_size=1
    )
    for api in (roomy, tight):
        api.successors(api.initial())
        api.successors(1)

    for state in range(3):
        assert roomy.key(state) == tight.key(state)
        assert roomy.ctx(state) == tight.ctx(state)
        assert roomy.is_winning(state) == tight.is_winning(state)


def test_search_task_memory_limit_is_configurable(monkeypatch) -> None:
    from scripts.run_puzzlescript_batched_gepa import (
        DEFAULT_SEARCH_TASK_MEMORY_LIMIT_MB,
        search_task_memory_limit_mb,
    )

    monkeypatch.delenv("LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB", raising=False)
    assert search_task_memory_limit_mb() == DEFAULT_SEARCH_TASK_MEMORY_LIMIT_MB
    monkeypatch.setenv("LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB", "512")
    assert search_task_memory_limit_mb() == 512
    monkeypatch.setenv("LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB", "0")
    assert search_task_memory_limit_mb() == 0
    monkeypatch.setenv("LLM_DESPARSIFIER_SEARCH_MEM_LIMIT_MB", "not-a-number")
    assert search_task_memory_limit_mb() == DEFAULT_SEARCH_TASK_MEMORY_LIMIT_MB


def test_memory_limit_round_trips(monkeypatch) -> None:
    import resource

    from scripts.run_puzzlescript_batched_gepa import (
        apply_search_task_memory_limit,
        restore_memory_limit,
    )

    before = resource.getrlimit(resource.RLIMIT_DATA)
    previous = apply_search_task_memory_limit(8192)
    try:
        if previous is not None:
            assert resource.getrlimit(resource.RLIMIT_DATA)[0] == 8192 * 1024 * 1024
    finally:
        restore_memory_limit(previous)
    assert resource.getrlimit(resource.RLIMIT_DATA) == before


def test_codex_error_detail_keeps_the_tail() -> None:
    """Codex echoes the prompt before reporting why it failed."""
    from scripts.run_puzzlescript_batched_gepa import truncate_middle_text

    text = "PROMPT" * 5000 + "RATE LIMIT REACHED"
    detail = truncate_middle_text(text, 400)
    assert len(detail) < len(text)
    assert detail.startswith("PROMPT")
    assert detail.endswith("RATE LIMIT REACHED")
    assert "truncated" in detail
    assert truncate_middle_text("short", 400) == "short"


def test_sibling_pairing_never_shows_the_scored_level() -> None:
    from scripts.run_puzzlescript_batched_gepa import assign_sibling_dev_levels

    for count in range(2, 12):
        levels = list(range(count))
        for seed in range(5):
            pairing = assign_sibling_dev_levels(levels, seed=seed)
            assert set(pairing) == set(levels)
            assert all(dev != level for level, dev in pairing.items())
            # Every level is used exactly once as a development level, so no
            # level is over-represented in what the agent gets to see.
            assert sorted(pairing.values()) == levels


def test_sibling_pairing_degrades_gracefully_for_one_level() -> None:
    from scripts.run_puzzlescript_batched_gepa import assign_sibling_dev_levels

    assert assign_sibling_dev_levels([7]) == {7: 7}
    assert assign_sibling_dev_levels([]) == {}


def test_task_visible_level_defaults_to_scored_level() -> None:
    from scripts.run_puzzlescript_batched_gepa import PuzzleScriptLevelTask

    plain = PuzzleScriptLevelTask(
        task_id=0, game="g", level=3, budget=10, env_description="", game_text_path="x"
    )
    assert plain.visible_level == 3
    assert plain.has_sibling_holdout is False

    held = PuzzleScriptLevelTask(
        task_id=0, game="g", level=3, budget=10, env_description="", game_text_path="x",
        dev_level=5,
    )
    assert held.visible_level == 5
    assert held.has_sibling_holdout is True


def test_workspace_hides_the_scored_level(tmp_path) -> None:
    from scripts.run_puzzlescript_batched_gepa import (
        PuzzleScriptLevelTask,
        build_codex_synthesis_workspace,
    )

    game = tmp_path / "game.puzzlescript"
    game.write_text("title t\n", encoding="utf-8")
    task = PuzzleScriptLevelTask(
        task_id=0, game="g", level=3, budget=10, env_description="",
        game_text_path=str(game), dev_level=5,
    )
    files = build_codex_synthesis_workspace(task, script_doctor=tmp_path)
    assert '"level": 5' in files["evaluate.py"]
    assert '"level": 3' not in files["evaluate.py"]
    assert "DIFFERENT, undisclosed level" in files["README.md"]
    # The old contract explicitly invited a memorized plan; it must be gone.
    assert "including a replay-validated plan" not in files["README.md"]


def test_replay_solves_are_excluded_from_the_efficiency_signal() -> None:
    from scripts.run_puzzlescript_batched_gepa import (
        _common_solve_efficiency_delta,
        _is_replay_solve,
        _is_searched_common_solve,
    )

    replay = {
        "solve_rate": 1.0, "baseline_solve_rate": 1.0,
        "solved_expanded_mean": 0.0, "baseline_solved_expanded_mean": 0.0,
    }
    assert _is_replay_solve(replay) is True
    assert _is_searched_common_solve(replay) is False
    assert _common_solve_efficiency_delta(replay, clip=1.0) == 0.0

    # A real search that halved its expansions must still register.
    searched = {
        "solve_rate": 1.0, "baseline_solve_rate": 1.0,
        "solved_expanded_mean": 1000.0, "baseline_solved_expanded_mean": 2000.0,
    }
    assert _is_replay_solve(searched) is False
    assert _is_searched_common_solve(searched) is True
    assert _common_solve_efficiency_delta(searched, clip=1.0) > 0.0

    # A candidate that replaces real search with a memorized plan is not a
    # measured efficiency win.
    became_replay = {
        "solve_rate": 1.0, "baseline_solve_rate": 1.0,
        "solved_expanded_mean": 0.0, "baseline_solved_expanded_mean": 2000.0,
    }
    assert _is_searched_common_solve(became_replay) is False
    assert _common_solve_efficiency_delta(became_replay, clip=1.0) == 0.0


def test_blind_budget_tracks_measured_difficulty() -> None:
    from scripts.run_puzzlescript_batched_gepa import blind_reference_budget

    solved = {"blind_solved": True, "blind_expanded": 800, "ceiling": 50000}
    assert blind_reference_budget(solved, multiplier=2.0, fallback=10000) == 1600
    # The ceiling still bounds a level whose blind cost is huge.
    huge = {"blind_solved": True, "blind_expanded": 40000, "ceiling": 50000}
    assert blind_reference_budget(huge, multiplier=2.0, fallback=10000) == 50000
    # No blind solve means no measured cost, so the ceiling stands.
    unsolved = {"blind_solved": False, "blind_expanded": 50000, "ceiling": 50000}
    assert blind_reference_budget(unsolved, multiplier=2.0, fallback=10000) == 50000
    # A level missing from the reference falls back to the flat budget.
    assert blind_reference_budget(None, multiplier=2.0, fallback=10000) == 10000


def test_speedup_is_measured_against_blind_not_the_previous_prompt() -> None:
    import math

    from scripts.run_puzzlescript_batched_gepa import row_blind_speedup_log2

    ref = {("g", 0): {"blind_solved": True, "blind_expanded": 999}}

    fast = {"game": "g", "level": 0, "solve_rate": 1.0, "solved_expanded_mean": 249.0}
    assert math.isclose(row_blind_speedup_log2(fast, ref), 2.0, rel_tol=1e-9)

    # A precomputed plan expands nothing. That is not an infinite speedup.
    replay = {"game": "g", "level": 0, "solve_rate": 1.0, "solved_expanded_mean": 0.0}
    assert row_blind_speedup_log2(replay, ref) == 0.0

    unsolved = {"game": "g", "level": 0, "solve_rate": 0.0, "solved_expanded_mean": 0.0}
    assert row_blind_speedup_log2(unsolved, ref) is None

    # Levels blind search never solved carry no reference.
    no_ref = {"game": "g", "level": 7, "solve_rate": 1.0, "solved_expanded_mean": 10.0}
    assert row_blind_speedup_log2(no_ref, ref) is None


def test_constrained_objective_rewards_speedup_and_gates_on_solves() -> None:
    from scripts.run_puzzlescript_batched_gepa import constrained_speedup_scores

    ref = {
        ("g", 0): {"blind_solved": True, "blind_expanded": 999},
        ("g", 1): {"blind_solved": True, "blind_expanded": 999},
    }

    def row(level, expanded, solved=1.0, base_solved=1.0):
        return {
            "game": "g", "level": level,
            "solve_rate": solved, "baseline_solve_rate": base_solved,
            "solved_expanded_mean": expanded,
        }

    feasible = [row(0, 249.0), row(1, 249.0)]
    scores = constrained_speedup_scores(feasible, ref)
    assert abs(sum(scores) / len(scores) - 2.0) < 1e-9

    # Halving expansions again doubles the log2 speedup.
    faster = [row(0, 124.0), row(1, 124.0)]
    assert sum(constrained_speedup_scores(faster, ref)) > sum(scores)

    # Losing a solve is infeasible no matter how fast the rest got.
    lost = [row(0, 1.0), row(1, 0.0, solved=0.0, base_solved=1.0)]
    lost_scores = constrained_speedup_scores(lost, ref)
    assert sum(lost_scores) / len(lost_scores) < -100.0

    # Slack lets a single loss through when it is explicitly allowed.
    slack_scores = constrained_speedup_scores(lost, ref, solve_slack=0.5)
    assert sum(slack_scores) / len(slack_scores) > -100.0


def test_objective_switch_routes_all_scoring_paths() -> None:
    from scripts.run_puzzlescript_batched_gepa import (
        OBJECTIVE_ADJUSTED,
        OBJECTIVE_SPEEDUP_CONSTRAINED,
        adjusted_candidate_scores,
        configure_objective,
    )

    ref = {("g", 0): {"blind_solved": True, "blind_expanded": 999}}
    rows = [{
        "game": "g", "level": 0,
        "solve_rate": 1.0, "baseline_solve_rate": 1.0,
        "solved_expanded_mean": 249.0, "baseline_solved_expanded_mean": 249.0,
    }]
    try:
        configure_objective(mode=OBJECTIVE_SPEEDUP_CONSTRAINED, blind_reference=ref)
        assert abs(adjusted_candidate_scores(rows)[0] - 2.0) < 1e-9
    finally:
        configure_objective(mode=OBJECTIVE_ADJUSTED)
    # Back on the default objective an unchanged candidate scores neutrally.
    assert abs(adjusted_candidate_scores(rows)[0]) < 1e-9


def test_constrained_objective_requires_a_reference() -> None:
    import pytest as _pytest

    from scripts.run_puzzlescript_batched_gepa import (
        OBJECTIVE_ADJUSTED,
        OBJECTIVE_SPEEDUP_CONSTRAINED,
        configure_objective,
    )

    try:
        with _pytest.raises(ValueError, match="blind reference"):
            configure_objective(mode=OBJECTIVE_SPEEDUP_CONSTRAINED, blind_reference={})
    finally:
        configure_objective(mode=OBJECTIVE_ADJUSTED)


def test_codex_quota_failure_is_fatal_immediately() -> None:
    from scripts.run_puzzlescript_batched_gepa import _CodexHealth

    health = _CodexHealth(budget=100)
    assert health.record_failure("connection reset") is None
    assert health.tripped is False
    reason = health.record_failure("You've hit your usage limit for this week")
    assert reason is not None and health.tripped is True


def test_codex_transport_failures_trip_after_the_budget() -> None:
    from scripts.run_puzzlescript_batched_gepa import _CodexHealth

    health = _CodexHealth(budget=3)
    assert health.record_failure("boom") is None
    assert health.record_failure("boom") is None
    # A success in between means the backend is alive, so the count restarts.
    health.record_success()
    assert health.record_failure("boom") is None
    assert health.record_failure("boom") is None
    assert health.record_failure("boom") is not None
    assert health.tripped is True


def test_transient_pushback_is_not_treated_as_exhausted_quota() -> None:
    from scripts.run_puzzlescript_batched_gepa import _CodexHealth, classify_codex_failure

    assert classify_codex_failure("429 Too Many Requests") == "transient"
    assert classify_codex_failure("stream error: connection reset") == "transient"
    assert classify_codex_failure("upstream returned 503") == "transient"
    assert classify_codex_failure("You've hit your usage limit") == "terminal"
    assert classify_codex_failure("insufficient_quota") == "terminal"
    assert classify_codex_failure("segmentation fault") == "unknown"

    # Rate limiting is routine at high concurrency; it must not end the run on
    # sight, but it still counts toward the consecutive-failure budget.
    health = _CodexHealth(budget=100)
    assert health.record_failure("429 Too Many Requests") is None
    assert health.tripped is False
    assert health.record_failure("You've hit your usage limit") is not None
    assert health.tripped is True


def test_synthesis_cache_key_covers_everything_the_agent_sees() -> None:
    from scripts.run_puzzlescript_batched_gepa import synthesis_cache_key

    base = dict(
        prompt="p", workspace_files={"a.py": "x"}, model="m",
        reasoning_effort="high", agentic=True, replicate=0,
    )
    key = synthesis_cache_key(**base)
    assert synthesis_cache_key(**base) == key

    for field, value in (
        ("prompt", "p2"),
        ("model", "m2"),
        ("reasoning_effort", "low"),
        ("agentic", False),
        ("replicate", 1),
    ):
        assert synthesis_cache_key(**{**base, field: value}) != key, field
    # A changed workspace file, and an added one, are both misses.
    assert synthesis_cache_key(**{**base, "workspace_files": {"a.py": "y"}}) != key
    assert synthesis_cache_key(**{**base, "workspace_files": {"a.py": "x", "b.py": "z"}}) != key


def test_synthesis_cache_round_trips_and_counts(tmp_path) -> None:
    from scripts.run_puzzlescript_batched_gepa import SynthesisCache

    disabled = SynthesisCache(None)
    assert disabled.enabled is False
    assert disabled.get("k") is None
    disabled.put("k", "code")  # must not raise

    cache = SynthesisCache(tmp_path / "store")
    assert cache.get("k") is None and cache.misses == 1
    cache.put("k", "def search_plan(api, seed):\n    return []\n")
    assert "search_plan" in cache.get("k")
    assert cache.hits == 1
    # Empty artifacts are never stored, so a failed synthesis cannot poison it.
    cache.put("empty", "   ")
    assert cache.get("empty") is None


def test_baseline_provenance_refuses_a_mismatched_seed(tmp_path) -> None:
    import json

    import pytest as _pytest

    from scripts.run_puzzlescript_batched_gepa import (
        PuzzleScriptLevelTask,
        baseline_provenance,
        check_baseline_provenance,
    )

    tasks = [
        PuzzleScriptLevelTask(
            task_id=0, game="g", level=1, budget=100,
            env_description="", game_text_path="x", dev_level=2,
        )
    ]
    expected = baseline_provenance(
        prompt_text="base", model="m", reasoning_effort="high",
        agentic=True, tasks=tasks,
    )

    outputs = tmp_path / "scoring_baseline_outputs.json"
    outputs.write_text("[]", encoding="utf-8")

    # No sidecar at all is a refusal, not a silent pass.
    with _pytest.raises(RuntimeError, match="provenance"):
        check_baseline_provenance(outputs, expected)

    sidecar = outputs.with_suffix(outputs.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(expected), encoding="utf-8")
    check_baseline_provenance(outputs, expected)  # matching: no raise

    # A baseline built under a different prompt must not seed this run.
    other = baseline_provenance(
        prompt_text="DIFFERENT", model="m", reasoning_effort="high",
        agentic=True, tasks=tasks,
    )
    sidecar.write_text(json.dumps(other), encoding="utf-8")
    with _pytest.raises(RuntimeError, match="prompt_sha256"):
        check_baseline_provenance(outputs, expected)
    check_baseline_provenance(outputs, expected, allow_mismatch=True)  # explicit override


def test_baseline_provenance_notices_a_changed_task_set() -> None:
    from scripts.run_puzzlescript_batched_gepa import PuzzleScriptLevelTask, baseline_provenance

    def task(level, budget, dev):
        return PuzzleScriptLevelTask(
            task_id=0, game="g", level=level, budget=budget,
            env_description="", game_text_path="x", dev_level=dev,
        )

    base = baseline_provenance(
        prompt_text="p", model="m", reasoning_effort="high",
        agentic=True, tasks=[task(1, 100, 2)],
    )
    for changed in (task(3, 100, 2), task(1, 999, 2), task(1, 100, 5)):
        other = baseline_provenance(
            prompt_text="p", model="m", reasoning_effort="high",
            agentic=True, tasks=[changed],
        )
        assert other["task_set_sha256"] != base["task_set_sha256"]


def _timed_row(level, base_t, cand_t, *, solved=1.0, base_solved=1.0,
               base_exp=1000.0, cand_exp=1000.0):
    return {
        "game": "g", "level": level,
        "solve_rate": solved, "baseline_solve_rate": base_solved,
        "solved_time_mean": cand_t, "baseline_solved_time_mean": base_t,
        "solved_expanded_mean": cand_exp, "baseline_solved_expanded_mean": base_exp,
    }


def test_base_relative_time_measures_improvement_over_the_seed_prompt() -> None:
    from scripts.run_puzzlescript_batched_gepa import (
        base_relative_time_scores,
        row_base_relative_time_log2,
    )

    # Half the wall time is one log2 of improvement.
    assert abs(row_base_relative_time_log2(_timed_row(0, 1.0, 0.5)) - 1.0) < 1e-9
    assert abs(row_base_relative_time_log2(_timed_row(0, 1.0, 2.0)) + 1.0) < 1e-9

    rows = [_timed_row(0, 1.0, 0.5), _timed_row(1, 2.0, 1.0)]
    scores = base_relative_time_scores(rows)
    assert abs(sum(scores) / len(scores) - 1.0) < 1e-9

    # An unchanged candidate scores exactly zero, not something near it.
    same = [_timed_row(0, 1.0, 1.0), _timed_row(1, 2.0, 2.0)]
    assert abs(sum(base_relative_time_scores(same))) < 1e-9


def test_instant_solves_are_clamped_not_discarded() -> None:
    from scripts.run_puzzlescript_batched_gepa import row_base_relative_time_log2

    # Two instant solves are a genuine tie.
    assert abs(row_base_relative_time_log2(_timed_row(0, 0.0, 0.0))) < 1e-9
    # Replacing an instant solve with a one-second one is a large regression,
    # and must not be silently dropped for being hard to time.
    assert row_base_relative_time_log2(_timed_row(0, 0.0, 1.0)) < -10.0
    # A level neither prompt solved carries no information either.
    assert row_base_relative_time_log2(_timed_row(0, 1.0, 1.0, solved=0.0)) is None


def test_strategy_switch_cannot_fake_an_improvement() -> None:
    """Regression: the false positive that was accepted in v9 iteration 1.

    The base prompt solved 8 levels with an internally-modelled search, which
    reports zero engine expansions, and the candidate converted them to engine
    search. Scored on expansions that read as a large gain while wall time was
    3.4x worse. Timing the solve must call that what it is: a regression.
    """
    from scripts.run_puzzlescript_batched_gepa import (
        base_relative_time_scores,
        constrained_speedup_scores,
    )

    # Eight levels: base solves instantly with its own model (0 expansions),
    # candidate uses the engine, expanding a lot and taking far longer.
    converted = [
        _timed_row(i, base_t=0.01, cand_t=0.30, base_exp=0.0, cand_exp=400.0)
        for i in range(8)
    ]
    reference = {("g", i): {"blind_solved": True, "blind_expanded": 5000} for i in range(8)}

    # Expansion-based scoring rewards the switch: base rows score 0.0 because
    # they expanded nothing, candidate rows score positively.
    assert sum(constrained_speedup_scores(converted, reference)) > 0.0

    # Wall time prices the work wherever it happened, so this is a regression.
    assert sum(base_relative_time_scores(converted)) < 0.0


def test_time_objective_still_gates_on_solve_rate() -> None:
    from scripts.run_puzzlescript_batched_gepa import base_relative_time_scores

    lost = [
        _timed_row(0, 1.0, 0.25),  # much faster
        _timed_row(1, 1.0, 1.0, solved=0.0, base_solved=1.0),  # but a solve is gone
    ]
    scores = base_relative_time_scores(lost)
    assert sum(scores) / len(scores) < -100.0
    assert sum(base_relative_time_scores(lost, solve_slack=0.5)) / len(lost) > -100.0


def test_objective_switch_reaches_the_time_objective() -> None:
    from scripts.run_puzzlescript_batched_gepa import (
        OBJECTIVE_ADJUSTED,
        OBJECTIVE_BASE_RELATIVE_TIME,
        adjusted_candidate_scores,
        configure_objective,
    )

    rows = [_timed_row(0, 1.0, 0.5)]
    try:
        configure_objective(mode=OBJECTIVE_BASE_RELATIVE_TIME)
        assert abs(adjusted_candidate_scores(rows)[0] - 1.0) < 1e-9
    finally:
        configure_objective(mode=OBJECTIVE_ADJUSTED)


def test_cleanup_trap_cancels_the_pool_before_touching_disk() -> None:
    """The pool must be cancelled even when the filesystem refuses writes.

    A controller that dies of an exhausted disk quota runs its EXIT trap on a
    filesystem that cannot create files. With `set -e`, a failing `touch` aborts
    the handler, so ordering the stop-file first left the array orphaned.
    """
    launcher = (ROOT / "sbatch" / "train_sokoban_search_code_gepa_cpu.s").read_text()
    body = launcher[launcher.index("cleanup() {") : launcher.index("trap cleanup EXIT")]
    scancel_at = body.index("scancel")
    touch_at = body.index("touch")
    assert scancel_at < touch_at, "scancel must run before any disk write"
    # Neither step may abort the handler under `set -e`.
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("touch ") or stripped.startswith("scancel "):
            assert stripped.endswith("|| true"), stripped


def test_default_seed_contract_offers_both_synthesis_routes():
    """The shared runner defaults to the dual-route search-code experiment."""
    assert runner.active_seed_contract() == runner.SEED_CONTRACT_DUAL_ROUTE
    text = runner.build_seed_candidate()[runner.HEURISTIC_COMPONENT]
    assert "search artifact" in text


def test_astar_seed_contract_pins_the_heuristic_only_experiment():
    """--seed-contract astar-heuristic reproduces the A*-heuristic experiment.

    The two experiments share one runner, so the heuristic-prompt experiment
    needs an explicit way to start from the A*-only contract instead of the
    dual-route seed that also invites custom search programs.
    """
    runner.configure_seed_contract(runner.SEED_CONTRACT_ASTAR_HEURISTIC)
    try:
        text = runner.build_seed_candidate()[runner.HEURISTIC_COMPONENT]
        assert "heuristic_cost_to_go" in text
        assert "search_plan" not in text
    finally:
        runner.configure_seed_contract(runner.SEED_CONTRACT_DUAL_ROUTE)
