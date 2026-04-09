"""Search primitives for the heuristic-only GEPA pipeline.

This package contains the search backend abstraction, XLand adapter utilities,
metric helpers, and artifact writers used by the heuristic-only runner. It is
needed because the refactor removes PPO from the supported path and makes search
evaluation the central execution mode, and it differs from the old package
layout by exporting multi-seed batch evaluation helpers instead of reward-based
planner entrypoints.
"""

from .jaxtar_backend import (
    JAXTAR_COMMIT,
    JAXTAR_GIT_URL,
    JAxtarSearchBackend,
    SearchBackend,
    SearchConfig,
    SearchTask,
)
from .metrics import (
    SearchBatchResult,
    SearchSeedResult,
    compute_seed_score,
    mean_job_scores,
    summarize_batch,
)
from .replay import write_json, write_text
from .xland_adapter import ACTION_NAMES, XLandTaskInstance, build_heuristic_ctx, build_task_instance

__all__ = [
    "ACTION_NAMES",
    "JAXTAR_COMMIT",
    "JAXTAR_GIT_URL",
    "JAxtarSearchBackend",
    "SearchBackend",
    "SearchBatchResult",
    "SearchConfig",
    "SearchSeedResult",
    "SearchTask",
    "XLandTaskInstance",
    "build_heuristic_ctx",
    "build_task_instance",
    "compute_seed_score",
    "mean_job_scores",
    "summarize_batch",
    "write_json",
    "write_text",
]
