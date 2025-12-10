"""Runtime patch(es) for Weave logging in threaded DSPy evals.

We see intermittent ValueErrors from Weave when DSPy threads try to
log scores after the prediction context has finished. These bubbles
out of the thread pool and halt GEPA. To keep runs alive, we install a
safe wrapper that simply drops late score logs instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any


def apply_safe_log_score_patch() -> None:
    """Wrap ScoreLogger.log_score to ignore late/finished errors.

    DSPy executes metrics inside a ThreadPoolExecutor. Occasionally a
    worker attempts to log a score after the ScoreLogger has already
    been finished, which raises ValueError("Cannot log score after
    finish has been called"). That exception aborts the whole GEPA
    iteration. We swallow that specific error while keeping all other
    behavior intact.
    """

    try:
        from weave.evaluation import eval_imperative as ei  # type: ignore
    except Exception:
        return

    if getattr(ei, "_llm_desparsifier_safe_log_score", False):
        return

    log = logging.getLogger(__name__)
    original = ei.ScoreLogger.log_score

    def safe_log_score(self: Any, scorer: Any, score: Any = ei.NOT_SET):  # type: ignore[arg-type]
        try:
            return original(self, scorer, score)
        except ValueError as exc:
            if "Cannot log score after finish has been called" in str(exc):
                log.debug("Dropping late score for scorer=%s (already finished)", scorer)
                return None
            raise

    ei.ScoreLogger.log_score = safe_log_score  # type: ignore[assignment]
    ei._llm_desparsifier_safe_log_score = True
