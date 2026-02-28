"""Top-level package namespace for llm_desparsifier."""

from . import rewards, rl, utils
from .rewards import RewardGenerator, create_reward_generator
from .rl.pipeline import (
    RewardGeneratorProtocol,
    TrainingResult,
    run_training_with_reward,
)

__all__ = [
    "rl",
    "rewards",
    "utils",
    "run_training_with_reward",
    "RewardGeneratorProtocol",
    "TrainingResult",
    "RewardGenerator",
    "create_reward_generator",
]
