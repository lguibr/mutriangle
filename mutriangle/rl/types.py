# File: mutriangle/rl/types.py
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Experience, GameHistory, StateType

logger = logging.getLogger(__name__)

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker."""

    model_config = arbitrary_types_config

    # MuZero: Store complete game history
    game_history: GameHistory

    final_score: float
    episode_steps: int
    trainer_step_at_episode_start: int

    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    # avg_tree_depth is now calculated and returned by mutrimcts
    avg_tree_depth: float = Field(..., ge=0)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context from the episode (e.g., triangles_cleared, avg_mcts_depth).",
    )
    game_over_reason: str | None = Field(
        default=None,
        description="Reason why the episode ended early (e.g., immediate game over, no valid actions).",
    )


SelfPlayResult.model_rebuild(force=True)
