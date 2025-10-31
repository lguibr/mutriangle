# File: mutriangle/tests/rl/test_self_play_result.py
"""
Unit tests for SelfPlayResult with game_over_reason field.
"""

import pytest
import numpy as np

from mutriangle.rl.types import SelfPlayResult
from mutriangle.utils.types import GameHistory, StateType


class TestSelfPlayResultWithGameOverReason:
    """Tests for SelfPlayResult with game_over_reason field."""

    def test_result_with_game_over_reason(self):
        """SelfPlayResult should accept and store game_over_reason."""
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        result = SelfPlayResult(
            game_history=empty_history,
            episode_experiences=[],
            final_score=0.0,
            episode_steps=0,
            trainer_step_at_episode_start=0,
            total_simulations=0,
            avg_root_visits=0.0,
            avg_tree_depth=0.0,
            context={},
            game_over_reason="Immediate game over: No valid actions",
        )

        assert result.game_over_reason == "Immediate game over: No valid actions"
        assert result.episode_steps == 0
        assert len(result.game_history["observations"]) == 0

    def test_result_without_game_over_reason(self):
        """SelfPlayResult should work with game_over_reason=None for successful episodes."""
        state: StateType = {
            "grid": np.zeros((1, 8, 15), dtype=np.float32),
            "other_features": np.zeros(30, dtype=np.float32),
        }

        successful_history: GameHistory = {
            "observations": [state],
            "actions": [5],
            "rewards": [0.1],
            "mcts_policies": [{5: 1.0}],
            "root_values": [0.0],
        }

        result = SelfPlayResult(
            game_history=successful_history,
            episode_experiences=[],
            final_score=0.1,
            episode_steps=1,
            trainer_step_at_episode_start=0,
            total_simulations=10,
            avg_root_visits=10.0,
            avg_tree_depth=2.0,
            context={"score": 0.1, "length": 1},
            game_over_reason=None,
        )

        assert result.game_over_reason is None
        assert result.episode_steps == 1
        assert len(result.game_history["observations"]) == 1

    def test_result_with_exception_reason(self):
        """SelfPlayResult should capture exception messages in game_over_reason."""
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        exception_msg = "Exception during episode: RuntimeError('MCTS failed')"

        result = SelfPlayResult(
            game_history=empty_history,
            episode_experiences=[],
            final_score=0.0,
            episode_steps=0,
            trainer_step_at_episode_start=0,
            total_simulations=0,
            avg_root_visits=0.0,
            avg_tree_depth=0.0,
            context={"error": True},
            game_over_reason=exception_msg,
        )

        assert result.game_over_reason == exception_msg
        assert "Exception during episode" in result.game_over_reason

    def test_result_validation_passes_with_game_over_reason(self):
        """Pydantic validation should pass with game_over_reason field."""
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        # This should not raise validation errors
        result_dict = {
            "game_history": empty_history,
            "episode_experiences": [],
            "final_score": 0.0,
            "episode_steps": 0,
            "trainer_step_at_episode_start": 0,
            "total_simulations": 0,
            "avg_root_visits": 0.0,
            "avg_tree_depth": 0.0,
            "context": {},
            "game_over_reason": "Test reason",
        }

        result = SelfPlayResult(**result_dict)
        assert isinstance(result, SelfPlayResult)
        assert result.game_over_reason == "Test reason"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



