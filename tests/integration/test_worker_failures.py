# File: mutriangle/tests/integration/test_worker_failures.py
"""
Integration tests for worker failure scenarios and circuit breaker functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock

from mutriangle.rl.types import SelfPlayResult
from mutriangle.training.loop import TrainingLoop, MAX_CONSECUTIVE_EMPTY_HISTORIES
from mutriangle.utils.types import GameHistory


class TestCircuitBreaker:
    """Tests for the circuit breaker functionality in TrainingLoop."""

    @pytest.fixture
    def mock_loop(self, monkeypatch):
        """Create a TrainingLoop with mocked components."""
        # Mock WorkerManager to avoid Ray actor initialization
        mock_worker_manager = Mock()
        mock_worker_manager.get_num_active_workers = Mock(return_value=0)
        mock_worker_manager.get_num_pending_tasks = Mock(return_value=0)

        # Mock LoopHelpers
        mock_loop_helpers = Mock()
        mock_loop_helpers.log_progress_eta = Mock()

        # Patch classes before importing
        monkeypatch.setattr(
            "mutriangle.training.loop.WorkerManager",
            Mock(return_value=mock_worker_manager),
        )
        monkeypatch.setattr(
            "mutriangle.training.loop.LoopHelpers", Mock(return_value=mock_loop_helpers)
        )

        # Create minimal mocked components
        mock_components = Mock()
        mock_components.train_config = Mock(
            MAX_TRAINING_STEPS=10,
            WORKER_UPDATE_FREQ_STEPS=10,
            CHECKPOINT_SAVE_FREQ_STEPS=100,
        )
        mock_components.trieye_config = Mock(
            persistence=Mock(
                SAVE_BUFFER=False,
                BUFFER_SAVE_FREQ_STEPS=100,
            )
        )
        mock_components.buffer = Mock()
        mock_components.buffer.min_size_to_train = 5
        mock_components.buffer.capacity = 10
        mock_components.buffer.is_ready = Mock(return_value=False)
        mock_components.buffer.__len__ = Mock(return_value=0)
        mock_components.buffer.add = Mock()
        mock_components.trainer = Mock()
        mock_components.trieye_actor = None
        mock_components.serializer = Mock()
        mock_components.nn = Mock()
        mock_components.env_config = Mock()
        mock_components.env_config.model_dump = Mock(return_value={})

        loop = TrainingLoop(mock_components)
        return loop

    def test_empty_history_tracking(self, mock_loop):
        """Circuit breaker should track consecutive empty histories per worker."""
        # Initially no failures
        assert mock_loop.consecutive_empty_histories == {}
        assert mock_loop.total_empty_histories == 0

        # Create empty GameHistory
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        # Create result with empty history
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
            game_over_reason="Test failure",
        )

        # Process the result - should track the failure but not raise yet
        worker_id = 0
        mock_loop._process_self_play_result(result, worker_id)

        assert mock_loop.consecutive_empty_histories[worker_id] == 1
        assert mock_loop.total_empty_histories == 1
        assert mock_loop.last_empty_history_reasons[worker_id] == "Test failure"

    def test_circuit_breaker_triggers_after_threshold(self, mock_loop):
        """Circuit breaker should raise exception after MAX_CONSECUTIVE_EMPTY_HISTORIES."""
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        worker_id = 0

        # Submit failures up to threshold - 1
        for i in range(MAX_CONSECUTIVE_EMPTY_HISTORIES - 1):
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
                game_over_reason=f"Test failure {i}",
            )
            mock_loop._process_self_play_result(result, worker_id)

        # Should not have raised yet
        assert (
            mock_loop.consecutive_empty_histories[worker_id]
            == MAX_CONSECUTIVE_EMPTY_HISTORIES - 1
        )

        # One more failure should trigger circuit breaker
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
            game_over_reason="Final test failure",
        )

        with pytest.raises(RuntimeError, match="Circuit breaker triggered"):
            mock_loop._process_self_play_result(result, worker_id)

    def test_successful_episode_resets_counter(self, mock_loop):
        """Successful episode should reset the consecutive failure counter."""
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        worker_id = 0

        # Add some failures
        for i in range(3):
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
                game_over_reason=f"Failure {i}",
            )
            mock_loop._process_self_play_result(result, worker_id)

        assert mock_loop.consecutive_empty_histories[worker_id] == 3

        # Now send a successful episode
        from mutriangle.utils.types import StateType
        import numpy as np

        state: StateType = {
            "grid": np.zeros((1, 8, 15), dtype=np.float32),
            "other_features": np.zeros(30, dtype=np.float32),
        }

        successful_history: GameHistory = {
            "observations": [state],
            "actions": [0],
            "rewards": [0.1],
            "mcts_policies": [{0: 1.0}],
            "root_values": [0.0],
        }

        success_result = SelfPlayResult(
            game_history=successful_history,
            episode_experiences=[],
            final_score=0.1,
            episode_steps=1,
            trainer_step_at_episode_start=0,
            total_simulations=4,
            avg_root_visits=4.0,
            avg_tree_depth=1.0,
            context={},
            game_over_reason=None,
        )

        mock_loop._process_self_play_result(success_result, worker_id)

        # Counter should be reset
        assert mock_loop.consecutive_empty_histories[worker_id] == 0

    def test_game_over_reason_propagated(self, mock_loop):
        """Game over reason from worker should be captured and stored."""
        empty_history: GameHistory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "mcts_policies": [],
            "root_values": [],
        }

        test_reason = "No valid actions at initialization"

        result = SelfPlayResult(
            game_history=empty_history,
            episode_experiences=[],
            final_score=0.0,
            episode_steps=0,
            trainer_step_at_episode_start=0,
            total_simulations=0,
            avg_root_visits=0.0,
            avg_tree_depth=0.0,
            context={"game_over_reason": test_reason},
            game_over_reason=test_reason,
        )

        worker_id = 0
        mock_loop._process_self_play_result(result, worker_id)

        assert mock_loop.last_empty_history_reasons[worker_id] == test_reason


class TestSelfPlayResultValidation:
    """Tests for SelfPlayResult validation with game_over_reason."""

    def test_self_play_result_with_game_over_reason(self):
        """SelfPlayResult should accept game_over_reason field."""
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
            game_over_reason="Test reason",
        )

        assert result.game_over_reason == "Test reason"

    def test_self_play_result_without_game_over_reason(self):
        """SelfPlayResult should work with game_over_reason=None."""
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
            game_over_reason=None,
        )

        assert result.game_over_reason is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
