# File: tests/training/test_loop_integration.py
"""Integration tests for MuZero training loop"""

import logging
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from trieye import (
    DEFAULT_METRICS,
    PersistenceConfig,
    Serializer,
    StatsConfig,
    TrieyeConfig,
)
from trieye.schemas import RawMetricEvent
from mutrimcts import SearchConfiguration

from mutriangle.config import (
    EnvConfig,
    ModelConfig,
    RunContext,
    TrainConfig,
)
from mutriangle.nn import NeuralNetwork
from mutriangle.rl import SelfPlayResult, Trainer
from mutriangle.rl.core.buffer import GameHistoryBuffer
from mutriangle.training import TrainingComponents, TrainingLoop
from mutriangle.training.loop_helpers import LoopHelpers
from mutriangle.training.worker_manager import WorkerManager
from mutriangle.utils.types import GameHistory, StateType, TrainingTarget

logger = logging.getLogger(__name__)


class MockSelfPlayWorker:
    """Mock worker for loop integration tests."""

    def __init__(self, actor_id: int, trieye_actor_mock: MagicMock | None):
        self.actor_id = actor_id
        self.trieye_actor_mock = trieye_actor_mock
        self.weights_set_count = 0
        self.last_weights_received: dict[str, Any] | None = None
        self.step_set_count = 0
        self.current_trainer_step = 0
        self.task_running = False
        self.set_weights = MagicMock(side_effect=self._set_weights_impl)
        self.set_current_trainer_step = MagicMock(
            side_effect=self._set_current_trainer_step_impl
        )
        self.run_episode = MagicMock(side_effect=self._run_episode_impl)

        self.set_weights.remote = self.set_weights
        self.set_current_trainer_step.remote = self.set_current_trainer_step
        self.run_episode.remote = self.run_episode

    def _run_episode_impl(self) -> SelfPlayResult:
        """Mock episode that returns a valid GameHistory."""
        self.task_running = True
        step_at_start = self.current_trainer_step
        time.sleep(0.01)

        # Create dummy state
        dummy_state: StateType = {
            "grid": np.zeros((1, 8, 15), dtype=np.float32),
            "other_features": np.zeros(30, dtype=np.float32),
        }

        # Create dummy GameHistory with multiple steps
        dummy_game_history: GameHistory = {
            "observations": [dummy_state, dummy_state, dummy_state],
            "actions": [0, 1],
            "rewards": [0.5, 1.0, 0.3],
            "mcts_policies": [{0: 1.0}, {1: 1.0}],
            "root_values": [0.5, 0.7],
        }

        avg_depth = 3.5
        episode_context = {
            "score": 1.0,
            "length": 2,
            "simulations": 10,
            "triangles_cleared": 0,
            "trainer_step": step_at_start,
            "avg_mcts_depth": avg_depth,
        }

        # Log events if trieye actor exists
        if self.trieye_actor_mock:
            self.trieye_actor_mock.log_event.remote(
                RawMetricEvent(
                    name="episode_end",
                    value=1.0,
                    global_step=step_at_start,
                    context=episode_context,
                )
            )

        result = SelfPlayResult(
            game_history=dummy_game_history,
            episode_experiences=[],
            final_score=1.0,
            episode_steps=2,
            trainer_step_at_episode_start=step_at_start,
            total_simulations=10,
            avg_root_visits=10.0,
            avg_tree_depth=avg_depth,
            context=episode_context,
        )
        self.task_running = False
        return result

    def _set_weights_impl(self, weights: dict) -> None:
        self.weights_set_count += 1
        self.last_weights_received = weights
        time.sleep(0.001)

    def _set_current_trainer_step_impl(self, global_step: int) -> None:
        self.step_set_count += 1
        self.current_trainer_step = global_step
        time.sleep(0.001)


@pytest.fixture
def mock_training_config(mock_train_config: TrainConfig) -> TrainConfig:
    """Fixture for TrainConfig suitable for integration tests."""
    cfg = mock_train_config.model_copy(deep=True)
    cfg.NUM_SELF_PLAY_WORKERS = 2
    cfg.WORKER_UPDATE_FREQ_STEPS = 5
    cfg.MIN_BUFFER_SIZE_TO_TRAIN = 2
    cfg.BATCH_SIZE = 2
    cfg.MAX_TRAINING_STEPS = 20
    cfg.USE_PER = False
    cfg.COMPILE_MODEL = False
    cfg.PROFILE_WORKERS = False
    cfg.CHECKPOINT_SAVE_FREQ_STEPS = 10
    cfg.UNROLL_STEPS = 3
    cfg.REWARD_LOSS_WEIGHT = 1.0
    cfg.DYNAMICS_GRADIENT_SCALE = 0.5
    return cfg


@pytest.fixture
def mock_run_context(tmp_path: Path) -> RunContext:
    """Fixture for a RunContext pointing to a temporary directory."""
    run_name = "test_loop_run"
    return RunContext.create(run_name=run_name, base_dir=tmp_path.parent)


@pytest.fixture
def mock_trieye_config(mock_run_context: RunContext) -> TrieyeConfig:
    """Fixture for a TrieyeConfig using paths from mock_run_context."""
    return TrieyeConfig(
        app_name=mock_run_context.app_name,
        run_name=mock_run_context.run_name,
        persistence=PersistenceConfig(
            ROOT_DATA_DIR=str(mock_run_context.data_root_dir.parent),
            APP_NAME=mock_run_context.app_name,
            RUN_NAME=mock_run_context.run_name,
            CHECKPOINT_SAVE_FREQ_STEPS=10,
            BUFFER_SAVE_FREQ_STEPS=0,
            SAVE_BUFFER=False,
        ),
        stats=StatsConfig(metrics=DEFAULT_METRICS),
    )


@pytest.fixture
def mock_mcts_config(mock_mcts_config: SearchConfiguration) -> SearchConfiguration:
    """Return mutrimcts SearchConfiguration."""
    return mock_mcts_config


@pytest.fixture
def mock_components(
    monkeypatch,
    mock_nn_interface: NeuralNetwork,
    mock_training_config: TrainConfig,
    mock_trieye_config: TrieyeConfig,
    mock_env_config: EnvConfig,
    mock_model_config: ModelConfig,
    mock_mcts_config: SearchConfiguration,
    mock_run_context: RunContext,
) -> TrainingComponents:
    """Fixture to create TrainingComponents with mocks for MuZero loop tests."""

    # Mock GameHistoryBuffer
    mock_buffer = MagicMock(spec=GameHistoryBuffer)
    mock_buffer.config = mock_training_config
    mock_buffer.capacity = mock_training_config.BUFFER_CAPACITY
    mock_buffer.min_size_to_train = mock_training_config.MIN_BUFFER_SIZE_TO_TRAIN
    mock_buffer.use_per = mock_training_config.USE_PER
    mock_buffer.is_ready.return_value = True
    mock_buffer.__len__.return_value = mock_training_config.MIN_BUFFER_SIZE_TO_TRAIN + 5

    # Create dummy training targets
    dummy_state: StateType = {
        "grid": np.zeros((1, 8, 15), dtype=np.float32),
        "other_features": np.zeros(30, dtype=np.float32),
    }
    dummy_target: TrainingTarget = {
        "observation": dummy_state,
        "actions": [0, 1, 2],
        "target_rewards": [0.5, 1.0, 0.3],
        "target_policies": [{0: 1.0}, {1: 1.0}, {2: 1.0}, {0: 1.0}],
        "target_values": [0.5, 0.7, 0.3, 0.1],
    }

    mock_buffer.sample.return_value = [dummy_target] * mock_training_config.BATCH_SIZE
    mock_buffer.get_contents.return_value = []

    monkeypatch.setattr(
        "mutriangle.rl.core.buffer.GameHistoryBuffer",
        lambda *args, **kwargs: mock_buffer,
    )

    # Mock Trainer
    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train_config = mock_training_config
    mock_trainer.env_config = mock_env_config
    mock_trainer.model_config = mock_model_config
    mock_trainer.nn = mock_nn_interface
    mock_trainer.device = mock_nn_interface.device
    mock_trainer.train_step.return_value = (
        {
            "total_loss": 0.1,
            "policy_loss": 0.05,
            "value_loss": 0.03,
            "reward_loss": 0.02,
        },
        np.array([0.1, 0.1]),
    )
    mock_trainer.get_current_lr.return_value = mock_training_config.LEARNING_RATE
    mock_trainer.optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_trainer.optimizer.state_dict.return_value = {"opt_state": "dummy"}
    monkeypatch.setattr(
        "mutriangle.rl.core.trainer.Trainer", lambda *args, **kwargs: mock_trainer
    )

    # Mock Trieye Actor
    mock_trieye_actor_handle = MagicMock(name="TrieyeActorMockHandle")
    mock_trieye_actor_handle.log_event = MagicMock(name="log_event_remote")
    mock_trieye_actor_handle.log_batch_events = MagicMock(
        name="log_batch_events_remote"
    )
    mock_trieye_actor_handle.save_training_state = MagicMock(
        name="save_training_state_remote"
    )
    mock_trieye_actor_handle.get_actor_name = MagicMock(
        name="get_actor_name_remote", return_value="mock_trieye_actor"
    )
    mock_trieye_actor_handle.get_run_base_dir_str = MagicMock(
        name="get_run_base_dir_str_remote",
        return_value=str(
            mock_run_context.data_root_dir / "runs" / mock_run_context.run_name
        ),
    )
    mock_trieye_actor_handle.log_event.remote = mock_trieye_actor_handle.log_event
    mock_trieye_actor_handle.log_batch_events.remote = (
        mock_trieye_actor_handle.log_batch_events
    )
    mock_trieye_actor_handle.save_training_state.remote = (
        mock_trieye_actor_handle.save_training_state
    )
    mock_trieye_actor_handle.get_actor_name.remote = (
        mock_trieye_actor_handle.get_actor_name
    )
    mock_trieye_actor_handle.get_run_base_dir_str.remote = (
        mock_trieye_actor_handle.get_run_base_dir_str
    )

    mock_serializer = MagicMock(spec=Serializer)
    mock_serializer.prepare_optimizer_state.return_value = {"opt_state": "dummy"}
    mock_serializer.prepare_buffer_data.return_value = MagicMock(buffer_list=[])

    # Mock WorkerManager
    mock_worker_manager_instance = MagicMock(spec=WorkerManager)
    mock_workers = [
        MockSelfPlayWorker(i, mock_trieye_actor_handle)
        for i in range(mock_training_config.NUM_SELF_PLAY_WORKERS)
    ]
    mock_worker_manager_instance.workers = mock_workers
    mock_worker_manager_instance.active_worker_indices = set(
        range(mock_training_config.NUM_SELF_PLAY_WORKERS)
    )
    mock_worker_tasks: dict[Any, int] = {}

    def mock_submit_task(worker_idx):
        if worker_idx in mock_worker_manager_instance.active_worker_indices:
            mock_task_ref = MagicMock(name=f"task_ref_w{worker_idx}")
            mock_worker_tasks[mock_task_ref] = worker_idx

    def mock_get_completed_tasks(*_args, **_kwargs):
        results = []
        if mock_worker_tasks:
            ref_to_complete = next(iter(mock_worker_tasks.keys()))
            worker_idx_to_complete = mock_worker_tasks.pop(ref_to_complete)

            if (
                worker_idx_to_complete
                in mock_worker_manager_instance.active_worker_indices
                and worker_idx_to_complete < len(mock_workers)
            ):
                result = mock_workers[worker_idx_to_complete].run_episode()
                results.append((worker_idx_to_complete, result))
        return results

    mock_worker_manager_instance.submit_task.side_effect = mock_submit_task
    mock_worker_manager_instance.get_completed_tasks.side_effect = (
        mock_get_completed_tasks
    )
    mock_worker_manager_instance.get_num_pending_tasks.side_effect = lambda: len(
        mock_worker_tasks
    )

    def mock_update_worker_networks(global_step: int):
        weights = mock_nn_interface.get_weights()
        for worker in mock_workers:
            if worker:
                worker.set_weights.remote(weights)
                worker.set_current_trainer_step.remote(global_step)

    mock_worker_manager_instance.update_worker_networks.side_effect = (
        mock_update_worker_networks
    )
    mock_worker_manager_instance.get_num_active_workers.return_value = (
        mock_training_config.NUM_SELF_PLAY_WORKERS
    )
    mock_worker_manager_instance.initialize_workers = MagicMock()

    def mock_submit_initial_tasks_impl():
        for worker_idx in mock_worker_manager_instance.active_worker_indices:
            mock_submit_task(worker_idx)

    mock_worker_manager_instance.submit_initial_tasks = MagicMock(
        side_effect=mock_submit_initial_tasks_impl
    )

    monkeypatch.setattr(
        "mutriangle.training.loop.WorkerManager",
        lambda *args, **kwargs: mock_worker_manager_instance,
    )

    mock_loop_helpers = MagicMock(spec=LoopHelpers)
    mock_loop_helpers.log_progress_eta = MagicMock()
    monkeypatch.setattr(
        "mutriangle.training.loop.LoopHelpers",
        lambda *args, **kwargs: mock_loop_helpers,
    )

    components = TrainingComponents(
        run_context=mock_run_context,
        nn=mock_nn_interface,
        buffer=mock_buffer,
        trainer=mock_trainer,
        trieye_actor=mock_trieye_actor_handle,
        trieye_config=mock_trieye_config,
        serializer=mock_serializer,
        train_config=mock_training_config,
        env_config=mock_env_config,
        model_config=mock_model_config,
        mcts_config=mock_mcts_config,
        profile_workers=mock_training_config.PROFILE_WORKERS,
    )

    return components


@pytest.fixture
def mock_training_loop(mock_components: TrainingComponents) -> TrainingLoop:
    """Fixture for an initialized TrainingLoop with mocked components."""
    loop = TrainingLoop(mock_components)
    loop.set_initial_state(0, 0, 0)
    return loop


def test_worker_weight_update_and_stats_logging(
    mock_training_loop: TrainingLoop, mock_components: TrainingComponents
):
    """Verify worker weight updates and event logging."""
    loop = mock_training_loop
    components = mock_components
    worker_manager = loop.worker_manager
    mock_workers = worker_manager.workers
    trieye_actor_mock = components.trieye_actor
    assert trieye_actor_mock is not None

    update_freq = components.train_config.WORKER_UPDATE_FREQ_STEPS
    max_steps = components.train_config.MAX_TRAINING_STEPS or 20

    loop.initialize_workers()
    loop.worker_manager.submit_initial_tasks()
    loop.run()

    total_expected_updates = max_steps // update_freq

    # Check weight update events
    weight_update_event_calls = [
        c
        for c in trieye_actor_mock.log_event.call_args_list
        if isinstance(c.args[0], RawMetricEvent)
        and c.args[0].name == "Progress/Weight_Updates_Total"
    ]
    assert len(weight_update_event_calls) == total_expected_updates

    # Verify workers were updated
    for worker in mock_workers:
        if worker is not None:
            set_step_mock = cast("MagicMock", worker.set_current_trainer_step)
            set_weights_mock = cast("MagicMock", worker.set_weights)

            assert set_step_mock.call_count == total_expected_updates
            assert set_weights_mock.call_count == total_expected_updates


def test_checkpoint_save_trigger(
    mock_training_loop: TrainingLoop, mock_components: TrainingComponents
):
    """Verify checkpoint save is triggered."""
    loop = mock_training_loop
    trieye_actor_mock = mock_components.trieye_actor
    assert trieye_actor_mock is not None

    save_freq = loop.trieye_config.persistence.CHECKPOINT_SAVE_FREQ_STEPS
    loop.train_config.MAX_TRAINING_STEPS = save_freq + 2

    loop.initialize_workers()
    loop.worker_manager.submit_initial_tasks()
    loop.run()

    save_calls = trieye_actor_mock.save_training_state.call_args_list
    assert len(save_calls) >= 1, "save_training_state was not called"
