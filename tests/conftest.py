# File: tests/conftest.py
import os

# Disable numba JIT compilation for tests to avoid caching issues
os.environ["NUMBA_DISABLE_JIT"] = "1"

import random
from typing import cast

import numpy as np
import pytest
import torch
import torch.optim as optim

# Import from trianglengin's top level
from trianglengin import EnvConfig

# Import trimcts config
from mutrimcts import SearchConfiguration

# Keep mutriangle imports
from mutriangle.config import (
    ModelConfig,
    TrainConfig,
)
from mutriangle.nn import NeuralNetwork
from mutriangle.rl import GameHistoryBuffer, Trainer
from mutriangle.utils.types import Experience, GameHistory, StateType, TrainingTarget

# Removed PersistenceConfig, StatsConfig imports

rng = np.random.default_rng()


@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    """Provides a default, *valid* trianglengin.EnvConfig for tests."""
    rows = 3
    cols = 3
    playable_range = [(0, 3), (0, 3), (0, 3)]
    return EnvConfig(
        ROWS=rows,
        COLS=cols,
        PLAYABLE_RANGE_PER_ROW=playable_range,
        NUM_SHAPE_SLOTS=1,
    )


@pytest.fixture(scope="session")
def mock_model_config(mock_env_config: EnvConfig) -> ModelConfig:
    """Provides a default ModelConfig compatible with mock_env_config (session-scoped)."""
    action_dim_int = int(
        mock_env_config.NUM_SHAPE_SLOTS * mock_env_config.ROWS * mock_env_config.COLS
    )
    expected_other_dim = 10

    return ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4],
        CONV_KERNEL_SIZES=[3],
        CONV_STRIDES=[1],
        CONV_PADDING=[1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=4,
        USE_TRANSFORMER=False,
        TRANSFORMER_DIM=16,
        TRANSFORMER_HEADS=2,
        TRANSFORMER_LAYERS=0,
        TRANSFORMER_FC_DIM=32,
        FC_DIMS_SHARED=[8],
        POLICY_HEAD_DIMS=[action_dim_int],
        VALUE_HEAD_DIMS=[1],
        OTHER_NN_INPUT_FEATURES_DIM=expected_other_dim,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=True,
        # MuZero-specific parameters
        HIDDEN_STATE_DIM=16,
        REPRESENTATION_HIDDEN_DIMS=[32],
        DYNAMICS_HIDDEN_DIMS=[32],
        PREDICTION_HIDDEN_DIMS=[32],
        NUM_REWARD_ATOMS=11,
        REWARD_MIN=-5.0,
        REWARD_MAX=5.0,
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    """Provides a default TrainConfig for tests (session-scoped)."""
    return TrainConfig(
        BATCH_SIZE=4,
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        USE_PER=False,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        REWARD_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.0,
        UNROLL_STEPS=3,
        DYNAMICS_GRADIENT_SCALE=0.5,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=100,
        PER_EPSILON=1e-5,
        MAX_TRAINING_STEPS=200,
        N_STEP_RETURNS=3,
        GAMMA=0.99,
        PROFILE_WORKERS=False,
        RUN_NAME="pytest_default_run",
    )


# Removed mock_persistence_config fixture


@pytest.fixture(scope="session")
def mock_mcts_config() -> SearchConfiguration:
    """Provides a default trimcts.SearchConfiguration for tests."""
    return SearchConfiguration(
        max_simulations=8,
        max_depth=5,
        cpuct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
        mcts_batch_size=4,
    )


@pytest.fixture(scope="session")
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    """Creates a mock StateType dictionary with correct shapes."""
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)

    return {
        "grid": rng.random(grid_shape, dtype=np.float32),
        "other_features": rng.random(other_shape, dtype=np.float32),
    }


@pytest.fixture(scope="session")
def mock_experience(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> Experience:
    """Creates a mock Experience tuple."""
    action_dim = int(
        mock_env_config.NUM_SHAPE_SLOTS * mock_env_config.ROWS * mock_env_config.COLS
    )
    policy_target = (
        dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
        else {0: 1.0}
    )
    value_target = random.uniform(-1, 1)
    return (mock_state_type, policy_target, value_target)


@pytest.fixture(scope="session")
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance with a mock model for testing."""
    device = torch.device("cpu")
    nn_interface = NeuralNetwork(
        mock_model_config, mock_env_config, mock_train_config, device
    )
    return nn_interface


@pytest.fixture(scope="session")
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance."""
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture(scope="session")
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    """Provides the optimizer from the mock_trainer."""
    return cast("optim.Optimizer", mock_trainer.optimizer)


@pytest.fixture
def mock_experience_buffer(mock_train_config: TrainConfig) -> GameHistoryBuffer:
    """Provides a GameHistoryBuffer instance for MuZero."""
    return GameHistoryBuffer(mock_train_config)


@pytest.fixture
def filled_mock_buffer(
    mock_experience_buffer: GameHistoryBuffer, mock_experience: Experience
) -> GameHistoryBuffer:
    """Provides a buffer filled with some mock game histories."""
    for i in range(mock_experience_buffer.min_size_to_train + 5):
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy() + i * 0.01,
            "other_features": mock_experience[0]["other_features"].copy() + i * 0.01,
        }
        exp_copy: Experience = (state_copy, mock_experience[1], random.uniform(-1, 1))
        mock_experience_buffer.add(exp_copy)
    return mock_experience_buffer


@pytest.fixture
def mock_game_history(mock_state_type: StateType) -> GameHistory:
    """Creates a mock GameHistory for MuZero tests."""
    return GameHistory(
        observations=[mock_state_type, mock_state_type, mock_state_type],
        actions=[0, 1, 2],
        rewards=[0.5, 1.0, 0.3],
        mcts_policies=[{0: 0.8, 1: 0.2}, {1: 0.9}, {2: 1.0}],
        root_values=[0.5, 0.7, 0.3],
    )


@pytest.fixture
def mock_training_target(mock_state_type: StateType) -> TrainingTarget:
    """Creates a mock TrainingTarget for MuZero tests."""
    return TrainingTarget(
        observation=mock_state_type,
        actions=[0, 1, 2],
        target_rewards=[0.5, 1.0, 0.3],
        target_policies=[{0: 1.0}, {1: 1.0}, {2: 1.0}, {0: 1.0}],
        target_values=[0.5, 0.7, 0.3, 0.1],
    )
