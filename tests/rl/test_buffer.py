# File: tests/rl/test_buffer.py
"""Tests for MuZero GameHistoryBuffer"""

import pytest
import numpy as np

from mutriangle.config import TrainConfig
from mutriangle.rl.core.buffer import GameHistoryBuffer
from mutriangle.utils.types import GameHistory, StateType
from mutriangle.utils.sumtree import SumTree

from tests.conftest import rng


@pytest.fixture
def uniform_train_config() -> TrainConfig:
    """TrainConfig for uniform buffer."""
    return TrainConfig(
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        BATCH_SIZE=4,
        UNROLL_STEPS=3,
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
    )


@pytest.fixture
def per_train_config() -> TrainConfig:
    """TrainConfig for PER buffer."""
    return TrainConfig(
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        BATCH_SIZE=4,
        UNROLL_STEPS=3,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=50,
        PER_EPSILON=1e-5,
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
        DYNAMICS_GRADIENT_SCALE=0.5,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        MAX_TRAINING_STEPS=200,
        N_STEP_RETURNS=3,
        GAMMA=0.99,
    )


@pytest.fixture
def dummy_game_history() -> GameHistory:
    """Create a dummy GameHistory for testing.

    Structure: 3 observations, 3 actions, 3 rewards, 3 policies, 3 root_values
    This represents a trajectory with 3 decision points.
    """
    state1: StateType = {
        "grid": np.zeros((1, 8, 15), dtype=np.float32),
        "other_features": np.zeros(30, dtype=np.float32),
    }
    state2: StateType = {
        "grid": np.ones((1, 8, 15), dtype=np.float32),
        "other_features": np.ones(30, dtype=np.float32),
    }
    state3: StateType = {
        "grid": np.ones((1, 8, 15), dtype=np.float32) * 0.5,
        "other_features": np.ones(30, dtype=np.float32) * 0.5,
    }

    return GameHistory(
        observations=[state1, state2, state3],  # 3 observations
        actions=[0, 1, 2],  # 3 actions
        rewards=[0.5, 1.0, 0.3],  # 3 rewards (n-step returns)
        mcts_policies=[{0: 0.8, 1: 0.2}, {1: 0.9, 2: 0.1}, {2: 1.0}],  # 3 policies
        root_values=[0.5, 0.7, 0.3],  # 3 root values
    )


@pytest.fixture
def uniform_buffer(uniform_train_config: TrainConfig) -> GameHistoryBuffer:
    """Provides an empty uniform GameHistoryBuffer."""
    return GameHistoryBuffer(uniform_train_config)


@pytest.fixture
def per_buffer(per_train_config: TrainConfig) -> GameHistoryBuffer:
    """Provides an empty PER GameHistoryBuffer."""
    return GameHistoryBuffer(per_train_config)


# --- Uniform Buffer Tests ---


def test_uniform_buffer_init(uniform_buffer: GameHistoryBuffer):
    """Test uniform buffer initialization."""
    assert not uniform_buffer.use_per
    assert uniform_buffer.capacity == 100
    assert len(uniform_buffer) == 0
    assert not uniform_buffer.is_ready()


def test_uniform_buffer_add(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test adding a single game history."""
    assert len(uniform_buffer) == 0
    uniform_buffer.add(dummy_game_history)
    assert len(uniform_buffer) == 1


def test_uniform_buffer_add_batch(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test adding multiple game histories."""
    batch = [dummy_game_history for _ in range(5)]
    uniform_buffer.add_batch(batch)
    assert len(uniform_buffer) == 5


def test_uniform_buffer_capacity(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test buffer respects capacity limit."""
    for i in range(uniform_buffer.capacity + 10):
        # Create unique game histories
        gh = GameHistory(
            observations=dummy_game_history["observations"].copy(),
            actions=dummy_game_history["actions"].copy(),
            rewards=[r + i for r in dummy_game_history["rewards"]],
            mcts_policies=dummy_game_history["mcts_policies"].copy(),
            root_values=dummy_game_history["root_values"].copy(),
        )
        uniform_buffer.add(gh)
    assert len(uniform_buffer) == uniform_buffer.capacity


def test_uniform_buffer_is_ready(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test buffer readiness threshold."""
    assert not uniform_buffer.is_ready()
    for _ in range(uniform_buffer.min_size_to_train):
        uniform_buffer.add(dummy_game_history)
    assert uniform_buffer.is_ready()


def test_uniform_buffer_sample(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test sampling training targets from uniform buffer."""
    # Fill buffer until ready
    for _ in range(uniform_buffer.min_size_to_train):
        uniform_buffer.add(dummy_game_history)

    unroll_steps = 3
    batch_targets = uniform_buffer.sample(
        batch_size=uniform_buffer.config.BATCH_SIZE,
        unroll_steps=unroll_steps,
        current_train_step=0,
    )

    assert batch_targets is not None
    assert isinstance(batch_targets, list)
    assert len(batch_targets) == uniform_buffer.config.BATCH_SIZE

    # Check structure of first target
    target = batch_targets[0]
    assert "observation" in target
    assert "actions" in target
    assert "target_rewards" in target
    assert "target_policies" in target
    assert "target_values" in target

    assert len(target["actions"]) == unroll_steps
    assert len(target["target_rewards"]) == unroll_steps
    assert len(target["target_policies"]) == unroll_steps + 1
    assert len(target["target_values"]) == unroll_steps + 1


def test_uniform_buffer_sample_not_ready(uniform_buffer: GameHistoryBuffer):
    """Test sampling returns None when buffer not ready."""
    batch_targets = uniform_buffer.sample(
        batch_size=uniform_buffer.config.BATCH_SIZE,
        unroll_steps=3,
    )
    assert batch_targets is None


# --- PER Buffer Tests ---


def test_per_buffer_init(per_buffer: GameHistoryBuffer):
    """Test PER buffer initialization."""
    assert per_buffer.use_per
    assert isinstance(per_buffer.tree, SumTree)
    assert per_buffer.capacity == 100
    assert len(per_buffer) == 0
    assert not per_buffer.is_ready()
    assert per_buffer.tree.max_priority == 1.0


def test_per_buffer_add(per_buffer: GameHistoryBuffer, dummy_game_history: GameHistory):
    """Test adding to PER buffer."""
    assert len(per_buffer) == 0
    per_buffer.add(dummy_game_history)
    assert len(per_buffer) == 1


def test_per_buffer_sample(
    per_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test sampling from PER buffer."""
    # Fill buffer until ready
    for _ in range(per_buffer.min_size_to_train):
        per_buffer.add(dummy_game_history)

    batch_targets = per_buffer.sample(
        batch_size=per_buffer.config.BATCH_SIZE,
        unroll_steps=3,
        current_train_step=10,
    )

    assert batch_targets is not None
    assert isinstance(batch_targets, list)
    assert len(batch_targets) == per_buffer.config.BATCH_SIZE


def test_per_buffer_sample_requires_step(
    per_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test PER sampling requires current_train_step."""
    for _ in range(per_buffer.min_size_to_train):
        per_buffer.add(dummy_game_history)

    with pytest.raises(ValueError, match="current_train_step required"):
        per_buffer.sample(batch_size=4, unroll_steps=3)


def test_per_buffer_beta_annealing(per_buffer: GameHistoryBuffer):
    """Test beta annealing calculation."""
    config = per_buffer.config
    assert per_buffer._calculate_beta(0) == config.PER_BETA_INITIAL

    anneal_steps = per_buffer.per_beta_anneal_steps
    mid_step = anneal_steps // 2
    expected_mid_beta = config.PER_BETA_INITIAL + 0.5 * (
        config.PER_BETA_FINAL - config.PER_BETA_INITIAL
    )
    assert per_buffer._calculate_beta(mid_step) == pytest.approx(expected_mid_beta)
    assert per_buffer._calculate_beta(anneal_steps) == config.PER_BETA_FINAL


def test_make_training_target_padding(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test that training targets are padded when near episode end."""
    uniform_buffer.add(dummy_game_history)

    # Sample from near the end (should require padding)
    T = len(dummy_game_history["actions"])
    start_idx = T - 1  # Last position
    unroll_steps = 5  # Requires padding

    target = uniform_buffer._make_training_target(
        dummy_game_history, start_idx, unroll_steps
    )

    assert len(target["actions"]) == unroll_steps
    assert len(target["target_rewards"]) == unroll_steps
    assert len(target["target_policies"]) == unroll_steps + 1
    assert len(target["target_values"]) == unroll_steps + 1


def test_get_contents(
    uniform_buffer: GameHistoryBuffer, dummy_game_history: GameHistory
):
    """Test getting buffer contents."""
    for _ in range(5):
        uniform_buffer.add(dummy_game_history)

    contents = uniform_buffer.get_contents()
    assert isinstance(contents, list)
    assert len(contents) == 5
    assert all(isinstance(gh, dict) for gh in contents)
    assert all("observations" in gh for gh in contents)
