# File: tests/rl/test_trainer.py
"""Tests for MuZero Trainer"""

import numpy as np
import pytest
import torch

from trianglengin import EnvConfig
from mutriangle.config import ModelConfig, TrainConfig
from mutriangle.nn import NeuralNetwork
from mutriangle.rl import Trainer
from mutriangle.utils.types import StateType, TrainingTarget


@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    mock_model_config.OTHER_NN_INPUT_FEATURES_DIM = 10
    return mock_model_config


@pytest.fixture
def train_config_uniform(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = False
    cfg.UNROLL_STEPS = 3
    cfg.REWARD_LOSS_WEIGHT = 1.0
    cfg.DYNAMICS_GRADIENT_SCALE = 0.5
    return cfg


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config_uniform: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing."""
    device = torch.device("cpu")
    nn_interface_instance = NeuralNetwork(
        model_config, env_config, train_config_uniform, device
    )
    nn_interface_instance.model.to(device)
    nn_interface_instance.model.eval()
    return nn_interface_instance


@pytest.fixture
def trainer_uniform(
    nn_interface: NeuralNetwork,
    train_config_uniform: TrainConfig,
    env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance configured for uniform sampling."""
    return Trainer(nn_interface, train_config_uniform, env_config)


@pytest.fixture
def dummy_training_targets(
    env_config: EnvConfig, model_config: ModelConfig
) -> list[TrainingTarget]:
    """Create dummy training targets for testing."""
    state: StateType = {
        "grid": np.zeros((1, env_config.ROWS, env_config.COLS), dtype=np.float32),
        "other_features": np.zeros(
            model_config.OTHER_NN_INPUT_FEATURES_DIM, dtype=np.float32
        ),
    }

    targets = []
    for _ in range(4):  # Batch size 4
        target = TrainingTarget(
            observation=state,
            actions=[0, 1, 2],  # 3 unroll steps
            target_rewards=[0.5, 1.0, 0.3],
            target_policies=[{0: 1.0}, {1: 1.0}, {2: 1.0}, {0: 1.0}],  # K+1
            target_values=[0.5, 0.7, 0.3, 0.1],  # K+1
        )
        targets.append(target)

    return targets


def test_trainer_initialization(trainer_uniform: Trainer):
    """Test trainer initializes correctly."""
    assert trainer_uniform.nn is not None
    assert trainer_uniform.model is not None
    assert trainer_uniform.optimizer is not None
    assert hasattr(trainer_uniform, "scheduler")
    assert hasattr(trainer_uniform, "reward_support")
    assert (
        trainer_uniform.num_reward_atoms
        == trainer_uniform.model_config.NUM_REWARD_ATOMS
    )


def test_observations_to_tensors(trainer_uniform: Trainer, env_config: EnvConfig):
    """Test converting observations to tensors."""
    observations = [
        {
            "grid": np.zeros((1, env_config.ROWS, env_config.COLS), dtype=np.float32),
            "other_features": np.zeros(30, dtype=np.float32),
        }
        for _ in range(3)
    ]

    grid_t, other_t = trainer_uniform._observations_to_tensors(observations)

    assert grid_t.shape == (3, 1, env_config.ROWS, env_config.COLS)
    assert other_t.shape == (3, 30)
    assert grid_t.device == trainer_uniform.device
    assert other_t.device == trainer_uniform.device


def test_policy_targets_to_tensor(trainer_uniform: Trainer, env_config: EnvConfig):
    """Test converting policy targets to tensor."""
    action_dim = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    policy_targets = [{0: 0.5, 1: 0.5}, {1: 1.0}]

    policy_tensor = trainer_uniform._policy_targets_to_tensor(
        policy_targets, batch_size=2
    )

    assert policy_tensor.shape == (2, action_dim)
    assert policy_tensor[0, 0] == pytest.approx(0.5)
    assert policy_tensor[0, 1] == pytest.approx(0.5)
    assert policy_tensor[1, 1] == pytest.approx(1.0)


def test_train_step_muzero(
    trainer_uniform: Trainer, dummy_training_targets: list[TrainingTarget]
):
    """Test MuZero training step with unrolled loss."""
    initial_params = [p.clone() for p in trainer_uniform.model.parameters()]

    result = trainer_uniform.train_step(dummy_training_targets)

    assert result is not None
    loss_info, td_errors = result

    # Check loss dictionary
    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert "reward_loss" in loss_info  # MuZero specific!
    assert "mean_td_error" in loss_info

    assert loss_info["total_loss"] > 0
    assert loss_info["reward_loss"] >= 0  # Reward loss should be present

    # Check TD errors
    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (len(dummy_training_targets),)

    # Check parameters changed
    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer_uniform.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change after training step"


def test_train_step_empty_batch(trainer_uniform: Trainer):
    """Test train_step with empty batch."""
    result = trainer_uniform.train_step([])
    assert result is None


def test_distributional_loss_calculation(trainer_uniform: Trainer):
    """Test distributional loss computation."""
    batch_size = 2
    pred_logits = torch.randn(batch_size, trainer_uniform.num_atoms)
    target_scalars = torch.tensor([0.5, -0.3])

    loss = trainer_uniform._distributional_loss(
        pred_logits,
        target_scalars,
        trainer_uniform.support,
        trainer_uniform.v_min,
        trainer_uniform.v_max,
        trainer_uniform.delta_z,
        trainer_uniform.num_atoms,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0


def test_get_current_lr(trainer_uniform: Trainer):
    """Test retrieving current learning rate."""
    lr = trainer_uniform.get_current_lr()
    assert isinstance(lr, float)
    assert lr == trainer_uniform.train_config.LEARNING_RATE
