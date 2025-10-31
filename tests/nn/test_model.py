# File: tests/nn/test_model.py
"""Tests for MuZero Network Architecture"""

import pytest
import torch

from trianglengin import EnvConfig
from mutriangle.config import ModelConfig
from mutriangle.nn import MuTriangleNet


@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def model(model_config: ModelConfig, env_config: EnvConfig) -> MuTriangleNet:
    """Provides an instance of the MuTriangleNet model."""
    return MuTriangleNet(model_config, env_config)


def test_model_initialization(
    model: MuTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test MuZero model initializes with three sub-networks."""
    assert model is not None
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)
    assert model.action_dim == action_dim_int

    # Check three sub-networks exist
    assert hasattr(model, "representation")
    assert hasattr(model, "dynamics")
    assert hasattr(model, "prediction")

    assert model.representation is not None
    assert model.dynamics is not None
    assert model.prediction is not None


def test_model_forward_pass_legacy(
    model: MuTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test legacy forward pass (for MCTS compatibility)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        policy_logits, value_logits = model(dummy_grid, dummy_other)

    assert policy_logits.shape == (batch_size, action_dim_int)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32


def test_initial_inference(
    model: MuTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test MuZero initial inference: h(o) -> s, f(s) -> p,v."""
    batch_size = 2
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        hidden_state, policy_logits, value_logits = model.initial_inference(
            dummy_grid, dummy_other
        )

    # Check hidden state
    assert hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert hidden_state.dtype == torch.float32

    # Check policy and value
    assert policy_logits.shape == (batch_size, action_dim_int)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)


def test_recurrent_inference(
    model: MuTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test MuZero recurrent inference: g(s,a) -> s',r; f(s') -> p,v."""
    batch_size = 2
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)

    # Create dummy hidden state
    hidden_state = torch.randn(batch_size, model_config.HIDDEN_STATE_DIM, device=device)

    # Create dummy actions
    actions = torch.randint(0, action_dim_int, (batch_size,), device=device)

    with torch.no_grad():
        next_hidden_state, reward_logits, policy_logits, value_logits = (
            model.recurrent_inference(hidden_state, actions)
        )

    # Check next hidden state
    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)

    # Check reward prediction
    assert reward_logits.shape == (batch_size, model_config.NUM_REWARD_ATOMS)

    # Check policy and value
    assert policy_logits.shape == (batch_size, action_dim_int)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)


def test_full_unroll(
    model: MuTriangleNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test complete MuZero unroll: initial + K recurrent steps."""
    batch_size = 2
    K = 5  # Unroll steps
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)

    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        # Step 0: Initial inference
        hidden_states, policy_0, value_0 = model.initial_inference(
            dummy_grid, dummy_other
        )

        assert hidden_states.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
        assert policy_0.shape == (batch_size, action_dim_int)
        assert value_0.shape == (batch_size, model_config.NUM_VALUE_ATOMS)

        # Steps 1..K: Recurrent inference
        for k in range(K):
            actions = torch.randint(0, action_dim_int, (batch_size,), device=device)
            hidden_states, reward_k, policy_k, value_k = model.recurrent_inference(
                hidden_states, actions
            )

            assert hidden_states.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
            assert reward_k.shape == (batch_size, model_config.NUM_REWARD_ATOMS)
            assert policy_k.shape == (batch_size, action_dim_int)
            assert value_k.shape == (batch_size, model_config.NUM_VALUE_ATOMS)


@pytest.mark.parametrize(
    "use_transformer", [False, True], ids=["CNN_Only", "CNN_Transformer"]
)
def test_model_with_transformer_toggle(use_transformer: bool, env_config: EnvConfig):
    """Test model with transformer enabled/disabled."""
    action_dim_int = int(env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS)

    model_config_test = ModelConfig(
        GRID_INPUT_CHANNELS=1,
        CONV_FILTERS=[4, 8],
        CONV_KERNEL_SIZES=[3, 3],
        CONV_STRIDES=[1, 1],
        CONV_PADDING=[1, 1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=8,
        USE_TRANSFORMER=use_transformer,
        TRANSFORMER_DIM=16,
        TRANSFORMER_HEADS=2,
        TRANSFORMER_LAYERS=1,
        TRANSFORMER_FC_DIM=32,
        FC_DIMS_SHARED=[16],
        POLICY_HEAD_DIMS=[action_dim_int],
        OTHER_NN_INPUT_FEATURES_DIM=10,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=True,
        HIDDEN_STATE_DIM=32,
        REPRESENTATION_HIDDEN_DIMS=[64],
        DYNAMICS_HIDDEN_DIMS=[64],
        PREDICTION_HIDDEN_DIMS=[64],
    )

    model = MuTriangleNet(model_config_test, env_config)
    batch_size = 2
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config_test.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config_test.OTHER_NN_INPUT_FEATURES_DIM)

    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        hidden_state, policy_logits, value_logits = model.initial_inference(
            dummy_grid, dummy_other
        )

    assert hidden_state.shape == (batch_size, model_config_test.HIDDEN_STATE_DIM)
    assert policy_logits.shape == (batch_size, action_dim_int)
    assert value_logits.shape == (batch_size, model_config_test.NUM_VALUE_ATOMS)
