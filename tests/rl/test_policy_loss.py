"""Tests for policy loss calculation to debug zero policy loss issue."""

import numpy as np
import pytest
import torch

from mutriangle.config import ModelConfig, TrainConfig
from mutriangle.rl import Trainer
from mutriangle.nn import NeuralNetwork
from mutriangle.utils.types import TrainingTarget, StateType
from trianglengin import EnvConfig


@pytest.fixture
def simple_trainer():
    """Create a minimal trainer for testing."""
    env_config = EnvConfig()
    model_config = ModelConfig()
    train_config = TrainConfig(NUM_SELF_PLAY_WORKERS=1, BATCH_SIZE=4)
    device = torch.device("cpu")

    nn = NeuralNetwork(model_config, env_config, train_config, device)
    trainer = Trainer(nn, train_config, env_config)
    return trainer, env_config


def test_policy_tensor_conversion_nonzero(simple_trainer):
    """Test that policy dict converts to non-zero tensor."""
    trainer, env_config = simple_trainer

    # Create a simple policy dict
    policy_dict = {15: 0.3, 42: 0.5, 7: 0.2}
    batch_size = 1

    tensor = trainer._policy_targets_to_tensor([policy_dict], batch_size)

    # Assertions
    assert tensor.shape[0] == batch_size
    assert tensor.sum() > 0.99, f"Tensor sum should be ~1.0, got {tensor.sum()}"
    assert (tensor > 0).sum() == 3, "Should have exactly 3 non-zero entries"
    assert abs(tensor[0, 15].item() - 0.3) < 1e-5
    assert abs(tensor[0, 42].item() - 0.5) < 1e-5
    assert abs(tensor[0, 7].item() - 0.2) < 1e-5


def test_policy_tensor_conversion_empty_dict(simple_trainer):
    """Test that empty policy dict produces all-zero tensor."""
    trainer, env_config = simple_trainer

    # Empty dict (the suspected bug)
    empty_policy = {}
    batch_size = 1

    tensor = trainer._policy_targets_to_tensor([empty_policy], batch_size)

    # This SHOULD produce zeros
    assert tensor.sum() == 0.0, "Empty dict should produce zero tensor"
    assert (tensor > 0).sum() == 0, "Should have no non-zero entries"


def test_policy_loss_with_valid_target(simple_trainer):
    """Test that policy loss is non-zero with valid policy targets."""
    trainer, env_config = simple_trainer

    # Create minimal training targets with valid policies
    dummy_obs: StateType = {
        "grid": np.zeros((1, env_config.ROWS, env_config.COLS), dtype=np.float32),
        "other_features": np.zeros(30, dtype=np.float32),
    }

    # Valid policy dict with 3 actions
    valid_policy = {15: 0.3, 42: 0.5, 7: 0.2}

    batch_targets = [
        TrainingTarget(
            observation=dummy_obs,
            actions=[15] * 5,  # K=5 actions
            target_rewards=[0.0] * 5,
            target_policies=[valid_policy] * 6,  # K+1 = 6 policies
            target_values=[0.0] * 6,
        )
    ] * 4  # batch_size=4

    # Run training step
    result = trainer.train_step(batch_targets)

    assert result is not None, "Training step should not return None"
    loss_info, td_errors = result

    # Policy loss should be NON-ZERO with valid targets
    policy_loss = loss_info["policy_loss"]
    assert policy_loss != 0.0, f"Policy loss should be non-zero, got {policy_loss}"
    assert policy_loss > 0, f"Policy loss should be positive, got {policy_loss}"
    # Cross-entropy can be high with random network (entropy of 360 actions)
    assert 0.1 < policy_loss < 50, f"Policy loss out of expected range: {policy_loss}"


def test_policy_loss_with_empty_target(simple_trainer):
    """Test that policy loss is ZERO with empty policy targets (the bug)."""
    trainer, env_config = simple_trainer

    dummy_obs: StateType = {
        "grid": np.zeros((1, env_config.ROWS, env_config.COLS), dtype=np.float32),
        "other_features": np.zeros(30, dtype=np.float32),
    }

    # EMPTY policy dict (suspected bug)
    empty_policy = {}

    batch_targets = [
        TrainingTarget(
            observation=dummy_obs,
            actions=[0] * 5,
            target_rewards=[0.0] * 5,
            target_policies=[empty_policy] * 6,  # Empty dicts!
            target_values=[0.0] * 6,
        )
    ] * 4

    result = trainer.train_step(batch_targets)
    assert result is not None
    loss_info, _ = result

    # With empty targets, policy loss SHOULD be zero (this is the bug we're testing for)
    policy_loss = loss_info["policy_loss"]
    assert policy_loss == 0.0 or abs(policy_loss) < 1e-6, (
        f"Empty policy target should give zero loss, got {policy_loss}"
    )


def test_mcts_policy_normalization():
    """Test that MCTS policy from mutrimcts sums to 1.0."""
    # This test requires actual mutrimcts - skip if not available
    pytest.importorskip("mutrimcts")

    from mutrimcts import run_mcts, SearchConfiguration
    from trianglengin import GameState, EnvConfig

    # Create a simple game state
    env_config = EnvConfig()
    game = GameState(env_config)

    # Create a mock network that returns valid outputs
    class MockNetwork:
        def initial_inference(self, observation):
            hidden = torch.zeros(128)
            policy = {i: 1.0 / 360 for i in range(360)}  # Uniform
            value = 0.0
            return hidden, policy, value

        def recurrent_inference(self, hidden_state, action):
            next_hidden = torch.zeros(128)
            reward = 0.0
            policy = {i: 1.0 / 360 for i in range(360)}
            value = 0.0
            return next_hidden, reward, policy, value

    mock_net = MockNetwork()
    config = SearchConfiguration(
        max_simulations=64,
        max_depth=10,
        cpuct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
        mcts_batch_size=16,
    )

    try:
        visit_counts, root_value, mcts_policy = run_mcts(game, mock_net, config)

        # Validate visit_counts
        assert isinstance(visit_counts, dict), (
            f"visit_counts should be dict, got {type(visit_counts)}"
        )
        assert len(visit_counts) > 0, "visit_counts should not be empty"
        total_visits = sum(visit_counts.values())
        assert total_visits > 0, (
            f"visit_counts sum should be positive, got {total_visits}"
        )

        # Validate mcts_policy
        assert isinstance(mcts_policy, dict), (
            f"mcts_policy should be dict, got {type(mcts_policy)}"
        )
        assert len(mcts_policy) > 0, (
            f"mcts_policy should not be empty (len={len(mcts_policy)})"
        )
        policy_sum = sum(mcts_policy.values())
        assert abs(policy_sum - 1.0) < 0.01, (
            f"mcts_policy should sum to ~1.0, got {policy_sum}"
        )

        # Validate they have same actions
        assert set(visit_counts.keys()) == set(mcts_policy.keys()), (
            "visit_counts and mcts_policy should have same actions"
        )

    except Exception as e:
        pytest.fail(f"mutrimcts.run_mcts raised exception: {e}")
