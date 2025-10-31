"""Test mutrimcts integration - verify MCTS returns valid policy."""
import torch
import pytest
from trianglengin import GameState, EnvConfig
from mutrimcts import run_mcts, SearchConfiguration


class MockMuZeroNetwork:
    """Mock network that returns valid outputs for MCTS."""
    
    def __init__(self):
        self.call_count = 0
    
    def initial_inference(self, observation):
        """Return valid initial inference outputs."""
        self.call_count += 1
        hidden_state = torch.randn(128)  # Hidden state
        
        # Return uniform policy over all 360 actions
        policy_dict = {i: 1.0/360 for i in range(360)}
        value = 0.0
        
        return hidden_state, policy_dict, value
    
    def recurrent_inference(self, hidden_state, action):
        """Return valid recurrent inference outputs."""
        self.call_count += 1
        next_hidden = torch.randn(128)
        reward = 0.0
        policy_dict = {i: 1.0/360 for i in range(360)}
        value = 0.0
        
        return next_hidden, reward, policy_dict, value


def test_mutrimcts_returns_valid_data():
    """Test that mutrimcts.run_mcts() returns valid visit_counts, root_value, mcts_policy."""
    env_config = EnvConfig()
    game = GameState(env_config, initial_seed=42)
    network = MockMuZeroNetwork()
    
    config = SearchConfiguration(
        max_simulations=64,
        max_depth=10,
        cpuct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
        mcts_batch_size=16,
    )
    
    # Run MCTS
    visit_counts, root_value, mcts_policy = run_mcts(game, network, config)
    
    # Verify visit_counts
    assert isinstance(visit_counts, dict), f"visit_counts should be dict, got {type(visit_counts)}"
    assert len(visit_counts) > 0, "visit_counts should not be empty"
    total_visits = sum(visit_counts.values())
    assert total_visits > 0, f"Total visits should be > 0, got {total_visits}"
    assert total_visits <= 64, f"Total visits should be <= 64, got {total_visits}"
    
    # Verify root_value
    assert isinstance(root_value, (float, int)), f"root_value should be float/int, got {type(root_value)}"
    
    # Verify mcts_policy (THE KEY TEST!)
    assert isinstance(mcts_policy, dict), f"mcts_policy should be dict, got {type(mcts_policy)}"
    assert len(mcts_policy) > 0, f"mcts_policy should NOT be empty! Got length {len(mcts_policy)}"
    
    policy_sum = sum(mcts_policy.values())
    assert abs(policy_sum - 1.0) < 0.01, f"mcts_policy should sum to ~1.0, got {policy_sum}"
    
    # Verify they have same actions
    assert set(visit_counts.keys()) == set(mcts_policy.keys()), \
        "visit_counts and mcts_policy should have same action keys"
    
    # Verify policy is normalized visits
    for action in visit_counts:
        expected_prob = visit_counts[action] / total_visits
        actual_prob = mcts_policy[action]
        assert abs(expected_prob - actual_prob) < 0.001, \
            f"Action {action}: expected {expected_prob:.4f}, got {actual_prob:.4f}"
    
    # Verify network was called (initial + expansions)
    assert network.call_count > 1, f"Network should be called multiple times, got {network.call_count}"
    
    print(f"✅ MCTS OK: {len(visit_counts)} actions, {total_visits} visits, policy_sum={policy_sum:.4f}")


def test_mcts_with_real_game():
    """Test MCTS with actual trianglengin game to ensure compatibility."""
    env_config = EnvConfig()
    game = GameState(env_config, initial_seed=123)
    
    assert not game.is_over(), "Game should not be over at start"
    assert len(game.valid_actions()) > 0, "Should have valid actions"
    
    network = MockMuZeroNetwork()
    config = SearchConfiguration(max_simulations=32, max_depth=8, cpuct=1.5)
    
    visit_counts, root_value, mcts_policy = run_mcts(game, network, config)
    
    assert len(visit_counts) > 0
    assert len(mcts_policy) > 0
    assert sum(mcts_policy.values()) > 0.99
    
    print(f"✅ Real game MCTS OK: {len(mcts_policy)} actions in policy")


def test_mcts_policy_not_empty():
    """Specific test for the bug: mcts_policy should NEVER be empty."""
    env_config = EnvConfig()
    game = GameState(env_config, initial_seed=999)
    network = MockMuZeroNetwork()
    config = SearchConfiguration(max_simulations=16, max_depth=5, cpuct=1.5)
    
    visit_counts, root_value, mcts_policy = run_mcts(game, network, config)
    
    # THE CRITICAL ASSERTION
    assert mcts_policy != {}, "mcts_policy must NOT be empty dict!"
    assert len(mcts_policy) >= len(visit_counts), \
        "mcts_policy should have at least as many entries as visit_counts"
    
    print(f"✅ mcts_policy not empty: {len(mcts_policy)} actions")

