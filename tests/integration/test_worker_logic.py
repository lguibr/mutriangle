"""Test worker episode logic - simulates what worker does without Ray."""
import torch
from trianglengin import GameState, EnvConfig
from mutrimcts import run_mcts, SearchConfiguration

from mutriangle.config import ModelConfig, TrainConfig
from mutriangle.nn import NeuralNetwork
from mutriangle.rl.self_play.mcts_helpers import select_action_from_visits
from mutriangle.features import extract_state_features
from mutriangle.utils.types import GameHistory


def test_episode_logic_with_real_mcts():
    """Simulate what worker.run_episode() does - verify MCTS policy is used correctly."""
    env_config = EnvConfig()
    model_config = ModelConfig()
    train_config = TrainConfig(NUM_SELF_PLAY_WORKERS=1, MAX_TRAINING_STEPS=10000)
    mcts_config = SearchConfiguration(
        max_simulations=16,
        max_depth=5,
        cpuct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
        mcts_batch_size=16,
    )
    
    device = torch.device("cpu")
    nn_eval = NeuralNetwork(model_config, env_config, train_config, device)
    nn_eval.model.eval()
    
    # Initialize game
    game = GameState(env_config, initial_seed=42)
    
    assert not game.is_over(), "Game should not be over at start"
    assert len(game.valid_actions()) > 0, "Should have valid actions"
    
    # Simulate episode collection (like worker does)
    observations = []
    actions_taken = []
    raw_rewards = []
    mcts_policies = []
    root_values = []
    
    max_steps = 5  # Just test first 5 steps for speed
    current_trainer_step = 500  # Simulating early training
    
    for step in range(max_steps):
        if game.is_over():
            break
        
        # Extract state features (like worker)
        state_features = extract_state_features(game, model_config)
        observations.append(state_features)
        
        # Run MCTS (THE CRITICAL PART)
        visit_counts, root_value, mcts_policy = run_mcts(
            initial_observation=game,
            network_interface=nn_eval,
            config=mcts_config,
        )
        
        # VERIFY MCTS OUTPUTS
        assert visit_counts, f"Step {step}: visit_counts is empty!"
        total_visits = sum(visit_counts.values())
        assert total_visits > 0, f"Step {step}: total_visits={total_visits}"
        
        # THE BUG CHECK: mcts_policy should NOT be empty
        assert mcts_policy, f"Step {step}: mcts_policy is EMPTY!"
        policy_sum = sum(mcts_policy.values())
        assert abs(policy_sum - 1.0) < 0.01, \
            f"Step {step}: mcts_policy sum={policy_sum:.4f}, should be ~1.0"
        
        # Store mcts_policy directly (like worker does after our fix)
        policy_target = mcts_policy
        mcts_policies.append(policy_target)
        
        # Store root_value from MCTS
        root_values.append(root_value)
        
        # Select action (with temperature)
        explore_steps = train_config.MAX_TRAINING_STEPS * 0.1
        selection_temp = 1.0 if current_trainer_step < explore_steps else 0.1
        
        action = select_action_from_visits(visit_counts, temperature=selection_temp)
        actions_taken.append(action)
        
        # Take step in game
        reward, done = game.step(action)
        raw_rewards.append(reward)
        
        if done:
            break
    
    # Verify we collected valid data
    assert len(observations) > 0, "Should have collected at least 1 observation"
    assert len(actions_taken) > 0, "Should have taken at least 1 action"
    assert len(mcts_policies) > 0, "Should have stored at least 1 MCTS policy"
    
    # Check all stored policies are valid
    for i, policy in enumerate(mcts_policies):
        assert len(policy) > 0, f"Policy {i} is EMPTY!"
        policy_sum = sum(policy.values())
        assert abs(policy_sum - 1.0) < 0.01, \
            f"Policy {i} sum={policy_sum:.4f}"
    
    # Create GameHistory
    game_history: GameHistory = {
        "observations": observations,
        "actions": actions_taken,
        "rewards": raw_rewards,
        "mcts_policies": mcts_policies,
        "root_values": root_values,
    }
    
    # Verify structure
    assert len(game_history['observations']) == len(game_history['actions'])
    assert len(game_history['mcts_policies']) == len(game_history['actions'])
    
    print(f"✅ Episode simulation: {len(observations)} steps, all policies valid")
    print(f"   First policy: {len(mcts_policies[0])} actions, sum={sum(mcts_policies[0].values()):.4f}")
    print(f"   Total visits first step: {sum([1 for k in visit_counts])} unique actions")


def test_mcts_policy_vs_visit_counts():
    """Verify that mcts_policy matches normalized visit_counts."""
    env_config = EnvConfig()
    game = GameState(env_config, initial_seed=100)
    
    class SimpleNetwork:
        def initial_inference(self, obs):
            return torch.zeros(128), {i: 1.0/360 for i in range(360)}, 0.0
        def recurrent_inference(self, h, a):
            return torch.zeros(128), 0.0, {i: 1.0/360 for i in range(360)}, 0.0
    
    network = SimpleNetwork()
    config = SearchConfiguration(max_simulations=32, max_depth=5, cpuct=1.5)
    
    visit_counts, root_value, mcts_policy = run_mcts(game, network, config)
    
    # Verify mcts_policy == normalized visit_counts
    total_visits = sum(visit_counts.values())
    for action, count in visit_counts.items():
        expected_prob = count / total_visits
        actual_prob = mcts_policy[action]
        assert abs(expected_prob - actual_prob) < 0.001, \
            f"Action {action}: visits={count}, expected_prob={expected_prob:.4f}, actual={actual_prob:.4f}"
    
    print(f"✅ mcts_policy correctly normalized from visit_counts")

