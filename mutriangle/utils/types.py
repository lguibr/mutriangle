# File: mutriangle/utils/types.py
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict


class StateType(TypedDict):
    """
    Represents the processed state features input to the neural network.
    Contains numerical arrays derived from the raw GameState.
    """

    grid: np.ndarray  # (C, H, W) float32, e.g., occupancy, death cells
    other_features: np.ndarray  # (OtherFeatDim,) float32, e.g., shape info, game stats


# Action representation (integer index)
ActionType = int

# Policy target from MCTS (visit counts distribution)
# Mapping from action index to its probability (normalized visit count)
PolicyTargetMapping = Mapping[ActionType, float]


# REMOVED StepInfo TypedDict (no longer needed directly here)


# --- MuZero Data Types ---


class GameHistory(TypedDict):
    """
    Complete trajectory from one episode.
    Stores all observations, actions, rewards, MCTS policies, and root values.
    """

    observations: list[StateType]  # s_0, s_1, ..., s_T
    actions: list[ActionType]  # a_0, a_1, ..., a_{T-1}
    rewards: list[float]  # r_0, r_1, ..., r_T (n-step returns)
    mcts_policies: list[PolicyTargetMapping]  # π_0, π_1, ..., π_{T-1}
    root_values: list[float]  # v_0, v_1, ..., v_{T-1}


class TrainingTarget(TypedDict):
    """
    Single training sample extracted from GameHistory for MuZero unrolled training.
    Contains observation at time t and K unroll steps of targets.
    """

    observation: StateType  # Initial observation at position t
    actions: list[ActionType]  # K actions starting from t
    target_rewards: list[float]  # K target rewards (n-step returns)
    target_policies: list[PolicyTargetMapping]  # K+1 policy targets (t to t+K)
    target_values: list[float]  # K+1 value targets (t to t+K)


# --- AlphaZero Data Types (Legacy, for compatibility) ---


Experience = tuple[StateType, PolicyTargetMapping, float]
# Represents one unit of experience stored in the replay buffer.
# 1. StateType: The processed features (grid, other_features) of the state s_t.
#               NOTE: This is NOT the raw GameState object.
# 2. PolicyTargetMapping: The MCTS-derived policy target pi(a|s_t) for state s_t.
# 3. float: The calculated N-step return G_t^n starting from state s_t, used
#           as the target for the value head during training.


# Batch of experiences for training
ExperienceBatch = list[Experience]


# Output type from the neural network's evaluate method (for MCTS interaction)
# 1. dict[ActionType, float]: Policy probabilities P(a|s) for the evaluated state.
# 2. float: The expected value V(s) calculated from the value distribution logits.
PolicyValueOutput = tuple[dict[int, float], float]


# REMOVED StatsCollectorData type alias


class PERBatchSample(TypedDict):
    """Output of the PER buffer's sample method."""

    batch: ExperienceBatch  # The sampled experiences
    indices: np.ndarray  # Tree indices of the sampled experiences (for priority update)
    weights: np.ndarray  # Importance sampling weights for the sampled experiences
