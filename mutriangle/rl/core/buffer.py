# File: rl/core/buffer.py
import logging
import random
from collections import deque

import numpy as np

from ...config import TrainConfig
from ...utils.sumtree import SumTree
from ...utils.types import (
    Experience,
    GameHistory,
    TrainingTarget,
)

logger = logging.getLogger(__name__)


class GameHistoryBuffer:
    """
    MuZero Experience Buffer storing complete GameHistory trajectories.
    Supports both uniform sampling and Prioritized Experience Replay (PER).
    Samples TrainingTarget objects for unrolled MuZero training.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.capacity = config.BUFFER_CAPACITY
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN
        self.use_per = config.USE_PER

        if self.use_per:
            self.tree = SumTree(self.capacity)
            self.per_alpha = config.PER_ALPHA
            self.per_beta_initial = config.PER_BETA_INITIAL
            self.per_beta_final = config.PER_BETA_FINAL
            # Ensure anneal steps is at least 1 to avoid division by zero
            self.per_beta_anneal_steps = max(
                1, config.PER_BETA_ANNEAL_STEPS or config.MAX_TRAINING_STEPS or 1
            )
            self.per_epsilon = config.PER_EPSILON
            logger.info(
                f"GameHistory buffer initialized with PER (alpha={self.per_alpha}, beta_init={self.per_beta_initial}). Capacity: {self.capacity}"
            )
        else:
            self.buffer: deque[GameHistory] = deque(maxlen=self.capacity)
            logger.info(
                f"GameHistory buffer initialized with uniform sampling. Capacity: {self.capacity}"
            )

    def _get_priority(self, error: float) -> float:
        """Calculates priority from TD error."""
        return float((np.abs(error) + self.per_epsilon) ** self.per_alpha)

    def add(self, game_history: GameHistory):
        """Adds a complete game history. Uses max priority if PER is enabled."""
        if self.use_per:
            max_p = self.tree.max_priority
            self.tree.add(max_p, game_history)
        else:
            self.buffer.append(game_history)

    def add_batch(self, game_histories: list[GameHistory]):
        """Adds a batch of game histories. Uses max priority if PER is enabled."""
        if self.use_per:
            max_p = self.tree.max_priority
            for gh in game_histories:
                self.tree.add(max_p, gh)
        else:
            self.buffer.extend(game_histories)

    def _calculate_beta(self, current_step: int) -> float:
        """Linearly anneals beta from initial to final value."""
        fraction = min(1.0, current_step / self.per_beta_anneal_steps)
        beta = self.per_beta_initial + fraction * (
            self.per_beta_final - self.per_beta_initial
        )
        return beta

    def _sample_game(
        self, current_train_step: int | None = None
    ) -> tuple[GameHistory | None, int, float]:
        """
        Sample a single game from buffer.
        Returns: (game_history, tree_idx, importance_weight)
        For uniform sampling, tree_idx and weight are dummy values.
        """
        if self.use_per:
            if current_train_step is None:
                raise ValueError("current_train_step required for PER sampling")

            # Sample using priority
            value = random.uniform(0, self.tree.total_priority)
            idx, p, data = self.tree.get_leaf(value)

            if not isinstance(data, dict) or "observations" not in data:
                logger.warning(
                    f"PER sampling encountered invalid data at index {idx}. Resampling."
                )
                # Retry once
                value = random.uniform(0, self.tree.total_priority)
                idx, p, data = self.tree.get_leaf(value)
                if not isinstance(data, dict) or "observations" not in data:
                    logger.error(f"PER resampling failed at index {idx}.")
                    return None, idx, 1.0

            # Calculate importance sampling weight
            sampling_prob = p / self.tree.total_priority
            beta = self._calculate_beta(current_train_step)
            weight = (
                (len(self) * sampling_prob) ** (-beta) if sampling_prob > 1e-9 else 0.0
            )

            return data, idx, weight
        else:
            # Uniform sampling
            if not self.buffer:
                return None, 0, 1.0
            game = random.choice(list(self.buffer))
            return game, 0, 1.0

    def _make_training_target(
        self, game: GameHistory, start_idx: int, unroll_steps: int
    ) -> TrainingTarget:
        """
        Create a training target from a position in game history with fresh n-step returns.

        Args:
            game: Complete game history
            start_idx: Starting position t in trajectory
            unroll_steps: Number of steps K to unroll
        Returns:
            TrainingTarget with observation_t and K steps of targets
        """
        T = len(game["actions"])
        raw_rewards = game["rewards"]  # Now stores raw rewards, not n-step returns
        root_values = game["root_values"]

        # Calculate fresh n-step returns for this sample
        n_step = self.config.N_STEP_RETURNS
        gamma = self.config.GAMMA

        # Extract data for K unroll steps (actual available steps, no padding yet)
        actual_steps = min(unroll_steps, T - start_idx)
        actions = game["actions"][start_idx : start_idx + actual_steps]
        target_policies = game["mcts_policies"][
            start_idx : start_idx + actual_steps + 1
        ]
        target_values_raw = root_values[start_idx : start_idx + actual_steps + 1]

        # Calculate n-step returns for each position in the unroll
        target_rewards = []
        for k in range(actual_steps):
            pos = start_idx + k
            G = 0.0
            # Sum discounted rewards for up to n steps
            for step in range(min(n_step, T - pos)):
                G += (gamma**step) * raw_rewards[pos + step]
            # Bootstrap from value if we didn't reach episode end
            if pos + n_step < T:
                bootstrap_idx = pos + n_step
                bootstrap_value = (
                    root_values[bootstrap_idx]
                    if bootstrap_idx < len(root_values)
                    else 0.0
                )
                G += (gamma**n_step) * bootstrap_value
            target_rewards.append(G)

        # Pad if necessary (when near end of episode) - pad with last valid values
        while len(actions) < unroll_steps:
            actions.append(actions[-1] if actions else 0)  # Repeat last action
            target_rewards.append(0.0)  # Zero for padding
            target_policies.append({})  # Empty policy
            target_values_raw.append(
                target_values_raw[-1] if target_values_raw else 0.0
            )

        # Ensure we have K+1 targets
        while len(target_policies) < unroll_steps + 1:
            target_policies.append({})
        while len(target_values_raw) < unroll_steps + 1:
            target_values_raw.append(0.0)

        return TrainingTarget(
            observation=game["observations"][start_idx],
            actions=actions,
            target_rewards=target_rewards,
            target_policies=target_policies,
            target_values=target_values_raw,
        )

    def sample(
        self, batch_size: int, unroll_steps: int, current_train_step: int | None = None
    ) -> list[TrainingTarget] | None:
        """
        Sample training targets from stored game histories.

        For each sample:
        1. Randomly select a game from buffer
        2. Randomly select position t in that game
        3. Extract observation_t, actions[t:t+K], targets for K steps

        Args:
            batch_size: Number of training targets to sample
            unroll_steps: Number of steps K to unroll for each target
            current_train_step: Current training step (required for PER)
        Returns:
            List of TrainingTarget objects, or None if buffer not ready
        """
        current_size = len(self)
        if current_size < 1 or current_size < self.min_size_to_train:
            return None

        batch_targets: list[TrainingTarget] = []
        sampled_indices: list[int] = []
        importance_weights: list[float] = []

        for _ in range(batch_size):
            # Sample a game
            game, tree_idx, weight = self._sample_game(current_train_step)

            if game is None or not game["actions"]:
                logger.warning("Sampled invalid or empty game, skipping sample.")
                continue

            # Sample position in game
            T = len(game["actions"])
            start_idx = random.randint(0, T - 1)

            # Create training target
            try:
                target = self._make_training_target(game, start_idx, unroll_steps)
                batch_targets.append(target)
                sampled_indices.append(tree_idx)
                importance_weights.append(weight)
            except Exception as e:
                logger.error(f"Error creating training target: {e}", exc_info=True)
                continue

        if not batch_targets:
            logger.warning("Failed to create any valid training targets.")
            return None

        # Store indices and weights for PER updates (if needed)
        # For now, we'll handle PER updates differently in MuZero
        # The trainer will need access to these for priority updates
        # We could return a dict similar to PERBatchSample, but for simplicity
        # we'll just return the targets list

        return batch_targets

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Updates the priorities of sampled game histories based on TD errors."""
        if not self.use_per:
            return

        if len(tree_indices) != len(td_errors):
            logger.error(
                f"Mismatch between tree_indices ({len(tree_indices)}) and td_errors ({len(td_errors)}) lengths."
            )
            return

        # Calculate priorities for each error
        priorities = np.array([self._get_priority(err) for err in td_errors])

        if not np.all(np.isfinite(priorities)):
            logger.warning("Non-finite priorities calculated. Clamping.")
            priorities = np.nan_to_num(
                priorities,
                nan=self.per_epsilon,
                posinf=self.tree.max_priority,
                neginf=self.per_epsilon,
            )
            priorities = np.maximum(priorities, self.per_epsilon)

        for idx, p in zip(tree_indices, priorities, strict=False):
            if not (0 <= idx < len(self.tree.tree)):
                logger.error(f"Invalid tree index {idx} provided for priority update.")
                continue
            self.tree.update(idx, p)

        # Update the overall max priority tracked by the tree
        if len(priorities) > 0:
            self.tree._max_priority = max(self.tree.max_priority, np.max(priorities))

    def __len__(self) -> int:
        """Returns the current number of game histories in the buffer."""
        return self.tree.n_entries if self.use_per else len(self.buffer)

    def is_ready(self) -> bool:
        """Checks if the buffer has enough samples to start training."""
        return len(self) >= self.min_size_to_train

    def get_contents(self) -> list[GameHistory]:
        """Returns the contents of the buffer as a list."""
        if self.use_per:
            # Extract valid data from the SumTree
            return [
                self.tree.data[i]
                for i in range(self.tree.n_entries)
                if isinstance(self.tree.data[i], dict)
                and "observations" in self.tree.data[i]
            ]
        else:
            return list(self.buffer)
