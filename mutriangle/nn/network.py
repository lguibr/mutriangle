# File: mutriangle/nn/network.py
import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

# Import GameState and EnvConfig from trianglengin's top level
from trianglengin import EnvConfig, GameState

# Keep mutriangle imports
from ..config import ModelConfig, TrainConfig
from ..features import extract_state_features

# Import ActionType for explicit type usage
from .model import MuTriangleNet

if TYPE_CHECKING:
    from ..utils.types import ActionType, StateType

logger = logging.getLogger(__name__)


class NetworkEvaluationError(Exception):
    """Custom exception for errors during network evaluation."""

    pass


class NeuralNetwork:
    """
    Wrapper for the PyTorch model providing evaluation and state management.
    Handles distributional value head (C51) by calculating expected value for MCTS.
    Optionally compiles the model using torch.compile().
    Conforms to mutrimcts.MuZeroNetworkInterface.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,  # Uses trianglengin.EnvConfig
        train_config: TrainConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.env_config = env_config  # Store trianglengin.EnvConfig
        self.train_config = train_config
        self.device = device
        # Pass trianglengin.EnvConfig to model
        # Store the original, uncompiled model reference separately
        self._orig_model = MuTriangleNet(model_config, env_config).to(device)
        self.model = self._orig_model  # Initially, model is the original model
        # Calculate action_dim manually
        self.action_dim = int(
            env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS
        )
        self.model.eval()

        # Value distribution support
        self.num_atoms = model_config.NUM_VALUE_ATOMS
        self.v_min = model_config.VALUE_MIN
        self.v_max = model_config.VALUE_MAX
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_atoms, device=self.device
        )

        # Reward distribution support (MuZero)
        self.num_reward_atoms = model_config.NUM_REWARD_ATOMS
        self.r_min = model_config.REWARD_MIN
        self.r_max = model_config.REWARD_MAX
        self.delta_r = (self.r_max - self.r_min) / (self.num_reward_atoms - 1)
        self.reward_support = torch.linspace(
            self.r_min, self.r_max, self.num_reward_atoms, device=self.device
        )

        if self.train_config.COMPILE_MODEL:
            if sys.platform == "win32":
                logger.warning(
                    "Model compilation requested but running on Windows. Skipping torch.compile()."
                )
            elif self.device.type == "mps":
                logger.warning(
                    "Model compilation requested but device is 'mps'. Skipping torch.compile()."
                )
            elif hasattr(torch, "compile"):
                try:
                    logger.info(
                        f"Attempting to compile model with torch.compile() on device '{self.device}'..."
                    )
                    # Compile the original model and store the result in self.model
                    self.model = torch.compile(self._orig_model)  # type: ignore
                    logger.info(
                        f"Model compiled successfully on device '{self.device}'."
                    )
                except Exception as e:
                    logger.warning(
                        f"torch.compile() failed on device '{self.device}': {e}. Proceeding without compilation.",
                        exc_info=False,
                    )
                    # Ensure self.model still refers to the original if compilation fails
                    self.model = self._orig_model
            else:
                logger.warning(
                    "torch.compile() requested but not available (requires PyTorch 2.0+). Proceeding without compilation."
                )
        else:
            logger.info(
                "Model compilation skipped (COMPILE_MODEL=False in TrainConfig)."
            )

    def _state_to_tensors(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from trianglengin.GameState and converts them to tensors."""
        state_dict: StateType = extract_state_features(state, self.model_config)
        grid_tensor = torch.from_numpy(state_dict["grid"]).unsqueeze(0).to(self.device)
        other_features_tensor = (
            torch.from_numpy(state_dict["other_features"]).unsqueeze(0).to(self.device)
        )
        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError(
                f"Non-finite values found in input grid_tensor for state {state}"
            )
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                f"Non-finite values found in input other_features_tensor for state {state}"
            )
        return grid_tensor, other_features_tensor

    def _batch_states_to_tensors(
        self, states: list[GameState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from a batch of trianglengin.GameStates and converts to batched tensors."""
        if not states:
            grid_shape = (
                0,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            )
            other_shape = (0, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
            return torch.empty(grid_shape, device=self.device), torch.empty(
                other_shape, device=self.device
            )

        batch_grid = []
        batch_other = []
        for state in states:
            state_dict: StateType = extract_state_features(state, self.model_config)
            batch_grid.append(state_dict["grid"])
            batch_other.append(state_dict["other_features"])

        grid_tensor = torch.from_numpy(np.stack(batch_grid)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(batch_other)).to(self.device)

        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values found in batched input grid_tensor"
            )
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values found in batched input other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    def _logits_to_expected_value(self, value_logits: torch.Tensor) -> torch.Tensor:
        """Calculates the expected value from the value distribution logits."""
        value_probs = F.softmax(value_logits, dim=1)
        support_expanded = self.support.expand_as(value_probs)
        expected_value = torch.sum(value_probs * support_expanded, dim=1, keepdim=True)
        return expected_value

    def _logits_to_expected_reward(self, reward_logits: torch.Tensor) -> torch.Tensor:
        """Calculates the expected reward from the reward distribution logits."""
        reward_probs = F.softmax(reward_logits, dim=1)
        support_expanded = self.reward_support.expand_as(reward_probs)
        expected_reward = torch.sum(
            reward_probs * support_expanded, dim=1, keepdim=True
        )
        return expected_reward

    @torch.inference_mode()
    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        """
        Evaluates a single trianglengin.GameState.
        Returns policy mapping (dict) and EXPECTED value from the distribution.
        Conforms to mutrimcts.MuZeroNetworkInterface.evaluate_state.
        Raises NetworkEvaluationError on issues.
        """
        self.model.eval()
        try:
            grid_tensor, other_features_tensor = self._state_to_tensors(state)
            policy_logits, value_logits = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected for state {state}. Logits: {policy_logits}"
                )
            if not torch.all(torch.isfinite(value_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite value_logits detected for state {state}: {value_logits}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax for state {state}. Logits were: {policy_logits}"
                )

            policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
            policy_probs = np.maximum(policy_probs, 0)
            prob_sum = np.sum(policy_probs)
            if abs(prob_sum - 1.0) > 1e-5:
                logger.warning(
                    f"Evaluate: Policy probabilities sum to {prob_sum:.6f} (not 1.0) for state {state.current_step}. Re-normalizing."
                )
                if prob_sum <= 1e-9:
                    valid_actions = state.valid_actions()
                    if valid_actions:
                        num_valid = len(valid_actions)
                        policy_probs = np.zeros_like(policy_probs)
                        uniform_prob = 1.0 / num_valid
                        for action_idx in valid_actions:
                            if 0 <= action_idx < len(policy_probs):
                                policy_probs[action_idx] = uniform_prob
                        logger.warning(
                            "Policy sum was zero, returning uniform over valid actions."
                        )
                    else:
                        raise NetworkEvaluationError(
                            f"Policy probability sum is near zero ({prob_sum}) for state {state.current_step} with no valid actions. Cannot normalize."
                        )
                else:
                    policy_probs /= prob_sum

            expected_value_tensor = self._logits_to_expected_value(value_logits)
            expected_value_scalar = expected_value_tensor.squeeze(0).item()

            action_policy: dict[ActionType, float] = {
                i: float(p) for i, p in enumerate(policy_probs)
            }

            num_non_zero = sum(1 for p in action_policy.values() if p > 1e-6)
            logger.debug(
                f"Evaluate Final Policy Dict (State {state.current_step}): {num_non_zero}/{self.action_dim} non-zero probs. Example: {list(action_policy.items())[:5]}"
            )

            return action_policy, expected_value_scalar

        except Exception as e:
            logger.error(
                f"Exception during single evaluation for state {state}: {e}",
                exc_info=True,
            )
            raise NetworkEvaluationError(
                f"Evaluation failed for state {state}: {e}"
            ) from e

    @torch.inference_mode()
    def evaluate_batch(
        self, states: list[GameState]
    ) -> list[tuple[dict[int, float], float]]:
        """
        Evaluates a batch of trianglengin.GameStates.
        Returns a list of (policy mapping (dict), EXPECTED value).
        Conforms to mutrimcts.MuZeroNetworkInterface.evaluate_batch.
        Raises NetworkEvaluationError on issues.
        """
        if not states:
            return []
        self.model.eval()
        logger.debug(f"Evaluating batch of {len(states)} states...")
        try:
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)
            policy_logits, value_logits = self.model(grid_tensor, other_features_tensor)

            if not torch.all(torch.isfinite(policy_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite policy_logits detected in batch evaluation. Logits shape: {policy_logits.shape}"
                )
            if not torch.all(torch.isfinite(value_logits)):
                raise NetworkEvaluationError(
                    f"Non-finite value_logits detected in batch value output. Value shape: {value_logits.shape}"
                )

            policy_probs_tensor = F.softmax(policy_logits, dim=1)

            if not torch.all(torch.isfinite(policy_probs_tensor)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax in batch. Logits shape: {policy_logits.shape}"
                )

            policy_probs = policy_probs_tensor.cpu().numpy()
            expected_values_tensor = self._logits_to_expected_value(value_logits)
            expected_values = expected_values_tensor.squeeze(1).cpu().numpy()

            results: list[tuple[dict[int, float], float]] = []
            for batch_idx in range(len(states)):
                probs_i = np.maximum(policy_probs[batch_idx], 0)
                prob_sum_i = np.sum(probs_i)
                if abs(prob_sum_i - 1.0) > 1e-5:
                    logger.warning(
                        f"EvaluateBatch: Policy probabilities sum to {prob_sum_i:.6f} (not 1.0) for sample {batch_idx}. Re-normalizing."
                    )
                    if prob_sum_i <= 1e-9:
                        valid_actions = states[batch_idx].valid_actions()
                        if valid_actions:
                            num_valid = len(valid_actions)
                            probs_i = np.zeros_like(probs_i)
                            uniform_prob = 1.0 / num_valid
                            for action_idx in valid_actions:
                                if 0 <= action_idx < len(probs_i):
                                    probs_i[action_idx] = uniform_prob
                            logger.warning(
                                f"Batch sample {batch_idx} policy sum was zero, returning uniform."
                            )
                        else:
                            raise NetworkEvaluationError(
                                f"Policy probability sum is near zero ({prob_sum_i}) for batch sample {batch_idx} with no valid actions. Cannot normalize."
                            )
                    else:
                        probs_i /= prob_sum_i

                policy_i: dict[ActionType, float] = {
                    i: float(p) for i, p in enumerate(probs_i)
                }
                value_i = float(expected_values[batch_idx])
                results.append((policy_i, value_i))

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(f"Batch evaluation failed: {e}") from e

        logger.debug(f"  Batch evaluation finished. Returning {len(results)} results.")
        return results

    @torch.inference_mode()
    def initial_inference(
        self, observation: GameState
    ) -> tuple[torch.Tensor, dict[int, float], float]:
        """
        MuZero initial inference: observation -> (hidden_state, policy_dict, value).
        Required by mutrimcts.MuZeroNetworkInterface.
        """
        self.model.eval()
        try:
            grid_tensor, other_features_tensor = self._state_to_tensors(observation)
            hidden_state, policy_logits, value_logits = self.model.initial_inference(
                grid_tensor, other_features_tensor
            )

            # Convert policy logits to dict
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            policy_dict = {i: float(p) for i, p in enumerate(policy_probs)}

            # Convert value logits to scalar
            expected_value = (
                self._logits_to_expected_value(value_logits).squeeze(0).item()
            )

            # Note: Don't clone() here - mutrimcts C++ needs original tensor
            # The @torch.inference_mode() decorator prevents gradient tracking anyway
            return hidden_state.squeeze(0), policy_dict, expected_value

        except Exception as e:
            logger.error(f"initial_inference failed: {e}", exc_info=True)
            raise NetworkEvaluationError(f"initial_inference failed: {e}") from e

    @torch.inference_mode()
    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: int
    ) -> tuple[torch.Tensor, float, dict[int, float], float]:
        """
        MuZero recurrent inference: (hidden_state, action) -> (next_hidden, reward, policy_dict, value).
        Required by mutrimcts.MuZeroNetworkInterface.
        """
        self.model.eval()
        try:
            # Add batch dimension if needed
            if hidden_state.dim() == 1:
                hidden_state = hidden_state.unsqueeze(0)

            # Action should be (batch_size,) = (1,) for single action
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

            next_hidden, reward_logits, policy_logits, value_logits = (
                self.model.recurrent_inference(hidden_state, action_tensor)
            )

            # Convert policy logits to dict
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            policy_dict = {i: float(p) for i, p in enumerate(policy_probs)}

            # Convert value and reward logits to scalars
            expected_value = (
                self._logits_to_expected_value(value_logits).squeeze(0).item()
            )
            expected_reward = (
                self._logits_to_expected_reward(reward_logits).squeeze(0).item()
            )

            # Note: Don't clone() here - mutrimcts C++ stores these tensors
            # The @torch.inference_mode() decorator prevents gradient tracking anyway
            return next_hidden.squeeze(0), expected_reward, policy_dict, expected_value

        except Exception as e:
            logger.error(f"recurrent_inference failed: {e}", exc_info=True)
            raise NetworkEvaluationError(f"recurrent_inference failed: {e}") from e

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        # Always get weights from the original underlying model
        return {k: v.cpu() for k, v in self._orig_model.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]):
        """Loads the model's state dictionary into the original underlying model."""
        try:
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            # Always load weights into the original underlying model
            self._orig_model.load_state_dict(weights_on_device)
            # Ensure the potentially compiled model (self.model) is also in eval mode
            self.model.eval()
            logger.debug("NN weights set successfully into underlying model.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise
