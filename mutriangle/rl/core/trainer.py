# File: rl/core/trainer.py
import logging
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.checkpoint import checkpoint

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig

# Keep mutriangle imports
from ...config import TrainConfig
from ...nn import NeuralNetwork
from ...utils.types import (
    ExperienceBatch,
    PERBatchSample,
    PolicyTargetMapping,
    TrainingTarget,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the neural network training process for MuZero.
    Implements the unrolled loss function with gradient scaling for dynamics.
    Supports Distributional RL (C51) for both value and reward predictions.
    """

    def __init__(
        self,
        nn_interface: NeuralNetwork,
        train_config: TrainConfig,
        env_config: EnvConfig,
    ):
        self.nn = nn_interface
        self.model = nn_interface.model
        self.train_config = train_config
        self.env_config = env_config
        self.model_config = nn_interface.model_config
        self.device = nn_interface.device
        self.optimizer = self._create_optimizer()
        self.scheduler: _LRScheduler | None = self._create_scheduler(self.optimizer)

        # Value distribution support
        self.num_atoms = self.nn.num_atoms
        self.v_min = self.nn.v_min
        self.v_max = self.nn.v_max
        self.delta_z = self.nn.delta_z
        self.support = self.nn.support.to(self.device)

        # Reward distribution support
        self.num_reward_atoms = self.nn.num_reward_atoms
        self.r_min = self.nn.r_min
        self.r_max = self.nn.r_max
        self.delta_r = self.nn.delta_r
        self.reward_support = self.nn.reward_support.to(self.device)

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer with separate parameter groups for policy and value heads."""
        base_lr = self.train_config.LEARNING_RATE
        policy_lr = self.train_config.POLICY_LEARNING_RATE or base_lr
        value_lr = self.train_config.VALUE_LEARNING_RATE or base_lr
        wd = self.train_config.WEIGHT_DECAY
        opt_type = self.train_config.OPTIMIZER_TYPE.lower()

        # Separate parameters into groups
        policy_head_params = []
        value_head_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "policy_head" in name:
                policy_head_params.append(param)
            elif "value_head" in name:
                value_head_params.append(param)
            else:
                base_params.append(param)

        param_groups = [
            {"params": base_params, "lr": base_lr},
            {"params": policy_head_params, "lr": policy_lr},
            {"params": value_head_params, "lr": value_lr},
        ]

        # Filter out groups with no parameters
        param_groups = [pg for pg in param_groups if pg["params"]]

        logger.info(f"Creating optimizer: {opt_type} with parameter groups:")
        for i, pg in enumerate(param_groups):
            num_params = sum(p.numel() for p in pg["params"])
            group_name = "Base"
            if pg["params"] is policy_head_params:
                group_name = "Policy Head"
            elif pg["params"] is value_head_params:
                group_name = "Value Head"
            logger.info(
                f"  - Group {i} ({group_name}): {num_params:,} params, LR: {pg['lr']}"
            )

        if opt_type == "adam":
            return optim.Adam(param_groups, weight_decay=wd)
        elif opt_type == "adamw":
            return optim.AdamW(param_groups, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(param_groups, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}"
            )

    def _create_scheduler(self, optimizer: optim.Optimizer) -> _LRScheduler | None:
        """Creates the learning rate scheduler based on TrainConfig."""
        scheduler_type_config = self.train_config.LR_SCHEDULER_TYPE
        scheduler_type: str | None = None
        if scheduler_type_config:
            scheduler_type = scheduler_type_config.lower()

        if not scheduler_type or scheduler_type == "none":
            logger.info("No LR scheduler configured.")
            return None

        logger.info(f"Creating LR scheduler: {scheduler_type}")
        if scheduler_type == "steplr":
            step_size = getattr(self.train_config, "LR_SCHEDULER_STEP_SIZE", 100000)
            gamma = getattr(self.train_config, "LR_SCHEDULER_GAMMA", 0.1)
            logger.info(f"  StepLR params: step_size={step_size}, gamma={gamma}")
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            )
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            if t_max is None:
                logger.warning(
                    "LR_SCHEDULER_T_MAX is None for CosineAnnealingLR. Scheduler might not work as expected."
                )
                t_max = self.train_config.MAX_TRAINING_STEPS or 1_000_000
            logger.info(f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}")
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                ),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type_config}")

    def _observations_to_tensors(
        self, observations: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of StateType observations to tensors.

        Args:
            observations: List of StateType dicts
        Returns:
            grid_tensor: (batch, C, H, W)
            other_features_tensor: (batch, feature_dim)
        """
        batch_size = len(observations)
        grids = []
        other_features = []

        for obs in observations:
            grids.append(obs["grid"])
            other_features.append(obs["other_features"])

        grid_tensor = torch.from_numpy(np.stack(grids)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(other_features)).to(
            self.device
        )

        return grid_tensor, other_features_tensor

    def _policy_targets_to_tensor(
        self, policy_targets: list[PolicyTargetMapping], batch_size: int
    ) -> torch.Tensor:
        """
        Convert policy target mappings to a tensor.

        Args:
            policy_targets: List of policy dicts
            batch_size: Size of batch
        Returns:
            policy_tensor: (batch, action_dim)
        """
        action_dim_int = int(
            self.env_config.NUM_SHAPE_SLOTS
            * self.env_config.ROWS
            * self.env_config.COLS
        )
        policy_tensor = torch.zeros(
            (batch_size, action_dim_int),
            dtype=torch.float32,
            device=self.device,
        )

        for i, policy_map in enumerate(policy_targets):
            for action, prob in policy_map.items():
                if 0 <= action < action_dim_int:
                    policy_tensor[i, action] = prob
                else:
                    logger.warning(
                        f"Action {action} out of bounds in policy target for sample {i}."
                    )

        return policy_tensor

    def _calculate_target_distribution(
        self,
        targets: torch.Tensor,
        support: torch.Tensor,
        v_min: float,
        v_max: float,
        delta: float,
        num_atoms: int,
    ) -> torch.Tensor:
        """
        Projects scalar targets onto fixed support atoms (for distributional RL).

        Args:
            targets: (batch,) tensor of scalar targets
            support: Support atoms tensor
            v_min, v_max, delta, num_atoms: Distribution parameters
        Returns:
            Target distribution (batch, num_atoms)
        """
        batch_size = targets.size(0)
        m = torch.zeros(
            (batch_size, num_atoms), dtype=torch.float32, device=self.device
        )
        target_clamped = targets.clamp(v_min, v_max)
        b = (target_clamped - v_min) / delta
        lower_idx = b.floor().long()
        u = b.ceil().long()
        # Ensure indices are within bounds [0, num_atoms - 1]
        lower_idx = torch.clamp(lower_idx, 0, num_atoms - 1)
        u = torch.clamp(u, 0, num_atoms - 1)

        # Handle cases where lower and upper indices are the same
        eq_mask = lower_idx == u
        ne_mask = ~eq_mask

        # Calculate weights for non-equal indices
        m_l = u[ne_mask].float() - b[ne_mask]
        m_u = b[ne_mask] - lower_idx[ne_mask].float()

        # Distribute probability mass
        batch_indices = torch.arange(batch_size, device=self.device)
        m[batch_indices[ne_mask], lower_idx[ne_mask]] += m_l
        m[batch_indices[ne_mask], u[ne_mask]] += m_u

        # Handle equal indices (assign full probability mass)
        m[batch_indices[eq_mask], lower_idx[eq_mask]] += 1.0

        return m

    def _distributional_loss(
        self,
        pred_logits: torch.Tensor,
        target_scalars: torch.Tensor,
        support: torch.Tensor,
        v_min: float,
        v_max: float,
        delta: float,
        num_atoms: int,
    ) -> torch.Tensor:
        """
        Calculate distributional cross-entropy loss.

        Args:
            pred_logits: (batch, num_atoms) prediction logits
            target_scalars: (batch,) target scalar values
            support, v_min, v_max, delta, num_atoms: Distribution parameters
        Returns:
            loss: scalar loss value
        """
        with torch.no_grad():
            target_distribution = self._calculate_target_distribution(
                target_scalars, support, v_min, v_max, delta, num_atoms
            )
        log_pred_dist = F.log_softmax(pred_logits, dim=1)
        loss_elementwise = -torch.sum(target_distribution * log_pred_dist, dim=1)
        return loss_elementwise.mean()

    def train_step(
        self, batch_targets: list[TrainingTarget]
    ) -> tuple[dict[str, float], np.ndarray] | None:
        """
        MuZero training step with unrolled dynamics.

        For each sample in batch:
        1. Initial inference: s_0 = h(o_0), get p_0, v_0
        2. Unroll K steps: for k=0..K-1:
            - s_{k+1}, r_k = g(s_k, a_k)
            - p_{k+1}, v_{k+1} = f(s_{k+1})
        3. Calculate losses at each step
        4. Scale gradients for dynamics network

        Args:
            batch_targets: List of TrainingTarget dicts
        Returns:
            (loss_info_dict, td_errors) or None if training fails
        """
        if not batch_targets:
            logger.warning("train_step called with empty batch.")
            return None

        self.model.train()
        batch_size = len(batch_targets)
        K = self.train_config.UNROLL_STEPS

        try:
            # Prepare observations
            observations = [target["observation"] for target in batch_targets]
            grid_t, other_t = self._observations_to_tensors(observations)
        except Exception as e:
            logger.error(f"Error preparing observations: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()

        # === STEP 0: Initial Inference ===
        hidden_states, policy_logits_0, value_logits_0 = self.model.initial_inference(
            grid_t, other_t
        )

        # Calculate initial losses
        target_policies_0 = [t["target_policies"][0] for t in batch_targets]
        policy_target_0 = self._policy_targets_to_tensor(target_policies_0, batch_size)

        target_values_0 = torch.tensor(
            [t["target_values"][0] for t in batch_targets],
            dtype=torch.float32,
            device=self.device,
        )

        # Policy loss (cross-entropy)
        log_probs_0 = F.log_softmax(policy_logits_0, dim=1)
        policy_target_0 = torch.nan_to_num(policy_target_0, nan=0.0)
        policy_loss_0 = -torch.sum(policy_target_0 * log_probs_0, dim=1).mean()

        # Value loss (distributional)
        value_loss_0 = self._distributional_loss(
            value_logits_0,
            target_values_0,
            self.support,
            self.v_min,
            self.v_max,
            self.delta_z,
            self.num_atoms,
        )

        total_policy_loss = policy_loss_0
        total_value_loss = value_loss_0
        total_reward_loss = torch.tensor(0.0, device=self.device)

        # === STEPS 1..K: Recurrent Inference ===
        for k in range(K):
            # Get actions for this step
            actions_k = torch.tensor(
                [t["actions"][k] for t in batch_targets],
                dtype=torch.long,
                device=self.device,
            )

            # Dynamics + Prediction
            # Use gradient checkpointing for memory efficiency without breaking MuZero gradients
            # This allows gradients to flow back through all K unroll steps to the representation network
            def recurrent_forward(h_state, act):
                return self.model.recurrent_inference(h_state, act)

            hidden_states, reward_logits_k, policy_logits_k, value_logits_k = (
                checkpoint(
                    recurrent_forward, hidden_states, actions_k, use_reentrant=False
                )
            )

            # Prepare targets for step k+1
            target_policies_k = [t["target_policies"][k + 1] for t in batch_targets]
            policy_target_k = self._policy_targets_to_tensor(
                target_policies_k, batch_size
            )

            target_values_k = torch.tensor(
                [t["target_values"][k + 1] for t in batch_targets],
                dtype=torch.float32,
                device=self.device,
            )

            target_rewards_k = torch.tensor(
                [t["target_rewards"][k] for t in batch_targets],
                dtype=torch.float32,
                device=self.device,
            )

            # Calculate losses for step k+1

            # Reward loss (distributional)
            reward_loss_k = self._distributional_loss(
                reward_logits_k,
                target_rewards_k,
                self.reward_support,
                self.r_min,
                self.r_max,
                self.delta_r,
                self.num_reward_atoms,
            )

            # Policy loss (cross-entropy)
            log_probs_k = F.log_softmax(policy_logits_k, dim=1)
            policy_target_k = torch.nan_to_num(policy_target_k, nan=0.0)
            policy_loss_k = -torch.sum(policy_target_k * log_probs_k, dim=1).mean()

            # Value loss (distributional)
            value_loss_k = self._distributional_loss(
                value_logits_k,
                target_values_k,
                self.support,
                self.v_min,
                self.v_max,
                self.delta_z,
                self.num_atoms,
            )

            total_reward_loss += reward_loss_k
            total_policy_loss += policy_loss_k
            total_value_loss += value_loss_k

        # === TOTAL LOSS ===
        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * total_policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * total_value_loss
            + self.train_config.REWARD_LOSS_WEIGHT * total_reward_loss
        )

        # === BACKPROPAGATION WITH GRADIENT SCALING ===
        total_loss.backward()

        # Scale gradients for dynamics network
        scale_factor = self.train_config.DYNAMICS_GRADIENT_SCALE
        for param in self.model.dynamics.parameters():
            if param.grad is not None:
                param.grad.data.mul_(scale_factor)

        # Gradient clipping
        if (
            self.train_config.GRADIENT_CLIP_VALUE is not None
            and self.train_config.GRADIENT_CLIP_VALUE > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # Clear intermediate tensors to free memory
        del grid_t, other_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # === CALCULATE TD ERRORS FOR PER ===
        with torch.no_grad():
            # Use value loss as TD error proxy
            # For MuZero, we could use the initial value loss as the priority signal
            td_errors = np.abs([total_value_loss.item()] * batch_size)

        # === PREPARE LOSS INFO ===
        loss_info = {
            "total_loss": total_loss.item(),
            "policy_loss": total_policy_loss.item(),
            "value_loss": total_value_loss.item(),
            "reward_loss": total_reward_loss.item(),
            "mean_td_error": float(np.mean(td_errors)),
        }

        return loss_info, td_errors

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer."""
        try:
            if self.optimizer and self.optimizer.param_groups:
                return float(self.optimizer.param_groups[0]["lr"])
            else:
                logger.warning("Optimizer or param_groups not available.")
                return 0.0
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(f"Could not retrieve learning rate from optimizer: {e}")
            return 0.0
