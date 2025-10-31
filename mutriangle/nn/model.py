# File: mutriangle/nn/model.py
import math
from typing import cast

import torch
import torch.nn as nn

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig  # UPDATED IMPORT

# Keep mutriangle ModelConfig import
from ..config import ModelConfig


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    use_batch_norm: bool,
    activation: type[nn.Module],
) -> nn.Sequential:
    """Creates a standard convolutional block."""
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_batch_norm,
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Standard Residual Block."""

    def __init__(
        self, channels: int, use_batch_norm: bool, activation: type[nn.Module]
    ):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1, use_batch_norm, activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    """Injects sinusoidal positional encoding. (Adapted from PyTorch tutorial)"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive for PositionalEncoding")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model / 2]
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        Returns:
            Tensor with added positional encoding.
        """
        pe_buffer = self.pe
        if not isinstance(pe_buffer, torch.Tensor):
            raise TypeError("PositionalEncoding buffer 'pe' is not a Tensor.")

        if x.shape[0] > pe_buffer.shape[0]:
            raise ValueError(
                f"Input sequence length {x.shape[0]} exceeds max_len {pe_buffer.shape[0]} of PositionalEncoding"
            )
        if x.shape[2] != pe_buffer.shape[2]:
            raise ValueError(
                f"Input embedding dimension {x.shape[2]} does not match PositionalEncoding dimension {pe_buffer.shape[2]}"
            )

        x = x + pe_buffer[: x.size(0)]
        return cast("torch.Tensor", self.dropout(x))


# --- ADDED: FiLM Layer Implementation ---
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Generates per-channel scaling (gamma) and bias (beta) from a context vector.
    """

    def __init__(self, context_dim: int, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        # A simple linear layer to project the context vector to the right dimension
        # We need 2 * num_channels outputs: one for gamma, one for beta for each channel
        self.generator = nn.Linear(context_dim, num_channels * 2)

    def forward(
        self, spatial_features: torch.Tensor, context_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies FiLM modulation.
        Args:
            spatial_features: Tensor of shape (batch_size, num_channels, height, width).
            context_vector: Tensor of shape (batch_size, context_dim).
        Returns:
            Modulated spatial features.
        """
        # Generate gamma and beta
        gamma_beta = self.generator(context_vector)

        # Split the output into gamma and beta
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        # Reshape gamma and beta for broadcasting with spatial features
        # They need to be (batch_size, num_channels, 1, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        # Apply the modulation
        return gamma * spatial_features + beta


# --- END ADDED ---


# --- MuZero Network Components ---


class RepresentationNetwork(nn.Module):
    """
    Representation function: h(o) -> s
    Converts raw observation to hidden state.
    Uses the existing CNN/ResNet/Transformer architecture.
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config

        activation_cls: type[nn.Module] = getattr(nn, model_config.ACTIVATION_FUNCTION)

        # --- CNN Body ---
        conv_layers: list[nn.Module] = []
        in_channels = model_config.GRID_INPUT_CHANNELS
        for i, out_channels in enumerate(model_config.CONV_FILTERS):
            conv_layers.append(
                conv_block(
                    in_channels,
                    out_channels,
                    model_config.CONV_KERNEL_SIZES[i],
                    model_config.CONV_STRIDES[i],
                    model_config.CONV_PADDING[i],
                    model_config.USE_BATCH_NORM,
                    activation_cls,
                )
            )
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_layers)

        # --- Residual Body ---
        res_layers: list[nn.Module] = []
        if model_config.NUM_RESIDUAL_BLOCKS > 0:
            res_channels = model_config.RESIDUAL_BLOCK_FILTERS
            if in_channels != res_channels:
                res_layers.append(
                    conv_block(
                        in_channels,
                        res_channels,
                        1,
                        1,
                        0,
                        model_config.USE_BATCH_NORM,
                        activation_cls,
                    )
                )
                in_channels = res_channels
            for _ in range(model_config.NUM_RESIDUAL_BLOCKS):
                res_layers.append(
                    ResidualBlock(
                        in_channels, model_config.USE_BATCH_NORM, activation_cls
                    )
                )
        self.res_body = nn.Sequential(*res_layers)
        self.cnn_output_channels = in_channels

        # --- Transformer Body (Optional) ---
        self.transformer_body = None
        self.pos_encoder = None
        self.input_proj: nn.Module = nn.Identity()

        if model_config.USE_TRANSFORMER and model_config.TRANSFORMER_LAYERS > 0:
            transformer_input_dim = model_config.TRANSFORMER_DIM
            if self.cnn_output_channels != transformer_input_dim:
                self.input_proj = nn.Conv2d(
                    self.cnn_output_channels, transformer_input_dim, kernel_size=1
                )
            else:
                self.input_proj = nn.Identity()

            self.pos_encoder = PositionalEncoding(transformer_input_dim, dropout=0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_input_dim,
                nhead=model_config.TRANSFORMER_HEADS,
                dim_feedforward=model_config.TRANSFORMER_FC_DIM,
                activation=model_config.ACTIVATION_FUNCTION.lower(),
                batch_first=False,
                norm_first=True,
            )
            transformer_norm = nn.LayerNorm(transformer_input_dim)
            self.transformer_body = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config.TRANSFORMER_LAYERS,
                norm=transformer_norm,
            )

        # --- FiLM Fusion (Optional) ---
        self.film_layer = None
        if model_config.USE_FILM_FUSION:
            spatial_channels = (
                model_config.TRANSFORMER_DIM
                if model_config.USE_TRANSFORMER and model_config.TRANSFORMER_LAYERS > 0
                else self.cnn_output_channels
            )
            self.film_layer = FiLMLayer(
                context_dim=model_config.OTHER_NN_INPUT_FEATURES_DIM,
                num_channels=spatial_channels,
            )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input_grid = torch.zeros(
                1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
            )
            spatial_out = self._forward_spatial(dummy_input_grid)
            self.flattened_spatial_size = spatial_out.numel()

        # --- Output layers to hidden state ---
        if model_config.USE_FILM_FUSION:
            combined_input_size = self.flattened_spatial_size
        else:
            combined_input_size = (
                self.flattened_spatial_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )

        # MLP to project to hidden state
        repr_layers: list[nn.Module] = []
        in_features = combined_input_size
        for hidden_dim in model_config.REPRESENTATION_HIDDEN_DIMS:
            repr_layers.append(nn.Linear(in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                repr_layers.append(nn.BatchNorm1d(hidden_dim))
            repr_layers.append(activation_cls())
            in_features = hidden_dim
        repr_layers.append(nn.Linear(in_features, model_config.HIDDEN_STATE_DIM))
        self.repr_mlp = nn.Sequential(*repr_layers)

    def _forward_spatial(self, grid_state: torch.Tensor) -> torch.Tensor:
        """Process spatial features through CNN/ResNet/Transformer."""
        conv_out = self.conv_body(grid_state)
        res_out = self.res_body(conv_out)

        if (
            self.model_config.USE_TRANSFORMER
            and self.transformer_body is not None
            and self.pos_encoder is not None
        ):
            proj_out = self.input_proj(res_out)
            b, d, h, w = proj_out.shape
            transformer_input = proj_out.flatten(2).permute(2, 0, 1)
            transformer_input = self.pos_encoder(transformer_input)
            transformer_output = self.transformer_body(transformer_input)
            return transformer_output.permute(1, 2, 0).view(b, d, h, w)
        else:
            return res_out

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert observation to hidden state.
        Args:
            grid_state: (batch, C, H, W)
            other_features: (batch, feature_dim)
        Returns:
            hidden_state: (batch, hidden_state_dim)
        """
        spatial_features = self._forward_spatial(grid_state)

        # Apply FiLM or concatenation
        if self.model_config.USE_FILM_FUSION and self.film_layer is not None:
            modulated_features = self.film_layer(spatial_features, other_features)
            flattened_features = modulated_features.reshape(
                modulated_features.size(0), -1
            )
            combined_features = flattened_features
        else:
            flattened_spatial = spatial_features.reshape(spatial_features.size(0), -1)
            combined_features = torch.cat([flattened_spatial, other_features], dim=1)

        hidden_state = self.repr_mlp(combined_features)
        return hidden_state


class DynamicsNetwork(nn.Module):
    """
    Dynamics function: g(s, a) -> (s', r)
    Predicts next hidden state and reward from current state and action.
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(
            env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS
        )

        activation_cls: type[nn.Module] = getattr(nn, model_config.ACTIVATION_FUNCTION)

        # Input: hidden_state + one-hot action
        input_dim = model_config.HIDDEN_STATE_DIM + self.action_dim

        # MLP for dynamics
        dynamics_layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim in model_config.DYNAMICS_HIDDEN_DIMS:
            dynamics_layers.append(nn.Linear(in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                dynamics_layers.append(nn.BatchNorm1d(hidden_dim))
            dynamics_layers.append(activation_cls())
            in_features = hidden_dim
        self.dynamics_mlp = nn.Sequential(*dynamics_layers)

        # Next hidden state head
        self.next_state_head = nn.Linear(in_features, model_config.HIDDEN_STATE_DIM)

        # Reward prediction head (distributional)
        reward_head_layers: list[nn.Module] = []
        reward_in_features = in_features
        reward_head_layers.append(nn.Linear(reward_in_features, 128))
        if model_config.USE_BATCH_NORM:
            reward_head_layers.append(nn.BatchNorm1d(128))
        reward_head_layers.append(activation_cls())
        reward_head_layers.append(nn.Linear(128, model_config.NUM_REWARD_ATOMS))
        self.reward_head = nn.Sequential(*reward_head_layers)

        # Reward support atoms
        self.num_reward_atoms = model_config.NUM_REWARD_ATOMS
        self.reward_min = model_config.REWARD_MIN
        self.reward_max = model_config.REWARD_MAX

    def forward(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward.
        Args:
            hidden_state: (batch, hidden_state_dim)
            action: (batch,) integer action indices
        Returns:
            next_hidden_state: (batch, hidden_state_dim)
            reward_logits: (batch, num_reward_atoms)
        """
        # One-hot encode action
        action_one_hot = torch.nn.functional.one_hot(
            action.long(), num_classes=self.action_dim
        ).float()

        # Concatenate state and action
        state_action = torch.cat([hidden_state, action_one_hot], dim=1)

        # Process through dynamics MLP
        dynamics_features = self.dynamics_mlp(state_action)

        # Predict next state and reward
        next_hidden_state = self.next_state_head(dynamics_features)
        reward_logits = self.reward_head(dynamics_features)

        return next_hidden_state, reward_logits


class PredictionNetwork(nn.Module):
    """
    Prediction function: f(s) -> (p, v)
    Predicts policy and value from hidden state.
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(
            env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS
        )

        activation_cls: type[nn.Module] = getattr(nn, model_config.ACTIVATION_FUNCTION)

        # Shared MLP
        pred_layers: list[nn.Module] = []
        in_features = model_config.HIDDEN_STATE_DIM
        for hidden_dim in model_config.PREDICTION_HIDDEN_DIMS:
            pred_layers.append(nn.Linear(in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                pred_layers.append(nn.BatchNorm1d(hidden_dim))
            pred_layers.append(activation_cls())
            in_features = hidden_dim
        self.prediction_mlp = nn.Sequential(*pred_layers)

        # Policy head
        policy_head_layers: list[nn.Module] = []
        policy_in_features = in_features
        for hidden_dim in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                policy_head_layers.append(nn.BatchNorm1d(hidden_dim))
            policy_head_layers.append(activation_cls())
            policy_in_features = hidden_dim
        policy_head_layers.append(nn.Linear(policy_in_features, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)

        # Value head (distributional)
        value_head_layers: list[nn.Module] = []
        value_in_features = in_features
        for hidden_dim in model_config.VALUE_HEAD_DIMS:
            value_head_layers.append(nn.Linear(value_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                value_head_layers.append(nn.BatchNorm1d(hidden_dim))
            value_head_layers.append(activation_cls())
            value_in_features = hidden_dim
        value_head_layers.append(
            nn.Linear(value_in_features, model_config.NUM_VALUE_ATOMS)
        )
        self.value_head = nn.Sequential(*value_head_layers)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value.
        Args:
            hidden_state: (batch, hidden_state_dim)
        Returns:
            policy_logits: (batch, action_dim)
            value_logits: (batch, num_value_atoms)
        """
        prediction_features = self.prediction_mlp(hidden_state)
        policy_logits = self.policy_head(prediction_features)
        value_logits = self.value_head(prediction_features)
        return policy_logits, value_logits


class MuTriangleNet(nn.Module):
    """
    Neural Network architecture for MuTriangle (MuZero).
    Includes optional Transformer Encoder block after CNN body.
    Supports Distributional Value Head (C51).
    """

    def __init__(
        self, model_config: ModelConfig, env_config: EnvConfig
    ):  # Uses trianglengin.EnvConfig
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        # Calculate action_dim manually
        self.action_dim = int(
            env_config.NUM_SHAPE_SLOTS * env_config.ROWS * env_config.COLS
        )

        # Initialize the three MuZero sub-networks
        self.representation = RepresentationNetwork(model_config, env_config)
        self.dynamics = DynamicsNetwork(model_config, env_config)
        self.prediction = PredictionNetwork(model_config, env_config)

    def initial_inference(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial step: convert observation to hidden state and predict policy/value.

        Args:
            grid_state: (batch, C, H, W)
            other_features: (batch, feature_dim)
        Returns:
            hidden_state: (batch, hidden_state_dim)
            policy_logits: (batch, action_dim)
            value_logits: (batch, num_value_atoms)
        """
        hidden_state = self.representation(grid_state, other_features)
        policy_logits, value_logits = self.prediction(hidden_state)
        return hidden_state, policy_logits, value_logits

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recurrent step: predict next state, reward, policy, value from hidden state and action.

        Args:
            hidden_state: (batch, hidden_state_dim)
            action: (batch,) integer action indices
        Returns:
            next_hidden_state: (batch, hidden_state_dim)
            reward_logits: (batch, num_reward_atoms)
            policy_logits: (batch, action_dim)
            value_logits: (batch, num_value_atoms)
        """
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)
        return next_hidden_state, reward_logits, policy_logits, value_logits

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy forward pass for backwards compatibility with MCTS.
        Uses initial_inference and returns policy/value.

        Returns: (policy_logits, value_logits)
        """
        _, policy_logits, value_logits = self.initial_inference(
            grid_state, other_features
        )
        return policy_logits, value_logits
