# File: mutriangle/config/model_config.py
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the Neural Network model (Pydantic model).
    --- TUNED FOR SMALLER CAPACITY (~3M Params Target, Laptop Feasible) ---
    NOTE: Increasing filters/layers/dims can improve final agent strength
          but will increase training time due to the NN evaluation bottleneck.
          Adjust based on hardware capabilities and performance requirements.
    """

    # Input channels for the grid (e.g., 1 for occupancy, more for history/colors)
    GRID_INPUT_CHANNELS: int = Field(
        default=1, gt=0, description="Number of input channels for the grid tensor."
    )

    # --- CNN Architecture Parameters ---
    CONV_FILTERS: list[int] = Field(
        default=[32, 64, 128], description="List of output filters for each Conv layer."
    )
    CONV_KERNEL_SIZES: list[int | tuple[int, int]] = Field(
        default=[3, 3, 3], description="Kernel size for each Conv layer."
    )
    CONV_STRIDES: list[int | tuple[int, int]] = Field(
        default=[1, 1, 1], description="Stride for each Conv layer."
    )
    CONV_PADDING: list[int | tuple[int, int] | str] = Field(
        default=[1, 1, 1], description="Padding for each Conv layer."
    )

    # --- Residual Block Parameters ---
    NUM_RESIDUAL_BLOCKS: int = Field(
        default=2, ge=0, description="Number of residual blocks after CNN body."
    )
    RESIDUAL_BLOCK_FILTERS: int = Field(
        default=128, gt=0, description="Number of filters within residual blocks."
    )

    # --- Transformer Parameters (Optional) ---
    USE_TRANSFORMER: bool = Field(
        default=True, description="Enable Transformer Encoder block after CNN/ResNet."
    )
    TRANSFORMER_DIM: int = Field(
        default=128, gt=0, description="Embedding dimension for the Transformer."
    )
    TRANSFORMER_HEADS: int = Field(
        default=4, gt=0, description="Number of attention heads in the Transformer."
    )
    TRANSFORMER_LAYERS: int = Field(
        default=2, ge=0, description="Number of layers in the Transformer Encoder."
    )
    TRANSFORMER_FC_DIM: int = Field(
        default=256,
        gt=0,
        description="Dimension of the feedforward layer in Transformer.",
    )

    # --- Feature Fusion ---
    USE_FILM_FUSION: bool = Field(
        default=True,
        description="Use FiLM layer for feature fusion instead of concatenation.",
    )

    # --- Fully Connected Layers ---
    FC_DIMS_SHARED: list[int] = Field(
        default=[128], description="Dimensions of shared fully connected layers."
    )

    # --- Policy Head ---
    POLICY_HEAD_DIMS: list[int] = Field(
        default=[128], description="Dimensions of hidden layers in the policy head."
    )

    # --- Distributional Value Head Parameters ---
    NUM_VALUE_ATOMS: int = Field(
        default=51,
        gt=1,
        description="Number of atoms for the distributional value head (C51).",
    )
    VALUE_MIN: float = Field(
        default=-10.0, description="Minimum value for the distributional value support."
    )
    VALUE_MAX: float = Field(
        default=10.0, description="Maximum value for the distributional value support."
    )

    # --- Value Head Dims ---
    VALUE_HEAD_DIMS: list[int] = Field(
        default=[128], description="Dimensions of hidden layers in the value head."
    )

    # --- MuZero Network Architecture Parameters ---
    HIDDEN_STATE_DIM: int = Field(
        default=128,
        gt=0,
        description="Dimension of the latent hidden state in MuZero networks",
    )
    REPRESENTATION_HIDDEN_DIMS: list[int] = Field(
        default=[256], description="Hidden layer dimensions in representation network"
    )
    DYNAMICS_HIDDEN_DIMS: list[int] = Field(
        default=[256], description="Hidden layer dimensions in dynamics network"
    )
    PREDICTION_HIDDEN_DIMS: list[int] = Field(
        default=[256], description="Hidden layer dimensions in prediction network"
    )
    NUM_REWARD_ATOMS: int = Field(
        default=51,
        gt=1,
        description="Number of atoms for distributional reward prediction",
    )
    REWARD_MIN: float = Field(
        default=-10.0, description="Minimum value for distributional reward support"
    )
    REWARD_MAX: float = Field(
        default=10.0, description="Maximum value for distributional reward support"
    )

    # --- Other Hyperparameters ---
    ACTIVATION_FUNCTION: Literal["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"] = Field(
        default="ReLU", description="Activation function for hidden layers."
    )
    USE_BATCH_NORM: bool = Field(
        default=True, description="Enable Batch Normalization in layers."
    )

    # --- Input Feature Dimension ---
    # Dimension of the non-grid feature vector concatenated after CNN/Transformer.
    # Must match the output of features.extractor.get_combined_other_features.
    OTHER_NN_INPUT_FEATURES_DIM: int = Field(
        default=30,
        gt=0,
        description="Dimension of the 'other_features' input vector (determined by feature extractor).",
    )

    @model_validator(mode="after")
    def check_conv_layers_consistency(self) -> "ModelConfig":
        if (
            hasattr(self, "CONV_FILTERS")
            and hasattr(self, "CONV_KERNEL_SIZES")
            and hasattr(self, "CONV_STRIDES")
            and hasattr(self, "CONV_PADDING")
        ):
            n_filters = len(self.CONV_FILTERS)
            if not (
                len(self.CONV_KERNEL_SIZES) == n_filters
                and len(self.CONV_STRIDES) == n_filters
                and len(self.CONV_PADDING) == n_filters
            ):
                raise ValueError(
                    "Lengths of CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES, and CONV_PADDING must match."
                )
        return self

    @model_validator(mode="after")
    def check_residual_filter_match(self) -> "ModelConfig":
        if (
            hasattr(self, "NUM_RESIDUAL_BLOCKS")
            and self.NUM_RESIDUAL_BLOCKS > 0
            and hasattr(self, "CONV_FILTERS")
            and self.CONV_FILTERS
            and hasattr(self, "RESIDUAL_BLOCK_FILTERS")
            and self.CONV_FILTERS[-1] != self.RESIDUAL_BLOCK_FILTERS
        ):
            pass  # Model handles projection if needed
        return self

    @model_validator(mode="after")
    def check_transformer_config(self) -> "ModelConfig":
        if hasattr(self, "USE_TRANSFORMER") and self.USE_TRANSFORMER:
            if not hasattr(self, "TRANSFORMER_LAYERS") or self.TRANSFORMER_LAYERS < 0:
                raise ValueError("TRANSFORMER_LAYERS cannot be negative.")
            if self.TRANSFORMER_LAYERS > 0:
                if not hasattr(self, "TRANSFORMER_DIM") or self.TRANSFORMER_DIM <= 0:
                    raise ValueError(
                        "TRANSFORMER_DIM must be positive if TRANSFORMER_LAYERS > 0."
                    )
                if (
                    not hasattr(self, "TRANSFORMER_HEADS")
                    or self.TRANSFORMER_HEADS <= 0
                ):
                    raise ValueError(
                        "TRANSFORMER_HEADS must be positive if TRANSFORMER_LAYERS > 0."
                    )
                if self.TRANSFORMER_DIM % self.TRANSFORMER_HEADS != 0:
                    raise ValueError(
                        f"TRANSFORMER_DIM ({self.TRANSFORMER_DIM}) must be divisible by TRANSFORMER_HEADS ({self.TRANSFORMER_HEADS})."
                    )
                if (
                    not hasattr(self, "TRANSFORMER_FC_DIM")
                    or self.TRANSFORMER_FC_DIM <= 0
                ):
                    raise ValueError(
                        "TRANSFORMER_FC_DIM must be positive if TRANSFORMER_LAYERS > 0."
                    )
        return self

    @model_validator(mode="after")
    def check_transformer_dim_consistency(self) -> "ModelConfig":
        if (
            hasattr(self, "USE_TRANSFORMER")
            and self.USE_TRANSFORMER
            and hasattr(self, "TRANSFORMER_LAYERS")
            and self.TRANSFORMER_LAYERS > 0
            and hasattr(self, "CONV_FILTERS")
            and self.CONV_FILTERS
            and hasattr(self, "TRANSFORMER_DIM")
        ):
            cnn_output_channels = (
                self.RESIDUAL_BLOCK_FILTERS
                if hasattr(self, "NUM_RESIDUAL_BLOCKS") and self.NUM_RESIDUAL_BLOCKS > 0
                else self.CONV_FILTERS[-1]
            )
            if cnn_output_channels != self.TRANSFORMER_DIM:
                pass  # Model handles projection
        return self

    @model_validator(mode="after")
    def check_value_distribution_params(self) -> "ModelConfig":
        if (
            hasattr(self, "VALUE_MIN")
            and hasattr(self, "VALUE_MAX")
            and self.VALUE_MIN >= self.VALUE_MAX
        ):
            raise ValueError("VALUE_MIN must be strictly less than VALUE_MAX.")
        return self

    @model_validator(mode="after")
    def check_reward_distribution_params(self) -> "ModelConfig":
        if (
            hasattr(self, "REWARD_MIN")
            and hasattr(self, "REWARD_MAX")
            and self.REWARD_MIN >= self.REWARD_MAX
        ):
            raise ValueError("REWARD_MIN must be strictly less than REWARD_MAX.")
        return self


ModelConfig.model_rebuild(force=True)
