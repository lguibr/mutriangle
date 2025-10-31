# File: mutriangle/config/train_config.py
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Get logger instance
logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    """
    Configuration for the training process (Pydantic model).
    --- TUNED FOR MORE SUBSTANTIAL LEARNING RUNS ---
    """

    RUN_NAME: str = Field(
        default_factory=lambda: f"train_{time.strftime('%Y%m%d_%H%M%S')}",
        description="Default name for the training run (can be overridden).",
    )
    LOAD_CHECKPOINT_PATH: str | None = Field(
        default=None, description="Path to a specific checkpoint file to load."
    )
    LOAD_BUFFER_PATH: str | None = Field(
        default=None, description="Path to a specific buffer file to load."
    )
    AUTO_RESUME_LATEST: bool = Field(
        default=True,
        description="Automatically resume from the latest checkpoint if found.",
    )
    # --- DEVICE: Defaults to 'auto' for automatic detection (CUDA > MPS > CPU) ---
    # This controls the device for the main Trainer process.
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        default="auto",
        description="Device for the main Trainer (NN updates).",  # Default is 'auto'
    )
    RANDOM_SEED: int = Field(
        default=42, description="Seed for random number generators."
    )

    # --- Training Loop ---
    MAX_TRAINING_STEPS: int | None = Field(
        default=100_000,
        # ge=1 constraint removed from MAX_TRAINING_STEPS as None is allowed
        description="Maximum number of training steps (optimizer steps). None for infinite.",
    )

    # --- Workers & Batching ---
    NUM_SELF_PLAY_WORKERS: int = Field(
        default=-1,  # Default to auto-detect sentinel
        # ge=1 constraint removed to allow -1 for auto
        description="Number of self-play workers. Set to -1 to auto-detect based on available CPU cores (recommended).",
    )
    # --- WORKER_DEVICE: Defaults to 'cpu' for self-play workers ---
    # Workers run MCTS and NN eval; CPU is often sufficient and avoids GPU contention.
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        default="cpu",
        description="Device for self-play workers (MCTS, NN eval).",  # Default is 'cpu'
    )
    BATCH_SIZE: int = Field(
        default=256, ge=1, description="Batch size for neural network training steps."
    )
    BUFFER_CAPACITY: int = Field(
        default=250_000,
        ge=1,
        description="Maximum number of experiences in the replay buffer.",
    )
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(
        default=25_000,
        ge=1,
        description="Minimum experiences needed in buffer before training starts.",
    )
    WORKER_UPDATE_FREQ_STEPS: int = Field(
        default=10,
        ge=1,
        description="Update worker networks every N training steps.",
    )

    # --- N-Step Returns ---
    N_STEP_RETURNS: int = Field(
        default=5, ge=1, description="Number of steps for N-step return calculation."
    )
    GAMMA: float = Field(
        default=0.99, gt=0, le=1.0, description="Discount factor for future rewards."
    )

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field(
        default="AdamW", description="Optimizer algorithm."
    )
    LEARNING_RATE: float = Field(
        default=2e-4, gt=0, description="Initial learning rate for the base model."
    )
    # --- ADDED: Two-Headed Learning Rates ---
    POLICY_LEARNING_RATE: float | None = Field(
        default=None,
        gt=0,
        description="Specific learning rate for the policy head. Defaults to LEARNING_RATE if None.",
    )
    VALUE_LEARNING_RATE: float | None = Field(
        default=None,
        gt=0,
        description="Specific learning rate for the value head. Defaults to LEARNING_RATE if None.",
    )
    # --- END ADDED ---
    WEIGHT_DECAY: float = Field(
        default=1e-4, ge=0, description="Weight decay (L2 penalty) for the optimizer."
    )
    GRADIENT_CLIP_VALUE: float | None = Field(
        default=1.0, description="Maximum norm for gradient clipping (None to disable)."
    )

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Literal["StepLR", "CosineAnnealingLR"] | None = Field(
        default="CosineAnnealingLR",
        description="Type of learning rate scheduler (or None).",
    )
    # T_MAX will be set automatically based on new MAX_TRAINING_STEPS
    LR_SCHEDULER_T_MAX: int | None = Field(
        default=None,
        description="T_max for CosineAnnealingLR (usually MAX_TRAINING_STEPS).",
    )
    LR_SCHEDULER_ETA_MIN: float = Field(
        default=1e-6, ge=0, description="Minimum learning rate for CosineAnnealingLR."
    )

    # --- Loss Weights ---
    POLICY_LOSS_WEIGHT: float = Field(
        default=1.0, ge=0, description="Weight for the policy loss component."
    )
    VALUE_LOSS_WEIGHT: float = Field(
        default=1.0, ge=0, description="Weight for the value loss component."
    )
    REWARD_LOSS_WEIGHT: float = Field(
        default=1.0, ge=0, description="Weight for the reward loss component (MuZero)."
    )
    ENTROPY_BONUS_WEIGHT: float = Field(
        default=0.001,
        ge=0,
        description="Weight for the policy entropy bonus (regularization).",
    )

    # --- MuZero Training Parameters ---
    UNROLL_STEPS: int = Field(
        default=5, ge=1, description="Number of steps to unroll in MuZero training."
    )
    DYNAMICS_GRADIENT_SCALE: float = Field(
        default=0.5,
        gt=0,
        le=1.0,
        description="Gradient scaling factor for dynamics network to stabilize training.",
    )

    # --- Checkpointing ---
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(
        default=2500, ge=1, description="Save checkpoint every N training steps."
    )

    # --- Prioritized Experience Replay (PER) ---
    USE_PER: bool = Field(
        default=True, description="Enable Prioritized Experience Replay."
    )
    PER_ALPHA: float = Field(
        default=0.6, ge=0, description="PER alpha (priority exponent)."
    )
    PER_BETA_INITIAL: float = Field(
        default=0.4,
        ge=0,
        le=1.0,
        description="PER beta initial value (importance sampling exponent).",
    )
    PER_BETA_FINAL: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="PER beta final value (importance sampling exponent).",
    )
    # Anneal steps will be set automatically based on MAX_TRAINING_STEPS
    PER_BETA_ANNEAL_STEPS: int | None = Field(
        default=None,
        description="Steps to anneal PER beta (usually MAX_TRAINING_STEPS).",
    )
    PER_EPSILON: float = Field(
        default=1e-5, gt=0, description="PER epsilon (small value added to priorities)."
    )

    # --- Model Compilation ---
    COMPILE_MODEL: bool = Field(
        default=True,
        description=(
            "Enable torch.compile() for potential speedup (Trainer only). Requires PyTorch 2.0+. "
            "May have initial overhead or compatibility issues on some setups/GPUs "
            "(especially non-CUDA backends like MPS). Set to False if encountering problems. "
            "The application will attempt compilation and fall back gracefully if it fails."
        ),
    )

    # --- Profiling ---
    PROFILE_WORKERS: bool = Field(
        default=False,
        description="Enable cProfile for worker 0 to generate .prof files.",
    )

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        if (
            hasattr(self, "MIN_BUFFER_SIZE_TO_TRAIN")
            and hasattr(self, "BUFFER_CAPACITY")
            and self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY
        ):
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        if (
            hasattr(self, "BATCH_SIZE")
            and hasattr(self, "BUFFER_CAPACITY")
            and self.BATCH_SIZE > self.BUFFER_CAPACITY
        ):
            raise ValueError("BATCH_SIZE cannot be greater than BUFFER_CAPACITY.")
        if (
            hasattr(self, "BATCH_SIZE")
            and hasattr(self, "MIN_BUFFER_SIZE_TO_TRAIN")
            and self.BATCH_SIZE > self.MIN_BUFFER_SIZE_TO_TRAIN
        ):
            pass  # Allow batch size > min buffer size, though sampling might fail initially
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        if (
            hasattr(self, "LR_SCHEDULER_TYPE")
            and self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and hasattr(self, "LR_SCHEDULER_T_MAX")
            and self.LR_SCHEDULER_T_MAX is None
        ):
            if (
                hasattr(self, "MAX_TRAINING_STEPS")
                and self.MAX_TRAINING_STEPS is not None
                and self.MAX_TRAINING_STEPS >= 1  # Check if MAX_TRAINING_STEPS is valid
            ):
                self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
                logger.info(
                    f"Set LR_SCHEDULER_T_MAX to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                )
            else:
                # Fallback if MAX_TRAINING_STEPS is None or invalid
                default_t_max = 100_000
                self.LR_SCHEDULER_T_MAX = default_t_max
                logger.warning(
                    f"MAX_TRAINING_STEPS is None or invalid ({self.MAX_TRAINING_STEPS}). Setting LR_SCHEDULER_T_MAX to default {default_t_max}"
                )

        if (
            hasattr(self, "LR_SCHEDULER_T_MAX")
            and self.LR_SCHEDULER_T_MAX is not None
            and self.LR_SCHEDULER_T_MAX <= 0
        ):
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        if (
            hasattr(self, "USE_PER")
            and self.USE_PER
            and hasattr(self, "PER_BETA_ANNEAL_STEPS")
            and self.PER_BETA_ANNEAL_STEPS is None
        ):
            if (
                hasattr(self, "MAX_TRAINING_STEPS")
                and self.MAX_TRAINING_STEPS is not None
                and self.MAX_TRAINING_STEPS >= 1  # Check if MAX_TRAINING_STEPS is valid
            ):
                self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
                logger.info(
                    f"Set PER_BETA_ANNEAL_STEPS to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                )
            else:
                # Fallback if MAX_TRAINING_STEPS is None or invalid
                default_anneal = 100_000
                self.PER_BETA_ANNEAL_STEPS = default_anneal
                logger.warning(
                    f"MAX_TRAINING_STEPS is None or invalid ({self.MAX_TRAINING_STEPS}). Setting PER_BETA_ANNEAL_STEPS to default {default_anneal}"
                )

        if (
            hasattr(self, "PER_BETA_ANNEAL_STEPS")
            and self.PER_BETA_ANNEAL_STEPS is not None
            and self.PER_BETA_ANNEAL_STEPS <= 0
        ):
            raise ValueError("PER_BETA_ANNEAL_STEPS must be positive if set.")
        return self

    @field_validator("GRADIENT_CLIP_VALUE")
    @classmethod
    def check_gradient_clip(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v

    @field_validator("PER_BETA_FINAL")
    @classmethod
    def check_per_beta_final(cls, v: float, info) -> float:
        # Use info.data for Pydantic v2 compatibility
        initial_beta = info.data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v

    @field_validator("MAX_TRAINING_STEPS")
    @classmethod
    def check_max_training_steps(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("MAX_TRAINING_STEPS must be None or >= 1.")
        return v


# Ensure model is rebuilt after changes
TrainConfig.model_rebuild(force=True)
