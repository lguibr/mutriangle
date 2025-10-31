# File: mutriangle/config_mgmt/cli_choices.py
"""
Defines the predefined choice lists used for interactive configuration
in the MuTriangle CLI. Separated for better maintainability.
Choices are limited to ~8 options for better usability.
"""

from typing import Any

# Define extensive choice lists (examples, expand as needed)
# Reduced to ~8 options per field
NUMERIC_CHOICES: dict[str, list[Any]] = {
    # ModelConfig
    "NUM_RESIDUAL_BLOCKS": [0, 1, 2, 4, 6, 8, 12, 16],
    "RESIDUAL_BLOCK_FILTERS": [32, 64, 96, 128, 192, 256, 384, 512],
    "TRANSFORMER_DIM": [32, 64, 96, 128, 192, 256, 384, 512],
    "TRANSFORMER_HEADS": [1, 2, 4, 8, 12, 16],  # Kept 6, common values
    "TRANSFORMER_LAYERS": [0, 1, 2, 3, 4, 6, 8, 12],
    "TRANSFORMER_FC_DIM": [64, 128, 256, 512, 768, 1024, 1536, 2048],
    "NUM_VALUE_ATOMS": [11, 21, 31, 51, 101, 201, 301, 401],
    "VALUE_MIN": [-50.0, -25.0, -20.0, -15.0, -10.0, -5.0, -1.0, 0.0],
    "VALUE_MAX": [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0],
    # TrainConfig
    "MAX_TRAINING_STEPS": [
        None,
        10_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
    ],
    # Add -1 sentinel for "auto"
    "NUM_SELF_PLAY_WORKERS": [-1, 1, 2, 4, 6, 8, 12, 16],
    "BATCH_SIZE": [32, 64, 96, 128, 192, 256, 384, 512],
    "BUFFER_CAPACITY": [
        10_000,
        25_000,
        50_000,
        100_000,
        200_000,
        250_000,
        500_000,
        1_000_000,
    ],
    "MIN_BUFFER_SIZE_TO_TRAIN": [
        100,
        500,
        1_000,
        5_000,
        10_000,
        20_000,
        25_000,
        50_000,
    ],
    "WORKER_UPDATE_FREQ_STEPS": [1, 5, 10, 20, 25, 50, 100, 200],
    "N_STEP_RETURNS": [1, 2, 3, 5, 7, 10, 15, 20],
    "GAMMA": [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0],
    "LEARNING_RATE": [1e-2, 5e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 5e-5],
    "WEIGHT_DECAY": [0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],  # Reduced to 7
    "GRADIENT_CLIP_VALUE": [None, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0],
    "LR_SCHEDULER_ETA_MIN": [0.0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5],  # Kept 6
    "POLICY_LOSS_WEIGHT": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "VALUE_LOSS_WEIGHT": [0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5],
    "ENTROPY_BONUS_WEIGHT": [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],  # Kept 7
    "CHECKPOINT_SAVE_FREQ_STEPS": [
        250,
        500,
        1000,
        2000,
        2500,
        5000,
        10000,
        20000,
    ],
    "PER_ALPHA": [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "PER_BETA_INITIAL": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
    "PER_BETA_FINAL": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Kept 7
    "PER_EPSILON": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],  # Kept 5
    # MuTriangleMCTSConfig
    "max_simulations": [16, 32, 50, 64, 100, 128, 200, 400],
    "max_depth": [3, 4, 5, 6, 8, 10, 15, 20],
    "cpuct": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    "dirichlet_alpha": [0.0, 0.03, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
    "dirichlet_epsilon": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
    "discount": [0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0],  # Kept 7
    "mcts_batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
    # StatsConfig
    "LOG_INTERVAL_STEPS": [1, 5, 10, 20, 25, 50, 100, 200],
    "PRINT_INTERVAL_SECONDS": [5, 10, 15, 20, 30, 60, 90, 120],
    "AGGREGATION_WINDOW_STEPS": [10, 20, 50, 100, 200, 500],  # Kept 6
    # PersistenceConfig
    "BUFFER_SAVE_FREQ_STEPS": [0, 500, 1000, 2500, 5000, 10000, 20000, 50000],
    "KEEP_N_CHECKPOINTS": [0, 1, 2, 3, 5, 10, 20],  # Kept 7
}

LIST_CHOICES: dict[str, list[Any]] = {
    # ModelConfig
    "CONV_FILTERS": [
        [16, 32],
        [32, 64],
        [64, 128],
        [16, 32, 64],
        [32, 64, 128],
        [64, 128, 256],
        [128, 256, 512],
    ],  # Reduced to 7
    "CONV_KERNEL_SIZES": [[3], [3, 3], [3, 3, 3], [5, 3, 3], [3, 3, 3, 3]],  # Kept 5
    "CONV_STRIDES": [[1], [1, 1], [1, 1, 1], [2, 1, 1], [1, 1, 1, 1]],  # Kept 5
    "CONV_PADDING": [[1], [1, 1], [1, 1, 1], ["same"], [1, 1, 1, 1]],  # Kept 5
    "FC_DIMS_SHARED": [
        [32],
        [64],
        [128],
        [256],
        [512],
        [128, 64],
        [256, 128],
        [512, 256],
    ],
    "POLICY_HEAD_DIMS": [[32], [64], [128], [256], [128, 64]],  # Kept 5
    "VALUE_HEAD_DIMS": [[32], [64], [128], [256], [128, 64]],  # Kept 5
}

LITERAL_CHOICES: dict[str, list[Any]] = {
    # ModelConfig
    "ACTIVATION_FUNCTION": ["ReLU", "GELU", "SiLU", "Tanh"],
    # TrainConfig
    "OPTIMIZER_TYPE": ["AdamW", "Adam", "SGD"],
    "LR_SCHEDULER_TYPE": [None, "CosineAnnealingLR", "StepLR"],
}

BOOL_CHOICES: dict[str, list[Any]] = {
    # ModelConfig
    "USE_TRANSFORMER": [True, False],
    "USE_BATCH_NORM": [True, False],
    # TrainConfig
    "AUTO_RESUME_LATEST": [True, False],
    "USE_PER": [True, False],
    "COMPILE_MODEL": [True, False],
    "PROFILE_WORKERS": [False, True],
    # PersistenceConfig
    "SAVE_BUFFER": [True, False],
    "SAVE_CHECKPOINTS": [True, False],
}
