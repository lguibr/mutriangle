# File: mutriangle/presets/definitions.py
# File: mutriangle/presets/definitions.py
"""
Defines preset configurations as Python objects.
Each function returns a fully instantiated SavedConfigSet.
"""

import datetime

from trianglengin import EnvConfig
from trieye import (
    DEFAULT_METRICS,
    PersistenceConfig,
    StatsConfig,
    TrieyeConfig,
)

from mutriangle.config import (
    APP_NAME,
    MuTriangleMCTSConfig,
    ModelConfig,
    TrainConfig,
)
from mutriangle.config.app_config import DATA_ROOT_DIR_NAME
from mutriangle.config_mgmt.schemas import Metadata, SavedConfigSet

# --- Common Env Config ---
# Corrected: Removed invalid/outdated arguments
DEFAULT_ENV_CONFIG = EnvConfig(
    ROWS=8,
    COLS=15,
    PLAYABLE_RANGE_PER_ROW=[
        (3, 12),  # Changed to tuple
        (2, 13),  # Changed to tuple
        (1, 14),  # Changed to tuple
        (0, 15),  # Changed to tuple
        (0, 15),  # Changed to tuple
        (1, 14),  # Changed to tuple
        (2, 13),  # Changed to tuple
        (3, 12),  # Changed to tuple
    ],
    NUM_SHAPE_SLOTS=3,
    REWARD_PER_CLEARED_TRIANGLE=1.0,
    REWARD_PER_PLACED_TRIANGLE=0.01,
    REWARD_PER_STEP_ALIVE=-0.001,
    PENALTY_GAME_OVER=-5.0,
)


def _create_metadata(name: str, description: str) -> Metadata:
    """Helper to create metadata."""
    now = datetime.datetime.now()
    return Metadata(
        name=name,
        description=description,
        created_at=now,
        last_modified_at=now,
        platform="preset_definition",  # Indicate origin
    )


def get_toy_config() -> SavedConfigSet:
    """Returns the 'toy' configuration set - minimal for testing."""
    name = "toy"
    run_name = f"{name}_run"
    max_steps = 100  # Increased training steps
    train_config = TrainConfig(
        RUN_NAME=run_name,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="auto",
        RANDOM_SEED=42,
        MAX_TRAINING_STEPS=max_steps,  # Use updated value
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        BATCH_SIZE=4,  # Keep batch size small for speed
        BUFFER_CAPACITY=50,  # Increased slightly
        MIN_BUFFER_SIZE_TO_TRAIN=10,  # Keep low for fast start
        WORKER_UPDATE_FREQ_STEPS=10,  # Update more often
        N_STEP_RETURNS=2,  # Shortened N-step
        GAMMA=0.99,
        OPTIMIZER_TYPE="AdamW",
        LEARNING_RATE=0.0005,
        WEIGHT_DECAY=0.0001,
        GRADIENT_CLIP_VALUE=1.0,
        LR_SCHEDULER_TYPE=None,  # Disable scheduler for short run
        LR_SCHEDULER_T_MAX=max_steps,  # Still needs a value if type was set
        LR_SCHEDULER_ETA_MIN=1e-06,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.001,
        CHECKPOINT_SAVE_FREQ_STEPS=25,  # Save more often
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=max_steps,  # Adjust anneal steps
        PER_EPSILON=1e-05,
        COMPILE_MODEL=False,  # Keep compilation off for testing speed
        PROFILE_WORKERS=True,  # Keep profiling ON for toy
    )
    # Use the fields defined in the toy.json for EnvConfig
    toy_env_config = EnvConfig(
        ROWS=8,
        COLS=15,
        PLAYABLE_RANGE_PER_ROW=[
            (3, 12),
            (2, 13),
            (1, 14),
            (0, 15),
            (0, 15),
            (1, 14),
            (2, 13),
            (3, 12),
        ],
        NUM_SHAPE_SLOTS=3,
        REWARD_PER_PLACED_TRIANGLE=0.01,
        REWARD_PER_CLEARED_TRIANGLE=0.5,
        REWARD_PER_STEP_ALIVE=0.005,
        PENALTY_GAME_OVER=-10.0,
    )
    return SavedConfigSet(
        metadata=_create_metadata(
            name,
            "Minimal preset for pipeline testing. Slightly longer run. Profiling ON.",
        ),
        env_config=toy_env_config,
        model_config_data=ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[8],
            CONV_KERNEL_SIZES=[3],
            CONV_STRIDES=[1],
            CONV_PADDING=[1],
            NUM_RESIDUAL_BLOCKS=0,
            RESIDUAL_BLOCK_FILTERS=8,
            USE_TRANSFORMER=False,
            TRANSFORMER_DIM=8,
            TRANSFORMER_HEADS=1,
            TRANSFORMER_LAYERS=0,
            TRANSFORMER_FC_DIM=16,
            FC_DIMS_SHARED=[8],
            POLICY_HEAD_DIMS=[8],
            NUM_VALUE_ATOMS=11,
            VALUE_MIN=-5.0,
            VALUE_MAX=5.0,
            VALUE_HEAD_DIMS=[8],
            ACTIVATION_FUNCTION="ReLU",
            USE_BATCH_NORM=True,
            OTHER_NN_INPUT_FEATURES_DIM=30,
        ),
        train_config=train_config,
        mcts_config=MuTriangleMCTSConfig(
            max_simulations=4,  # Increased slightly
            max_depth=3,  # Increased slightly
            cpuct=1.25,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            discount=1.0,
            mcts_batch_size=2,  # Increased slightly
        ),
        trieye_config=TrieyeConfig(
            app_name=APP_NAME,
            run_name=run_name,
            stats=StatsConfig(processing_interval_seconds=0.5, metrics=DEFAULT_METRICS),
            persistence=PersistenceConfig(
                ROOT_DATA_DIR=DATA_ROOT_DIR_NAME,
                RUN_NAME=run_name,
                MLFLOW_TRACKING_URI=f"file:{DATA_ROOT_DIR_NAME}/mlruns",
                SAVE_BUFFER=True,
                BUFFER_SAVE_FREQ_STEPS=25,  # Save buffer more often
            ),
        ),
    )


def get_test_config() -> SavedConfigSet:
    """Returns the 'test' configuration set - fast iteration between toy and simple."""
    name = "test"
    run_name = f"{name}_run"
    max_steps = 5000
    train_config = TrainConfig(
        RUN_NAME=run_name,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
        DEVICE="auto",
        RANDOM_SEED=42,
        MAX_TRAINING_STEPS=max_steps,
        NUM_SELF_PLAY_WORKERS=3,  # Fixed count for fast testing
        WORKER_DEVICE="cpu",
        BATCH_SIZE=64,
        BUFFER_CAPACITY=5000,
        MIN_BUFFER_SIZE_TO_TRAIN=1000,  # Warmup at 2000
        WORKER_UPDATE_FREQ_STEPS=15,
        N_STEP_RETURNS=3,
        GAMMA=0.99,
        OPTIMIZER_TYPE="AdamW",
        LEARNING_RATE=0.0004,
        WEIGHT_DECAY=0.0001,
        GRADIENT_CLIP_VALUE=1.0,
        LR_SCHEDULER_TYPE="CosineAnnealingLR",
        LR_SCHEDULER_T_MAX=max_steps,
        LR_SCHEDULER_ETA_MIN=1e-06,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.001,
        CHECKPOINT_SAVE_FREQ_STEPS=1000,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=max_steps,
        PER_EPSILON=1e-05,
        COMPILE_MODEL=False,  # Faster startup
        PROFILE_WORKERS=False,
    )
    return SavedConfigSet(
        metadata=_create_metadata(
            name, "Fast test preset for quick iteration (2-5 min to training)."
        ),
        env_config=DEFAULT_ENV_CONFIG,
        model_config_data=ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=1,  # Smaller model
            RESIDUAL_BLOCK_FILTERS=64,
            USE_TRANSFORMER=False,
            TRANSFORMER_DIM=64,
            TRANSFORMER_HEADS=2,
            TRANSFORMER_LAYERS=0,
            TRANSFORMER_FC_DIM=128,
            FC_DIMS_SHARED=[64],
            POLICY_HEAD_DIMS=[64],
            NUM_VALUE_ATOMS=51,
            VALUE_MIN=-10.0,
            VALUE_MAX=10.0,
            VALUE_HEAD_DIMS=[64],
            ACTIVATION_FUNCTION="ReLU",
            USE_BATCH_NORM=True,
            OTHER_NN_INPUT_FEATURES_DIM=30,
        ),
        train_config=train_config,
        mcts_config=MuTriangleMCTSConfig(
            max_simulations=16,  # Faster MCTS
            max_depth=8,
            cpuct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            discount=1.0,
            mcts_batch_size=8,
        ),
        trieye_config=TrieyeConfig(
            app_name=APP_NAME,
            run_name=run_name,
            stats=StatsConfig(
                processing_interval_seconds=1.0,
                metrics=DEFAULT_METRICS,
            ),
            persistence=PersistenceConfig(
                ROOT_DATA_DIR=DATA_ROOT_DIR_NAME,
                RUN_NAME=run_name,
                MLFLOW_TRACKING_URI=f"file:{DATA_ROOT_DIR_NAME}/mlruns",
                SAVE_BUFFER=True,
                BUFFER_SAVE_FREQ_STEPS=1000,
            ),
        ),
    )


def get_simple_config() -> SavedConfigSet:
    """Returns the 'simple' configuration set."""
    name = "simple"
    run_name = f"{name}_run"
    max_steps = 100_000
    train_config = TrainConfig(
        RUN_NAME=run_name,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=True,
        DEVICE="auto",
        RANDOM_SEED=42,
        MAX_TRAINING_STEPS=max_steps,
        NUM_SELF_PLAY_WORKERS=6,  # Fixed count for 16GB container: 6×300MB + 1.5GB main + 630MB Ray ≈ 4GB safe
        WORKER_DEVICE="cpu",
        BATCH_SIZE=128,  # Memory optimized
        BUFFER_CAPACITY=100_000,  # Increased from 50k for better sample diversity
        MIN_BUFFER_SIZE_TO_TRAIN=10_000,  # Proportional to buffer capacity
        WORKER_UPDATE_FREQ_STEPS=20,
        N_STEP_RETURNS=5,
        GAMMA=0.99,
        OPTIMIZER_TYPE="AdamW",
        LEARNING_RATE=0.0003,
        WEIGHT_DECAY=0.0001,
        GRADIENT_CLIP_VALUE=1.0,
        LR_SCHEDULER_TYPE="CosineAnnealingLR",
        LR_SCHEDULER_T_MAX=max_steps,
        LR_SCHEDULER_ETA_MIN=1e-06,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.001,
        CHECKPOINT_SAVE_FREQ_STEPS=5000,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=max_steps,
        PER_EPSILON=1e-05,
        COMPILE_MODEL=True,
        PROFILE_WORKERS=False,
    )
    return SavedConfigSet(
        metadata=_create_metadata(
            name, "Lightweight preset for faster runs. Transformer OFF, PER ON."
        ),
        env_config=DEFAULT_ENV_CONFIG,
        model_config_data=ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64, 128],
            CONV_KERNEL_SIZES=[3, 3, 3],
            CONV_STRIDES=[1, 1, 1],
            CONV_PADDING=[1, 1, 1],
            NUM_RESIDUAL_BLOCKS=2,
            RESIDUAL_BLOCK_FILTERS=128,
            USE_TRANSFORMER=False,  # Transformer OFF
            TRANSFORMER_DIM=128,
            TRANSFORMER_HEADS=4,
            TRANSFORMER_LAYERS=0,
            TRANSFORMER_FC_DIM=256,
            FC_DIMS_SHARED=[128],
            POLICY_HEAD_DIMS=[128],
            NUM_VALUE_ATOMS=51,
            VALUE_MIN=-10.0,
            VALUE_MAX=10.0,
            VALUE_HEAD_DIMS=[128],
            ACTIVATION_FUNCTION="ReLU",
            USE_BATCH_NORM=True,
            OTHER_NN_INPUT_FEATURES_DIM=30,
        ),
        train_config=train_config,
        mcts_config=MuTriangleMCTSConfig(
            max_simulations=64,
            max_depth=10,
            cpuct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            discount=1.0,
            mcts_batch_size=16,
        ),
        trieye_config=TrieyeConfig(
            app_name=APP_NAME,
            run_name=run_name,
            stats=StatsConfig(
                LOG_INTERVAL_STEPS=20,
                PRINT_INTERVAL_SECONDS=30,
                AGGREGATION_WINDOW_STEPS=100,
                metrics=DEFAULT_METRICS,
            ),
            persistence=PersistenceConfig(
                ROOT_DATA_DIR=DATA_ROOT_DIR_NAME,
                RUN_NAME=run_name,
                MLFLOW_TRACKING_URI=f"file:{DATA_ROOT_DIR_NAME}/mlruns",
                SAVE_BUFFER=True,
                BUFFER_SAVE_FREQ_STEPS=20000,
            ),
        ),
    )


def get_medium_config() -> SavedConfigSet:
    """Returns the 'medium' configuration set."""
    name = "medium"
    run_name = f"{name}_run"
    max_steps = 500_000
    train_config = TrainConfig(
        RUN_NAME=run_name,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=True,
        DEVICE="auto",
        RANDOM_SEED=42,
        MAX_TRAINING_STEPS=max_steps,
        NUM_SELF_PLAY_WORKERS=-1,
        WORKER_DEVICE="cpu",
        BATCH_SIZE=256,
        BUFFER_CAPACITY=250_000,
        MIN_BUFFER_SIZE_TO_TRAIN=25_000,
        WORKER_UPDATE_FREQ_STEPS=10,
        N_STEP_RETURNS=5,
        GAMMA=0.99,
        OPTIMIZER_TYPE="AdamW",
        LEARNING_RATE=0.0002,
        WEIGHT_DECAY=0.0001,
        GRADIENT_CLIP_VALUE=1.0,
        LR_SCHEDULER_TYPE="CosineAnnealingLR",
        LR_SCHEDULER_T_MAX=max_steps,
        LR_SCHEDULER_ETA_MIN=1e-06,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.001,
        CHECKPOINT_SAVE_FREQ_STEPS=2500,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=max_steps,
        PER_EPSILON=1e-05,
        COMPILE_MODEL=True,
        PROFILE_WORKERS=False,
    )
    return SavedConfigSet(
        metadata=_create_metadata(
            name, "Balanced preset for standard training. Transformer ON, PER ON."
        ),
        env_config=DEFAULT_ENV_CONFIG,
        model_config_data=ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[64, 128, 256],
            CONV_KERNEL_SIZES=[3, 3, 3],
            CONV_STRIDES=[1, 1, 1],
            CONV_PADDING=[1, 1, 1],
            NUM_RESIDUAL_BLOCKS=4,
            RESIDUAL_BLOCK_FILTERS=256,
            USE_TRANSFORMER=True,  # Transformer ON
            TRANSFORMER_DIM=256,
            TRANSFORMER_HEADS=8,
            TRANSFORMER_LAYERS=4,
            TRANSFORMER_FC_DIM=512,
            FC_DIMS_SHARED=[256],
            POLICY_HEAD_DIMS=[256],
            NUM_VALUE_ATOMS=51,
            VALUE_MIN=-15.0,
            VALUE_MAX=15.0,
            VALUE_HEAD_DIMS=[256],
            ACTIVATION_FUNCTION="GELU",
            USE_BATCH_NORM=True,
            OTHER_NN_INPUT_FEATURES_DIM=30,
        ),
        train_config=train_config,
        mcts_config=MuTriangleMCTSConfig(
            max_simulations=128,
            max_depth=15,
            cpuct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            discount=1.0,
            mcts_batch_size=32,
        ),
        trieye_config=TrieyeConfig(
            app_name=APP_NAME,
            run_name=run_name,
            stats=StatsConfig(
                LOG_INTERVAL_STEPS=25,
                PRINT_INTERVAL_SECONDS=30,
                AGGREGATION_WINDOW_STEPS=100,
                metrics=DEFAULT_METRICS,
            ),
            persistence=PersistenceConfig(
                ROOT_DATA_DIR=DATA_ROOT_DIR_NAME,
                RUN_NAME=run_name,
                MLFLOW_TRACKING_URI=f"file:{DATA_ROOT_DIR_NAME}/mlruns",
                SAVE_BUFFER=True,
                BUFFER_SAVE_FREQ_STEPS=25000,
            ),
        ),
    )


def get_large_config() -> SavedConfigSet:
    """Returns the 'large' configuration set."""
    name = "large"
    run_name = f"{name}_run"
    max_steps = 1_000_000
    train_config = TrainConfig(
        RUN_NAME=run_name,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=True,
        DEVICE="auto",
        RANDOM_SEED=42,
        MAX_TRAINING_STEPS=max_steps,
        NUM_SELF_PLAY_WORKERS=-1,
        WORKER_DEVICE="cpu",
        BATCH_SIZE=512,
        BUFFER_CAPACITY=500_000,
        MIN_BUFFER_SIZE_TO_TRAIN=50_000,
        WORKER_UPDATE_FREQ_STEPS=10,
        N_STEP_RETURNS=10,
        GAMMA=0.99,
        OPTIMIZER_TYPE="AdamW",
        LEARNING_RATE=0.0001,
        WEIGHT_DECAY=0.0001,
        GRADIENT_CLIP_VALUE=1.0,
        LR_SCHEDULER_TYPE="CosineAnnealingLR",
        LR_SCHEDULER_T_MAX=max_steps,
        LR_SCHEDULER_ETA_MIN=1e-06,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        ENTROPY_BONUS_WEIGHT=0.001,
        CHECKPOINT_SAVE_FREQ_STEPS=2500,
        USE_PER=True,
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        PER_BETA_FINAL=1.0,
        PER_BETA_ANNEAL_STEPS=max_steps,
        PER_EPSILON=1e-05,
        COMPILE_MODEL=True,
        PROFILE_WORKERS=False,
    )
    return SavedConfigSet(
        metadata=_create_metadata(
            name, "Robust preset for high performance. Large capacity/batch/sims."
        ),
        env_config=DEFAULT_ENV_CONFIG,
        model_config_data=ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[128, 256, 512],
            CONV_KERNEL_SIZES=[3, 3, 3],
            CONV_STRIDES=[1, 1, 1],
            CONV_PADDING=[1, 1, 1],
            NUM_RESIDUAL_BLOCKS=8,
            RESIDUAL_BLOCK_FILTERS=512,
            USE_TRANSFORMER=True,
            TRANSFORMER_DIM=512,
            TRANSFORMER_HEADS=8,
            TRANSFORMER_LAYERS=6,
            TRANSFORMER_FC_DIM=1024,
            FC_DIMS_SHARED=[512, 256],
            POLICY_HEAD_DIMS=[256],
            NUM_VALUE_ATOMS=101,
            VALUE_MIN=-20.0,
            VALUE_MAX=20.0,
            VALUE_HEAD_DIMS=[256],
            ACTIVATION_FUNCTION="GELU",
            USE_BATCH_NORM=True,
            OTHER_NN_INPUT_FEATURES_DIM=30,
        ),
        train_config=train_config,
        mcts_config=MuTriangleMCTSConfig(
            max_simulations=400,
            max_depth=20,
            cpuct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            discount=1.0,
            mcts_batch_size=64,
        ),
        trieye_config=TrieyeConfig(
            app_name=APP_NAME,
            run_name=run_name,
            stats=StatsConfig(
                LOG_INTERVAL_STEPS=50,
                PRINT_INTERVAL_SECONDS=60,
                AGGREGATION_WINDOW_STEPS=200,
                metrics=DEFAULT_METRICS,
            ),
            persistence=PersistenceConfig(
                ROOT_DATA_DIR=DATA_ROOT_DIR_NAME,
                RUN_NAME=run_name,
                MLFLOW_TRACKING_URI=f"file:{DATA_ROOT_DIR_NAME}/mlruns",
                SAVE_BUFFER=True,
                BUFFER_SAVE_FREQ_STEPS=50000,
            ),
        ),
    )


# Dictionary mapping preset names to their getter functions
PRESET_DEFINITIONS = {
    "toy": get_toy_config,
    "test": get_test_config,
    "simple": get_simple_config,
    "medium": get_medium_config,
    "large": get_large_config,
}
