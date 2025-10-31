# File: tests/utils/test_memory.py
"""
Tests for memory estimation and calculation utilities.
"""

import pytest

from mutriangle.config import ModelConfig
from mutriangle.utils.memory import (
    calculate_max_workers,
    calculate_recommended_workers,
    estimate_worker_memory,
    get_available_memory,
)


class TestEstimateWorkerMemory:
    """Test worker memory estimation."""

    def test_small_model(self):
        """Test memory estimation for small model."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[16, 32],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=0,
            RESIDUAL_BLOCK_FILTERS=32,
            USE_TRANSFORMER=False,
            TRANSFORMER_DIM=32,
            TRANSFORMER_HEADS=2,
            TRANSFORMER_LAYERS=0,
            TRANSFORMER_FC_DIM=64,
            FC_DIMS_SHARED=[32],
            POLICY_HEAD_DIMS=[32],
            NUM_VALUE_ATOMS=11,
            VALUE_MIN=-5.0,
            VALUE_MAX=5.0,
            VALUE_HEAD_DIMS=[32],
            ACTIVATION_FUNCTION="ReLU",
            USE_BATCH_NORM=True,
            OTHER_NN_INPUT_FEATURES_DIM=30,
        )

        memory_mb = estimate_worker_memory(config)

        # Small model should be < 200MB
        assert 50 < memory_mb < 200
        assert isinstance(memory_mb, int)

    def test_medium_model(self):
        """Test memory estimation for medium model."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64, 128],
            CONV_KERNEL_SIZES=[3, 3, 3],
            CONV_STRIDES=[1, 1, 1],
            CONV_PADDING=[1, 1, 1],
            NUM_RESIDUAL_BLOCKS=2,
            RESIDUAL_BLOCK_FILTERS=128,
            USE_TRANSFORMER=False,
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
        )

        memory_mb = estimate_worker_memory(config)

        # Medium model should be 200-400MB
        assert 150 < memory_mb < 500
        assert isinstance(memory_mb, int)

    def test_large_model_with_transformer(self):
        """Test memory estimation for large model with transformer."""
        config = ModelConfig(
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
        )

        memory_mb = estimate_worker_memory(config)

        # Large model with transformer should be > 400MB
        assert memory_mb > 400
        assert isinstance(memory_mb, int)


class TestCalculateMaxWorkers:
    """Test maximum worker calculation."""

    def test_sufficient_memory(self):
        """Test calculation with sufficient memory."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=1,
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
        )

        # 16GB available
        max_workers = calculate_max_workers(
            available_mb=16000,
            model_config=config,
            reserve_mb=2048,
            safety_margin=0.3,
        )

        # Should be able to fit several workers
        assert max_workers >= 3
        assert isinstance(max_workers, int)

    def test_low_memory(self):
        """Test calculation with low memory."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[128, 256],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=4,
            RESIDUAL_BLOCK_FILTERS=256,
            USE_TRANSFORMER=True,
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
        )

        # Only 4GB available
        max_workers = calculate_max_workers(
            available_mb=4000,
            model_config=config,
            reserve_mb=2048,
            safety_margin=0.3,
        )

        # Should return at least 1 even with low memory
        assert max_workers >= 1
        assert isinstance(max_workers, int)

    def test_insufficient_memory(self):
        """Test calculation with insufficient memory (after reserves)."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=1,
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
        )

        # Very low memory (1GB)
        max_workers = calculate_max_workers(
            available_mb=1000,
            model_config=config,
            reserve_mb=2048,
            safety_margin=0.3,
        )

        # Should always return at least 1
        assert max_workers == 1

    def test_custom_safety_margin(self):
        """Test calculation with custom safety margin."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=1,
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
        )

        # 8GB available, 50% safety margin
        max_workers_high_margin = calculate_max_workers(
            available_mb=8000,
            model_config=config,
            reserve_mb=1024,
            safety_margin=0.5,
        )

        # 8GB available, 10% safety margin
        max_workers_low_margin = calculate_max_workers(
            available_mb=8000,
            model_config=config,
            reserve_mb=1024,
            safety_margin=0.1,
        )

        # Lower margin should allow more workers
        assert max_workers_low_margin > max_workers_high_margin


class TestGetAvailableMemory:
    """Test available memory detection."""

    def test_get_available_memory(self):
        """Test that we can get available memory."""
        available_mb = get_available_memory()

        # Should return a positive integer
        assert isinstance(available_mb, int)
        assert available_mb > 0

        # Should be at least 512MB (sanity check)
        assert available_mb >= 512


class TestCalculateRecommendedWorkers:
    """Test recommended worker calculation."""

    def test_recommended_workers(self):
        """Test calculation of recommended workers."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=1,
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
        )

        recommended = calculate_recommended_workers(
            model_config=config,
            cpu_count=8,
            use_ray_memory=False,  # Use system memory (don't need Ray)
        )

        # Should return a positive integer
        assert isinstance(recommended, int)
        assert recommended >= 1

        # With 8 CPUs, should get at most 6 (8 - 2 reserved)
        assert recommended <= 6

    def test_recommended_workers_low_cpu(self):
        """Test calculation with low CPU count."""
        config = ModelConfig(
            GRID_INPUT_CHANNELS=1,
            CONV_FILTERS=[32, 64],
            CONV_KERNEL_SIZES=[3, 3],
            CONV_STRIDES=[1, 1],
            CONV_PADDING=[1, 1],
            NUM_RESIDUAL_BLOCKS=1,
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
        )

        recommended = calculate_recommended_workers(
            model_config=config,
            cpu_count=2,
            use_ray_memory=False,
        )

        # With 2 CPUs, should get at least 1 (2 - 2 reserved = 0, but min is 1)
        assert recommended >= 1
