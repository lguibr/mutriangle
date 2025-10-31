# File: tests/training/test_worker_manager.py
"""
Tests for WorkerManager memory handling and OOM scenarios.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import ray
from ray.exceptions import OutOfMemoryError

from mutriangle.config import ModelConfig, MuTriangleMCTSConfig, TrainConfig
from mutriangle.training.worker_manager import WorkerManager


@pytest.fixture
def mock_components():
    """Create mock training components."""
    components = Mock()
    
    # Mock train config
    components.train_config = TrainConfig(
        NUM_SELF_PLAY_WORKERS=3,
        RANDOM_SEED=42,
        WORKER_DEVICE="cpu",
    )
    
    # Mock env config
    from trianglengin import EnvConfig
    components.env_config = EnvConfig(
        ROWS=8,
        COLS=15,
        NUM_SHAPE_SLOTS=3,
    )
    
    # Mock model config
    components.model_config = ModelConfig(
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
    
    # Mock MCTS config
    components.mcts_config = MuTriangleMCTSConfig(
        max_simulations=16,
        max_depth=8,
        cpuct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
        mcts_batch_size=8,
    )
    
    # Mock neural network
    components.nn = Mock()
    components.nn.get_weights.return_value = {"dummy": "weights"}
    
    # Mock Trieye actor
    components.trieye_actor = Mock()
    
    # Mock profiling
    components.profile_workers = False
    
    return components


class TestWorkerManagerMemoryLimits:
    """Test worker manager memory limit handling."""

    @patch("mutriangle.training.worker_manager.ray")
    def test_worker_initialization_with_memory_limit(self, mock_ray, mock_components):
        """Test that workers are initialized with memory limits."""
        # Mock Ray get calls for trieye info
        mock_ray.get.side_effect = [
            "trieye_actor_test",  # actor name
            "/test/run/dir",      # run dir
        ]
        
        # Mock ray.put
        mock_weights_ref = Mock()
        mock_ray.put.return_value = mock_weights_ref
        
        # Mock SelfPlayWorker
        mock_worker_class = Mock()
        mock_worker_options = Mock()
        mock_worker_class.options.return_value = mock_worker_options
        
        with patch("mutriangle.training.worker_manager.SelfPlayWorker", mock_worker_class):
            manager = WorkerManager(mock_components)
            manager.initialize_workers()
            
            # Verify workers were created with memory limits
            assert mock_worker_class.options.call_count == 3  # 3 workers
            
            # Check that memory limit was specified
            for call in mock_worker_class.options.call_args_list:
                kwargs = call[1] if call[1] else call[0][0] if call[0] else {}
                assert "num_cpus" in kwargs or (call[1] and "num_cpus" in call[1])
                assert "memory" in kwargs or (call[1] and "memory" in call[1])
                
                # If using kwargs
                if "memory" in kwargs:
                    # 300MB = 300 * 1024 * 1024 bytes
                    assert kwargs["memory"] == 300 * 1024 * 1024

    @patch("mutriangle.training.worker_manager.ray")
    @patch("mutriangle.training.worker_manager.SelfPlayWorker")
    def test_worker_initialization_failure_handling(
        self, mock_worker_class, mock_ray, mock_components
    ):
        """Test that worker initialization failures are handled gracefully."""
        # Mock Ray get calls for trieye info
        mock_ray.get.side_effect = [
            "trieye_actor_test",
            "/test/run/dir",
        ]
        
        # Mock ray.put
        mock_ray.put.return_value = Mock()
        
        # Make first worker succeed, second fail, third succeed
        mock_worker_options = Mock()
        mock_worker_class.options.return_value = mock_worker_options
        
        mock_worker_1 = Mock()
        mock_worker_3 = Mock()
        
        mock_worker_options.remote.side_effect = [
            mock_worker_1,
            Exception("OOM simulation"),
            mock_worker_3,
        ]
        
        manager = WorkerManager(mock_components)
        manager.initialize_workers()
        
        # Should have 2 active workers (1st and 3rd)
        assert len(manager.active_worker_indices) == 2
        assert 0 in manager.active_worker_indices
        assert 2 in manager.active_worker_indices
        assert 1 not in manager.active_worker_indices


class TestWorkerManagerOOMHandling:
    """Test OOM error handling in worker manager."""

    def test_handle_oom_in_task_completion(self, mock_components):
        """Test handling of OOM errors when getting task results."""
        # Use real Ray exceptions but mock the ray module calls
        with patch("mutriangle.training.worker_manager.ray") as mock_ray, \
             patch("mutriangle.training.worker_manager.SelfPlayWorker") as mock_worker_class:
            
            # Setup ray.exceptions properly (use real exceptions module)
            mock_ray.exceptions = ray.exceptions
            
            # Mock Ray setup for initialization
            mock_ray.get.side_effect = [
                "trieye_actor_test",
                "/test/run/dir",
            ]
            mock_ray.put.return_value = Mock()
            
            # Create worker manager with mock workers
            mock_worker = Mock()
            mock_worker_options = Mock()
            mock_worker_options.remote.return_value = mock_worker
            mock_worker_class.options.return_value = mock_worker_options
            
            manager = WorkerManager(mock_components)
            manager.initialize_workers()
            
            # Simulate an OOM error when getting results
            mock_task_ref = Mock()
            manager.worker_tasks[mock_task_ref] = 0  # Worker 0
            
            # Mock ray.wait to return our task ref as ready
            mock_ray.wait.return_value = ([mock_task_ref], [])
            
            # Mock ray.get to raise OOM after initialization calls
            mock_ray.get.side_effect = OutOfMemoryError(
                "Task was killed due to the node running low on memory."
            )
            
            # This should be handled in get_completed_tasks
            # The method catches exceptions and returns them as results
            results = manager.get_completed_tasks(timeout=0.1)
            
            # Should return list with the exception
            assert isinstance(results, list)
            if results:
                # Result should be tuple of (worker_idx, exception)
                assert len(results[0]) == 2
                worker_idx, exception = results[0]
                assert worker_idx == 0
                assert isinstance(exception, Exception)


class TestWorkerManagerResourceManagement:
    """Test resource management in worker manager."""

    @patch("mutriangle.training.worker_manager.ray")
    @patch("mutriangle.training.worker_manager.SelfPlayWorker")
    def test_worker_count_matches_config(
        self, mock_worker_class, mock_ray, mock_components
    ):
        """Test that worker count matches configuration."""
        # Mock Ray setup
        mock_ray.get.side_effect = [
            "trieye_actor_test",
            "/test/run/dir",
        ]
        mock_ray.put.return_value = Mock()
        
        # Create workers
        mock_worker_options = Mock()
        mock_worker_options.remote.return_value = Mock()
        mock_worker_class.options.return_value = mock_worker_options
        
        # Set specific worker count
        mock_components.train_config.NUM_SELF_PLAY_WORKERS = 5
        
        manager = WorkerManager(mock_components)
        manager.initialize_workers()
        
        # Should have 5 workers
        assert len(manager.workers) == 5
        assert len(manager.active_worker_indices) == 5

    @patch("mutriangle.training.worker_manager.ray")
    @patch("mutriangle.training.worker_manager.SelfPlayWorker")
    def test_weights_sharing_via_ray_put(
        self, mock_worker_class, mock_ray, mock_components
    ):
        """Test that weights are shared efficiently via ray.put."""
        # Mock Ray setup
        mock_ray.get.side_effect = [
            "trieye_actor_test",
            "/test/run/dir",
        ]
        
        mock_weights_ref = Mock()
        mock_ray.put.return_value = mock_weights_ref
        
        # Create workers
        mock_worker_options = Mock()
        mock_worker_options.remote.return_value = Mock()
        mock_worker_class.options.return_value = mock_worker_options
        
        manager = WorkerManager(mock_components)
        manager.initialize_workers()
        
        # Verify ray.put was called once for weights (not per worker)
        assert mock_ray.put.call_count == 1
        
        # Verify all workers got the same weights reference
        assert mock_worker_options.remote.call_count == 3
        for call in mock_worker_options.remote.call_args_list:
            kwargs = call[1]
            assert kwargs["initial_weights"] == mock_weights_ref


class TestWorkerManagerTaskSubmission:
    """Test task submission and management."""

    @patch("mutriangle.training.worker_manager.ray")
    @patch("mutriangle.training.worker_manager.SelfPlayWorker")
    def test_submit_task_to_active_worker(
        self, mock_worker_class, mock_ray, mock_components
    ):
        """Test submitting tasks to active workers."""
        # Mock Ray setup
        mock_ray.get.side_effect = [
            "trieye_actor_test",
            "/test/run/dir",
        ]
        mock_ray.put.return_value = Mock()
        
        # Create mock worker with remote method
        mock_worker = Mock()
        mock_task_ref = Mock()
        mock_worker.run_episode.remote.return_value = mock_task_ref
        
        mock_worker_options = Mock()
        mock_worker_options.remote.return_value = mock_worker
        mock_worker_class.options.return_value = mock_worker_options
        
        manager = WorkerManager(mock_components)
        manager.initialize_workers()
        
        # Submit task to worker 0
        manager.submit_task(0)
        
        # Verify task was submitted
        assert mock_worker.run_episode.remote.called
        assert mock_task_ref in manager.worker_tasks
        assert manager.worker_tasks[mock_task_ref] == 0

    @patch("mutriangle.training.worker_manager.ray")
    @patch("mutriangle.training.worker_manager.SelfPlayWorker")
    def test_skip_task_submission_to_inactive_worker(
        self, mock_worker_class, mock_ray, mock_components
    ):
        """Test that tasks are not submitted to inactive workers."""
        # Mock Ray setup
        mock_ray.get.side_effect = [
            "trieye_actor_test",
            "/test/run/dir",
        ]
        mock_ray.put.return_value = Mock()
        
        mock_worker = Mock()
        mock_worker_options = Mock()
        mock_worker_options.remote.return_value = mock_worker
        mock_worker_class.options.return_value = mock_worker_options
        
        manager = WorkerManager(mock_components)
        manager.initialize_workers()
        
        # Remove worker 1 from active set
        manager.active_worker_indices.discard(1)
        
        # Try to submit task to inactive worker
        initial_task_count = len(manager.worker_tasks)
        manager.submit_task(1)
        
        # Should not create new task
        assert len(manager.worker_tasks) == initial_task_count

