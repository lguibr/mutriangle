# File: mutriangle/utils/memory.py
"""
Memory estimation and calculation utilities for Ray workers and training.
Helps prevent OOM errors by calculating safe worker counts based on available memory.
"""

import logging
from typing import TYPE_CHECKING

import psutil
import ray

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


def estimate_worker_memory(model_config: "ModelConfig") -> int:
    """
    Estimate memory usage per worker in MB based on model configuration.
    
    Args:
        model_config: Model configuration object
        
    Returns:
        Estimated memory in MB per worker
    """
    # Base memory for worker overhead and MCTS tree
    base_memory_mb = 50
    
    # Estimate model size based on filters and blocks
    if model_config.CONV_FILTERS:
        max_filters = max(model_config.CONV_FILTERS)
    else:
        max_filters = 64
    
    # Model memory scales with filters and residual blocks
    model_memory_mb = (max_filters / 64) * 100  # ~100MB for 64 filters baseline
    model_memory_mb *= (1 + model_config.NUM_RESIDUAL_BLOCKS * 0.2)  # Add 20% per residual block
    
    # Transformer adds significant memory
    if model_config.USE_TRANSFORMER:
        transformer_memory_mb = model_config.TRANSFORMER_DIM * model_config.TRANSFORMER_LAYERS * 0.1
        model_memory_mb += transformer_memory_mb
    
    # MCTS tree memory (rough estimate)
    mcts_memory_mb = 50
    
    total_mb = int(base_memory_mb + model_memory_mb + mcts_memory_mb)
    
    # Add safety margin (20%)
    total_mb = int(total_mb * 1.2)
    
    logger.debug(
        f"Estimated worker memory: base={base_memory_mb}MB, model={model_memory_mb:.1f}MB, "
        f"mcts={mcts_memory_mb}MB, total_with_margin={total_mb}MB"
    )
    
    return total_mb


def get_available_memory() -> int:
    """
    Get available system memory in MB.
    
    Returns:
        Available memory in MB
    """
    try:
        # Get system memory
        mem = psutil.virtual_memory()
        available_mb = int(mem.available / (1024 * 1024))
        
        logger.debug(
            f"System memory: total={mem.total / (1024**3):.1f}GB, "
            f"available={available_mb}MB ({mem.percent}% used)"
        )
        
        return available_mb
    except Exception as e:
        logger.warning(f"Failed to get available memory: {e}")
        # Conservative fallback
        return 4096  # 4GB


def get_ray_available_memory() -> int:
    """
    Get available memory from Ray cluster in MB.
    
    Returns:
        Available memory in MB, or None if Ray not initialized
    """
    try:
        if not ray.is_initialized():
            logger.warning("Ray not initialized, cannot get Ray memory")
            return get_available_memory()
        
        resources = ray.cluster_resources()
        # Ray tracks memory in bytes
        memory_bytes = resources.get("memory", 0)
        memory_mb = int(memory_bytes / (1024 * 1024))
        
        logger.debug(f"Ray cluster memory: {memory_mb}MB")
        
        return memory_mb
    except Exception as e:
        logger.warning(f"Failed to get Ray memory: {e}")
        return get_available_memory()


def calculate_max_workers(
    available_mb: int,
    model_config: "ModelConfig",
    reserve_mb: int = 2048,
    safety_margin: float = 0.3,
) -> int:
    """
    Calculate maximum safe number of workers based on available memory.
    
    Args:
        available_mb: Available memory in MB
        model_config: Model configuration object
        reserve_mb: Memory to reserve for main process and Ray (default 2GB)
        safety_margin: Additional safety margin as fraction (default 0.3 = 30%)
        
    Returns:
        Maximum safe number of workers
    """
    worker_mb = estimate_worker_memory(model_config)
    
    # Calculate usable memory after reserves
    usable_mb = available_mb - reserve_mb
    
    # Apply safety margin
    usable_mb = int(usable_mb * (1.0 - safety_margin))
    
    if usable_mb <= 0:
        logger.warning(
            f"Insufficient memory: available={available_mb}MB, reserve={reserve_mb}MB"
        )
        return 1  # Minimum 1 worker
    
    max_workers = usable_mb // worker_mb
    
    # Ensure at least 1 worker
    max_workers = max(1, max_workers)
    
    logger.info(
        f"Memory calculation: available={available_mb}MB, reserve={reserve_mb}MB, "
        f"usable={usable_mb}MB, worker_est={worker_mb}MB → max_workers={max_workers}"
    )
    
    return max_workers


def calculate_recommended_workers(
    model_config: "ModelConfig",
    cpu_count: int | None = None,
    use_ray_memory: bool = True,
) -> int:
    """
    Calculate recommended number of workers based on memory and CPU availability.
    
    Args:
        model_config: Model configuration object
        cpu_count: Available CPU cores (if None, auto-detect)
        use_ray_memory: If True, use Ray memory info; otherwise use system memory
        
    Returns:
        Recommended number of workers
    """
    # Get available memory
    if use_ray_memory:
        available_mb = get_ray_available_memory()
    else:
        available_mb = get_available_memory()
    
    # Calculate memory-based limit
    memory_limit = calculate_max_workers(available_mb, model_config)
    
    # Get CPU limit
    if cpu_count is None:
        try:
            import os
            cpu_count = os.cpu_count() or 4
        except Exception:
            cpu_count = 4
    
    # Reserve 2 cores for main process and Ray
    cpu_limit = max(1, cpu_count - 2)
    
    # Take minimum of memory and CPU limits
    recommended = min(memory_limit, cpu_limit)
    
    logger.info(
        f"Recommended workers: memory_limit={memory_limit}, cpu_limit={cpu_limit}, "
        f"recommended={recommended}"
    )
    
    return recommended

