"""
Core RL components: Trainer, Buffer.
The Orchestrator logic has been moved to the mutriangle.training module.
"""

from .trainer import Trainer

__all__ = [
    "Trainer",
]
