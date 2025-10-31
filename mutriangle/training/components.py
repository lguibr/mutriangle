# File: mutriangle/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ray

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig
from trieye import Serializer  # Import Serializer from trieye
from mutrimcts import SearchConfiguration

# Keep mutriangle imports

if TYPE_CHECKING:
    from trieye import TrieyeConfig  # Import TrieyeConfig

    from mutriangle.config import (
        ModelConfig,
        RunContext,  # Import RunContext
        TrainConfig,
    )
    from mutriangle.nn import NeuralNetwork
    from mutriangle.rl import GameHistoryBuffer, Trainer


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    run_context: "RunContext"  # Add RunContext
    nn: "NeuralNetwork"
    buffer: "GameHistoryBuffer"
    trainer: "Trainer"
    trieye_actor: ray.actor.ActorHandle
    trieye_config: "TrieyeConfig"
    serializer: Serializer
    train_config: "TrainConfig"
    env_config: EnvConfig
    model_config: "ModelConfig"
    mcts_config: SearchConfiguration
    profile_workers: bool
