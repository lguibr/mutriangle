# File: mutriangle/config/mcts_config.py
"""
Configuration for MCTS parameters specific to MuTriangle,
mirroring mutrimcts.SearchConfiguration for easy control.
"""

from pydantic import BaseModel, ConfigDict, Field
from mutrimcts import SearchConfiguration  # Import base config for reference

# Restore default simulations to a lower value for faster testing/profiling
DEFAULT_MAX_SIMULATIONS = 64
DEFAULT_MAX_DEPTH = 8
DEFAULT_CPUCT = 1.5
DEFAULT_MCTS_BATCH_SIZE = 32  # Default batch size for network evals within MCTS


class MuTriangleMCTSConfig(BaseModel):
    """MCTS Search Configuration managed within MuTriangle."""

    # Core Search Parameters
    max_simulations: int = Field(
        default=DEFAULT_MAX_SIMULATIONS,
        description="Maximum number of MCTS simulations per move.",
        gt=0,
    )
    max_depth: int = Field(
        default=DEFAULT_MAX_DEPTH,
        description="Maximum depth for tree traversal during simulation.",
        gt=0,
    )

    # UCT Parameters (PUCT style)
    cpuct: float = Field(
        default=DEFAULT_CPUCT,
        description="Constant determining the level of exploration (PUCT). Higher is more exploration.",
    )

    # Dirichlet Noise (for root node exploration)
    dirichlet_alpha: float = Field(
        default=0.3, description="Alpha parameter for Dirichlet noise (shape).", ge=0
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        description="Weight of Dirichlet noise in root prior probabilities (0 to disable).",
        ge=0,
        le=1.0,
    )

    # Discount Factor (Primarily for MuZero/Value Propagation)
    discount: float = Field(
        default=1.0,
        description="Discount factor (gamma) for future rewards/values within MCTS (usually 1.0 for AZ).",
        ge=0.0,
        le=1.0,
    )

    # Batching for Network Evaluations within MCTS
    mcts_batch_size: int = Field(
        default=DEFAULT_MCTS_BATCH_SIZE,
        description="Number of leaf nodes to collect in C++ MCTS before calling network evaluate_batch.",
        gt=0,
    )

    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(validate_assignment=True)

    def to_mutrimcts_config(self) -> SearchConfiguration:
        """Converts this config to the mutrimcts.SearchConfiguration."""
        return SearchConfiguration(
            max_simulations=self.max_simulations,
            max_depth=self.max_depth,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            discount=self.discount,
            mcts_batch_size=self.mcts_batch_size,  # Pass batch size
        )


# Ensure model is rebuilt after changes
MuTriangleMCTSConfig.model_rebuild(force=True)
