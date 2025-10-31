# File: mutriangle/config/run_context.py
"""Defines the RunContext model for managing run-specific identifiers and the root data path."""

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Import constants from app_config
from .app_config import APP_NAME, DATA_ROOT_DIR_NAME

logger = logging.getLogger(__name__)


class RunContext(BaseModel):
    """
    Holds run-specific context: identifiers and the root data directory.

    Path derivation for specific artifacts (checkpoints, logs, etc.) is handled
    by the Trieye library based on this context. Instances should be treated
    as immutable after creation.
    """

    model_config = ConfigDict(frozen=True)  # Make instances immutable

    app_name: str = Field(default=APP_NAME, description="Application name.")
    run_name: str = Field(..., description="Unique identifier for this run.")
    data_root_dir: Path = Field(
        ..., description="Absolute path to the root data directory for the application."
    )

    @classmethod
    def create(cls, run_name: str, base_dir: Path | None = None) -> "RunContext":
        """Factory method to create a RunContext instance."""
        if base_dir is None:
            base_dir = Path.cwd()
        # Derive the data root specific to the application
        data_root = (base_dir / DATA_ROOT_DIR_NAME / APP_NAME).resolve()
        return cls(run_name=run_name, data_root_dir=data_root)


# Ensure model is rebuilt after changes
RunContext.model_rebuild(force=True)
