# File: mutriangle/config_mgmt/schemas.py
"""Pydantic schemas for configuration management."""

import datetime
import platform
from importlib import metadata as importlib_metadata
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator
from trianglengin import EnvConfig
from trieye import TrieyeConfig

# Import MuTriangle specific configs
from mutriangle.config import (
    MuTriangleMCTSConfig,
    ModelConfig,
    TrainConfig,
)

# Import central constants
from mutriangle.config.app_config import APP_NAME, DATA_ROOT_DIR_NAME

# Get version using importlib.metadata
try:
    mutriangle_version = importlib_metadata.version("mutriangle")
except importlib_metadata.PackageNotFoundError:
    mutriangle_version = "0.0.0-unknown"


class Metadata(BaseModel):
    """Metadata associated with a saved configuration set."""

    name: str = Field(..., description="Unique name for the configuration set.")
    description: str | None = Field(
        default=None, description="Optional description for the configuration."
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now, description="Timestamp of creation."
    )
    last_modified_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp of last modification.",
    )
    mutriangle_version: str = Field(
        default=mutriangle_version, description="MuTriangle version when saved."
    )
    platform: str = Field(
        default_factory=platform.platform, description="Platform where saved."
    )


class SavedConfigSet(BaseModel):
    """A bundle containing all necessary configurations and metadata."""

    metadata: Metadata = Field(
        ..., description="Metadata about this configuration set."
    )
    env_config: Annotated[
        EnvConfig,
        Field(
            ...,
            description="Environment configuration (grid size, rewards - mostly read-only).",
        ),
    ]
    model_config_data: Annotated[
        ModelConfig, Field(..., description="Neural network model architecture.")
    ]
    train_config: Annotated[
        TrainConfig, Field(..., description="Training loop hyperparameters.")
    ]
    mcts_config: Annotated[
        MuTriangleMCTSConfig,
        Field(..., description="Monte Carlo Tree Search parameters."),
    ]
    trieye_config: Annotated[
        TrieyeConfig,
        Field(..., description="Statistics logging and persistence settings."),
    ]

    @field_validator("train_config", "trieye_config", mode="before")
    @classmethod
    def sync_run_name_before(cls, v: Any, info) -> Any:
        """Attempt to sync run_name before individual model validation."""
        # This ensures that if a user edits one run_name in the JSON,
        # the other is updated before Pydantic validates each sub-model.
        if isinstance(v, dict) and "run_name" in v:
            run_name = v["run_name"]
            if "train_config" in info.data and isinstance(
                info.data["train_config"], dict
            ):
                info.data["train_config"]["RUN_NAME"] = run_name
            if "trieye_config" in info.data and isinstance(
                info.data["trieye_config"], dict
            ):
                info.data["trieye_config"]["run_name"] = run_name
                if "persistence" in info.data["trieye_config"] and isinstance(
                    info.data["trieye_config"]["persistence"], dict
                ):
                    info.data["trieye_config"]["persistence"]["RUN_NAME"] = run_name
        elif isinstance(v, dict) and "RUN_NAME" in v:  # Check TrainConfig key
            run_name = v["RUN_NAME"]
            if "train_config" in info.data and isinstance(
                info.data["train_config"], dict
            ):
                info.data["train_config"]["RUN_NAME"] = run_name
            if "trieye_config" in info.data and isinstance(
                info.data["trieye_config"], dict
            ):
                info.data["trieye_config"]["run_name"] = run_name
                if "persistence" in info.data["trieye_config"] and isinstance(
                    info.data["trieye_config"]["persistence"], dict
                ):
                    info.data["trieye_config"]["persistence"]["RUN_NAME"] = run_name
        return v

    @field_validator("trieye_config", mode="before")
    @classmethod
    def ensure_correct_trieye_paths_before(cls, v: Any) -> Any:
        """Ensure TrieyeConfig paths point to the unified data dir before validation."""
        # This validator is less critical now as RunContext overrides paths at runtime,
        # but it helps ensure consistency in the saved JSON file itself.
        if isinstance(v, dict):
            # Ensure app_name is correct
            v["app_name"] = APP_NAME

            # Ensure persistence paths reflect the standard structure, even if
            # the exact root path isn't known here. The runtime RunContext
            # will provide the absolute path.
            expected_root_placeholder = DATA_ROOT_DIR_NAME  # Use the constant
            expected_mlflow_uri_placeholder = f"file:{DATA_ROOT_DIR_NAME}/mlruns"

            if "persistence" not in v or not isinstance(v["persistence"], dict):
                v["persistence"] = {}  # Initialize if missing

            # Set placeholder paths in the dict before validation
            v["persistence"]["ROOT_DATA_DIR"] = expected_root_placeholder
            v["persistence"]["MLFLOW_TRACKING_URI"] = expected_mlflow_uri_placeholder
            # RUN_NAME should be synced by sync_run_name_before

        return v

    def update_last_modified(self):
        """Updates the last_modified_at timestamp."""
        self.metadata.last_modified_at = datetime.datetime.now()


# Rebuild models to ensure validators are registered
Metadata.model_rebuild(force=True)
SavedConfigSet.model_rebuild(force=True)
