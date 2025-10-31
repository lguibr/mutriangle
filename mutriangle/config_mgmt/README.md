# File: alphatriangle/config_mgmt/README.md
# Configuration Management Module (alphatriangle.config_mgmt)

## Purpose and Architecture

This module provides the underlying logic and storage mechanisms for managing AlphaTriangle configuration sets. The command-line interface is now integrated directly into the main alphatriangle command structure, rather than being a separate subcommand.

-   **Persistence:** Configuration sets are saved as JSON files in the .alphatriangle_data/configs/ directory (within the unified data directory).
-   **Default Configuration:** A .default_config_marker file within .alphatriangle_data/configs/ tracks the default configuration.
-   **Presets:** Predefined configurations (toy, simple, medium, large) are defined as Python functions in the [alphatriangle/presets/definitions.py](../presets/definitions.py) module. These are automatically instantiated and saved as JSON files to the user's .alphatriangle_data/configs/ directory on the first run if they don't exist using ensure_presets_saved. The simple preset is set as the default if no other default is configured.
-   **Schema ([schemas.py](schemas.py)):** Defines the SavedConfigSet Pydantic model, which bundles all configuration sub-models (EnvConfig, ModelConfig, TrainConfig, AlphaTriangleMCTSConfig, TrieyeConfig) along with metadata.
-   **Storage ([storage.py](storage.py)):** Handles file I/O (load/save/list/delete JSON), default marker management, preset saving, and launching the system editor. Defines constants like DATA_ROOT_DIR_NAME.
-   **CLI Logic:**
    -   **Commands ([commands/](commands/)):** Contains the *implementation logic* for configuration actions (create, list, show, edit, delete, validate, default). These functions are imported and decorated by the main Typer app in alphatriangle/cli.py. They utilize functions from storage.py and cli_utils.py.
    -   **Utilities ([cli_utils.py](cli_utils.py)):** Contains shared helpers (like the interactive prompter using questionary), constants defining field behavior (READ_ONLY_FIELDS, SIMPLE_FIELDS, etc.), and type hints used by the command logic.
    -   **Choices ([cli_choices.py](cli_choices.py)):** Contains the dictionaries defining the predefined options presented during interactive configuration (NUMERIC_CHOICES, LIST_CHOICES, etc.).
-   **Integration:** The main alphatriangle CLI uses these components to provide top-level commands like alphatriangle create, alphatriangle list, alphatriangle default, etc. The alphatriangle train command uses the storage functions to load specified or default configurations.

## Exposed Interfaces

-   **CLI Commands (via alphatriangle ...):**
    -   alphatriangle create [--overwrite] (Interactive, can start from preset)
    -   alphatriangle list
    -   alphatriangle show <name> (Table view)
    -   alphatriangle edit <name>
    -   alphatriangle delete <name>
    -   alphatriangle validate <name>
    -   alphatriangle default [<name>] (Show or set default)
-   **Python API (primarily internal):**
    -   SavedConfigSet (Pydantic Model)
    -   Functions in storage.py.
    -   Logic functions in commands/*.py.

## Dependencies

-   **[alphatriangle.config](../config/README.md)**: ModelConfig, TrainConfig, AlphaTriangleMCTSConfig, RunContext, APP_NAME.
-   **[alphatriangle.presets](../presets/README.md)**: PRESET_DEFINITIONS.
-   **trianglengin**: EnvConfig.
-   **trieye**: TrieyeConfig.
-   **pydantic**: For defining schemas (SavedConfigSet, Metadata).
-   **typer**: Used by the command logic functions for type hints and potentially error handling (typer.Exit).
-   **rich**: Used by command logic for console output (tables, panels).
-   **questionary**: Used by cli_utils for interactive prompts.
-   **Standard Libraries:** json, logging, os, shutil, subprocess, sys, pathlib, datetime.

---

**Note:** Please keep this README updated when changing the storage mechanism, the SavedConfigSet schema, or the logic within the command implementation functions.
