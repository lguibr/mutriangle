# File: alphatriangle/config/README.md
# Configuration Module (`alphatriangle.config`)

## Purpose and Architecture

This module centralizes configuration parameters for the AlphaTriangle agent itself, *excluding* statistics logging and data persistence which are now handled by the `trieye` library. It uses separate **Pydantic models** for different aspects of the agent (model, training loop, MCTS) to promote modularity, clarity, and automatic validation.

**Core environment configuration (`EnvConfig`) is imported directly from the `trianglengin` library.**
**Statistics and Persistence configuration (`StatsConfig`, `PersistenceConfig`) are defined and managed within the `trieye` library via `TrieyeConfig`.**
**Run-specific context, especially paths, is managed by the `RunContext` model.**

-   **Modularity:** Separating configurations makes it easier to manage parameters for different components.
-   **Type Safety & Validation:** Using Pydantic models (`BaseModel`) provides strong type hinting, automatic parsing, and validation of configuration values based on defined types and constraints (e.g., `Field(gt=0)`).
-   **Run Context ([`run_context.py`](run_context.py)):** The `RunContext` model defines the application name, run name, and root data directory for a specific training run. It uses computed properties to derive all necessary sub-paths (configs, MLflow, runs, checkpoints, etc.), ensuring consistency. An instance is created at the start of a training run and passed down to components like `TrieyeConfig`.
-   **Validation Script:** The [`validation.py`](validation.py) script instantiates the AlphaTriangle-specific configuration models (including importing and validating `trianglengin.EnvConfig`), triggering Pydantic's validation, and prints a summary. **Note:** It does *not* validate `TrieyeConfig` directly; `trieye` handles its own validation upon actor initialization using paths provided by the `RunContext`.
-   **Dynamic Defaults:** Some configurations, like `RUN_NAME` in `TrainConfig`, use `default_factory` for dynamic defaults (e.g., timestamp). This default is often overridden by the `RunContext`.
-   **Tuned Defaults:** The default values in `TrainConfig` and `ModelConfig` are tuned for substantial learning runs. `AlphaTriangleMCTSConfig` defaults to 64 simulations.

## Exposed Interfaces

-   **Pydantic Models:**
    -   `EnvConfig` (Imported from `trianglengin`): Environment parameters (grid size, shapes, rewards like `REWARD_PER_CLEARED_TRIANGLE`, `REWARD_PER_STEP_ALIVE`, `PENALTY_GAME_OVER`, etc.).
    -   [`ModelConfig`](model_config.py): Neural network architecture parameters.
    -   [`TrainConfig`](train_config.py): Training loop hyperparameters (batch size, learning rate, workers, PER settings, etc.).
    -   [`AlphaTriangleMCTSConfig`](mcts_config.py): MCTS parameters (simulations, exploration constants, temperature).
    -   [`RunContext`](run_context.py): Manages run-specific identifiers and paths.
-   **Constants:**
    -   [`APP_NAME`](app_config.py): The name of the application (used by `trieye` and `RunContext`).
-   **Functions:**
    -   `print_config_info_and_validate(mcts_config_instance: AlphaTriangleMCTSConfig | None)`: Validates and prints a summary of AlphaTriangle-specific configurations.

## Dependencies

This module primarily defines configurations and relies heavily on **Pydantic**.

-   **`pydantic`**: The core library used for defining models and validation.
-   **`trianglengin`**: Imports `EnvConfig`.
-   **`trimcts`**: Used by `AlphaTriangleMCTSConfig` (implicitly, as it mirrors the structure).
-   **Standard Libraries:** `typing`, `time`, `os`, `logging`, `pathlib`.

---

**Note:** Please keep this README updated when adding, removing, or significantly modifying configuration parameters or the structure of the Pydantic models within this module, including the `RunContext`.

