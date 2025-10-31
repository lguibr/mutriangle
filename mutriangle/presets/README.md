# File: alphatriangle/presets/README.md
# Preset Configurations Module (`alphatriangle.presets`)

This module provides predefined configuration sets for `alphatriangle`. These presets offer starting points for different use cases, from quick testing to serious training runs.

## Architecture

Instead of storing presets as static JSON files within the package, they are now defined as Python functions within [`definitions.py`](definitions.py). Each function (e.g., `get_toy_config()`) returns a fully instantiated `SavedConfigSet` object, ensuring that the presets are always valid according to the latest Pydantic models.

This approach offers several advantages:

*   **Validation:** Presets are validated against the Pydantic models at definition time.
*   **Maintainability:** Easier to update presets when configuration models change.
*   **Testability:** Presets can be imported and used directly in tests.
*   **Consistency:** Reduces the risk of errors from outdated or malformed JSON files within the package source.

## Available Presets

The following presets are defined in [`definitions.py`](definitions.py):

*   **`toy`**: (`get_toy_config()`) Extremely minimal settings designed for rapid testing of the end-to-end pipeline. Starts training very quickly, finishes quickly, enables profiling and all features. **Not intended for actual learning.**
*   **`simple`**: (`get_simple_config()`) A lightweight configuration suitable for faster runs on less powerful hardware. Disables the Transformer block for speed but keeps PER enabled. Capable of learning but optimized for speed over peak performance.
*   **`medium`**: (`get_medium_config()`) A balanced configuration suitable for standard training runs. Enables the Transformer and PER with moderate capacity, batch size, and MCTS simulations. Offers good learning potential.
*   **`large`**: (`get_large_config()`) A robust configuration designed for achieving high performance, assuming sufficient compute resources. Uses large capacities, batch sizes, and high MCTS simulations. Enables all features for maximum learning potential, but will be the slowest.

## How Presets are Used

1.  **First Run:** When you run `alphatriangle` for the first time, the `ensure_presets_saved` function (in `alphatriangle.config_mgmt.storage`) is called.
2.  **Instantiation & Saving:** This function imports the preset definitions (e.g., `get_toy_config`), calls them to get the `SavedConfigSet` objects, and saves each one as a JSON file (e.g., `toy.config.json`) into your local configuration directory (`.alphatriangle_data/configs/`) if it doesn't already exist.
3.  **Default:** If no default configuration is set, `simple` is automatically set as the default after the presets are saved.
4.  **Usage:** You can then use these saved JSON configurations like any other configuration created via `alphatriangle create`:
    *   `alphatriangle train simple`
    *   `alphatriangle show medium`
    *   `alphatriangle edit large` (opens the saved JSON file)
    *   `alphatriangle default toy`

This system combines the robustness of code-defined presets with the user-friendliness and editability of JSON files stored locally in the user's data directory. The package itself does not ship static preset JSON files.