# File: mutriangle/config_mgmt/storage.py
# File: mutriangle/config_mgmt/storage.py
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

# Import APP_NAME and DATA_ROOT_DIR_NAME from the correct location
from mutriangle.config.app_config import APP_NAME, DATA_ROOT_DIR_NAME

# Import preset definitions
from mutriangle.presets import PRESET_DEFINITIONS

# Import the schema
from .schemas import SavedConfigSet

logger = logging.getLogger(__name__)
console = Console()

CONFIG_SUBDIR_NAME: str = "configs"
DEFAULT_MARKER_FILENAME: str = ".default_config_marker"


def get_app_data_root_dir(base_dir: Path | None = None) -> Path:
    """Gets the absolute path to the application-specific data root directory."""
    if base_dir is None:
        base_dir = Path.cwd()
    # Use the imported constants to build the app-specific path
    return (base_dir / DATA_ROOT_DIR_NAME / APP_NAME).resolve()


def get_config_dir(base_dir: Path | None = None) -> Path:
    """Gets the absolute path to the configuration storage directory."""
    # Use the app-specific root directory
    app_data_root = get_app_data_root_dir(base_dir)
    config_dir = app_data_root / CONFIG_SUBDIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path(name: str, base_dir: Path | None = None) -> Path:
    """Gets the full path for a configuration file."""
    config_dir = get_config_dir(base_dir)
    return config_dir / f"{name}.config.json"


def get_default_marker_path(base_dir: Path | None = None) -> Path:
    """Gets the path to the default configuration marker file."""
    config_dir = get_config_dir(base_dir)
    return config_dir / DEFAULT_MARKER_FILENAME


def config_exists(name: str, base_dir: Path | None = None) -> bool:
    """Checks if a configuration file exists."""
    return get_config_path(name, base_dir).exists()


def save_config(config_set: SavedConfigSet, overwrite: bool = False) -> bool:
    """Saves a configuration set to a JSON file."""
    config_path = get_config_path(config_set.metadata.name)
    if config_path.exists() and not overwrite:
        logger.error(
            f"Configuration file {config_path} already exists. Use overwrite=True."
        )
        return False
    try:
        config_set.update_last_modified()
        # Use model_dump_json for Pydantic v2
        json_data = config_set.model_dump_json(indent=2)
        config_path.write_text(json_data, encoding="utf-8")
        logger.info(
            f"Configuration '{config_set.metadata.name}' saved to {config_path}"
        )
        return True
    except (OSError, TypeError, ValidationError) as e:
        logger.error(
            f"Error saving configuration '{config_set.metadata.name}' to {config_path}: {e}",
            exc_info=True,
        )
        return False


def load_config(name: str) -> SavedConfigSet | None:
    """Loads a configuration set from a JSON file."""
    config_path = get_config_path(name)
    if not config_path.exists():
        # Try loading from presets if not found locally (e.g., for tests)
        if name in PRESET_DEFINITIONS:
            logger.info(
                f"Config '{name}' not found locally, loading from preset definition."
            )
            try:
                return PRESET_DEFINITIONS[name]()
            except Exception as e:
                logger.error(f"Error loading preset definition '{name}': {e}")
                return None
        else:
            logger.error(f"Configuration file not found: {config_path}")
            return None
    try:
        json_data = config_path.read_text(encoding="utf-8")
        # Use model_validate_json for Pydantic v2
        config_set = SavedConfigSet.model_validate_json(json_data)
        logger.info(f"Configuration '{name}' loaded from {config_path}")
        return config_set
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        logger.error(
            f"Error loading or validating configuration '{name}' from {config_path}: {e}",
            exc_info=True,
        )
        console.print(
            f"[bold red]Error loading or validating configuration '{name}':[/] {e}"
        )
        return None


def list_configs() -> list[tuple[str, datetime.datetime | None]]:
    """Lists all saved configuration files and their modification times."""
    config_dir = get_config_dir()
    configs = []
    try:
        for item in config_dir.glob("*.config.json"):
            if item.is_file():
                name = item.stem.replace(".config", "")
                try:
                    mod_time = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                except OSError:
                    mod_time = None
                configs.append((name, mod_time))
        configs.sort()
    except OSError as e:
        logger.error(f"Error listing configurations in {config_dir}: {e}")
    return configs


def delete_config(name: str) -> bool:
    """Deletes a configuration file and removes the default marker if it matches."""
    config_path = get_config_path(name)
    try:
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Deleted configuration file: {config_path}")
            # Check if this was the default and remove marker
            if get_default_config_name() == name:
                marker_path = get_default_marker_path()
                if marker_path.exists():
                    marker_path.unlink()
                    logger.info(f"Removed default marker for deleted config '{name}'.")
            return True
        else:
            logger.warning(f"Configuration file not found for deletion: {config_path}")
            return False
    except OSError as e:
        logger.error(f"Error deleting configuration file {config_path}: {e}")
        return False


def get_default_config_name() -> str | None:
    """Reads the name of the default configuration from the marker file."""
    marker_path = get_default_marker_path()
    try:
        if marker_path.exists():
            return marker_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error(f"Error reading default config marker {marker_path}: {e}")
    return None


def set_default_config_marker(name: str) -> bool:
    """Writes the name of the default configuration to the marker file."""
    marker_path = get_default_marker_path()
    try:
        marker_path.write_text(name.strip(), encoding="utf-8")
        logger.info(f"Set default configuration marker to: {name}")
        return True
    except OSError as e:
        logger.error(f"Error writing default config marker {marker_path}: {e}")
        return False


def ensure_presets_saved(set_default_if_missing: bool = True) -> None:
    """
    Loads preset definitions and saves them as JSON files in the user's
    config directory if they don't already exist.
    """
    get_config_dir()  # Ensure the correct directory is created/used
    saved_any = False

    for name, preset_func in PRESET_DEFINITIONS.items():
        dest_path = get_config_path(name)
        if not dest_path.exists():
            try:
                preset_config_set = preset_func()
                if save_config(preset_config_set, overwrite=False):
                    logger.info(f"Saved preset '{name}' definition to {dest_path}")
                    saved_any = True
                else:
                    logger.error(f"Failed to save preset '{name}' to {dest_path}")
            except Exception as e:
                logger.error(
                    f"Error instantiating or saving preset '{name}': {e}",
                    exc_info=True,
                )

    # Set 'simple' as default only if we saved presets AND no default exists
    if (
        saved_any
        and set_default_if_missing
        and get_default_config_name() is None
        and config_exists("simple")
    ):
        logger.info("Setting 'simple' preset as the default configuration.")
        set_default_config_marker("simple")


def launch_editor(filepath: Path) -> bool:
    """Launches the default system editor for the given file."""
    editor = os.environ.get("EDITOR")
    if not editor:
        # Fallback for different OS
        if sys.platform.startswith("win"):
            editor = "notepad"
        elif sys.platform.startswith("darwin"):
            editor = "open"  # Opens with default app, usually TextEdit
        else:  # Linux/other Unix
            # Try common editors
            for cmd in ["nvim", "vim", "nano", "gedit", "kate"]:
                if shutil.which(cmd):
                    editor = cmd
                    break
            else:
                logger.error(
                    "No $EDITOR set and could not find a default editor (vim, nano, etc.)."
                )
                console.print(
                    "[bold red]Error:[/bold red] Could not find a text editor. Please set the $EDITOR environment variable."
                )
                return False

    logger.info(f"Launching editor '{editor}' for file: {filepath}")
    try:
        # Use subprocess.run for better control and error handling
        process = subprocess.run([editor, str(filepath)], check=False)
        if process.returncode == 0:
            logger.info(f"Editor exited successfully for {filepath}.")
            return True
        else:
            logger.error(
                f"Editor command '{editor} {filepath}' exited with code {process.returncode}."
            )
            console.print(
                f"[bold red]Error:[/bold red] Editor command failed (code {process.returncode})."
            )
            return False
    except FileNotFoundError:
        logger.error(f"Editor command '{editor}' not found.")
        console.print(
            f"[bold red]Error:[/bold red] Editor command '[cyan]{editor}[/]' not found. Check your $EDITOR variable or PATH."
        )
        return False
    except Exception as e:
        logger.error(f"Error launching editor '{editor}' for {filepath}: {e}")
        console.print(f"[bold red]Error launching editor:[/bold red] {e}")
        return False
