# File: mutriangle/config_mgmt/commands/__init__.py
"""Exports CLI command logic functions for config management."""

# Import command logic
# Re-export necessary storage functions for use within command modules
# This helps mypy resolve imports correctly within this subpackage.
from ..storage import (
    DATA_ROOT_DIR_NAME,
    config_exists,
    delete_config,
    ensure_presets_saved,  # Corrected function name
    get_config_dir,
    get_config_path,
    get_default_config_name,
    launch_editor,
    list_configs,
    load_config,
    save_config,
    set_default_config_marker,
)
from ._create import create
from ._default import show_or_set_default
from ._delete import delete_saved_config
from ._edit import edit
from ._list import list_saved_configs
from ._show import show
from ._validate import validate

__all__ = [
    # Command functions
    "create",
    "list_saved_configs",
    "show",
    "edit",
    "delete_saved_config",
    "validate",
    "show_or_set_default",
    # Re-exported storage functions
    "DATA_ROOT_DIR_NAME",
    "config_exists",
    "delete_config",
    "ensure_presets_saved",  # Corrected function name
    "get_config_dir",
    "get_config_path",
    "get_default_config_name",
    "launch_editor",
    "list_configs",
    "load_config",
    "save_config",
    "set_default_config_marker",
]
