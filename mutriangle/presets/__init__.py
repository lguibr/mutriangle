# File: mutriangle/presets/__init__.py
"""
Presets Module.

Provides predefined configuration sets for MuTriangle.
"""

from .definitions import (
    PRESET_DEFINITIONS,
    get_large_config,
    get_medium_config,
    get_simple_config,
    get_toy_config,
)

__all__ = [
    "get_toy_config",
    "get_simple_config",
    "get_medium_config",
    "get_large_config",
    "PRESET_DEFINITIONS",
]
