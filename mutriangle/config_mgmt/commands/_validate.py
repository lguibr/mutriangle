# File: mutriangle/config_mgmt/commands/_validate.py
"""The 'validate' command for configuration management."""

import logging

import typer
from rich.console import Console

# Import storage functions from the package __init__
from mutriangle.config_mgmt.cli_utils import ConfigNameArg, _confirm_config_name

from . import load_config

logger = logging.getLogger(__name__)
console = Console()


def validate(name: ConfigNameArg):
    """
    ✔️ Validate the structure and types of a saved configuration file.
    """
    safe_name = _confirm_config_name(name)
    console.print(f"Validating configuration '[cyan]{safe_name}[/]'...")
    loaded_set = load_config(safe_name)
    if loaded_set:
        console.print("[green]✅ Configuration validated successfully.[/]")
    else:
        # load_config already prints errors
        console.print("[bold red]Validation failed. See errors above.[/]")
        raise typer.Exit(code=1)
