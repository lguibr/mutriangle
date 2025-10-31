# File: mutriangle/config_mgmt/commands/_edit.py
"""The 'edit' command for configuration management."""

import logging

import typer
from rich.console import Console

from mutriangle.config_mgmt.cli_utils import ConfigNameArg, _confirm_config_name

# Import storage functions from the package __init__
from . import get_config_path, launch_editor, load_config, save_config

logger = logging.getLogger(__name__)
console = Console()


def edit(name: ConfigNameArg):
    """
    ✏️ Edit an existing configuration set using the default system editor ($EDITOR).

    Opens the configuration JSON file for manual editing.
    The file is validated upon editor exit. Use this for complex changes.
    """
    safe_name = _confirm_config_name(name)
    config_path = get_config_path(safe_name)
    if not config_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Configuration '[cyan]{safe_name}[/]' not found."
        )
        raise typer.Exit(code=1)

    console.print(f"Attempting to edit '[cyan]{safe_name}[/]'...")
    if launch_editor(config_path):
        console.print("Validating configuration after edit...")
        # Validate after edit
        loaded_set = load_config(safe_name)
        if loaded_set:
            # Save again to update last_modified timestamp
            save_config(loaded_set, overwrite=True)
            console.print(
                "[green]✅ Configuration validated and saved successfully after edit.[/]"
            )
        else:
            console.print(
                "[bold red]Error:[/bold red] Configuration failed validation after edit. Please fix the JSON file manually."
            )
            raise typer.Exit(code=1)
    else:
        console.print(
            "[yellow]Editor launch failed or was cancelled. No changes validated.[/]"
        )
