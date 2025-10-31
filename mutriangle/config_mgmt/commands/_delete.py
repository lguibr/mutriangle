# File: mutriangle/config_mgmt/commands/_delete.py
"""The 'delete' command for configuration management."""

import logging

import typer
from rich.console import Console
from rich.prompt import Confirm

from mutriangle.config_mgmt.cli_utils import ConfigNameArg, _confirm_config_name

# Import storage functions from the package __init__
from . import config_exists, delete_config, get_default_config_name

logger = logging.getLogger(__name__)
console = Console()


def delete_saved_config(name: ConfigNameArg):
    """
    🗑️ Delete a saved configuration set.
    """
    safe_name = _confirm_config_name(name)
    if not config_exists(safe_name):
        console.print(
            f"[bold red]Error:[/bold red] Configuration '[cyan]{safe_name}[/]' not found."
        )
        raise typer.Exit(code=1)

    is_default = get_default_config_name() == safe_name
    confirm_message = (
        f"Are you sure you want to delete configuration '[cyan]{safe_name}[/]'?"
    )
    if is_default:
        confirm_message += " [bold yellow](This is the current default!)[/]"

    if Confirm.ask(confirm_message, default=False):
        if delete_config(safe_name):  # delete_config now handles marker removal
            console.print(
                f"[green]✅ Configuration '[cyan]{safe_name}[/]' deleted successfully.[/]"
            )
            if is_default:
                console.print("[yellow]⚠️ Default configuration marker removed.[/]")
        else:
            console.print("[bold red]Error:[/bold red] Failed to delete configuration.")
            raise typer.Exit(code=1)
    else:
        console.print("Deletion cancelled.")
