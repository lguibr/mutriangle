# File: mutriangle/config_mgmt/commands/_default.py
"""The 'default' command logic."""

import logging

import typer
from rich.console import Console

from mutriangle.config_mgmt.cli_utils import _confirm_config_name

# Import storage functions from the package __init__
from . import (
    config_exists,
    get_default_config_name,
    set_default_config_marker,
)

logger = logging.getLogger(__name__)
console = Console()


def show_or_set_default(
    name: str | None = typer.Argument(
        None,
        help="Name of the config to set as default. If omitted, shows the current default.",
    ),
):
    """
    ⭐ Show or set the default configuration for training runs.

    If NAME is provided, sets that configuration as the default.
    If NAME is omitted, displays the currently set default configuration.
    The default is used by `mutriangle train` when no specific config is given.
    """
    if name:
        # Set default
        safe_name = _confirm_config_name(name)
        if not config_exists(safe_name):
            console.print(
                f"[bold red]❌ Error:[/bold red] Configuration '[cyan]{safe_name}[/]' not found."
            )
            raise typer.Exit(code=1)

        if set_default_config_marker(safe_name):
            console.print(
                f"[green]✅ Configuration '[cyan]{safe_name}[/]' is now the default.[/]"
            )
        else:
            console.print(
                f"[bold red]❌ Error:[/bold red] Failed to set '[cyan]{safe_name}[/]' as default."
            )
            raise typer.Exit(code=1)
    else:
        # Show default
        default_name = get_default_config_name()
        if default_name:
            console.print(
                f"⭐ Current default configuration: [bold cyan]{default_name}[/]"
            )
            console.print(
                "   Run `mutriangle train` without arguments to use this configuration."
            )
        else:
            console.print("💡 No default configuration is currently set.")
            console.print("   Use `mutriangle default <name>` to set one, or")
            console.print(
                "   use `mutriangle train <name>` to specify a config for a run."
            )
