# File: mutriangle/config_mgmt/commands/_list.py
"""The 'list' command for configuration management."""

import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import storage functions from the package __init__
from . import get_config_dir, list_configs

logger = logging.getLogger(__name__)
console = Console()


def list_saved_configs():
    """
    📄 List all saved configuration sets.
    """
    config_dir = get_config_dir()
    console.print(
        Panel(
            f"🗂️ Stored Configurations ([dim]Location: {config_dir}[/])",
            title="[bold blue]Saved Configurations[/]",
            border_style="blue",
            expand=False,
        )
    )
    configs = list_configs()
    if not configs:
        console.print("  No configurations found. Use `mutriangle create` to make one!")
        return

    table = Table(title=None, show_header=True, header_style="bold magenta")
    # Removed Default column
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Last Modified", style="yellow")

    for name, mod_time in configs:
        mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S") if mod_time else "N/A"
        table.add_row(name, mod_time_str)  # Removed default marker

    console.print(table)
    console.print(
        "\n💡 Use `mutriangle default` to see or set the default configuration."
    )
