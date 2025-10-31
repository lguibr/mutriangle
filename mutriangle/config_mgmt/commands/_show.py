# File: mutriangle/config_mgmt/commands/_show.py
"""The 'show' command for configuration management."""

import logging
from typing import Any

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Use absolute imports for sibling modules/packages
from mutriangle.config_mgmt.cli_utils import ConfigNameArg, _confirm_config_name
from mutriangle.config_mgmt.storage import get_config_path, load_config

logger = logging.getLogger(__name__)
console = Console()


def _format_value(value: Any) -> str:
    """Formats values for display in the table."""
    if isinstance(value, list) and len(value) > 5:
        return f"[List with {len(value)} items]"
    if isinstance(value, dict) and len(value) > 5:
        return f"{{Dict with {len(value)} keys}}"
    if isinstance(value, str) and len(value) > 60:
        return f"{value[:57]}..."
    return str(value)


def _add_section_to_table(table: Table, model: BaseModel):
    """Adds fields from a Pydantic model to the Rich table."""
    try:
        # Use model_fields for Pydantic v2
        for field_name, field_info in model.model_fields.items():
            value = getattr(model, field_name, "N/A")
            description = field_info.description or "[dim]No description[/]"
            table.add_row(f"[field]{field_name}[/]", _format_value(value), description)
    except AttributeError:
        # Fallback for older Pydantic or unexpected types
        for field_name, value in model.model_dump().items():
            table.add_row(
                f"[field]{field_name}[/]", _format_value(value), "[dim]N/A[/]"
            )


def show(name: ConfigNameArg):
    """
    👀 Show the details of a specific configuration set in tables.
    """
    safe_name = _confirm_config_name(name)
    config_set = load_config(safe_name)

    if not config_set:
        # Error already logged by load_config
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Viewing Configuration: [bold cyan]{safe_name}[/]",
            title="📄 Configuration Details",
            border_style="blue",
            expand=False,
        )
    )

    # --- Metadata ---
    meta_table = Table(
        title="[bold]📊 Metadata[/]", show_header=False, box=None, padding=(0, 1)
    )
    meta_table.add_column("Key", style="dim")
    meta_table.add_column("Value")
    meta_table.add_row("Name:", config_set.metadata.name)
    meta_table.add_row("Description:", config_set.metadata.description or "[dim]N/A[/]")
    meta_table.add_row("Created:", str(config_set.metadata.created_at))
    meta_table.add_row("Modified:", str(config_set.metadata.last_modified_at))
    meta_table.add_row("Version:", config_set.metadata.mutriangle_version)
    meta_table.add_row("Platform:", config_set.metadata.platform)
    console.print(meta_table)

    # --- Config Sections ---
    sections = {
        "🌳 Environment (EnvConfig)": config_set.env_config,
        "🧠 Model (ModelConfig)": config_set.model_config_data,
        "🏋️ Training (TrainConfig)": config_set.train_config,
        "🌲 MCTS (MuTriangleMCTSConfig)": config_set.mcts_config,
        "👁️ Stats & Persistence (TrieyeConfig)": config_set.trieye_config,
    }

    for title, model_instance in sections.items():
        section_table = Table(
            title=f"[bold]{title}[/]",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        section_table.add_column("Parameter", style="cyan", no_wrap=True, min_width=25)
        section_table.add_column("Value", style="yellow", min_width=20)
        section_table.add_column("Description", style="dim")

        _add_section_to_table(section_table, model_instance)
        console.print(section_table)
        console.print("")  # Spacer

    console.print(f"💡 Full JSON saved at: [dim]{get_config_path(safe_name)}[/]")
