# File: mutriangle/config_mgmt/commands/_create.py
import logging
import sys
import time

import questionary
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from trianglengin import EnvConfig
from trieye import TrieyeConfig

from mutriangle.config import (
    APP_NAME,
    MuTriangleMCTSConfig,
    ModelConfig,
    RunContext,
    TrainConfig,
)
from mutriangle.config_mgmt.cli_utils import (
    OverwriteOption,
    _confirm_config_name,
    _interactive_config_section,
    _make_choice,
)
from mutriangle.config_mgmt.schemas import Metadata, SavedConfigSet

# Import preset definitions using absolute path
from mutriangle.presets import PRESET_DEFINITIONS

# Import storage functions from the package __init__
from . import (
    config_exists,
    get_config_path,
    save_config,
    set_default_config_marker,
)

logger = logging.getLogger(__name__)
console = Console()


def create(
    overwrite: OverwriteOption = False,
):
    """
    💾 Create a new configuration set interactively.

    Guides you through setting up EnvConfig, ModelConfig, TrainConfig,
    MuTriangleMCTSConfig, and TrieyeConfig using selection prompts.
    Optionally start from a predefined preset.
    Saves the result in the `.mutriangle_data/MuTriangle/configs` directory.
    """
    console.print(
        Panel(
            "✨ Welcome to the MuTriangle Interactive Configuration Setup! ✨\n"
            "I'll guide you through setting the parameters for your training run.\n"
            "Use [bold]arrow keys[/] and [bold]Enter[/] for selections.\n"
            "Choose 'Simple' configuration for recommended settings, 'Full' for all options, or 'Default'.",
            title="[bold green]🚀 Configuration Creation 🚀[/]",
            border_style="green",
            expand=False,
        )
    )

    while True:
        name_input = questionary.text(
            "🏷️ First, let's give this configuration set a unique name (e.g., 'baseline_fast', 'transformer_large'):"
        ).ask()
        if name_input is None:
            console.print("\nConfiguration cancelled.")
            raise typer.Exit()
        safe_name = _confirm_config_name(name_input)
        if config_exists(safe_name) and not overwrite:
            console.print(
                f"[bold red]Error:[/bold red] Configuration '[cyan]{safe_name}[/]' already exists. Use --overwrite or choose a different name."
            )
            if not questionary.confirm("Try a different name?", default=True).ask():
                raise typer.Exit()
        elif config_exists(safe_name) and overwrite:
            console.print(
                f"[yellow]⚠️ Configuration '[cyan]{safe_name}[/]' exists and will be overwritten.[/]"
            )
            break
        else:
            break

    description = questionary.text(
        "📝 Enter an optional description for this configuration:", default=""
    ).ask()
    if description is None:
        raise typer.Exit()

    try:
        start_from_preset = questionary.confirm(
            "🚀 Start from a predefined preset (toy, simple, medium, large)?",
            default=False,
        ).ask()
        if start_from_preset is None:
            raise typer.Exit()

        if start_from_preset:
            preset_name = questionary.select(
                "Select a preset to start from:",
                choices=[_make_choice(name, name) for name in PRESET_DEFINITIONS],
            ).ask()
            if preset_name is None:
                raise typer.Exit()
            try:
                base_config = PRESET_DEFINITIONS[preset_name]()
                console.print(
                    f"✅ Starting configuration based on preset: '{preset_name}'"
                )
                default_env = base_config.env_config.model_copy(deep=True)
                default_model = base_config.model_config_data.model_copy(deep=True)
                default_train = base_config.train_config.model_copy(deep=True)
                default_mcts = base_config.mcts_config.model_copy(deep=True)
                default_trieye = base_config.trieye_config.model_copy(deep=True)
                # Ensure run name matches the new config name
                default_train.RUN_NAME = f"{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}"
                default_trieye.run_name = default_train.RUN_NAME
                if hasattr(default_trieye, "persistence"):
                    default_trieye.persistence.RUN_NAME = default_train.RUN_NAME
                    default_trieye.persistence.APP_NAME = (
                        APP_NAME  # Ensure app name is correct
                    )

            except KeyError:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid preset name '{preset_name}'."
                )
                raise typer.Exit(code=1)
            except Exception as e:
                console.print(f"[bold red]Error loading preset '{preset_name}':[/] {e}")
                raise typer.Exit(code=1)
        else:
            console.print("✅ Starting configuration from code defaults.")
            default_env = EnvConfig()
            default_model = ModelConfig()
            default_train = TrainConfig()
            default_mcts = MuTriangleMCTSConfig()
            # Create a temporary RunContext to get default paths/names
            temp_run_context = RunContext.create(
                run_name=f"{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            default_train.RUN_NAME = temp_run_context.run_name
            default_trieye = TrieyeConfig(
                app_name=temp_run_context.app_name, run_name=temp_run_context.run_name
            )
            # Ensure persistence config has correct root dir based on context
            if hasattr(default_trieye, "persistence"):
                default_trieye.persistence.ROOT_DATA_DIR = str(
                    temp_run_context.data_root_dir.parent
                )

        env_config = _interactive_config_section(EnvConfig, "Environment", default_env)
        model_config = _interactive_config_section(ModelConfig, "Model", default_model)
        train_config = _interactive_config_section(
            TrainConfig, "Training", default_train
        )
        mcts_config = _interactive_config_section(
            MuTriangleMCTSConfig, "MCTS", default_mcts
        )
        trieye_config = _interactive_config_section(
            TrieyeConfig, "Statistics & Persistence (Trieye)", default_trieye
        )

        metadata = Metadata(name=safe_name, description=description or None)
        final_run_name = train_config.RUN_NAME
        trieye_config.run_name = final_run_name
        if hasattr(trieye_config, "persistence"):
            trieye_config.persistence.RUN_NAME = final_run_name
        trieye_config.app_name = APP_NAME
        final_run_context = RunContext.create(run_name=final_run_name)
        if hasattr(trieye_config, "persistence"):
            trieye_config.persistence.ROOT_DATA_DIR = str(
                final_run_context.data_root_dir.parent
            )
            # Remove MLFLOW_TRACKING_URI assignment here - let Trieye handle it
            # trieye_config.persistence.MLFLOW_TRACKING_URI = final_run_context.mlflow_tracking_uri

        final_config_set = SavedConfigSet(
            metadata=metadata,
            env_config=env_config,
            model_config_data=model_config,
            train_config=train_config,
            mcts_config=mcts_config,
            trieye_config=trieye_config,
        )

        try:
            SavedConfigSet.model_validate(final_config_set.model_dump())
            console.print(
                "\n[green]✅ Final configuration bundle validated successfully.[/]"
            )
        except ValidationError as e:
            console.print(
                "\n[bold red]Error:[/bold red] Final configuration bundle failed validation:"
            )
            console.print(e)
            console.print("Saving cancelled.")
            raise typer.Exit(code=1)

        if save_config(final_config_set, overwrite=overwrite):
            console.print(
                f"[green]💾 Configuration '[cyan]{safe_name}[/]' saved successfully.[/]"
            )
            config_path = get_config_path(safe_name)
            console.print(f"   Saved to: [dim]{config_path}[/]")

            set_as_default = questionary.confirm(
                f"\n❓ Set '[cyan]{safe_name}[/]' as the default configuration for training runs?",
                default=False,
            ).ask()
            if set_as_default:
                if set_default_config_marker(safe_name):
                    console.print(
                        f"✅ '[cyan]{safe_name}[/]' is now the default configuration."
                    )
                else:
                    console.print(f"[red]Error setting '{safe_name}' as default.[/]")

            console.print("\n💡 You can load this configuration for training using:")
            console.print(f"   [bold]mutriangle train {safe_name}[/]")
            console.print("💡 You can manually edit the JSON file using:")
            console.print(f"   [bold]mutriangle edit {safe_name}[/]")
            console.print("💡 You can change the default configuration using:")
            console.print("   [bold]mutriangle default <name>[/]")
        else:
            console.print("[bold red]Error:[/bold red] Failed to save configuration.")
            raise typer.Exit(code=1)

    except typer.Exit:
        console.print("\nConfiguration creation cancelled.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during interactive config creation: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred: {e}")
        raise typer.Exit(code=1)
