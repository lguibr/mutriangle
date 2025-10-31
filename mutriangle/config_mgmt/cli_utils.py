# File: mutriangle/config_mgmt/cli_utils.py
"""Utility functions and shared definitions for the config management CLI."""

import json
import logging
import sys
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import questionary
import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from trianglengin import EnvConfig
from trieye import PersistenceConfig, StatsConfig, TrieyeConfig

from mutriangle.config import (
    MuTriangleMCTSConfig,
    ModelConfig,
    RunContext,
    TrainConfig,
)

# Import choices from the new file
from .cli_choices import BOOL_CHOICES, LIST_CHOICES, LITERAL_CHOICES, NUMERIC_CHOICES

logger = logging.getLogger(__name__)
console = Console()

# --- Argument/Option Annotations ---
ConfigNameArg = Annotated[
    str,
    typer.Argument(
        ..., help="The unique name for the configuration set (e.g., 'baseline_fast')."
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite", "-o", help="Overwrite the configuration if it already exists."
    ),
]

# --- Constants ---
T = TypeVar("T", bound=BaseModel)
AUTO_WORKER_SENTINEL = -1

# Fields to display but not edit interactively
READ_ONLY_FIELDS: dict[type[BaseModel], set[str]] = {
    EnvConfig: {"ROWS", "COLS", "PLAYABLE_RANGE_PER_ROW", "NUM_SHAPE_SLOTS"},
    TrainConfig: {"DEVICE", "WORKER_DEVICE", "RUN_NAME"},
    # TrieyeConfig paths are now handled by defaults/validation via RunContext
    TrieyeConfig: {"app_name", "run_name"},
    # Remove MLFLOW_TRACKING_URI as it's derived
    PersistenceConfig: {"ROOT_DATA_DIR", "RUN_NAME", "APP_NAME", "MLFLOW_TRACKING_URI"},
    StatsConfig: set(),
    ModelConfig: {"OTHER_NN_INPUT_FEATURES_DIM", "GRID_INPUT_CHANNELS"},
}

# Fields whose defaults might depend on others (handle after initial prompts)
DEPENDENT_FIELDS: dict[type[BaseModel], set[str]] = {
    TrainConfig: {"LR_SCHEDULER_T_MAX", "PER_BETA_ANNEAL_STEPS"},
}

# Fields prompted in "Simple" configuration mode
SIMPLE_FIELDS: dict[type[BaseModel], set[str]] = {
    EnvConfig: set(),
    ModelConfig: {
        "NUM_RESIDUAL_BLOCKS",
        "USE_TRANSFORMER",
        "TRANSFORMER_LAYERS",
        "TRANSFORMER_HEADS",
    },
    TrainConfig: {
        "MAX_TRAINING_STEPS",
        "NUM_SELF_PLAY_WORKERS",
        "BATCH_SIZE",
        "BUFFER_CAPACITY",
        "MIN_BUFFER_SIZE_TO_TRAIN",
        "LEARNING_RATE",
        "N_STEP_RETURNS",
        "USE_PER",
        "COMPILE_MODEL",
    },
    MuTriangleMCTSConfig: {"max_simulations", "cpuct", "mcts_batch_size"},
    TrieyeConfig: {"stats", "persistence"},
    # Use Trieye defaults for StatsConfig in simple mode
    StatsConfig: set(),
    PersistenceConfig: {
        "CHECKPOINT_SAVE_FREQ_STEPS",
        "BUFFER_SAVE_FREQ_STEPS",
        "SAVE_BUFFER",
        "AUTO_RESUME_LATEST",
    },
}

CNN_FIELD_NAMES: set[str] = {
    "CONV_FILTERS",
    "CONV_KERNEL_SIZES",
    "CONV_STRIDES",
    "CONV_PADDING",
    "RESIDUAL_BLOCK_FILTERS",
}


# --- Guided Choices Generation ---


def _make_choice(title: str, value: Any, checked: bool = False) -> questionary.Choice:
    """Helper to create questionary.Choice objects."""
    display_title = title
    if value == AUTO_WORKER_SENTINEL:
        display_title = "Auto (Detect Cores)"
    elif isinstance(value, list | dict):
        try:
            display_title = f"{title}: {json.dumps(value)}"
        except TypeError:
            display_title = f"{title}: <non-serializable>"
    return questionary.Choice(title=display_title, value=value, checked=checked)


def _generate_choices(
    current_value: Any, options: list[Any]
) -> list[questionary.Choice]:
    """Generates questionary choices from a list of option values."""
    choices = []
    has_current = False
    for opt_val in options:
        title = "None" if opt_val is None else str(opt_val)
        if opt_val == AUTO_WORKER_SENTINEL:
            title = "Auto (Detect Cores)"

        is_current = opt_val == current_value
        choices.append(_make_choice(title, opt_val, checked=is_current))
        if is_current:
            has_current = True

    if not has_current and current_value is not None:
        title = f"Current/Custom: {current_value}"
        if current_value == AUTO_WORKER_SENTINEL:
            title = "Current/Custom: Auto (Detect Cores)"
        choices.append(_make_choice(title, current_value, checked=True))

    return choices


# --- Helper Functions ---


def _confirm_config_name(name: str) -> str:
    """Validates and potentially modifies config name."""
    safe_name = name.replace(" ", "_").strip().lower()
    if not safe_name:
        console.print("[bold red]Error:[/bold red] Configuration name cannot be empty.")
        raise typer.Exit(code=1)
    if safe_name != name:
        console.print(
            f"[yellow]⚠️ Warning:[/yellow] Name normalized to '[cyan]{safe_name}[/]'."
        )
    return safe_name


def _ask_select(
    message: str | Text,
    choices: list[questionary.Choice],
    default: Any = None,
    **kwargs,
) -> Any:
    """Wrapper for questionary.select to handle cancellation and find default."""
    default_choice = None
    if default is not None:
        for choice in choices:
            if choice.value == default:
                default_choice = choice
                break
        if default_choice is None:
            logger.warning(
                f"Default value {default} not found in choices for '{message}'. No default will be pre-selected."
            )

    message_str = str(message)

    result = questionary.select(
        message_str, choices=choices, default=default_choice, **kwargs
    ).ask()
    if result is None:
        console.print("\nConfiguration cancelled by user.")
        sys.exit(0)
    return result


def _interactive_config_section(
    model_cls: type[T], section_name: str, defaults: T
) -> T:
    """Interactively configure fields for a Pydantic model section using Questionary select."""
    console.print(
        Panel(
            f"⚙️ Configuring: [bold cyan]{section_name}[/]",
            border_style="blue",
            expand=False,
        )
    )
    read_only = READ_ONLY_FIELDS.get(model_cls, set())
    dependent = DEPENDENT_FIELDS.get(model_cls, set())
    simple_mode_fields = SIMPLE_FIELDS.get(model_cls, set())

    if read_only:
        console.print("\n[bold dim]ℹ️ Informational Parameters (Not Editable Here):[/]")
        for field_name in sorted(read_only):
            if field_name in model_cls.model_fields:
                field_info = model_cls.model_fields[field_name]
                value = getattr(defaults, field_name, "N/A")
                desc = field_info.description or "Core parameter"
                console.print(f"  - [dim]{field_name}: {value} ({desc})[/]")
        console.print("")

    config_mode = _ask_select(
        f"Configure [bold]{section_name}[/] settings?",
        choices=[
            _make_choice("Simple (Recommended)", "Simple"),
            _make_choice("Full (Advanced)", "Full"),
            _make_choice("Use Defaults", "Default"),
        ],
        default="Simple",
        instruction=" (Choose level of detail)",
    )

    if config_mode == "Default":
        console.print(f"✅ Using default settings for {section_name}.")
        try:
            return model_cls(**defaults.model_dump())
        except ValidationError as e:
            console.print(
                f"[bold red]Error validating default {section_name} configuration:[/]"
            )
            console.print(e)
            raise typer.Exit(code=1)

    custom_values = defaults.model_dump()
    processed_fields = set()

    if model_cls is TrieyeConfig:
        console.print("\n[bold]Sub-Sections:[/]")
        proposed_run_name = "default_run"
        if hasattr(defaults, "run_name") and isinstance(
            getattr(defaults, "run_name", None), str
        ):
            proposed_run_name = defaults.run_name
        else:
            logger.warning(
                f"Could not determine proposed run_name from defaults ({type(defaults)}). Using 'default_run'."
            )

        temp_run_context = RunContext.create(run_name=proposed_run_name)

        stats_defaults = getattr(defaults, "stats", StatsConfig())
        persistence_defaults = getattr(
            defaults,
            "persistence",
            PersistenceConfig(
                APP_NAME=temp_run_context.app_name,
                # Pass parent dir, Trieye derives full path
                ROOT_DATA_DIR=str(temp_run_context.data_root_dir.parent),
                RUN_NAME=temp_run_context.run_name,
            ),
        )

        stats_config = _interactive_config_section(
            StatsConfig, "Statistics", stats_defaults
        )
        persistence_config = _interactive_config_section(
            PersistenceConfig, "Persistence", persistence_defaults
        )

        custom_values["stats"] = stats_config.model_dump()
        custom_values["persistence"] = persistence_config.model_dump()
        processed_fields.update({"stats", "persistence"})

    if model_cls is ModelConfig:
        console.print(
            "\n[dim]ℹ️ CNN/Residual layers use defaults. Edit JSON manually for custom architectures.[/]"
        )
        processed_fields.update(CNN_FIELD_NAMES)

    console.print("\n[bold]Parameters:[/]")
    fields_to_process = sorted(
        [
            f
            for f in model_cls.model_fields
            if f not in read_only and f not in dependent and f not in processed_fields
        ]
    )

    for field_name in fields_to_process:
        if config_mode == "Simple" and field_name not in simple_mode_fields:
            continue

        field_info = model_cls.model_fields[field_name]
        current_value = custom_values.get(field_name)
        field_type = field_info.annotation
        description = field_info.description or "No description"
        origin_type = get_origin(field_type)
        type_args = get_args(field_type)

        prompt_text = Text("Select ")
        prompt_text.append(field_name, style="cyan")
        desc_short = (
            (description[:75] + "...") if len(description) > 78 else description
        )
        prompt_text.append(f" ({desc_short})", style="dim")

        choices: list[questionary.Choice] = []
        options: list[Any] | None = None

        try:
            is_optional = origin_type is Union and type(None) in type_args
            base_type: type[Any] | None = None

            if is_optional:
                non_none_args = [t for t in type_args if t is not type(None)]
                if len(non_none_args) == 1:
                    potential_base_type = non_none_args[0]
                    if (
                        isinstance(potential_base_type, type)
                        or get_origin(potential_base_type) is not None
                    ):
                        base_type = potential_base_type
                    else:
                        logger.warning(
                            f"Could not resolve non-None type for Optional field {field_name}: {potential_base_type}. Skipping."
                        )
                        continue
                else:
                    logger.warning(
                        f"Complex Optional type for field {field_name}: {field_type}. Skipping."
                    )
                    continue
            elif isinstance(field_type, type) or get_origin(field_type) is not None:
                base_type = field_type
            else:
                logger.warning(
                    f"Field {field_name} has non-standard type annotation: {field_type}. Skipping."
                )
                continue

            if base_type is not None:
                origin_type = get_origin(base_type)

            current_type_to_check = base_type if base_type is not None else field_type

            if current_type_to_check is bool:
                options = cast(
                    "list[Any] | None", BOOL_CHOICES.get(field_name, [True, False])
                )
            elif origin_type is Literal:
                options = cast(
                    "list[Any] | None",
                    LITERAL_CHOICES.get(
                        field_name,
                        [str(arg) for arg in get_args(current_type_to_check)],
                    ),
                )
            elif origin_type is list:
                options = cast("list[Any] | None", LIST_CHOICES.get(field_name))
            elif current_type_to_check is int or current_type_to_check is float:
                options = cast("list[Any] | None", NUMERIC_CHOICES.get(field_name))

            if is_optional and options is not None and None not in options:
                options = [None] + options

            if options:
                choices = _generate_choices(current_value, options)
            else:
                console.print(
                    f"  [dim]Skipping field '{field_name}' - no predefined choices found for type {field_type}.[/]"
                )
                continue

            if not choices:
                console.print(
                    f"  [dim]Skipping field '{field_name}' - could not generate choices.[/]"
                )
                continue

            user_value = _ask_select(
                prompt_text,
                choices=choices,
                default=current_value,
                use_shortcuts=True,
                instruction=" (Use arrows or number keys, Enter)",
            )
            custom_values[field_name] = user_value

        except typer.Abort:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error prompting for {field_name}: {e}", exc_info=True
            )
            console.print(
                f"  [red]An unexpected error occurred ({e}). Using default value: {current_value}[/]"
            )
            custom_values[field_name] = current_value

    if dependent:
        console.print("\n[bold dim]ℹ️ Setting dependent fields based on choices...[/]")
        try:
            temp_model = model_cls(**custom_values)
            for field_name in dependent:
                if field_name in temp_model.model_fields:
                    updated_value = getattr(temp_model, field_name)
                    if custom_values.get(field_name) != updated_value:
                        console.print(
                            f"  - [dim]Updated [field]{field_name}[/] to: {updated_value}[/]"
                        )
                        custom_values[field_name] = updated_value
        except ValidationError as e:
            console.print(
                f"[bold red]Error validating dependent fields for {section_name}:[/]"
            )
            console.print(e)

    try:
        validated_model = model_cls(**custom_values)
        console.print(f"✅ Settings for [bold]{section_name}[/] configured.")
        return validated_model
    except ValidationError as e:
        console.print(f"[bold red]Error validating {section_name} configuration:[/]")
        console.print(e)
        console.print(
            "Using default settings for this section due to validation errors."
        )
        return model_cls(**defaults.model_dump())
