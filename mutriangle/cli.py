# File: mutriangle/cli.py
import logging
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata  # For version
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError  # Import for validation error handling
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Import Trieye config - MAKE SURE THESE ARE THE CORRECT CLASSES FROM TRIEYE
from trieye import PersistenceConfig, TrieyeConfig

# Import mutriangle specific configs and runner
from mutriangle.config import (
    APP_NAME,
    RunContext,
    TrainConfig,
)
from mutriangle.config.app_config import DATA_ROOT_DIR_NAME  # Import constant

# Import command logic functions from config_mgmt
from mutriangle.config_mgmt.commands import (
    create as create_config_logic,
)
from mutriangle.config_mgmt.commands import (
    delete_saved_config as delete_config_logic,
)
from mutriangle.config_mgmt.commands import edit as edit_config_logic
from mutriangle.config_mgmt.commands import list_saved_configs as list_configs_logic
from mutriangle.config_mgmt.commands import show as show_config_logic
from mutriangle.config_mgmt.commands import (
    show_or_set_default as default_config_logic,
)
from mutriangle.config_mgmt.commands import validate as validate_config_logic

# Import storage utils using absolute path
from mutriangle.config_mgmt.storage import (
    ensure_presets_saved,
    get_default_config_name,
)
from mutriangle.config_mgmt.storage import (
    load_config as load_saved_config,
)
from mutriangle.logging_config import setup_logging
from mutriangle.training.runner import run_training

# Initialize Rich Console
console = Console()
logger = logging.getLogger(__name__)

# --- Instantiate the main Typer App FIRST ---
app = typer.Typer(
    name="mutriangle",
    add_completion=True,  # Enable completion
    rich_markup_mode="markdown",
    pretty_exceptions_show_locals=False,
    help="▲ MuTriangle ▲ - MuZero agent for a triangle puzzle game.",
)


# --- Helper Function: Run External UI ---
def _run_external_ui(
    command_name: str, command_args: list[str], ui_name: str, default_url: str
):
    """Runs an external UI command and handles common errors."""
    logger = logging.getLogger(__name__)
    executable = shutil.which(command_name)
    if not executable:
        console.print(
            f"[bold red]❌ Error:[/bold red] Could not find '{command_name}'. Is {ui_name} installed and in your PATH?"
        )
        raise typer.Exit(code=1)
    full_command = [executable] + command_args
    console.print(
        Panel(
            f"🚀 Launching [bold cyan]{ui_name}[/]...\n   ❯ Command: [dim]{' '.join(full_command)}[/]\n   ❯ Access URL (approx): [link={default_url}]{default_url}[/]",
            title="External UI",
            border_style="blue",
            expand=False,
        )
    )
    try:
        process = subprocess.run(
            full_command, check=False, capture_output=True, text=True
        )  # Capture output
        if process.returncode != 0:
            console.print(
                f"[bold red]Error:[/bold red] {ui_name} command failed (code {process.returncode})."
            )
            # Print stderr if available
            if process.stderr:
                console.print("[dim]Error output:[/]")
                console.print(f"[dim]{process.stderr.strip()}[/]")
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ Error:[/bold red] '{executable}' not found.")
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:
        console.print(f"\n[yellow]🟡 {ui_name} interrupted.[/]")
        raise typer.Exit(code=0) from None
    except Exception as e:
        console.print(f"[bold red]❌ Error launching {ui_name}:[/] {e}")
        logger.error(f"Error launching {ui_name}", exc_info=True)
        raise typer.Exit(code=1) from e


# --- Main Callback for Guided Experience ---
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show application version.",
            is_eager=True,
            callback=lambda v: v,  # Simple callback to satisfy Typer
        ),
    ] = False,
):
    """
    Run without subcommands for a guided experience (run default config or create one).
    Use subcommands for specific actions like training, monitoring, or config management.
    """
    # Ensure presets are available in the user's config dir on first run
    ensure_presets_saved(set_default_if_missing=True)

    if version:
        try:
            pkg_version = importlib_metadata.version("mutriangle")
        except importlib_metadata.PackageNotFoundError:
            pkg_version = "0.0.0-unknown (not installed)"
        console.print(f"▲ MuTriangle Version: [bold cyan]{pkg_version}[/]")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print(
            Panel(
                "👋 Welcome to MuTriangle!",
                title="[bold green]MuTriangle CLI[/]",
                border_style="green",
                expand=False,
            )
        )
        default_name = get_default_config_name()

        if default_name:
            console.print(
                f"\n⭐ Found default configuration: [bold cyan]{default_name}[/]"
            )
            if Confirm.ask(
                "🚀 Run training with the default configuration?", default=True
            ):
                # Pass empty dict for overrides when running default from guided start
                _run_training_with_config(config_name=default_name, cli_overrides={})
            else:
                console.print("\nOkay. To manage configurations, use commands like:")
                console.print("  - mutriangle list")
                console.print("  - mutriangle default <name>")
                console.print("  - mutriangle create")
                console.print("  - mutriangle train <name>")
        else:
            console.print("\n💡 No default configuration found.")
            if Confirm.ask(
                "Would you like to create a new configuration interactively?",
                default=True,
            ):
                # Call the create command logic directly
                create_config_logic(overwrite=False)
            else:
                console.print(
                    "\nOkay. You can create a configuration later using mutriangle create."
                )
                console.print(
                    f"Presets (toy, simple, medium, large) are available in {DATA_ROOT_DIR_NAME}/{APP_NAME}/configs/."
                )

        raise typer.Exit()


# --- CLI Option Annotations (Shared) ---
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEED = 42
DEFAULT_PROFILE = False

LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level", "-l", help="Set the logging level.", case_sensitive=False
    ),
]
SeedOption = Annotated[
    int, typer.Option("--seed", "-s", help="Random seed (overrides config).")
]
ProfileOption = Annotated[
    bool,
    typer.Option(
        "--profile",
        help="Enable cProfile for worker 0 (overrides config).",
        is_flag=True,
    ),
]
RunNameOption = Annotated[
    str | None, typer.Option("--run-name", help="Custom run name (overrides config).")
]
HostOption = Annotated[str, typer.Option(help="The network address to listen on.")]
PortOption = Annotated[int, typer.Option(help="The port to listen on.")]
ConfigNameArg = Annotated[
    str, typer.Argument(..., help="The unique name for the configuration set.")
]
OptionalConfigNameArg = Annotated[
    str | None,
    typer.Argument(
        help="Configuration set name. If omitted, uses the default or code defaults."
    ),
]
OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite", "-o", help="Overwrite the configuration if it already exists."
    ),
]


# --- Helper: Run Training ---
def _run_training_with_config(config_name: str | None, cli_overrides: dict[str, Any]):
    """Helper to load config and run training, applying CLI overrides."""
    log_level = cli_overrides.get("log_level", DEFAULT_LOG_LEVEL)
    seed_override = cli_overrides.get("seed")
    profile_override = cli_overrides.get("profile")
    run_name_override = cli_overrides.get("run_name")

    setup_logging(log_level)
    logger = logging.getLogger(__name__)  # Ensure logger is obtained after setup

    loaded_config_set = None
    config_source_display = "[dim]Code Defaults[/]"
    if config_name:
        loaded_config_set = load_saved_config(config_name)
        if not loaded_config_set:
            console.print(
                f"[bold red]Error:[/bold red] Failed to load configuration set '[cyan]{config_name}[/]'."
            )
            raise typer.Exit(code=1)
        config_source_display = f"[cyan]{config_name}[/]" + (
            " (Default)" if config_name == get_default_config_name() else ""
        )

    # --- Determine Base Configs ---
    if loaded_config_set:
        train_config_base = loaded_config_set.train_config
        trieye_config_base = loaded_config_set.trieye_config
        model_config_loaded = loaded_config_set.model_config_data
        mcts_config_loaded = loaded_config_set.mcts_config
    else:
        logger.warning("No configuration specified or found. Using code defaults.")
        train_config_base = TrainConfig()
        # Create default TrieyeConfig (paths will be set later by RunContext)
        trieye_config_base = TrieyeConfig(app_name=APP_NAME)
        model_config_loaded = None
        mcts_config_loaded = None

    # --- Determine Effective Seed ---
    if seed_override is not None:
        effective_seed = seed_override
        logger.info(f"Overriding RANDOM_SEED with CLI value: {effective_seed}")
    else:
        effective_seed = train_config_base.RANDOM_SEED

    # --- Determine Effective Profile Setting ---
    if profile_override is not None:
        effective_profile = profile_override
        logger.info(f"Overriding PROFILE_WORKERS with CLI value: {effective_profile}")
    else:
        effective_profile = train_config_base.PROFILE_WORKERS

    # --- Apply Overrides to TrainConfig ---
    train_config_override = train_config_base.model_copy(deep=True)
    train_config_override.RANDOM_SEED = effective_seed
    train_config_override.PROFILE_WORKERS = effective_profile

    # --- Determine Effective Run Name ---
    effective_run_name: str
    if run_name_override:
        effective_run_name = run_name_override
        logger.info(f"Using run_name from CLI override: '{effective_run_name}'")
    else:
        # Use run_name from loaded TrainConfig or its default factory
        effective_run_name = train_config_override.RUN_NAME
        logger.info(
            f"Using run_name from {'loaded' if loaded_config_set else 'default'} config: '{effective_run_name}'"
        )

    # --- Create RunContext ---
    base_dir_for_context = Path.cwd()
    run_context = RunContext.create(
        run_name=effective_run_name, base_dir=base_dir_for_context
    )
    logger.info(
        f"Run Context created. App: {run_context.app_name}, Run: {run_context.run_name}, Data Root: {run_context.data_root_dir}"
    )

    # --- Update TrainConfig run_name based on context ---
    train_config_override.RUN_NAME = run_context.run_name

    # --- Reconstruct TrieyeConfig ---
    try:
        trieye_config_override = trieye_config_base.model_copy(deep=True)
        trieye_config_override.app_name = run_context.app_name
        trieye_config_override.run_name = run_context.run_name
        if not hasattr(trieye_config_override, "persistence"):
            trieye_config_override.persistence = PersistenceConfig()
        trieye_config_override.persistence.APP_NAME = run_context.app_name
        trieye_config_override.persistence.RUN_NAME = run_context.run_name
        trieye_config_override.persistence.ROOT_DATA_DIR = str(
            run_context.data_root_dir.parent
        )
        # --- REMOVE direct MLflow URI assignment ---
        # mlflow_uri = f"file:{(run_context.data_root_dir / 'mlruns').resolve()}"
        # trieye_config_override.persistence.MLFLOW_TRACKING_URI = mlflow_uri # <-- REMOVED
        # logger.info(f"Set MLflow tracking URI to: {mlflow_uri}") # <-- REMOVED
        # --- End Removal ---

        TrieyeConfig.model_validate(trieye_config_override.model_dump())
        logger.info("Reconstructed TrieyeConfig validated successfully.")
    except ValidationError as e:
        logger.error(
            f"Validation Error during TrieyeConfig reconstruction: {e}. "
            f"Check preset file structure ('{config_name}') or Trieye defaults.",
            exc_info=True,
        )
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error(
            f"Unexpected error reconstructing TrieyeConfig: {e}", exc_info=True
        )
        raise typer.Exit(code=1) from e

    console.print(
        Panel(
            f"🚀 Starting Training Run: '[bold cyan]{run_context.run_name}[/]'\n"
            f"   Config Source: {config_source_display}\n"
            f"   App Data Root: [dim]{run_context.data_root_dir}[/]\n"
            f"   Seed: {effective_seed}, Log Level: {log_level.upper()}, Profiling: {'✅ Enabled' if effective_profile else '❌ Disabled'}",
            title="[bold green]Training Setup[/]",
            border_style="green",
            expand=False,
        )
    )

    # --- Run Training ---
    exit_code = run_training(
        run_context=run_context,
        log_level_str=log_level,
        train_config_override=train_config_override,
        trieye_config_override=trieye_config_override,
        model_config_override=model_config_loaded,
        mcts_config_override=mcts_config_loaded,
        profile=effective_profile,  # Pass the *effective* profile setting
    )

    if exit_code == 0:
        console.print(
            Panel(
                f"✅ Training run '[bold cyan]{run_context.run_name}[/]' completed successfully.",
                title="[bold green]Training Finished[/]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"❌ Training run '[bold cyan]{run_context.run_name}[/]' failed with exit code {exit_code}.",
                title="[bold red]Training Failed[/]",
                border_style="red",
            )
        )
    sys.exit(exit_code)


# --- Training Command ---
@app.command()
def train(
    config_name: OptionalConfigNameArg = None,
    # CLI overrides - Use defaults matching the constants
    log_level: LogLevelOption = DEFAULT_LOG_LEVEL,
    seed: SeedOption = DEFAULT_SEED,
    profile: ProfileOption = DEFAULT_PROFILE,
    run_name: RunNameOption = None,
):
    """
    🚀 Run the MuTriangle training pipeline.

    Loads configuration: uses <config_name> if provided, otherwise the default config
    (set via mutriangle default), or finally code defaults if neither exists.
    CLI options (--seed, --profile, --run-name) override loaded values.
    """
    # Collect overrides only if they differ from the defined defaults
    cli_overrides: dict[str, Any] = {}
    if log_level != DEFAULT_LOG_LEVEL:
        cli_overrides["log_level"] = log_level
    if seed != DEFAULT_SEED:
        cli_overrides["seed"] = seed
    if profile != DEFAULT_PROFILE:  # Explicitly check if the FLAG was passed
        cli_overrides["profile"] = profile
    if run_name is not None:
        cli_overrides["run_name"] = run_name

    # Determine which config to use (passed arg > default > None)
    config_to_use = config_name
    if not config_to_use:
        config_to_use = get_default_config_name()
        if not config_to_use:
            logger.info(
                "No config name provided and no default set. Will use code defaults."
            )

    _run_training_with_config(config_name=config_to_use, cli_overrides=cli_overrides)


# --- Monitoring Commands ---
@app.command(
    name="stats", help="📈 Launch TensorBoard UI pointing to the runs directory."
)
def stats_cmd(host: HostOption = "127.0.0.1", port: PortOption = 6006):
    setup_logging("INFO")
    # Use RunContext to get the correct base 'runs' directory for mutriangle
    try:
        # Create a temporary context just to get the path structure
        temp_context = RunContext.create(run_name="placeholder_for_paths")
        # Point TensorBoard to the parent of the specific run dirs
        tensorboard_parent_dir = temp_context.data_root_dir / "runs"
    except Exception as e:
        logger.error(
            f"Could not determine TensorBoard log directory using RunContext: {e}"
        )
        console.print(
            "[bold red]Error:[/bold red] Could not determine TensorBoard log directory."
        )
        raise typer.Exit(code=1)

    if not tensorboard_parent_dir.exists() or not any(tensorboard_parent_dir.iterdir()):
        console.print(
            f"[yellow]Warning:[/yellow] TensorBoard parent directory not found or empty: [dim]{tensorboard_parent_dir}[/]"
        )
    else:
        console.print(
            f"Pointing TensorBoard to runs data at: [dim]{tensorboard_parent_dir}[/]"
        )

    command_args = [
        "--logdir",
        str(tensorboard_parent_dir),  # Use the correct parent 'runs' dir
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        _run_external_ui(
            "tensorboard", command_args, "TensorBoard UI", f"http://{host}:{port}"
        )
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(
                f"[yellow]TensorBoard UI failed (Exit Code: {e.exit_code}). Is port {port} in use?[/]"
            )
        sys.exit(e.exit_code)


@app.command(name="artifacts", help="📊 Launch the MLflow UI for experiment tracking.")
def artifacts_cmd(host: HostOption = "127.0.0.1", port: PortOption = 5000):
    setup_logging("INFO")
    # Use RunContext to get the correct MLflow paths for mutriangle
    try:
        temp_context = RunContext.create(run_name="placeholder_for_paths")
        # Derive MLflow paths based on the app's data root
        mlflow_path = temp_context.data_root_dir / "mlruns"
        mlflow_uri = f"file:{mlflow_path.resolve()}"
    except Exception as e:
        logger.error(f"Could not determine MLflow URI/path using RunContext: {e}")
        console.print(
            "[bold red]Error:[/bold red] Could not determine MLflow URI/path."
        )
        raise typer.Exit(code=1)

    if not mlflow_path.exists() or not any(mlflow_path.iterdir()):
        console.print(
            f"[yellow]Warning:[/yellow] MLflow directory not found or empty: [dim]{mlflow_path}[/]"
        )
    else:
        console.print(f"Found MLflow data at: [dim]{mlflow_path}[/]")
    command_args = [
        "ui",
        "--backend-store-uri",
        mlflow_uri,  # Use correct URI
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        _run_external_ui("mlflow", command_args, "MLflow UI", f"http://{host}:{port}")
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(
                f"[yellow]MLflow UI failed (Exit Code: {e.exit_code}). Is port {port} in use?[/]"
            )
        sys.exit(e.exit_code)


@app.command(name="monitor", help="☀️ Provides instructions to view the Ray Dashboard.")
def monitor_cmd(host: HostOption = "127.0.0.1", port: PortOption = 8265):
    setup_logging("INFO")
    try:
        temp_context = RunContext.create(run_name="placeholder_for_paths")
        logs_base_dir_str = str(temp_context.data_root_dir / "runs")
        log_dir_name = "logs"  # Default name from Trieye
    except Exception as e:
        logger.error(
            f"Could not determine log directory structure using RunContext: {e}"
        )
        # Fallback display path (string)
        logs_base_dir_str = f"{DATA_ROOT_DIR_NAME}/{APP_NAME}/runs"
        log_dir_name = "logs"  # Fallback display name

    console.print(
        Panel(
            f"💡 To view the Ray Dashboard:\n\n1. Run [bold]mutriangle train[/].\n2. Check the console output or logs in {logs_base_dir_str}/<run_name>/{log_dir_name}/.\n3. Look for a line like: '[bold cyan]Ray Dashboard running at: http://<address>:<port>[/]' \n4. Open that URL.\n\n[dim]Default URL is often http://{host}:{port}, but may differ.[/]",
            title="[bold yellow]Ray Dashboard Instructions[/]",
            border_style="yellow",
            expand=False,
        )
    )


# --- Config Management Commands ---
@app.command()
def create(overwrite: OverwriteOption = False):
    """💾 Create a new configuration set interactively."""
    create_config_logic(overwrite=overwrite)


@app.command("list")
def list_cmd():
    """📄 List all saved configuration sets."""
    list_configs_logic()


@app.command()
def show(name: ConfigNameArg):
    """👀 Show the details of a specific configuration set."""
    show_config_logic(name)


@app.command()
def edit(name: ConfigNameArg):
    """✏️ Edit an existing configuration set using $EDITOR."""
    edit_config_logic(name)


@app.command()
def delete(name: ConfigNameArg):
    """🗑️ Delete a saved configuration set."""
    delete_config_logic(name)


@app.command()
def validate(name: ConfigNameArg):
    """✔️ Validate the structure and types of a saved configuration file."""
    validate_config_logic(name)


@app.command()
def default(name: OptionalConfigNameArg = None):
    """⭐ Show or set the default configuration for training runs."""
    default_config_logic(name)


# --- External Script Runner (Example) ---
# Keep analyze_profiles as an external script for now

if __name__ == "__main__":
    app()
