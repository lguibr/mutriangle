# File: mutriangle/training/setup.py
import contextlib
import logging
import os
from pathlib import Path  # Ensure Path is imported
from typing import cast

import ray
import torch

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig
from trieye import (  # Import from Trieye
    Serializer,
    TrieyeActor,
    TrieyeConfig,
)

# Keep mutriangle imports
from .. import config, utils
from ..config import (
    MuTriangleMCTSConfig,
    ModelConfig,
    RunContext,
    TrainConfig,
)
from ..config_mgmt.cli_utils import AUTO_WORKER_SENTINEL  # Import sentinel
from ..nn import NeuralNetwork
from ..rl import GameHistoryBuffer, Trainer
from .components import TrainingComponents

logger = logging.getLogger(__name__)


def setup_training_components(
    run_context: RunContext,
    train_config_override: "TrainConfig",
    trieye_config_override: TrieyeConfig,
    profile: bool,
    model_config_override: ModelConfig | None = None,
    mcts_config_override: MuTriangleMCTSConfig | None = None,
) -> tuple[TrainingComponents | None, bool]:
    """
    Initializes Ray, detects cores, updates config, initializes TrieyeActor,
    and returns the TrainingComponents bundle. Uses RunContext for identifiers.
    Sets MLFLOW_TRACKING_URI environment variable before actor init.
    """
    ray_initialized_here = False
    detected_cpu_cores: int | None = None
    dashboard_started_successfully = False
    trieye_actor_handle: ray.actor.ActorHandle | None = None
    original_mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")  # Store original

    try:
        # --- Ray Initialization ---
        if not ray.is_initialized():
            try:
                logger.info("Attempting to initialize Ray with dashboard...")
                ray.init(
                    logging_level=logging.WARNING,
                    log_to_driver=False,
                    include_dashboard=True,
                )
                ray_initialized_here = True
                dashboard_started_successfully = True
                logger.info(
                    "Ray initialized by setup_training_components WITH dashboard attempt."
                )
            except Exception as e_dash:
                if "Cannot include dashboard with missing packages" in str(e_dash):
                    logger.warning(
                        "Ray dashboard dependencies missing. Retrying Ray initialization without dashboard. Install 'ray[default]' for dashboard support."
                    )
                    try:
                        ray.init(
                            logging_level=logging.WARNING,
                            log_to_driver=False,
                            include_dashboard=False,
                        )
                        ray_initialized_here = True
                        dashboard_started_successfully = False
                        logger.info(
                            "Ray initialized by setup_training_components WITHOUT dashboard."
                        )
                    except Exception as e_no_dash:
                        logger.critical(
                            f"Failed to initialize Ray even without dashboard: {e_no_dash}",
                            exc_info=True,
                        )
                        raise RuntimeError("Ray initialization failed") from e_no_dash
                else:
                    logger.critical(
                        f"Failed to initialize Ray (with dashboard attempt): {e_dash}",
                        exc_info=True,
                    )
                    raise RuntimeError("Ray initialization failed") from e_dash

            if dashboard_started_successfully:
                logger.info(
                    "Ray Dashboard *should* be running. Check Ray startup logs for the exact URL (usually http://127.0.0.1:8265)."
                )
            elif ray_initialized_here:
                logger.info(
                    "Ray Dashboard is NOT running (missing dependencies). Install 'ray[default]' to enable it."
                )
        else:
            logger.info("Ray already initialized.")
            logger.info(
                "Ray Dashboard status in existing session unknown. Check Ray logs or http://127.0.0.1:8265."
            )
            ray_initialized_here = False

        # --- Resource Detection ---
        cores_to_reserve = 2
        available_cores = 0
        try:
            resources = ray.cluster_resources()
            available_cores = int(resources.get("CPU", 0))
            logger.info(f"Ray detected {available_cores} total CPU cores.")
        except Exception as e:
            logger.warning(
                f"Could not get Ray cluster resources: {e}. Falling back to os.cpu_count()."
            )
            try:
                os_cores = os.cpu_count()
                if os_cores:
                    available_cores = os_cores
                    logger.info(f"os.cpu_count() reported {available_cores} cores.")
                else:
                    logger.error("os.cpu_count() returned None or 0.")
            except NotImplementedError:
                logger.error("os.cpu_count() is not implemented on this system.")

        detected_cpu_cores = max(
            1, available_cores - cores_to_reserve
        )  # Ensure at least 1
        logger.info(
            f"Reserving {cores_to_reserve}. Available for workers: {detected_cpu_cores}."
        )

        # --- Use Configurations Passed In ---
        train_config = train_config_override
        trieye_config = trieye_config_override
        env_config = EnvConfig()
        model_config = model_config_override or config.ModelConfig()
        mutriangle_mcts_config = mcts_config_override or MuTriangleMCTSConfig()
        logger.info(
            f"Using Model Config: {'Loaded/Override' if model_config_override else 'Default'}"
        )
        logger.info(
            f"Using MCTS Config: {'Loaded/Override' if mcts_config_override else 'Default'}"
        )

        # --- Adjust Worker Count ---
        requested_workers = train_config.NUM_SELF_PLAY_WORKERS
        actual_workers = 1

        if requested_workers == AUTO_WORKER_SENTINEL:
            logger.info(
                f"NUM_SELF_PLAY_WORKERS set to 'auto'. Using detected cores: {detected_cpu_cores}"
            )
            actual_workers = detected_cpu_cores
        elif requested_workers > 0:
            actual_workers = min(requested_workers, detected_cpu_cores)
            if actual_workers != requested_workers:
                logger.info(
                    f"Adjusting requested workers ({requested_workers}) to available cores ({detected_cpu_cores}). Using {actual_workers} workers."
                )
            else:
                logger.info(f"Using requested number of workers: {requested_workers}")
        else:
            logger.warning(
                f"Invalid NUM_SELF_PLAY_WORKERS ({requested_workers}) from config. Defaulting to 1 worker."
            )
            actual_workers = 1

        train_config.NUM_SELF_PLAY_WORKERS = actual_workers
        logger.info(f"Final worker count set to: {train_config.NUM_SELF_PLAY_WORKERS}")

        # --- Validate MuTriangle Configurations ---
        config.print_config_info_and_validate(mutriangle_mcts_config)

        # --- Create mutrimcts SearchConfiguration ---
        mutrimcts_mcts_config = mutriangle_mcts_config.to_mutrimcts_config()
        logger.info(
            f"Created mutrimcts.SearchConfiguration with max_simulations={mutrimcts_mcts_config.max_simulations}, mcts_batch_size={mutrimcts_mcts_config.mcts_batch_size}"
        )

        # --- Setup Devices and Seeds ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)
        worker_device = utils.get_device(train_config.WORKER_DEVICE)
        logger.info(f"Determined Training Device: {device}")
        logger.info(f"Determined Worker Device: {worker_device}")
        logger.info(f"Model Compilation Enabled: {train_config.COMPILE_MODEL}")
        logger.info(f"Worker Profiling Enabled: {profile}")

        # --- Set MLflow URI ENV VAR before Actor Init ---
        # This ensures the actor process picks up the correct URI from the environment
        mlflow_uri = trieye_config.persistence.MLFLOW_TRACKING_URI
        logger.info(
            f"Setting MLFLOW_TRACKING_URI environment variable to: {mlflow_uri}"
        )
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

        # --- Initialize Trieye Actor ---
        actor_name = f"trieye_actor_{run_context.run_name}"
        logger.debug(
            f"Attempting to initialize TrieyeActor with name '{actor_name}' and config: {trieye_config.model_dump_json(indent=2)}"
        )

        try:
            trieye_actor_handle = ray.get_actor(actor_name)
            logger.info(f"Reconnected to existing TrieyeActor '{actor_name}'.")
        except ValueError:
            logger.info(f"Creating new TrieyeActor '{actor_name}'.")
            trieye_actor_handle = TrieyeActor.options(
                name=actor_name, lifetime="detached"
            ).remote(config=trieye_config)
            if trieye_actor_handle:
                try:
                    # Wait for actor readiness (e.g., by checking MLflow run ID)
                    ray.get(trieye_actor_handle.get_mlflow_run_id.remote(), timeout=15)
                    logger.info(f"TrieyeActor '{actor_name}' created and ready.")
                except Exception as ready_err:
                    logger.error(
                        f"Error waiting for TrieyeActor '{actor_name}' to become ready: {ready_err}",
                        exc_info=True,
                    )
                    # Ensure env var is restored even if actor init fails
                    if original_mlflow_uri is None:
                        if "MLFLOW_TRACKING_URI" in os.environ:
                            del os.environ["MLFLOW_TRACKING_URI"]
                    else:
                        os.environ["MLFLOW_TRACKING_URI"] = original_mlflow_uri
                    with contextlib.suppress(Exception):
                        ray.kill(trieye_actor_handle)
                    raise RuntimeError(
                        f"TrieyeActor '{actor_name}' failed readiness check."
                    ) from ready_err
            else:
                raise RuntimeError(
                    f"Failed to get handle for TrieyeActor '{actor_name}' after creation."
                ) from None

        if not trieye_actor_handle:
            raise RuntimeError(
                f"TrieyeActor '{actor_name}' handle is invalid."
            ) from None

        # --- Initialize Core MuTriangle Components ---
        serializer = Serializer()
        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = GameHistoryBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)

        # --- Bundle Components ---
        components = TrainingComponents(
            run_context=run_context,
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            trieye_actor=cast("ray.actor.ActorHandle", trieye_actor_handle),
            trieye_config=trieye_config,
            serializer=serializer,
            train_config=train_config,
            env_config=env_config,
            model_config=model_config,
            mcts_config=mutrimcts_mcts_config,
            profile_workers=profile,
        )

        # Restore env var immediately after actor is confirmed to be ready
        if original_mlflow_uri is None:
            if "MLFLOW_TRACKING_URI" in os.environ:
                del os.environ["MLFLOW_TRACKING_URI"]
                logger.info("Unset MLFLOW_TRACKING_URI environment variable.")
        else:
            os.environ["MLFLOW_TRACKING_URI"] = original_mlflow_uri
            logger.info(
                f"Restored MLFLOW_TRACKING_URI environment variable to: {original_mlflow_uri}"
            )

        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        if trieye_actor_handle:
            try:
                ray.kill(trieye_actor_handle)
            except Exception as kill_err:
                logger.error(
                    f"Error killing TrieyeActor during setup cleanup: {kill_err}"
                )
        if ray_initialized_here and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down due to setup error.")
            except Exception as ray_err:
                logger.error(f"Error shutting down Ray during setup cleanup: {ray_err}")
        # Restore original env var if it existed, even on error
        if original_mlflow_uri is None:
            if "MLFLOW_TRACKING_URI" in os.environ:
                del os.environ["MLFLOW_TRACKING_URI"]
        else:
            os.environ["MLFLOW_TRACKING_URI"] = original_mlflow_uri
        return None, ray_initialized_here
    # Removed finally block as cleanup is handled within try/except


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
