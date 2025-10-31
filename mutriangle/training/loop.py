# File: mutriangle/training/loop.py
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from trieye.schemas import RawMetricEvent  # Import from trieye

from ..rl import SelfPlayResult
from .loop_helpers import LoopHelpers
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    import numpy as np

    from ..utils.types import PERBatchSample
    from .components import TrainingComponents


logger = logging.getLogger(__name__)

# Circuit breaker constants
MAX_CONSECUTIVE_EMPTY_HISTORIES = 10
MAX_TOTAL_EMPTY_HISTORIES = 100  # Absolute limit before giving up


class TrainingLoop:
    """
    Manages the core asynchronous training loop logic: coordinating worker tasks,
    processing results, triggering training steps, and interacting with TrieyeActor.
    Runs headless.
    """

    def __init__(
        self,
        components: "TrainingComponents",
    ):
        self.components = components
        self.train_config = components.train_config
        self.trieye_config = components.trieye_config
        self.buffer = components.buffer
        self.trainer = components.trainer
        self.trieye_actor = components.trieye_actor
        self.serializer = components.serializer  # Get serializer from components

        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.weight_update_count = 0
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.training_exception: Exception | None = None
        self._buffer_ready_logged = False
        self._warmup_logged = False
        self._training_started_logged = False
        self._last_phase_log_time = 0.0

        # Circuit breaker for empty GameHistory tracking
        self.consecutive_empty_histories: dict[int, int] = {}
        self.total_empty_histories = 0
        self.last_empty_history_reasons: dict[int, str] = {}

        self.worker_manager = WorkerManager(components)
        self.loop_helpers = LoopHelpers(components, self._get_loop_state)

        logger.info("TrainingLoop initialized (Headless, using Trieye).")

    def _get_loop_state(self) -> dict[str, Any]:
        """Provides current loop state to helpers."""
        return {
            "global_step": self.global_step,
            "episodes_played": self.episodes_played,
            "total_simulations_run": self.total_simulations_run,
            "weight_update_count": self.weight_update_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "num_active_workers": self.worker_manager.get_num_active_workers(),
            "num_pending_tasks": self.worker_manager.get_num_pending_tasks(),
            "start_time": self.start_time,
            "num_workers": self.train_config.NUM_SELF_PLAY_WORKERS,
        }

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        self.weight_update_count = (
            global_step // self.train_config.WORKER_UPDATE_FREQ_STEPS
            if self.train_config.WORKER_UPDATE_FREQ_STEPS > 0
            else 0
        )
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}, WeightUpdates={self.weight_update_count}"
        )

    def initialize_workers(self):
        """Initializes self-play workers using WorkerManager."""
        self.worker_manager.initialize_workers()

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _send_event_async(
        self, name: str, value: float | int, context: dict | None = None
    ):
        """Helper to send a raw metric event to the Trieye actor asynchronously."""
        if self.trieye_actor:
            event = RawMetricEvent(
                name=name,
                value=value,
                global_step=self.global_step,  # Use loop's global step for events sent from loop
                timestamp=time.time(),
                context=context or {},
            )
            try:
                self.trieye_actor.log_event.remote(event)
            except Exception as e:
                logger.error(f"Failed to send event '{name}' to Trieye actor: {e}")

    def _send_batch_events_async(self, events: list[RawMetricEvent]):
        """Helper to send a batch of raw metric events asynchronously."""
        if self.trieye_actor and events:
            try:
                self.trieye_actor.log_batch_events.remote(events)
            except Exception as e:
                logger.error(f"Failed to send batch events to Trieye actor: {e}")

    def _process_self_play_result(self, result: SelfPlayResult, worker_id: int):
        """Processes a validated result from a worker."""
        logger.debug(
            f"Processing result from worker {worker_id} (Ep Steps: {result.episode_steps}, Score: {result.final_score:.2f})"
        )

        # MuZero: Process GameHistory instead of experiences
        game_history = result.game_history

        # Validate GameHistory structure
        if (
            not game_history["observations"]
            or not game_history["actions"]
            or not game_history["rewards"]
        ):
            # Track empty history for circuit breaker
            self.consecutive_empty_histories[worker_id] = (
                self.consecutive_empty_histories.get(worker_id, 0) + 1
            )
            self.total_empty_histories += 1

            # Extract game over reason from result context if available
            game_over_reason = "Unknown"
            if hasattr(result, "game_over_reason") and result.game_over_reason:
                game_over_reason = result.game_over_reason
            elif result.context and "game_over_reason" in result.context:
                game_over_reason = result.context["game_over_reason"]

            self.last_empty_history_reasons[worker_id] = game_over_reason

            consecutive_count = self.consecutive_empty_histories[worker_id]

            logger.error(
                f"Worker {worker_id}: Self-play episode produced empty GameHistory "
                f"(Steps: {result.episode_steps}, Score: {result.final_score:.2f}, "
                f"Reason: {game_over_reason}). "
                f"Consecutive failures: {consecutive_count}/{MAX_CONSECUTIVE_EMPTY_HISTORIES}, "
                f"Total: {self.total_empty_histories}/{MAX_TOTAL_EMPTY_HISTORIES}"
            )

            # Circuit breaker: Check if we've hit the threshold
            if consecutive_count >= MAX_CONSECUTIVE_EMPTY_HISTORIES:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"CRITICAL: Worker {worker_id} produced {consecutive_count} consecutive empty GameHistory objects.\n"
                    f"Last game-over reason: {game_over_reason}\n"
                    f"Total empty histories across all workers: {self.total_empty_histories}\n"
                    f"\n"
                    f"This typically indicates:\n"
                    f"  1. GameState initialization failure (check trianglengin installation)\n"
                    f"  2. Invalid environment configuration (check PLAYABLE_RANGE_PER_ROW)\n"
                    f"  3. Shapes cannot be placed on empty grid (check shape templates)\n"
                    f"\n"
                    f"Diagnostic steps:\n"
                    f"  - Run: trianglengin play --seed 42\n"
                    f"  - Check config: {self.components.env_config.model_dump()}\n"
                    f"  - Review logs for GameState initialization errors\n"
                    f"{'=' * 80}\n"
                )
                logger.critical(error_msg)
                raise RuntimeError(
                    f"Circuit breaker triggered: Worker {worker_id} failed {consecutive_count} "
                    f"consecutive times. Last reason: {game_over_reason}. "
                    f"See logs for diagnostic information."
                )

            # Also check total failures across all workers
            if self.total_empty_histories >= MAX_TOTAL_EMPTY_HISTORIES:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"CRITICAL: Total empty GameHistory count reached {self.total_empty_histories}.\n"
                    f"All workers are failing to produce valid episodes.\n"
                    f"Recent failures by worker: {self.last_empty_history_reasons}\n"
                    f"\n"
                    f"The environment configuration appears to be fundamentally broken.\n"
                    f"Please verify trianglengin is installed correctly and the config is valid.\n"
                    f"{'=' * 80}\n"
                )
                logger.critical(error_msg)
                raise RuntimeError(
                    f"Circuit breaker triggered: {self.total_empty_histories} total empty histories. "
                    f"System cannot produce valid episodes."
                )

            return

        # Reset consecutive failure counter on successful episode
        if worker_id in self.consecutive_empty_histories:
            prev_count = self.consecutive_empty_histories[worker_id]
            if prev_count > 0:
                logger.info(
                    f"Worker {worker_id}: Recovered from {prev_count} consecutive empty histories."
                )
            self.consecutive_empty_histories[worker_id] = 0

        try:
            self.buffer.add(game_history)
            buffer_size = len(self.buffer)
            logger.debug(
                f"Added GameHistory from worker {worker_id} to buffer (Length: {len(game_history['observations'])}, Buffer size: {buffer_size})."
            )
            
            # Log phase transitions
            if (
                not self._buffer_ready_logged
                and buffer_size >= self.buffer.min_size_to_train
            ):
                logger.info(
                    f"✅ BUFFER READY: Size {buffer_size}/{self.buffer.min_size_to_train} reached | "
                    f"Entering WARMUP phase (need {self.buffer.min_size_to_train * 2} for training)"
                )
                self._buffer_ready_logged = True

        except Exception as e:
            logger.error(
                f"Error adding GameHistory to buffer from worker {worker_id}: {e}",
                exc_info=True,
            )
            return  # Don't update counters if add failed

        self.episodes_played += 1
        self.total_simulations_run += result.total_simulations

        # Send loop-level progress events
        self._send_event_async("Progress/Episodes_Played", self.episodes_played)
        self._send_event_async("Progress/Total_Simulations", self.total_simulations_run)
        # Note: Episode-specific events (score, length) are sent by the worker

    def _trigger_save_state(self, is_best: bool = False, save_buffer: bool = False):
        """Triggers the Trieye actor to save the current training state."""
        if not self.trieye_actor:
            logger.error("Cannot save state: TrieyeActor handle is missing.")
            return

        logger.info(
            f"Requesting state save via Trieye at step {self.global_step}. Best={is_best}, SaveBuffer={save_buffer}"
        )
        try:
            # Prepare data using the local serializer
            nn_state = self.components.nn.get_weights()
            opt_state = self.serializer.prepare_optimizer_state(self.trainer.optimizer.state_dict())

            # --- CORRECTED BUFFER CONTENT ---
            # Get buffer contents as a list before passing to serializer
            buffer_content_list = self.buffer.get_contents()
            buffer_data = self.serializer.prepare_buffer_data(buffer_content_list)
            # --- END CORRECTED BUFFER CONTENT ---

            if buffer_data is None and save_buffer:
                logger.error("Failed to prepare buffer data, cannot save buffer.")
                save_buffer = False  # Don't attempt to save None

            # Fire-and-forget call to the actor
            self.trieye_actor.save_training_state.remote(
                nn_state_dict=nn_state,
                optimizer_state_dict=opt_state,
                buffer_content=(
                    buffer_data.buffer_list if buffer_data else []
                ),  # Pass the list
                global_step=self.global_step,
                episodes_played=self.episodes_played,
                total_simulations_run=self.total_simulations_run,
                is_best=is_best,
                save_buffer=save_buffer,
                model_config_dict=self.components.model_config.model_dump(),
                env_config_dict=self.components.env_config.model_dump(),
            )
        except Exception as e_save:
            logger.error(
                f"Failed to trigger save state via Trieye at step {self.global_step}: {e_save}",
                exc_info=True,
            )

    def _run_training_step(self) -> bool:
        """Runs one training step."""
        buffer_size = len(self.buffer)
        
        # Phase 1: Collection - buffer not ready
        if not self.buffer.is_ready():
            # Log phase info every 10 seconds
            if time.time() - self._last_phase_log_time > 10.0:
                logger.info(
                    f"📦 COLLECTION PHASE: Buffer {buffer_size}/{self.buffer.min_size_to_train} "
                    f"| Workers generating episodes | Training blocked"
                )
                self._last_phase_log_time = time.time()
            return False
        
        # Phase 2: Warmup - buffer ready but not enough diversity for dynamics training
        warmup_threshold = self.buffer.min_size_to_train * 2
        if buffer_size < warmup_threshold:
            if not self._warmup_logged or time.time() - self._last_phase_log_time > 10.0:
                progress_pct = (buffer_size / warmup_threshold) * 100
                logger.info(
                    f"🔥 WARMUP PHASE: Buffer {buffer_size}/{warmup_threshold} ({progress_pct:.1f}%) "
                    f"| Building diverse dataset for dynamics model | Training blocked"
                )
                self._warmup_logged = True
                self._last_phase_log_time = time.time()
            return False
        
        # Phase 3: Active training
        if not self._training_started_logged:
            logger.info(
                f"🚀 TRAINING STARTED: Buffer {buffer_size} | Warmup complete | "
                f"Now sampling & training in parallel with episode collection"
            )
            self._training_started_logged = True

        # MuZero: Sample training targets with unroll steps
        batch_targets = self.buffer.sample(
            batch_size=self.train_config.BATCH_SIZE,
            unroll_steps=self.train_config.UNROLL_STEPS,
            current_train_step=self.global_step,
        )
        if not batch_targets:
            return False

        train_result: tuple[dict[str, float], np.ndarray] | None = (
            self.trainer.train_step(batch_targets)
        )
        if train_result:
            loss_info, td_errors = train_result
            prev_step = self.global_step
            self.global_step += 1
            if prev_step == 0:
                logger.info(
                    f"--- First training step completed (Global Step: {self.global_step}) ---"
                )
            self._send_event_async("step_completed", 1.0)  # Event for rate calculation

            # Note: PER priority updates for MuZero would need game-level indices
            # For simplicity, we skip PER updates for now in MuZero
            # if self.train_config.USE_PER:
            #     self.buffer.update_priorities(indices, td_errors)
            #     per_beta = self.buffer._calculate_beta(self.global_step)
            #     self._send_event_async("PER/Beta", per_beta)

            # Send loss events to Trieye
            events_batch = []
            # Map internal loss keys to desired metric names
            loss_name_map = {
                "total_loss": "Loss/Total",
                "policy_loss": "Loss/Policy",
                "value_loss": "Loss/Value",
                "reward_loss": "Loss/Reward",
                "mean_td_error": "Loss/Mean_Abs_TD_Error",
            }
            for key, value in loss_info.items():
                metric_name = loss_name_map.get(key)
                if metric_name:
                    events_batch.append(
                        RawMetricEvent(
                            name=metric_name,  # Use mapped name
                            value=value,
                            global_step=self.global_step,
                            # Add source context if needed by Trieye metric config
                            context={"source": "trainer"},
                        )
                    )
                else:
                    logger.warning(f"Unmapped loss key from trainer: {key}")

            current_lr = self.trainer.get_current_lr()
            events_batch.append(
                RawMetricEvent(
                    name="LearningRate",
                    value=current_lr,
                    global_step=self.global_step,
                    context={"source": "trainer"},
                )
            )

            self._send_batch_events_async(events_batch)

            # Update worker networks periodically
            if (
                self.train_config.WORKER_UPDATE_FREQ_STEPS > 0
                and self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS == 0
            ):
                logger.info(
                    f"🔄 Step {self.global_step}: Syncing new weights to {self.worker_manager.get_num_active_workers()} workers"
                )
                try:
                    self.worker_manager.update_worker_networks(self.global_step)
                    self.weight_update_count += 1
                    self._send_event_async(
                        "Progress/Weight_Updates_Total", self.weight_update_count
                    )
                except Exception as update_err:
                    logger.error(
                        f"Failed to update worker networks at step {self.global_step}: {update_err}"
                    )

            # Log training metrics periodically
            if self.global_step == 1:
                logger.info(
                    f"✨ First training step completed! Step {self.global_step} | "
                    f"Policy Loss={loss_info.get('policy_loss', 0.0):.4f}, "
                    f"Value Loss={loss_info.get('value_loss', 0.0):.4f}, "
                    f"Reward Loss={loss_info.get('reward_loss', 0.0):.4f}"
                )
            elif self.global_step % 50 == 0:
                logger.info(
                    f"📈 Step {self.global_step} | "
                    f"P={loss_info.get('policy_loss', 0.0):.3f} "
                    f"V={loss_info.get('value_loss', 0.0):.3f} "
                    f"R={loss_info.get('reward_loss', 0.0):.3f} "
                    f"Tot={loss_info.get('total_loss', 0.0):.3f}"
                )
            return True
        else:
            logger.warning(f"Training step {self.global_step + 1} failed.")
            return False

    def run(self):
        """Main training loop."""
        max_steps_info = (
            f"Target steps: {self.train_config.MAX_TRAINING_STEPS}"
            if self.train_config.MAX_TRAINING_STEPS is not None
            else "Running indefinitely (no MAX_TRAINING_STEPS)"
        )
        logger.info(f"Starting TrainingLoop run... {max_steps_info}")
        self.start_time = time.time()

        try:
            self.worker_manager.submit_initial_tasks()

            while not self.stop_requested.is_set():
                # Check for completion
                if (
                    self.train_config.MAX_TRAINING_STEPS is not None
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached MAX_TRAINING_STEPS ({self.train_config.MAX_TRAINING_STEPS}). Stopping loop."
                    )
                    self.training_complete = True
                    self.request_stop()
                    break

                # Run training step if buffer is ready
                trained_this_step = False
                trained_this_step = self._run_training_step()
                
                if not trained_this_step:
                    time.sleep(0.05)  # Short sleep if not training yet

                # Trigger checkpoint save
                # --- FIX: Access CHECKPOINT_SAVE_FREQ_STEPS from train_config ---
                if (
                    trained_this_step
                    and self.train_config.CHECKPOINT_SAVE_FREQ_STEPS > 0
                    and self.global_step % self.train_config.CHECKPOINT_SAVE_FREQ_STEPS
                    == 0
                ):
                    self._trigger_save_state(is_best=False, save_buffer=False)
                # --- END FIX ---

                # Trigger buffer save
                if (
                    trained_this_step
                    and self.components.trieye_config.persistence.SAVE_BUFFER
                    and self.components.trieye_config.persistence.BUFFER_SAVE_FREQ_STEPS
                    > 0
                    and self.global_step
                    % self.components.trieye_config.persistence.BUFFER_SAVE_FREQ_STEPS
                    == 0
                ):
                    self._trigger_save_state(is_best=False, save_buffer=True)

                if self.stop_requested.is_set():
                    break

                # Get results from workers
                wait_timeout = 0.1 if self.buffer.is_ready() else 0.5
                completed_tasks = self.worker_manager.get_completed_tasks(wait_timeout)

                for worker_id, result_or_error in completed_tasks:
                    if isinstance(result_or_error, SelfPlayResult):
                        try:
                            self._process_self_play_result(result_or_error, worker_id)
                        except Exception as proc_err:
                            logger.error(
                                f"Error processing result from worker {worker_id}: {proc_err}",
                                exc_info=True,
                            )
                    elif isinstance(result_or_error, Exception):
                        logger.error(
                            f"Worker {worker_id} task failed with exception: {result_or_error}"
                        )
                    else:
                        logger.error(
                            f"Received unexpected item from completed tasks for worker {worker_id}: {type(result_or_error)}"
                        )

                    # Resubmit task to the worker
                    self.worker_manager.submit_task(worker_id)

                if self.stop_requested.is_set():
                    break

                # Log progress and system stats
                self.loop_helpers.log_progress_eta()
                loop_state = self._get_loop_state()
                self._send_event_async("Buffer/Size", loop_state["buffer_size"])
                self._send_event_async(
                    "System/Num_Active_Workers", loop_state["num_active_workers"]
                )
                self._send_event_async(
                    "System/Num_Pending_Tasks", loop_state["num_pending_tasks"]
                )

                # Trigger Trieye processing
                if self.trieye_actor:
                    # Fire-and-forget call
                    self.trieye_actor.process_and_log.remote(self.global_step)

                # Memory monitoring - log every 1000 steps (not every loop!)
                if self.global_step > 0 and self.global_step % 1000 == 0:
                    try:
                        import psutil
                        import torch
                        
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        mem_mb = mem_info.rss / 1024 / 1024
                        logger.info(f"💾 Memory: {mem_mb:.1f} MB process RSS")
                        
                        # PyTorch memory if CUDA
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024 / 1024
                            reserved = torch.cuda.memory_reserved() / 1024 / 1024
                            logger.info(
                                f"💾 GPU: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved"
                            )
                        
                        # Send memory metrics to Trieye
                        self._send_event_async("System/Process_Memory_MB", mem_mb)
                    except Exception as e:
                        logger.debug(f"Memory monitoring failed: {e}")

                # Small sleep if nothing happened this iteration
                if (
                    not completed_tasks
                    and not trained_this_step
                    and not self.buffer.is_ready()
                ):
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received in TrainingLoop. Stopping.")
            self.request_stop()
        except Exception as e:
            logger.critical(f"Unhandled exception in TrainingLoop: {e}", exc_info=True)
            self.training_exception = e
            self.request_stop()
        finally:
            if (
                self.training_exception
                or self.stop_requested.is_set()
                and not self.training_complete
            ):
                self.training_complete = False
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Cleans up worker actors. TrieyeActor cleanup handled by runner."""
        self.worker_manager.cleanup_actors()
