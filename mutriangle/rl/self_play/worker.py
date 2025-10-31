import cProfile
import logging
import random
import time
from collections import deque
from functools import partial  # Import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray
from ray.actor import ActorHandle
from trianglengin import EnvConfig, GameState
from trieye.schemas import RawMetricEvent  # Import from trieye
from mutrimcts import SearchConfiguration, run_mcts

from ...config import ModelConfig, TrainConfig
from ...features import extract_state_features
from ...nn import NeuralNetwork
from ...utils import get_device, set_random_seeds

# MuZero: No augmentation imports needed
from ..types import SelfPlayResult
from .mcts_helpers import (
    PolicyGenerationError,
    get_policy_target_from_visits,
    select_action_from_visits,
)

if TYPE_CHECKING:
    from ...utils.types import (
        Experience,
        GameHistory,
        PolicyTargetMapping,
        StateType,
    )

    MctsTreeHandle = Any

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running self-play episodes using mutrimcts and a NN.
    Uses trianglengin.GameState for MuZero MCTS in learned latent space.
    Collects complete GameHistory trajectories for MuZero training.
    Sends raw metric events to the TrieyeActor using its name.
    Saves profiles to the run-specific directory provided.
    Note: Tree reuse not currently supported in mutrimcts.
    """

    def __init__(
        self,
        actor_id: int,
        env_config: EnvConfig,
        mcts_config: SearchConfiguration,
        model_config: ModelConfig,
        train_config: TrainConfig,
        trieye_actor_name: str,
        run_base_dir: str,  # Accepts run base dir string
        initial_weights: dict | None = None,
        seed: int | None = None,
        worker_device_str: str = "cpu",
        profile_this_worker: bool = False,
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.trieye_actor_name = trieye_actor_name
        self.run_base_dir = Path(run_base_dir)  # Store as Path
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str
        self.profile_this_worker = profile_this_worker

        self.n_step = self.train_config.N_STEP_RETURNS
        self.gamma = self.train_config.GAMMA
        self.current_trainer_step = 0

        global logger
        logger = logging.getLogger(__name__)

        set_random_seeds(self.seed)
        self.device = get_device(self.worker_device_str)

        self.nn_evaluator = NeuralNetwork(
            model_config=self.model_config,
            env_config=self.env_config,
            train_config=self.train_config,
            device=self.device,
        )

        if initial_weights:
            self.set_weights(initial_weights)
        else:
            self.nn_evaluator.model.eval()

        # Cache the actor handle after first lookup
        self._trieye_actor_handle: ActorHandle | None = None

        logger.debug(f"INIT: MCTS Config: {self.mcts_config.model_dump()}")
        logger.info(
            f"Worker {self.actor_id} initialized on device {self.device}. Seed: {self.seed}. LogLevel: {logging.getLevelName(logger.getEffectiveLevel())}. TrieyeActor Name: {self.trieye_actor_name}. Run Base Dir: {self.run_base_dir}"
        )
        logger.debug(f"Worker {self.actor_id} init complete.")

        self.profiler: cProfile.Profile | None = None
        if self.profile_this_worker:
            self.profiler = cProfile.Profile()
            logger.warning(f"Worker {self.actor_id}: Profiling ENABLED.")
        else:
            logger.info(f"Worker {self.actor_id}: Profiling DISABLED.")

    def _get_trieye_actor(self) -> ActorHandle | None:
        """Gets the TrieyeActor handle, caching it after the first lookup."""
        if self._trieye_actor_handle is None:
            try:
                self._trieye_actor_handle = ray.get_actor(self.trieye_actor_name)
                logger.info(
                    f"Worker {self.actor_id}: Successfully got TrieyeActor handle."
                )
            except ValueError:
                logger.error(
                    f"Worker {self.actor_id}: Could not find TrieyeActor named '{self.trieye_actor_name}'. Logging disabled."
                )
                self._trieye_actor_handle = None
            except Exception as e:
                logger.error(
                    f"Worker {self.actor_id}: Error getting TrieyeActor handle: {e}"
                )
                self._trieye_actor_handle = None
        return self._trieye_actor_handle

    def set_weights(self, weights: dict):
        """Updates the neural network weights."""
        try:
            self.nn_evaluator.set_weights(weights)
            logger.info(f"Worker {self.actor_id}: Weights updated.")
        except Exception as e:
            logger.error(
                f"Worker {self.actor_id}: Failed to set weights: {e}", exc_info=True
            )

    def set_current_trainer_step(self, global_step: int):
        """Sets the global step corresponding to the current network weights."""
        self.current_trainer_step = global_step
        logger.info(f"Worker {self.actor_id}: Trainer step set to {global_step}")

    def _send_event_async(
        self, name: str, value: float | int, context: dict | None = None
    ):
        """Helper to send a raw metric event to the Trieye actor asynchronously."""
        actor_handle = self._get_trieye_actor()
        if actor_handle:
            event = RawMetricEvent(
                name=name,
                value=value,
                global_step=self.current_trainer_step,
                timestamp=time.time(),
                context=context or {},
            )
            try:
                actor_handle.log_event.remote(event)
            except Exception as e:
                logger.error(
                    f"Worker {self.actor_id}: Failed to send event '{name}' to Trieye actor: {e}"
                )
        else:
            logger.warning(
                f"Worker {self.actor_id}: Cannot send event '{name}', TrieyeActor handle not available."
            )

    def run_episode(self) -> SelfPlayResult:
        """Runs a single episode of self-play using MuZero MCTS in learned latent space."""
        step_at_start = self.current_trainer_step
        logger.info(
            f"Worker {self.actor_id}: Starting run_episode. Seed: {self.seed}. Using network from trainer step: {step_at_start}"
        )
        if self.profiler:
            self.profiler.enable()

        result: SelfPlayResult | None = None
        episode_seed = self.seed + random.randint(0, 1000)
        game: GameState | None = None
        final_score = 0.0
        final_step = 0
        total_sims_episode = 0
        total_triangles_cleared_episode = 0
        avg_mcts_depth_episode = 0.0
        mcts_step_count = 0

        try:
            self.nn_evaluator.model.eval()
            logger.debug(
                f"Worker {self.actor_id}: Initializing GameState with seed {episode_seed}..."
            )
            game = GameState(self.env_config, initial_seed=episode_seed)
            logger.debug(
                f"Worker {self.actor_id}: GameState initialized. Initial step: {game.current_step}, Initial score: {game.game_score()}"
            )

            if game.is_over():
                reason = game.get_game_over_reason() or "Unknown reason at start"

                # Enhanced diagnostics
                shapes = game.get_shapes()
                shapes_info = [
                    f"Slot {i}: {len(s.triangles) if s else 'None'} triangles"
                    for i, s in enumerate(shapes)
                ]
                logger.error(
                    f"Worker {self.actor_id}: Game over immediately after reset (Seed: {episode_seed}).\n"
                    f"  Reason: {reason}\n"
                    f"  Grid shape: {self.env_config.ROWS}x{self.env_config.COLS}\n"
                    f"  Shapes: {shapes_info}\n"
                    f"  Valid actions: {len(game.valid_actions())}\n"
                    f"  Config: NUM_SHAPE_SLOTS={self.env_config.NUM_SHAPE_SLOTS}"
                )

                episode_end_context = {
                    "score": game.game_score(),
                    "length": 0,
                    "simulations": 0,
                    "triangles_cleared": 0,
                    "trainer_step": step_at_start,
                    "avg_mcts_depth": 0.0,
                    "game_over_reason": reason,
                }
                self._send_event_async("episode_end", 1.0, context=episode_end_context)

                # Return with empty GameHistory
                empty_history: GameHistory = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "mcts_policies": [],
                    "root_values": [],
                }
                return SelfPlayResult(
                    game_history=empty_history,
                    episode_experiences=[],
                    final_score=game.game_score(),
                    episode_steps=0,
                    trainer_step_at_episode_start=step_at_start,
                    total_simulations=0,
                    avg_root_visits=0.0,
                    avg_tree_depth=0.0,
                    context=episode_end_context,
                    game_over_reason=f"Immediate game over: {reason}",
                )
            if not game.valid_actions():
                # Enhanced diagnostics
                shapes = game.get_shapes()
                shapes_info = [
                    f"Slot {i}: {len(s.triangles) if s else 'None'} triangles"
                    for i, s in enumerate(shapes)
                ]
                grid_data = game.get_grid_data_np()
                occupied_count = (
                    grid_data["occupied"].sum() if "occupied" in grid_data else "N/A"
                )

                logger.error(
                    f"Worker {self.actor_id}: Game not over, but NO valid actions at start (Seed: {episode_seed}).\n"
                    f"  Grid shape: {self.env_config.ROWS}x{self.env_config.COLS}\n"
                    f"  Shapes: {shapes_info}\n"
                    f"  Occupied cells: {occupied_count}\n"
                    f"  Is game over: {game.is_over()}\n"
                    f"  Config: NUM_SHAPE_SLOTS={self.env_config.NUM_SHAPE_SLOTS}"
                )

                reason = "No valid actions at initialization"
                episode_end_context = {
                    "score": game.game_score(),
                    "length": 0,
                    "simulations": 0,
                    "triangles_cleared": 0,
                    "trainer_step": step_at_start,
                    "avg_mcts_depth": 0.0,
                    "game_over_reason": reason,
                }
                self._send_event_async("episode_end", 1.0, context=episode_end_context)

                # Return with empty GameHistory
                empty_history: GameHistory = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "mcts_policies": [],
                    "root_values": [],
                }
                return SelfPlayResult(
                    game_history=empty_history,
                    final_score=game.game_score(),
                    episode_steps=0,
                    trainer_step_at_episode_start=step_at_start,
                    total_simulations=0,
                    avg_root_visits=0.0,
                    avg_tree_depth=0.0,
                    context=episode_end_context,
                    game_over_reason=reason,
                )

            # --- MuZero: Simple lists for trajectory collection ---
            observations: list[StateType] = []
            actions: list[ActionType] = []
            raw_rewards: list[float] = []
            mcts_policies: list[PolicyTargetMapping] = []
            root_values: list[float] = []
            last_total_visits = 0

            logger.info(
                f"Worker {self.actor_id}: Starting episode loop with seed {episode_seed}"
            )

            while not game.is_over():
                current_step_in_loop = game.current_step
                logger.debug(
                    f"Worker {self.actor_id}: --- Starting Step {current_step_in_loop} ---"
                )
                step_start_time = time.monotonic()

                if game.is_over():
                    logger.warning(
                        f"Worker {self.actor_id}: State terminal before MCTS (Step {current_step_in_loop}). Exiting loop."
                    )
                    break

                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Running MCTS ({self.mcts_config.max_simulations} sims)..."
                )
                mcts_start_time = time.monotonic()
                visit_counts: dict[int, int] = {}
                root_value: float = 0.0
                mcts_policy: dict[int, float] = {}
                try:
                    visit_counts, root_value, mcts_policy = run_mcts(
                        initial_observation=game,
                        network_interface=self.nn_evaluator,
                        config=self.mcts_config,
                    )
                    avg_depth_this_step = 0.0  # Depth not returned by mutrimcts API
                    
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: MCTS returned visit_counts: {len(visit_counts)} actions, Avg Depth: {avg_depth_this_step:.2f}"
                    )
                except Exception as mcts_err:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: mutrimcts failed: {mcts_err}",
                        exc_info=True,
                    )
                    break
                mcts_duration = time.monotonic() - mcts_start_time
                last_total_visits = sum(visit_counts.values())
                total_sims_episode += last_total_visits
                if last_total_visits > 0:
                    avg_mcts_depth_episode += avg_depth_this_step
                    mcts_step_count += 1
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: MCTS finished ({mcts_duration:.3f}s). Total visits: {last_total_visits}"
                )

                self._send_event_async(
                    "mcts_step",
                    float(last_total_visits),
                    context={"game_step": current_step_in_loop},
                )
                self._send_event_async(
                    "mcts_avg_depth",
                    avg_depth_this_step,
                    context={"game_step": current_step_in_loop},
                )

                if not visit_counts:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: MCTS returned empty visit counts. Cannot proceed."
                    )
                    break
                

                action_selection_start_time = time.monotonic()
                # Temperature annealing based on TRAINING progress, not game steps
                temp = 1.0
                selection_temp = 1.0
                if self.train_config.MAX_TRAINING_STEPS is not None:
                    explore_steps = self.train_config.MAX_TRAINING_STEPS * 0.1
                    # Use trainer step (global training progress), not game step
                    selection_temp = (
                        1.0 if self.current_trainer_step < explore_steps else 0.1
                    )
                    # Also anneal policy target temperature
                    temp = selection_temp

                action: int = -1
                try:
                    # Use mcts_policy directly from MCTS (already normalized)
                    # This is more accurate and avoids redundant computation
                    policy_target = mcts_policy
                    
                    # Select action from visit counts with temperature
                    action = select_action_from_visits(
                        visit_counts, temperature=selection_temp
                    )
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Policy target generated. Action selected: {action}"
                    )
                except PolicyGenerationError as policy_err:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Policy/Action selection failed: {policy_err}",
                        exc_info=False,
                    )
                    break
                except Exception as policy_err:
                    logger.error(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: Unexpected policy error: {policy_err}",
                        exc_info=True,
                    )
                    break
                action_selection_duration = (
                    time.monotonic() - action_selection_start_time
                )
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Action selection time: {action_selection_duration:.4f}s"
                )

                feature_start_time = time.monotonic()
                try:
                    state_features: StateType = extract_state_features(
                        game, self.model_config
                    )
                except Exception as e:
                    logger.error(
                        f"Worker {self.actor_id}: Feature extraction error (Step {current_step_in_loop}): {e}",
                        exc_info=True,
                    )
                    break
                feature_duration = time.monotonic() - feature_start_time
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Feature extraction time: {feature_duration:.4f}s"
                )

                # --- MuZero: Store observation, policy, and root value ---
                observations.append(state_features)
                mcts_policies.append(policy_target)

                # Use root value from MCTS (already backed up from simulations)
                root_values.append(root_value)

                # Store action
                actions.append(action)
                # --- END MuZero collection ---

                game_step_start_time = time.monotonic()
                step_reward, done = 0.0, False
                cleared_triangles_this_step = 0
                try:
                    step_reward, done = game.step(action)
                    cleared_triangles_this_step = game.get_last_cleared_triangles()  # type: ignore [attr-defined]
                    total_triangles_cleared_episode += cleared_triangles_this_step
                    logger.debug(
                        f"Worker {self.actor_id}: Step {current_step_in_loop}: game.step({action}) -> Reward: {step_reward:.3f}, Done: {done}, Cleared: {cleared_triangles_this_step}"
                    )
                except Exception as step_err:
                    logger.error(
                        f"Worker {self.actor_id}: Game step error (Action {action}, Step {current_step_in_loop}): {step_err}",
                        exc_info=True,
                    )
                    break
                game_step_duration = time.monotonic() - game_step_start_time
                logger.debug(
                    f"Worker {self.actor_id}: Step {current_step_in_loop}: Game step time: {game_step_duration:.4f}s"
                )

                # --- MuZero: Store raw reward ---
                raw_rewards.append(step_reward)
                # --- END MuZero collection ---

                self._send_event_async(
                    "step_reward",
                    step_reward,
                    context={"game_step": current_step_in_loop},
                )
                if cleared_triangles_this_step > 0:
                    self._send_event_async(
                        "triangles_cleared_step",
                        cleared_triangles_this_step,
                        context={"game_step": current_step_in_loop},
                    )

                self._send_event_async(
                    "current_score",
                    game.game_score(),
                    context={"game_step": current_step_in_loop},
                )

                step_duration = time.monotonic() - step_start_time
                logger.debug(
                    f"Worker {self.actor_id}: --- Finished Step {current_step_in_loop}. Duration: {step_duration:.3f}s ---"
                )

                if done:
                    logger.info(
                        f"Worker {self.actor_id}: Game ended naturally at step {game.current_step}. Reason: {game.get_game_over_reason()}"
                    )
                    break

            final_score = game.game_score() if game else 0.0
            final_step = game.current_step if game else 0
            logger.info(
                f"Worker {self.actor_id}: Episode loop finished. Final Score: {final_score:.2f}, Final Step: {final_step}, Total Cleared: {total_triangles_cleared_episode}"
            )

            # --- MuZero: Store raw rewards (n-step returns calculated during sampling) ---
            # This allows fresh return calculation with current value estimates
            logger.debug(
                f"Worker {self.actor_id}: Collected {len(raw_rewards)} raw rewards for GameHistory"
            )
            # --- END MuZero processing ---

            final_avg_depth = (
                avg_mcts_depth_episode / mcts_step_count if mcts_step_count > 0 else 0.0
            )

            # --- MuZero: Create GameHistory ---
            # Store raw rewards (not n-step returns) for fresh calculation during sampling
            game_history: GameHistory = {
                "observations": observations,
                "actions": actions,
                "rewards": raw_rewards,  # Store raw rewards
                "mcts_policies": mcts_policies,
                "root_values": root_values,
            }

            if not observations:
                logger.warning(
                    f"Worker {self.actor_id}: Episode finished with 0 observations collected. Score: {final_score}, Steps: {final_step}"
                )
            # --- END GameHistory creation ---

            episode_end_context = {
                "score": final_score,
                "length": final_step,
                "simulations": total_sims_episode,
                "triangles_cleared": total_triangles_cleared_episode,
                "trainer_step": step_at_start,
                "avg_mcts_depth": final_avg_depth,
            }
            self._send_event_async(
                name="episode_end", value=1.0, context=episode_end_context
            )

            result = SelfPlayResult(
                game_history=game_history,
                final_score=final_score,
                episode_steps=final_step,
                trainer_step_at_episode_start=step_at_start,
                total_simulations=total_sims_episode,
                avg_root_visits=float(last_total_visits),
                avg_tree_depth=final_avg_depth,
                context=episode_end_context,
                game_over_reason=None,  # Normal completion
            )
            logger.info(
                f"Worker {self.actor_id}: Episode result created. GameHistory length: {len(game_history['observations'])}"
            )

        except Exception as e:
            logger.critical(
                f"Worker {self.actor_id}: Unhandled exception in run_episode: {e}",
                exc_info=True,
            )
            final_score = game.game_score() if game else 0.0
            final_step = game.current_step if game else 0
            final_avg_depth = (
                avg_mcts_depth_episode / mcts_step_count if mcts_step_count > 0 else 0.0
            )
            episode_end_context = {
                "score": final_score,
                "length": final_step,
                "simulations": total_sims_episode,
                "triangles_cleared": total_triangles_cleared_episode,
                "trainer_step": step_at_start,
                "avg_mcts_depth": final_avg_depth,
                "error": True,
            }
            self._send_event_async(
                "episode_end",
                1.0,
                context=episode_end_context,
            )
            # Return with empty GameHistory on error
            empty_history: GameHistory = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "mcts_policies": [],
                "root_values": [],
            }
            result = SelfPlayResult(
                game_history=empty_history,
                final_score=final_score,
                episode_steps=final_step,
                trainer_step_at_episode_start=step_at_start,
                total_simulations=total_sims_episode,
                avg_root_visits=0.0,
                avg_tree_depth=final_avg_depth,
                context=episode_end_context,
                game_over_reason=f"Exception during episode: {str(e)[:100]}",
            )

        finally:
            if self.profiler:
                self.profiler.disable()
                # Use run_base_dir which points to the specific run directory
                profile_dir = self.run_base_dir / "profile_data"
                profile_dir.mkdir(parents=True, exist_ok=True)
                profile_filename = (
                    profile_dir / f"worker_{self.actor_id}_ep_{episode_seed}.prof"
                )
                try:
                    self.profiler.dump_stats(str(profile_filename))
                    logger.warning(
                        f"Worker {self.actor_id}: Profiling stats saved to {profile_filename}"
                    )
                except Exception as e:
                    logger.error(
                        f"Worker {self.actor_id}: Failed to save profile stats: {e}"
                    )

        if result is None:
            logger.error(
                f"Worker {self.actor_id}: run_episode finished without setting a result. Returning empty."
            )
            final_avg_depth = (
                avg_mcts_depth_episode / mcts_step_count if mcts_step_count > 0 else 0.0
            )
            episode_end_context = {
                "score": 0.0,
                "length": 0,
                "simulations": 0,
                "triangles_cleared": 0,
                "trainer_step": step_at_start,
                "avg_mcts_depth": final_avg_depth,
                "error": True,
            }
            self._send_event_async(
                "episode_end",
                1.0,
                context=episode_end_context,
            )
            # Return with empty GameHistory
            empty_history: GameHistory = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "mcts_policies": [],
                "root_values": [],
            }
            result = SelfPlayResult(
                game_history=empty_history,
                final_score=0.0,
                episode_steps=0,
                trainer_step_at_episode_start=step_at_start,
                total_simulations=0,
                avg_root_visits=0.0,
                avg_tree_depth=final_avg_depth,
                context=episode_end_context,
            )

        logger.info(
            f"Worker {self.actor_id}: Finished run_episode. Returning result with GameHistory length: {len(result.game_history['observations'])}"
        )
        return result

