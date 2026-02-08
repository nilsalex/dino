"""Main training loop with locally trained DQN and decoupled background training."""


# ruff: noqa: E402

import gi

gi.require_version("Gst", "1.0")

import random
import time

import torch
from gi.repository import Gst

from src.actors.local_inference_model import LocalInferenceModel
from src.agents.local_trainer import LocalDQNTrainer
from src.buffers.thread_safe_buffer import ThreadSafeExperienceBuffer
from src.capture.frame_processor import FrameProcessor
from src.capture.gstreamer import GStreamerPipeline
from src.core.config import Config
from src.core.game_config import get_game_config
from src.env.game_interface import GameInterface
from src.env.state_monitor import StateMonitor
from src.training.local_training_thread import LocalTrainingThread
from src.utils.metrics import MetricsTracker


def pull_frame_from_appsink(appsink: Gst.Element, frame_processor: FrameProcessor) -> bool:
    sample = appsink.emit("pull-sample")  # type: ignore[arg-type]
    if not sample:
        return False

    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if success:
        try:
            import numpy as np

            frame_arr = np.ndarray(
                shape=(frame_processor.config.output_height, frame_processor.config.output_width),
                dtype=np.uint8,
                buffer=map_info.data,
            )
            frame_processor.add_frame(frame_arr)
            return True
        finally:
            buffer.unmap(map_info)

    return False


def main():
    config = Config()
    game_config = get_game_config(config.game_name)

    print(f"Game: {config.game_name}")
    print(f"Actions: {game_config.action_names}")
    print(f"Frame stack: {game_config.frame_stack}, Frame skip: {game_config.frame_skip}")

    print("Creating local trainer...")
    local_trainer = LocalDQNTrainer(config, game_config.n_actions)

    print("Initializing thread-safe experience buffer...")
    buffer = ThreadSafeExperienceBuffer(config)

    print("Initializing local inference model...")
    local_model = LocalInferenceModel(
        n_actions=game_config.n_actions,
        device=config.device or torch.device("cpu"),
        frame_stack=game_config.frame_stack,
    )

    game_interface = GameInterface(game_config.action_keys)
    state_monitor = StateMonitor()
    frame_processor = FrameProcessor(config, game_config.frame_stack)
    metrics_tracker = MetricsTracker()

    def on_weights_updated(state_dict: dict[str, torch.Tensor]) -> None:
        """Callback to update local model when weights update."""

        local_model.update_state_dict(state_dict)

    print("Starting background training thread...")
    training_thread = LocalTrainingThread(config, buffer, local_trainer, on_weights_updated)
    training_thread.start()

    epsilon = config.epsilon_start
    step_count = 0
    episode_count = 0

    is_evaluating = True
    eval_step_count = 0
    eval_episode_count = 1
    eval_episodes_remaining = 5
    best_eval_score = 0

    total_reward = 0.0
    curr_reward = 0.0
    episode_steps = 0

    min_episode_steps = 20

    print("Starting local training...")
    print("Make sure the game is open and visible!")
    if config.game_name == "dino":
        print("Make sure Chrome Dino game is open and visible!")
    print("[EVAL] Starting initial greedy evaluation episode (baseline)\\n")

    gst_pipeline = GStreamerPipeline(config)
    gst_pipeline.create_pipeline()
    gst_pipeline.start()
    print("GStreamer pipeline started")

    pipeline = gst_pipeline.pipeline
    if pipeline is None:
        raise RuntimeError("GStreamer pipeline not initialized")

    appsink = pipeline.get_by_name("appsink")
    if appsink is None:
        raise RuntimeError("Could not get appsink from GStreamer pipeline")

    previous_state: torch.Tensor | None = None
    is_resetting: bool = False
    reset_delay_counter: int = 0
    reset_delay_frames: int = 30
    reset_cooldown_counter: int = 0
    reset_cooldown_frames: int = 10

    frame_skip_counter = 0
    current_action: int | None = None
    waiting_for_game_over: bool = False
    was_episode_limit_reached: bool = False

    try:
        while episode_count < config.max_episodes:
            pull_frame_from_appsink(appsink, frame_processor)

            current_state = frame_processor.get_state()

            if current_state is None or previous_state is None:
                previous_state = current_state
                time.sleep(0.001)
                continue

            is_game_over = state_monitor.is_game_over(frame_processor.frame_buffer)

            # Handle episode termination
            if (
                episode_steps >= config.max_episode_steps
                and not waiting_for_game_over
                and not was_episode_limit_reached
            ):
                print(f"[SUCCESS] Episode reached {config.max_episode_steps} steps, waiting for game over...")
                # Add success bonus reward when we hit the limit
                curr_reward += 100.0
                # Record bonus as terminal transition in replay buffer
                if current_action is not None and previous_state is not None and current_state is not None:
                    buffer.add(
                        previous_state.squeeze(0),
                        current_action,
                        100.0,
                        current_state.squeeze(0),
                        True,
                    )
                waiting_for_game_over = True
                was_episode_limit_reached = True

            # If waiting for natural game over, skip all logic until it happens
            if waiting_for_game_over:
                if not is_game_over:
                    continue
                waiting_for_game_over = False

            # Handle game over and surgical reset
            if is_game_over:
                if not is_resetting:
                    is_resetting = True
                    reset_delay_counter = 0
                    reset_cooldown_counter = 0
                elif reset_delay_counter < reset_delay_frames:
                    reset_delay_counter += 1
                else:
                    # Wait has passed, send jump action to leave game over state
                    game_interface.execute_action(1)
                    reset_cooldown_counter = 0

                # Skip normal action and recording during reset
                continue
            elif is_resetting:
                # Game is running again, but wait for cooldown to pass
                reset_cooldown_counter += 1
                if reset_cooldown_counter >= reset_cooldown_frames:
                    # Reset complete - game is now stably running
                    is_resetting = False
                    episode_terminated_by_limit = was_episode_limit_reached
                else:
                    # Still in cooldown, skip action and recording
                    continue

                # Handle episode completion
                if is_evaluating:
                    if eval_step_count > 5:
                        print(
                            f"\n[EVAL] Episode {eval_episode_count} complete. Steps: {eval_step_count}\n",
                            end="",
                            flush=True,
                        )
                        if eval_step_count > best_eval_score:
                            best_eval_score = eval_step_count
                            print(f"[BEST] New evaluation score: {best_eval_score}\n")
                        eval_episodes_remaining -= 1
                        eval_episode_count += 1
                        eval_step_count = 0
                        if eval_episodes_remaining > 0:
                            print(f"[EVAL] Starting next eval episode ({eval_episodes_remaining} remaining)\n")
                            previous_state = None
                            current_action = None
                            frame_skip_counter = 0
                            continue
                        else:
                            is_evaluating = False
                            eval_episodes_remaining = 5
                    else:
                        print(f"[EVAL] Episode too short ({eval_step_count} steps), restarting eval...")
                        eval_step_count = 0
                        previous_state = None
                        current_action = None
                        frame_skip_counter = 0
                        continue
                else:
                    if episode_steps >= min_episode_steps:
                        episode_count += 1
                        total_reward += curr_reward
                        status = "[SUCCESS]" if episode_terminated_by_limit else "[GAME OVER]"
                        print(
                            f"\r\x1b[K{status} Episode {episode_count} complete. "
                            f"Steps: {step_count}, Reward: {curr_reward:.2f}\n",
                            end="",
                            flush=True,
                        )
                        curr_reward = 0.0
                        episode_steps = 0

                        if episode_count % 50 == 0:
                            is_evaluating = True
                            eval_episode_count += 1
                            eval_episodes_remaining = 5
                            print("[EVAL] Starting 5 consecutive greedy evaluation episodes\n")
                    else:
                        print(f"[WARN] Episode too short ({episode_steps} steps), not counting")
                        curr_reward = 0.0
                        episode_steps = 0

                current_action = None
                frame_skip_counter = 0
                waiting_for_game_over = False
                was_episode_limit_reached = False
                continue

            # Local inference for fast action selection
            processing_start = time.perf_counter()

            frame_skip_counter += 1

            if frame_skip_counter % game_config.frame_skip == 0 or current_action is None:
                if is_evaluating or (buffer.size() > 0 and random.random() >= epsilon):
                    action = local_model.get_action(previous_state)
                else:
                    action = random.randint(0, game_config.n_actions - 1)
                current_action = action

                # Decay epsilon
                if step_count < config.epsilon_decay:
                    epsilon = max(
                        config.epsilon_end,
                        config.epsilon_start
                        - (config.epsilon_start - config.epsilon_end) * (step_count / config.epsilon_decay),
                    )
            else:
                action = current_action

            game_interface.execute_action(int(action))
            processing_time = time.perf_counter() - processing_start

            if is_evaluating:
                eval_step_count += 1
                step_count += 1
            else:
                # Reward: survival (0.1 per step)
                reward = 0.1

                # Record transition (terminal only on natural game over)
                buffer.add(previous_state.squeeze(0), action, reward, current_state.squeeze(0), is_game_over)
                curr_reward += reward
                episode_steps += 1

                # Update target network locally
                if step_count > 0 and step_count % 1000 == 0:
                    local_trainer.update_target()

                step_count += 1

            metrics = metrics_tracker.update(
                processing_time, frame_processor.queue_size, frame_processor.max_queue_size
            )

            if step_count % 10 == 0:
                training_stats = training_thread.get_stats()
                eval_tag = "[EVAL] " if is_evaluating else ""
                line = (
                    f"{eval_tag}"
                    f"FPS:{metrics.fps:4.0f} "
                    f"Inf:{metrics.process_time * 1000:4.1f}ms "
                    f"Tr:{training_stats['training_count']:5d} "
                    f"Sync:{training_stats['weight_sync_count']:4d} "
                    f"Q:{metrics.queue_size:2d}/{metrics.queue_max_size} "
                    f"Epi:{episode_count:3d} "
                    f"Step:{step_count:5d} "
                    f"Eps:{epsilon:.2f} "
                    f"Buf:{buffer.size():5d} "
                    f"Rwd:{curr_reward if not is_evaluating else 0.0:.1f} "
                    f"Loss:{training_stats['last_loss']:.4f} "
                    f"Q:{training_stats['q_mean']:.2f}"
                )
                print("\x1b[K" + line, end="\r", flush=True)

            previous_state = current_state
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print(f"\nReplay buffer: {buffer.size()} / {buffer.max_size} transitions")
        if buffer.size() > 0:
            # Print last 10 transitions to verify rewards are recorded
            sample_size = min(10, buffer.size())
            with buffer.lock:
                print(f"Last {sample_size} transitions in buffer:")
                for i in range(-sample_size, 0):
                    action = buffer._actions[i]
                    reward = buffer._rewards[i]
                    done = buffer._dones[i]
                    print(f"  [{i}]: action={action}, reward={reward}, done={done}")
        print("Stopping training thread...")
        training_thread.stop()
        gst_pipeline.stop()
        game_interface.close()


if __name__ == "__main__":
    main()
