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
from src.env.xlib_interface import XlibGameInterface
from src.training.local_training_thread import LocalTrainingThread
from src.utils.metrics import MetricsTracker
from src.utils.tensorboard_logger import TensorBoardLogger


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
    print(f"Input method: {'Xlib' if config.headless else 'evdev'}")
    if config.headless:
        print(f"Headless mode: display {config.display_name}")
        print(f"Browser window: {config.browser_width}x{config.browser_height}")
        print(f"Crop region: ({config.crop_x},{config.crop_y}) {config.crop_width}x{config.crop_height}")
        if config.udp_port > 0:
            print(f"UDP full view: port {config.udp_port}")
        if config.udp_port_agent > 0:
            print(f"UDP agent view: port {config.udp_port_agent}")

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

    if config.headless:
        if game_config.action_keys_str is None:
            raise ValueError(f"Game {config.game_name} does not have action_keys_str configured")
        action_keys_xlib = [ord(k) if k else 0 for k in game_config.action_keys_str]
        print(f"Using Xlib input on display {config.display_name}")
        game_interface = XlibGameInterface(
            action_keys=action_keys_xlib,
            reset_key=game_config.reset_key,
            display_name=config.display_name,
        )
    else:
        game_interface = GameInterface(game_config.action_keys)

    state_monitor = StateMonitor()
    frame_processor = FrameProcessor(config, game_config.frame_stack)
    metrics_tracker = MetricsTracker()

    action_counts = dict.fromkeys(range(game_config.n_actions), 0)

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

    min_episode_steps = 1

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
    reset_delay_frames: int = 1
    reset_cooldown_counter: int = 0
    reset_cooldown_frames: int = 5

    frame_skip_counter = 0
    current_action: int | None = None

    tb_logger = TensorBoardLogger(log_dir="runs")
    print(f"TensorBoard logging initialized at {tb_logger.log_dir}")

    try:
        while episode_count < config.max_episodes:
            pull_frame_from_appsink(appsink, frame_processor)

            current_state = frame_processor.get_state()

            if current_state is None or previous_state is None:
                previous_state = current_state
                time.sleep(0.001)
                continue

            is_game_over = state_monitor.is_game_over(frame_processor.game_over_buffer)

            # Handle game over and surgical reset
            if is_game_over and not is_resetting:
                # Add terminal transition with penalty for game over
                if not is_evaluating and previous_state is not None and current_action is not None:
                    reward = -1.0
                    buffer.add(previous_state.squeeze(0), current_action, reward, current_state.squeeze(0), done=True)
                    curr_reward += reward

                is_resetting = True
                reset_delay_counter = 0
                reset_cooldown_counter = 0
                continue
            elif is_game_over and reset_delay_counter < reset_delay_frames:
                reset_delay_counter += 1
                continue
            elif is_game_over:
                # Wait has passed, send space to reset the game
                game_interface.reset_game()
                reset_cooldown_counter = 0

                # Skip normal action and recording during reset
                continue
            elif is_resetting:
                # Game is running again, but wait for cooldown to pass
                reset_cooldown_counter += 1
                if reset_cooldown_counter >= reset_cooldown_frames:
                    # Reset complete - game is now stably running
                    is_resetting = False
                else:
                    # Still in cooldown, skip action and recording
                    continue

                # Handle episode completion
                if is_evaluating:
                    if eval_step_count > 1:
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
                        status = "[GAME OVER]"
                        episode_reward = curr_reward
                        episode_length = episode_steps
                        print(
                            f"\r\x1b[K{status} Episode {episode_count} complete. "
                            f"Steps: {step_count}, Reward: {episode_reward:.2f}\n",
                            end="",
                            flush=True,
                        )
                        tb_logger.log_episode(episode_count, episode_reward, episode_length)
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
                action_counts[action] += 1

                # Reward: survival steps are neutral (0), only game over gives penalty (-1)
                reward = 0.0

                # Record transition (terminal only on natural game over)
                buffer.add(previous_state.squeeze(0), action, reward, current_state.squeeze(0), is_game_over)
                curr_reward += reward
                episode_steps += 1

                # Update target network locally
                if step_count > 0 and step_count % 1000 == 0:
                    local_trainer.update_target()

                # Save checkpoint
                if step_count % config.checkpoint_freq == 0:
                    checkpoint_path = config.checkpoint_path / f"checkpoint_{step_count}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    local_trainer.save_checkpoint(str(checkpoint_path), step_count)
                    print(f"\n[CHECKPOINT] Saved checkpoint to {checkpoint_path}")

                step_count += 1

            metrics = metrics_tracker.update(
                processing_time, frame_processor.queue_size, frame_processor.max_queue_size
            )

            training_stats = training_thread.get_stats()
            loss = training_stats["last_loss"]

            if not is_evaluating and training_stats["training_count"] > 0 and step_count % 10 == 0:
                tb_logger.log_training_metrics(
                    step_count,
                    loss=loss,
                    q_mean=training_stats["q_mean"],
                    q_max=None,
                )

            if not is_evaluating and step_count % 100 == 0:
                tb_logger.log_system_metrics(
                    step_count,
                    epsilon=epsilon,
                    fps=metrics.fps,
                    buffer_size=buffer.size(),
                )
                tb_logger.log_action_distribution(step_count, action_counts, game_config.action_names)
                action_counts = dict.fromkeys(range(game_config.n_actions), 0)

            eval_tag = "[EVAL] " if is_evaluating else ""

            if is_evaluating:
                action_str = "--/--/--"
            else:
                total_actions = sum(action_counts.values())
                action_freqs = [
                    action_counts.get(i, 0) / total_actions if total_actions > 0 else 0
                    for i in range(game_config.n_actions)
                ]
                action_str = "/".join(f"{f:.2f}" for f in action_freqs)

            line1 = (
                f"{eval_tag}"
                f"FPS:{metrics.fps:4.0f} "
                f"Inf:{metrics.process_time * 1000:4.1f}ms "
                f"Tr:{training_stats['training_count']:5d} "
                f"Sync:{training_stats['weight_sync_count']:4d} "
                f"Q:{metrics.queue_size:2d}/{metrics.queue_max_size} "
                f"Epi:{episode_count:3d} "
                f"Step:{step_count:5d} "
                f"Eps:{epsilon:.2f}"
            )
            line2 = (
                f"Buf:{buffer.size():5d} "
                f"Rwd:{curr_reward if not is_evaluating else 0.0:.1f} "
                f"Loss:{training_stats['last_loss']:.4f} "
                f"Q:{training_stats['q_mean']:.2f} "
                f"Act:{action_str}"
            )
            print(f"\x1b[K{line1}\n\x1b[K{line2}\x1b[A", end="\r", flush=True)

            previous_state = current_state
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        tb_logger.close()
        print(f"\\nReplay buffer: {buffer.size()} / {buffer.max_size} transitions")
        if buffer.size() > 0:
            # Print last 50 transitions to verify rewards are recorded
            sample_size = min(50, buffer.size())
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
