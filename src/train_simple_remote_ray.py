"""Main training loop with decoupled background training and local inference."""


# ruff: noqa: E402

import gi

gi.require_version("Gst", "1.0")

import random
import time

import ray
import torch
from gi.repository import Gst

from src.actors.local_inference_model import LocalInferenceModel
from src.agents.simple_ray_trainer import SimpleRemoteTrainer
from src.buffers.thread_safe_buffer import ThreadSafeExperienceBuffer
from src.capture.frame_processor import FrameProcessor
from src.capture.gstreamer import GStreamerPipeline
from src.core.config import Config
from src.env.game_interface import GameInterface
from src.env.state_monitor import StateMonitor
from src.training.training_thread import TrainingThread
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

    ray_address = "ray://localhost:10001"
    print(f"Connecting to Ray server at {ray_address}")
    ray.init(address=ray_address, ignore_reinit_error=True)

    print("Creating remote trainer...")
    remote_trainer = SimpleRemoteTrainer.remote(n_actions=config.n_actions)

    print("Initializing thread-safe experience buffer...")
    buffer = ThreadSafeExperienceBuffer(config)

    print("Initializing local inference model...")
    local_model = LocalInferenceModel(n_actions=config.n_actions, device=config.device or torch.device("cpu"))

    game_interface = GameInterface()
    state_monitor = StateMonitor()
    frame_processor = FrameProcessor(config)
    metrics_tracker = MetricsTracker()

    def on_weights_updated(state_dict: dict[str, torch.Tensor]) -> None:
        """Callback to update local model when weights sync from remote."""

        local_model.update_state_dict(state_dict)

    print("Starting background training thread...")
    training_thread = TrainingThread(config, buffer, remote_trainer, on_weights_updated)
    training_thread.start()

    epsilon = config.epsilon_start
    step_count = 0
    episode_count = 0

    is_evaluating = True
    eval_step_count = 0
    eval_episode_count = 1
    best_eval_score = 0

    print("Starting training with remote GPU...")
    print("Make sure Chrome Dino game is open and visible!")
    print(f"[EVAL] Starting initial greedy evaluation episode (baseline)\n")

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
    reset_phase = 0

    try:
        while episode_count < config.max_episodes:
            pull_frame_from_appsink(appsink, frame_processor)

            current_state = frame_processor.get_state()

            if current_state is None or previous_state is None:
                previous_state = current_state
                time.sleep(0.001)
                continue

            is_game_over = state_monitor.is_game_over(frame_processor.frame_buffer)

            if is_game_over:
                if reset_phase == 0:
                    game_interface.reset_game()
                    reset_phase = 1
                elif reset_phase == 1:
                    reset_phase = 2
                elif reset_phase == 2:
                    reset_phase = 3
            else:
                if reset_phase == 3:
                    if is_evaluating:
                        if eval_step_count > 5:
                            print(
                                f"\\n[EVAL] Episode {eval_episode_count} complete. Steps: {eval_step_count}\\n",
                                end="",
                                flush=True,
                            )
                            if eval_step_count > best_eval_score:
                                best_eval_score = eval_step_count
                                print(f"[BEST] New evaluation score: {best_eval_score}\\n")
                            is_evaluating = False
                            eval_step_count = 0
                        else:
                            print(f"[EVAL] Episode too short ({eval_step_count} steps), restarting eval...")
                            reset_phase = 1
                            eval_step_count = 0
                            previous_state = None
                            continue
                    else:
                        episode_count += 1
                        print(f"\\nEpisode {episode_count} complete. Steps: {step_count}\\n", end="", flush=True)

                        if episode_count % 50 == 0:
                            is_evaluating = True
                            eval_episode_count += 1
                            print(f"[EVAL] Starting greedy evaluation episode {eval_episode_count}\\n")
                    reset_phase = 0
                elif reset_phase > 0:
                    reset_phase = 0
                elif reset_phase > 0:
                    reset_phase = 0

            # Local inference for fast action selection
            processing_start = time.perf_counter()
            if is_evaluating or (buffer.size() > 0 and random.random() >= epsilon):
                action = local_model.get_action(previous_state)
            else:
                action = random.randint(0, config.n_actions - 1)

            # Decay epsilon
            if step_count < config.epsilon_decay:
                epsilon = max(
                    config.epsilon_end,
                    config.epsilon_start
                    - (config.epsilon_start - config.epsilon_end) * (step_count / config.epsilon_decay),
                )

            game_interface.execute_action(int(action))
            processing_time = time.perf_counter() - processing_start

            if is_evaluating:
                eval_step_count += 1
                step_count += 1
            else:
                reward = 0.1 + (-0.02 if action != 0 else 0)
                buffer.add(previous_state.squeeze(0), action, reward, current_state.squeeze(0), is_game_over)

                # Update target network remotely periodically
                if step_count > 0 and step_count % 1000 == 0:
                    remote_trainer.update_target.remote()  # type: ignore[arg-type]

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
                    f"Loss:{training_stats['last_loss']:.4f}"
                )
                print("\x1b[K" + line, end="\r", flush=True)

            previous_state = current_state
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Stopping training thread...")
        training_thread.stop()
        gst_pipeline.stop()
        game_interface.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
