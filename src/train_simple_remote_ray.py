"""Main training loop with remote GPU batch training and local inference."""


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
from src.capture.frame_processor import FrameProcessor
from src.capture.gstreamer import GStreamerPipeline
from src.core.config import Config
from src.env.game_interface import GameInterface
from src.env.state_monitor import StateMonitor
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

    ray_address = "ray://52.208.110.9:10001"
    print(f"Connecting to Ray server at {ray_address}")
    ray.init(address=ray_address, ignore_reinit_error=True)

    print("Creating remote trainer...")
    remote_trainer = SimpleRemoteTrainer.remote(n_actions=config.n_actions)

    print("Initializing local inference model...")
    local_model = LocalInferenceModel(n_actions=config.n_actions, device=config.device or torch.device("cpu"))

    game_interface = GameInterface()
    state_monitor = StateMonitor()
    frame_processor = FrameProcessor(config)
    metrics_tracker = MetricsTracker()

    epsilon = config.epsilon_start
    step_count = 0
    episode_count = 0

    buffer_states = []
    buffer_actions = []
    buffer_rewards = []
    buffer_next_states = []
    buffer_dones = []

    print("Starting training with remote GPU...")
    print("Make sure Chrome Dino game is open and visible!")

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
    last_loss = 0.0

    try:
        while episode_count < 1000:
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
                    episode_count += 1
                    print(f"\nEpisode {episode_count} complete. Steps: {step_count}\n", end="", flush=True)
                    reset_phase = 0
                elif reset_phase > 0:
                    reset_phase = 0

            # Local inference for fast action selection
            processing_start = time.perf_counter()
            if len(buffer_states) > 0 and random.random() >= epsilon:
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

            # Store experience (squeeze batch dimension)
            reward = 0.1 + (-0.02 if action != 0 else 0)
            buffer_states.append(previous_state.squeeze(0))
            buffer_actions.append(action)
            buffer_rewards.append(reward)
            buffer_next_states.append(current_state.squeeze(0))
            buffer_dones.append(is_game_over)

            # Trim buffer
            if len(buffer_states) > config.replay_buffer_size:
                buffer_states.pop(0)
                buffer_actions.pop(0)
                buffer_rewards.pop(0)
                buffer_next_states.pop(0)
                buffer_dones.pop(0)

            # Batch training on remote GPU every N steps
            training_latency = 0.0
            if len(buffer_states) >= config.min_buffer_size and step_count % 100 == 0:
                indices = random.sample(range(len(buffer_states)), config.batch_size)

                state_batch = [buffer_states[i].detach().cpu().numpy() for i in indices]
                action_batch = [buffer_actions[i] for i in indices]
                reward_batch = [buffer_rewards[i] for i in indices]
                next_state_batch = [buffer_next_states[i].detach().cpu().numpy() for i in indices]
                done_batch = [buffer_dones[i] for i in indices]

                train_start = time.perf_counter()
                try:
                    loss_result = ray.get(
                        remote_trainer.train_step.remote(  # type: ignore[arg-type]
                            state_batch, action_batch, reward_batch, next_state_batch, done_batch
                        )
                    )
                    training_latency = time.perf_counter() - train_start
                    if isinstance(loss_result, dict):
                        last_loss = loss_result.get("loss", last_loss)
                except Exception as e:
                    print(f"Training error: {e}")

            # Update target network and sync weights periodically
            if step_count > 0 and step_count % 1000 == 0:
                remote_trainer.update_target.remote()  # type: ignore[arg-type]

                # Sync model weights from remote to local for inference
                remote_state = ray.get(remote_trainer.get_model_state.remote())  # type: ignore[arg-type]
                local_model.update_state_dict(remote_state)

            step_count += 1

            metrics = metrics_tracker.update(
                processing_time, frame_processor.queue_size, frame_processor.max_queue_size
            )

            if step_count % 10 == 0:
                line = (
                    f"FPS:{metrics.fps:4.0f} "
                    f"Inf:{metrics.process_time * 1000:4.1f}ms "
                    f"Tr:{training_latency * 1000:4.1f}ms "
                    f"Q:{metrics.queue_size:2d}/{metrics.queue_max_size} "
                    f"Epi:{episode_count:3d} "
                    f"Step:{step_count:5d} "
                    f"Eps:{epsilon:.2f} "
                    f"Buf:{len(buffer_states):5d} "
                    f"Loss:{last_loss:.4f}"
                )
                print("\x1b[K" + line, end="\r", flush=True)

            previous_state = current_state
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        gst_pipeline.stop()
        game_interface.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
