# ruff: noqa: E402
import gi

gi.require_version("Gst", "1.0")

import time

import numpy as np
from gi.repository import Gst

from src.agents.dqn_agent import DQNAgent
from src.capture.frame_processor import FrameProcessor
from src.capture.gstreamer import GStreamerPipeline
from src.core.config import Config
from src.core.types import Experience, State
from src.env.game_interface import GameInterface
from src.env.state_monitor import StateMonitor
from src.replay.buffer import ReplayBuffer
from src.utils.metrics import MetricsTracker


def pull_frame_from_appsink(appsink: Gst.Element, frame_processor: FrameProcessor) -> None:
    sample = appsink.emit("pull-sample")  # type: ignore[arg-type]
    if not sample:
        return

    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if success:
        try:
            frame_arr = np.ndarray(
                shape=(frame_processor.config.output_height, frame_processor.config.output_width),
                dtype=np.uint8,
                buffer=map_info.data,
            )
            frame_processor.add_frame(frame_arr)
        finally:
            buffer.unmap(map_info)


def process_frames(  # type: ignore[misc]
    game_interface: GameInterface,
    state_monitor: StateMonitor,
    agent: DQNAgent,
    frame_processor: FrameProcessor,
    metrics_tracker: MetricsTracker,
    replay_buffer: ReplayBuffer | None = None,
    reset_phase: list[int] | None = None,
    state: State | None = None,
    next_state: State | None = None,
) -> bool:
    if reset_phase is None:
        reset_phase = [0]

    process_start = time.perf_counter()

    if state is None or next_state is None:
        return True

    is_game_over = state_monitor.is_game_over(frame_processor.frame_buffer)
    reset_status = ""
    loss_str = ""

    if is_game_over:
        if reset_phase[0] == 0:
            game_interface.reset_game()
            reset_phase[0] = 1
            reset_status = "RESET "
        elif reset_phase[0] == 1:
            reset_status = "RESET "
            reset_phase[0] = 2
        elif reset_phase[0] == 2:
            reset_phase[0] = 0

    if not is_game_over and reset_phase[0] > 0:
        reset_phase[0] = 0

    should_save_checkpoint = agent.step_count > 0 and agent.step_count % agent.config.checkpoint_freq == 0
    if should_save_checkpoint:
        agent.save(str(agent.config.checkpoint_path / f"dqn_checkpoint_{agent.step_count}.pth"))
        agent.save(str(agent.config.checkpoint_path / "dqn_checkpoint.pth"))

    action = agent.act(state)
    game_interface.execute_action(action)

    reward = 0.1

    loss_str = ""
    if replay_buffer:
        replay_buffer.push(Experience(state, action, reward, next_state, False))

        if len(replay_buffer) >= agent.config.min_buffer_size:
            batch = replay_buffer.sample(agent.config.batch_size)
            training_metrics = agent.train_batch(batch)
            loss_str = f" | Loss: {training_metrics['loss']:.4f} | Epsilon: {training_metrics['epsilon']:.3f}"

    process_time = time.perf_counter() - process_start
    metrics = metrics_tracker.update(process_time, frame_processor.queue_size, frame_processor.max_queue_size)

    status = "OVER " if is_game_over else ""
    line = (
        f"{status}{reset_status}FPS:{metrics.fps:5.1f} "
        f"Avg:{metrics.avg_frame_time * 1000:5.1f}ms "
        f"P:{metrics.process_time * 1000:5.1f}ms "
        f"Q:{metrics.queue_size}/{metrics.queue_max_size} "
        f"B:{len(replay_buffer) if replay_buffer else 0}"
        f"{loss_str}"
    )
    print(f"\033[K{line}", end="\r", flush=True)

    return True


def main():
    config = Config()
    print("Starting DINO RL agent...")

    config.checkpoint_path.mkdir(parents=True, exist_ok=True)
    game_interface = GameInterface()
    state_monitor = StateMonitor()
    agent = DQNAgent(config)
    frame_processor = FrameProcessor(config)
    metrics_tracker = MetricsTracker()
    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    checkpoint_file = config.checkpoint_path / "dqn_checkpoint.pth"
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        agent.load(str(checkpoint_file))

    gst_pipeline = GStreamerPipeline(config)
    gst_pipeline.create_pipeline()
    gst_pipeline.start()
    print("GStreamer pipeline started")

    appsink = gst_pipeline.pipeline.get_by_name("appsink")  # type: ignore[arg-type]

    reset_phase = [0]
    previous_state = None

    try:
        while True:
            pull_frame_from_appsink(appsink, frame_processor)  # type: ignore[arg-type]
            current_state = frame_processor.get_state()

            if current_state is not None and previous_state is not None:
                process_frames(
                    game_interface,
                    state_monitor,
                    agent,
                    frame_processor,
                    metrics_tracker,
                    replay_buffer,
                    reset_phase,
                    previous_state,
                    current_state,
                )
            elif current_state is None:
                print("Buffering... ", end="\r", flush=True)

            previous_state = current_state
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gst_pipeline.stop()
        game_interface.close()


if __name__ == "__main__":
    main()
