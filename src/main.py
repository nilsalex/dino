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
from src.env.game_interface import GameInterface
from src.env.state_monitor import StateMonitor
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
) -> bool:
    process_start = time.perf_counter()

    state = frame_processor.get_state()

    if state is None:
        if not frame_processor.buffer_ready():
            print(f"Buffering frames... {len(frame_processor.frame_buffer)}/{frame_processor.config.frame_stack}")
        return True

    if state_monitor.is_game_over(frame_processor.frame_buffer):
        print("GAME OVER")
        return True

    action = agent.act(state)
    print(f"Executing action: {action}", end="\n")
    game_interface.execute_action(action)

    process_time = time.perf_counter() - process_start
    metrics = metrics_tracker.update(process_time, frame_processor.queue_size, frame_processor.max_queue_size)

    print(
        f"FPS: {metrics.fps:5.1f} | "
        f"Avg: {metrics.avg_frame_time * 1000:5.1f}ms | "
        f"Proc: {metrics.process_time * 1000:5.1f}ms | "
        f"Queue: {metrics.queue_size}/{metrics.queue_max_size} ({metrics.queue_latency_ms:4.1f}ms)",
        end="\r",
    )

    return True


def main():
    config = Config()

    game_interface = GameInterface()
    state_monitor = StateMonitor()
    agent = DQNAgent(config)
    frame_processor = FrameProcessor(config)
    metrics_tracker = MetricsTracker()

    gst_pipeline = GStreamerPipeline(config)
    gst_pipeline.create_pipeline()
    gst_pipeline.start()

    appsink = gst_pipeline.pipeline.get_by_name("appsink")  # type: ignore[arg-type]

    try:
        while True:
            pull_frame_from_appsink(appsink, frame_processor)  # type: ignore[arg-type]
            process_frames(game_interface, state_monitor, agent, frame_processor, metrics_tracker)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gst_pipeline.stop()
        game_interface.close()


if __name__ == "__main__":
    main()
