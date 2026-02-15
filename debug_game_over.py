"""Debug script to test game over detection while playing manually."""

from collections import deque

import gi

gi.require_version("Gst", "1.0")

import time

import numpy as np
from gi.repository import Gst

from src.capture.gstreamer import GStreamerPipeline
from src.core.config import Config
from src.env.state_monitor import StateMonitor


def main() -> None:
    config = Config()
    pipeline = GStreamerPipeline(config)
    pipeline.create_pipeline()

    state_monitor = StateMonitor(game_over_window=config.game_over_window)
    frame_buffer: deque[np.ndarray] = deque(maxlen=config.game_over_window + 1)
    frame_count = 0

    def on_sample(sink: object, _data: object) -> int:
        nonlocal frame_count
        sample = sink.emit("try-pull-sample", 100000000)
        if not sample:
            return 0

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return 0

        info = sample.get_caps().get_structure(0)
        w, h = info.get_value("width"), info.get_value("height")
        frame = np.ndarray((h, w), buffer=map_info.data, dtype=np.uint8).copy()
        buffer.unmap(map_info)

        frame_buffer.append(frame)
        frame_count += 1

        frames = list(frame_buffer)
        is_over, offset = state_monitor.check_game_over(frames)

        if len(frames) >= 2:
            stacked = np.stack(frames[-2:], axis=0)
            variance = np.mean(np.var(stacked, axis=0)) * 10000

            if is_over and offset > 0:
                print(f"GAME OVER | frame={frame_count} | offset={offset}")
            elif is_over:
                print(f"GAME OVER (continued) | frame={frame_count}")
            else:
                print(f"RUNNING | frame={frame_count} | variance={variance:.4f}")

        return 0

    pipeline.set_sample_callback("appsink", on_sample)
    print(f"Watching {config.crop_width}x{config.crop_height} at ({config.crop_x},{config.crop_y})")
    print(f"Game over window: {config.game_over_window}")
    print("Play manually. Ctrl+C to stop.")
    print("-" * 60)
    pipeline.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
