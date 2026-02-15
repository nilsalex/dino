"""Debug script to test game over detection while playing manually."""

from collections import deque

import gi

gi.require_version("Gst", "1.0")

import time

import numpy as np
from gi.repository import Gst

from src.capture.gstreamer import GStreamerPipeline
from src.core.config import Config
from src.core.constants import GAME_OVER_VARIANCE_THRESHOLD
from src.env.pending_buffer import PendingTransitionBuffer
from src.env.state_monitor import StateMonitor


def main() -> None:
    config = Config()
    pipeline = GStreamerPipeline(config)
    pipeline.create_pipeline()

    state_monitor = StateMonitor(game_over_window=config.game_over_window)
    pending_buffer = PendingTransitionBuffer(max_size=config.pending_buffer_size)
    frame_buffer: deque[np.ndarray] = deque(maxlen=config.game_over_window + 1)
    frame_count = 0

    def on_sample(sink: object, _data: object) -> int:
        nonlocal frame_count
        sample = sink.emit("try-pull-sample", 100000000)  # type: ignore[union-attr]
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

        pending_buffer.add(frame_count, 0)  # type: ignore[arg-type]

        if len(frames) >= 2:
            stacked = np.stack(frames[-2:], axis=0)
            variance = np.mean(np.var(stacked, axis=0)) * 10000

            if is_over and offset > 0:
                committed = []

                def commit(s: int, a: int, r: float, ns: int, d: bool) -> None:
                    committed.append((s, a, r, ns, d))

                pending_buffer.commit(commit, next_state=frame_count, game_over_offset=offset)  # type: ignore[arg-type]

                print(f"GAME OVER | frame={frame_count} | offset={offset}")
                for t in committed:
                    status = "TERMINAL" if t[4] else "normal"
                    print(f"  -> state={t[0]} action={t[1]} reward={t[2]:.1f} next_state={t[3]} done={t[4]} [{status}]")
            elif is_over:
                print(f"GAME OVER (continued) | frame={frame_count}")
            else:
                if len(pending_buffer) >= config.pending_buffer_size:
                    pending_buffer.commit(
                        lambda s, a, r, ns, d: None,
                        next_state=frame_count,  # type: ignore[arg-type]
                        game_over_offset=0,
                    )
                print(f"RUNNING | frame={frame_count} | variance={variance:.4f}")

        return 0

    pipeline.set_sample_callback("appsink", on_sample)
    print(f"Watching {config.crop_width}x{config.crop_height} at ({config.crop_x},{config.crop_y})")
    print(f"Game over window: {config.game_over_window}")
    print(f"Pending buffer size: {config.pending_buffer_size}")
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
