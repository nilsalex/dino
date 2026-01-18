import gi

from screenshare import init_screenshare

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
import threading
import time
from queue import Queue

import numpy as np
from gi.repository import GObject, Gst  # type: ignore [missing-source]

# Initialize GStreamer
Gst.init(None)

screenshare_params = init_screenshare()

# Pipeline setup
pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "  # Match OpenCV default
    "appsink name=sink emit-signals=false sync=false"
)

appsink = pipeline.get_by_name("sink")  # type: ignore [attr-defined]

# -----------------------------------------------------------------------------
# CAPABILITIES RESOLUTION: Get actual frame format at runtime
# -----------------------------------------------------------------------------
pipeline.set_state(Gst.State.PAUSED)  # Pause to negotiate caps

print("waiting for first frame")

# Wait for first buffer to probe dimensions
sample = appsink.emit("pull-preroll")  # Blocking until first frame

print("got first frame")

if not sample:
    raise RuntimeError("Could not get initial sample to detect caps")

caps = sample.get_caps()
structure = caps.get_structure(0)

width = structure.get_int("width").value
height = structure.get_int("height").value
fmt = structure.get_string("format").value

print(f"Resolution detected: {width}x{height} [{fmt}]")

# Preallocate reusable buffers (zero-copy optimization)
frame_buffer = np.empty((height, width, 3), dtype=np.uint8)
timestamp_ns = None
frame_duration_ns = None

pipeline.set_state(Gst.State.PLAYING)

# -----------------------------------------------------------------------------
# FRAME QUEUE & WORKER THREAD
# -----------------------------------------------------------------------------
frame_queue = Queue(maxsize=2)  # Small buffer to prevent stalling
exit_flag = threading.Event()


def frame_puller_worker():
    """Dedicated thread for low-latency frame pulling"""
    while not exit_flag.is_set():
        sample = appsink.emit("try-pull-sample", 0)  # Non-blocking

        if sample is None:
            GObject.idle_add(GObject.timeout_add(10, lambda: None))  # Minimal sleep
            continue

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            continue

        try:
            # Copy frame data directly into preallocated buffer
            np.copyto(
                frame_buffer, np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data, order="C")
            )

            # Extract timing info
            ts = buffer.pts  # Presentation timestamp (nanoseconds)
            dur = buffer.duration

            # Put into queue with metadata
            frame_queue.put(
                (
                    frame_buffer.copy(),  # Copy if processing takes longer than frame interval
                    ts,
                    dur,
                )
            )

        finally:
            buffer.unmap(map_info)


pull_thread = threading.Thread(target=frame_puller_worker)
pull_thread.start()

# -----------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------
try:
    print("starting loop")
    while True:
        frame, timestamp, duration = frame_queue.get()

        # PROCESS FRAME HERE (example)
        # processed_frame = reverse_colors(frame)  # Replace with your processing

        # Timing diagnostics
        if timestamp_ns is not None:
            latency_ms = (time.monotonic_ns() - timestamp_ns) / 1e6
            fps = 1e9 / (timestamp - timestamp_ns) if timestamp - timestamp_ns > 0 else 0
            print(f"[Frame {timestamp / 1e9:.3f}s] FPS: {fps:.1f} | Proc Latency: {latency_ms:.1f}ms")

        timestamp_ns = timestamp  # Update previous timestamp

except KeyboardInterrupt:
    pass
finally:
    exit_flag.set()
    pull_thread.join(timeout=1.0)
    pipeline.set_state(Gst.State.NULL)
