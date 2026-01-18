import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
import threading
from queue import Queue

import cv2
import numpy as np
from gi.repository import Gst  # type: ignore [missing-source]

from screenshare import init_screenshare

# Initialize GStreamer
Gst.init(None)

screenshare_params = init_screenshare()

# Pipeline setup
pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "appsink name=sink emit-signals=false sync=false"
)

appsink = pipeline.get_by_name("sink")  # type: ignore [attr-defined]

# Shared state variables
frame_buffer = np.empty((0, 0, 0), dtype=np.uint8)
width, height = 0, 0
caps_initialized = threading.Event()


# -----------------------------------------------------------------------------
# CAPS PROBING WITH PAD PROBE
# -----------------------------------------------------------------------------
def on_have_caps(pad, _info):
    global frame_buffer, width, height

    caps = pad.get_current_caps()
    if not caps:
        return Gst.PadProbeReturn.OK

    structure = caps.get_structure(0)
    width = structure.get_int("width").value
    height = structure.get_int("height").value
    fmt = structure.get_string("format").value

    print(f"Detected resolution: {width}x{height} ({fmt})")

    # Preallocate buffer based on actual caps
    fmt_to_cv = {"BGRx": cv2.COLOR_BGRA2BGR, "RGBx": cv2.COLOR_RGBA2RGB, "BGR": None}
    if fmt not in fmt_to_cv:
        raise RuntimeError(f"Unsupported video format: {fmt}")

    frame_buffer = np.empty((height, width, 3), dtype=np.uint8)
    caps_initialized.set()

    return Gst.PadProbeReturn.REMOVE  # Remove probe after first caps


# Attach probe to appsink's sink pad
sink_pad = appsink.get_static_pad("sink")
sink_pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, on_have_caps)

# -----------------------------------------------------------------------------
# FRAME PULLING THREAD
# -----------------------------------------------------------------------------
frame_queue = Queue(maxsize=2)
exit_flag = threading.Event()


def frame_puller_worker():
    """Dedicated thread for frame acquisition"""
    while not exit_flag.is_set():
        print("worker")
        if not caps_initialized.is_set():
            continue  # Wait for caps initialization

        sample = appsink.emit("pull-sample")  # Blocking pull with minimal latency
        if sample is None:
            continue

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            continue

        try:
            # Copy into preallocated buffer
            frame_array = np.ndarray(
                shape=(height, width, 3 if frame_buffer.ndim == 3 else 1), dtype=np.uint8, buffer=map_info.data
            )
            np.copyto(frame_buffer, frame_array)

            # Extract timing metadata
            pts = buffer.pts
            duration = buffer.duration

            frame_queue.put((frame_buffer.copy(), pts, duration))  # Copy for safety

        finally:
            buffer.unmap(map_info)


pull_thread = threading.Thread(target=frame_puller_worker)
pull_thread.start()

print("thread started")

# -----------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------
try:
    # Start pipeline AFTER everything is initialized
    pipeline.set_state(Gst.State.PLAYING)

    print("state is playing")

    # Core processing loop
    while True:
        frame, pts, duration = frame_queue.get()

        # PROCESS YOUR FRAME HERE (example)
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Timing diagnostics
        timestamp_sec = pts / Gst.SECOND
        interval_ns = duration if duration != Gst.CLOCK_TIME_NONE else 16_666_666  # Default 60 Hz
        print(f"Frame @ {timestamp_sec:.3f}s | Interval: {interval_ns / 1e6:.2f}ms")

except KeyboardInterrupt:
    pass
finally:
    exit_flag.set()
    pull_thread.join(timeout=1.0)
    pipeline.set_state(Gst.State.NULL)
