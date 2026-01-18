import gi

gi.require_version("Gst", "1.0")
import threading
from queue import Queue

import cv2
import numpy as np
from gi.repository import Gst  # type: ignore

from screenshare import init_screenshare

screenshare_params = init_screenshare()

Gst.init(None)
pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "appsink name=sink emit-signals=false sync=false"
)
appsink = pipeline.get_by_name("sink")  # type: ignore [attr-defined]

# Shared state
frame_buffer = None
caps_initialized = threading.Event()


# -----------------------------------------------------------------------------
# RELIABLE CAPS INITIALIZATION USING BLOCKING PULL
# -----------------------------------------------------------------------------
def initialize_caps():
    global frame_buffer

    pipeline.set_state(Gst.State.PLAYING)
    sample = appsink.emit("pull-sample")  # Blocking first frame

    if not sample:
        raise RuntimeError("Failed to get initial sample")

    caps = sample.get_caps()
    if not caps or caps.is_empty():
        raise RuntimeError("No caps detected in initial sample")

    # Iterate through all potential structures in caps
    for i in range(caps.get_size()):
        structure = caps.get_structure(i)
        if structure.has_name("video/x-raw"):
            break
    else:
        raise RuntimeError("No video/x-raw caps found")

    # SAFE field extraction
    width = structure.get_value("width") or 1920
    height = structure.get_value("height") or 1080
    fmt = structure.get_value("format") or "BGRx"  # PipeWire default

    print(f"Detected caps: {width}x{height} ({fmt})")
    print(f"Full structure: {structure.to_string()}")

    # Default to BGRx if unknown format
    valid_formats = ["BGR", "RGB", "BGRx", "RGBx"]
    if fmt not in valid_formats:
        print(f"Warning: Format {fmt} not in {valid_formats}, using BGRx")
        fmt = "BGRx"

    channels = 3 if fmt in ["BGR", "RGB"] else 4
    frame_buffer = np.empty((height, width, channels), dtype=np.uint8)
    caps_initialized.set()
    return fmt, width, height


fmt, width, height = initialize_caps()

print(f"{fmt}, {width}, {height}")

# -----------------------------------------------------------------------------
# FRAME PULLING THREAD (NOW SAFE WITH KNOWN CAPS)
# -----------------------------------------------------------------------------
frame_queue = Queue(maxsize=2)
exit_flag = threading.Event()


def frame_puller():
    """Thread: Pull frames continuously"""
    while not exit_flag.is_set():
        sample = appsink.emit("try-pull-sample", 10 * Gst.MSECOND)  # 10ms timeout

        if sample is None:
            continue

        buffer = sample.get_buffer()
        if buffer is None:
            continue

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            continue

        try:
            # Convert to OpenCV-compatible format if needed
            if fmt in ["BGRx", "RGBx"]:
                # For 4-channel formats, create view and convert
                frame = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=map_info.data)
                cv_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR if fmt == "BGRx" else cv2.COLOR_RGBA2RGB)
            else:
                # Direct 3-channel copy
                np.copyto(frame_buffer, np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data))
                cv_frame = frame_buffer

            frame_queue.put((cv_frame.copy(), buffer.pts, buffer.duration))

        finally:
            buffer.unmap(map_info)


pull_thread = threading.Thread(target=frame_puller)
pull_thread.start()

# -----------------------------------------------------------------------------
# PROCESSING LOOP
# -----------------------------------------------------------------------------
try:
    while True:
        frame, pts, duration = frame_queue.get()

        # Your processing here (example edge detection)
        processed = cv2.Canny(frame, 100, 200)

        # Diagnostics
        print(f"Frame {pts / 1e9:.3f}s | Size: {frame.shape[1]}x{frame.shape[0]}")

except KeyboardInterrupt:
    exit_flag.set()
    pull_thread.join(timeout=0.5)
    pipeline.set_state(Gst.State.NULL)
