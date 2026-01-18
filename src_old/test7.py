import gi

from screenshare import init_screenshare

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
import threading
import time
from queue import Queue

import numpy as np
from gi.repository import Gst

screenshare_params = init_screenshare()
Gst.init()

# Pipeline setup using dynamic parameters
pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "video/x-raw,format=RGBx ! "  # Force known format
    "appsink name=sink emit-signals=true sync=false"
)
appsink = pipeline.get_by_name("sink")

# Enable dynamic caps resolution
appsink.set_property("async", True)
appsink.set_property("sync", False)
appsink.set_property("enable-last-sample", False)


# ------------------------------------------------------
# DYNAMIC CAPS DETECTION
# ------------------------------------------------------
def initialize_dynamic_caps():
    """Capture first frame to detect screen resolution"""
    # Start pipeline temporarily
    pipeline.set_state(Gst.State.PLAYING)

    # Wait for first buffer (blocking but with timeout)
    sample = appsink.emit("try-pull-sample", 3 * Gst.SECOND)
    if not sample:
        raise RuntimeError("Failed to get initial frame for caps detection")

    caps = sample.get_caps()
    structure = caps.get_structure(0)

    # Extract parameters with defaults
    width = structure.get_value("width") or 1920
    height = structure.get_value("height") or 1080
    fmt = structure.get_value("format") or "RGBx"

    print(f"Dynamic resolution detected: {width}x{height} [{fmt}]")

    # Stop and reset pipeline for fresh start
    pipeline.set_state(Gst.State.NULL)
    return width, height, fmt


width, height, fmt = initialize_dynamic_caps()

# Preallocate buffer based on detected caps
channels = 4 if fmt.endswith("x") else 3
frame_buffer = np.empty((height, width, channels), dtype=np.uint8)

# ------------------------------------------------------
# ADJUSTED PIPELINE WITH DYNAMIC CAPS
# ------------------------------------------------------
# Recreate pipeline with format-preserving chain
pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    f"video/x-raw,width={width},height={height},format={fmt} ! "  # Use detected values
    "appsink name=sink emit-signals=true sync=false"
)
appsink = pipeline.get_by_name("sink")

# ------------------------------------------------------
# FRAME PULLING THREAD (DYNAMIC BUFFER SIZE)
# ------------------------------------------------------
frame_queue = Queue(maxsize=2)
exit_event = threading.Event()


def pull_frames():
    while not exit_event.is_set():
        sample = appsink.emit("pull-sample")

        if not sample:
            time.sleep(0.01)
            continue

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)

        if success:
            try:
                # Copy directly into pre-sized buffer
                frame = np.ndarray(shape=(height, width, channels), dtype=np.uint8, buffer=map_info.data)
                frame_queue.put(frame.copy())
            finally:
                buffer.unmap(map_info)

    print("Frame pulling stopped")


pull_thread = threading.Thread(target=pull_frames)

# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------
if __name__ == "__main__":
    try:
        pipeline.set_state(Gst.State.PLAYING)
        pull_thread.start()

        frame_count = 0
        while True:
            frame = frame_queue.get()
            frame_count += 1

            # PROCESSING EXAMPLE:
            # if fmt == "RGBx":
            #     rgb_frame = frame[:, :, :3]  # Slice to RGB

            print(f"Processing {width}x{height} frame: {frame_count}", end="\r")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        exit_event.set()
        pull_thread.join(timeout=0.5)
        pipeline.set_state(Gst.State.NULL)
