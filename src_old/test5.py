import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
import threading
import time
from queue import Queue

import numpy as np
from gi.repository import Gst

from screenshare import init_screenshare

screenshare_params = init_screenshare()

Gst.init(None)

# Create identity element with signal
identity = Gst.ElementFactory.make("identity")
identity.set_property("signal-handoffs", True)
identity.connect("handoff", lambda *args: print("Buffer flowing!"))

# Build pipeline with approved node ID
PIPELINE_STR = (
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "video/x-raw,format=RGBx ! "  # Force format
    "identity name=debug ! "
    "appsink name=sink emit-signals=false sync=false"
)

pipeline = Gst.parse_launch(PIPELINE_STR)
appsink = pipeline.get_by_name("sink")

# Retrieve and configure debug element
debug = pipeline.get_by_name("debug")
debug.connect("handoff", lambda *args: print("Buffer detected!"))


# Configure appsink for optimal performance
appsink.set_property("enable-last-sample", False)
appsink.set_property("max-buffers", 2)
appsink.set_property("drop", True)

# -----------------------------------------------------------
# CAPS INITIALIZATION (HARDCODED FOR SESSION)
# -----------------------------------------------------------
width, height = 1920, 1200
fmt = "RGBx"
frame_buffer = np.empty((height, width, 4), dtype=np.uint8)

# -----------------------------------------------------------
# VERIFIED FRAME PULLER THREAD
# -----------------------------------------------------------
frame_queue = Queue(maxsize=2)
exit_flag = threading.Event()


def frame_puller():
    """Optimized thread using blocking pull"""
    print("Puller: Starting")
    while not exit_flag.is_set():
        sample = appsink.emit("pull-sample")  # BLOCKING WAIT

        if not sample:
            time.sleep(0.1)
            continue

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                # Direct buffer copy without format conversion
                np.copyto(frame_buffer, np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=map_info.data))
                frame_queue.put(frame_buffer.copy())
            finally:
                buffer.unmap(map_info)

    print("Puller: Exiting")


pull_thread = threading.Thread(target=frame_puller, daemon=True)

# -----------------------------------------------------------
# START PIPELINE
# -----------------------------------------------------------
pipeline.set_state(Gst.State.PLAYING)
pull_thread.start()

# -----------------------------------------------------------
# MAIN LOOP (SIMPLIFIED)
# -----------------------------------------------------------
try:
    frame_count = 0
    while True:
        frame = frame_queue.get(timeout=3)
        frame_count += 1
        print(f"Received frame {frame_count}")

except Exception as e:
    print(f"Terminated: {e!s}")
finally:
    exit_flag.set()
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")
