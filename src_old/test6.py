import gi

from screenshare import init_screenshare

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
import threading
from queue import Queue

import numpy as np
from gi.repository import GLib, Gst

# Initialize screenshare FIRST
screenshare_params = init_screenshare()
Gst.init()

# Pipeline setup with verified params
PIPELINE_STR = (
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "video/x-raw,width=1920,height=1200,format=RGBx ! "  # Match your screen resolution
    "appsink name=sink emit-signals=true sync=false drop=true"
)

pipeline = Gst.parse_launch(PIPELINE_STR)
appsink = pipeline.get_by_name("sink")

# Force buffer format
appsink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGBx"))
appsink.set_property("max-buffers", 2)

# Preallocate buffer (RGBx uses 4 channels)
height, width = 1200, 1920
frame_buffer = np.empty((height, width, 4), dtype=np.uint8)


# ------------------------------------------------------
# DEBUGGING ESSENTIALS
# ------------------------------------------------------
def bus_watch(bus, message):
    if message.type == Gst.MessageType.EOS:
        print("End-of-stream")
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err.message}")
        print(f"Debug info: {debug}")
    return True


bus = pipeline.get_bus()
bus.add_watch(0, bus_watch)

# ------------------------------------------------------
# RELIABLE FRAME PULLING
# ------------------------------------------------------
frame_queue = Queue(maxsize=2)
exit_event = threading.Event()


def pull_frames():
    print("Pull thread: Running")
    while not exit_event.is_set():
        sample = appsink.emit("pull-sample")  # BLOCKING PULL

        if not sample:
            continue

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)

        if success:
            try:
                # Direct memory copy (no format conversion)
                np.copyto(frame_buffer, np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=map_info.data))
                frame_queue.put((frame_buffer.copy(), buffer.pts))
            finally:
                buffer.unmap(map_info)

    print("Pull thread: Exiting")


pull_thread = threading.Thread(target=pull_frames)

# Start everything
pipeline.set_state(Gst.State.PLAYING)
pull_thread.start()

# ------------------------------------------------------
# MAIN PROCESSING LOOP
# ------------------------------------------------------
try:
    last_ts = None
    frame_count = 0

    while True:
        frame, pts = frame_queue.get(timeout=3)
        frame_count += 1

        # DISPLAY TIMING INFO
        if last_ts:
            interval_ms = (pts - last_ts) / 1_000_000
            print(f"Frame {frame_count}: Interval {interval_ms:.2f}ms | Size {frame.shape}")

        last_ts = pts

        # PROCESS FRAME HERE
        # rgb_frame = frame[:, :, :3]  # Convert RGBx to RGB (slice last channel)

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Fatal: {e!s}")
finally:
    exit_event.set()
    pull_thread.join(timeout=0.5)
    pipeline.set_state(Gst.State.NULL)
    GLib.source_remove(bus_watch)
