import gi

gi.require_version("Gst", "1.0")
import os
import sys
import time
from queue import Empty, Queue

import numpy as np
import torch
from evdev import UInput
from evdev import ecodes as e
from gi.repository import GLib, Gst
from PIL import Image

from dqn import DQN
from screenshare import get_main_loop, init_screenshare

SAVE_FRAMES = False
SAVE_DIR = "./debug_frames"

os.makedirs(SAVE_DIR, exist_ok=True)

# ========================
# CONFIGURATION
# ========================
CROP_LEFT = 20
CROP_RIGHT = 980
CROP_TOP = 100
CROP_BOTTOM = 800
OUTPUT_WIDTH = 150
OUTPUT_HEIGHT = 50
CHANNELS = 3
FPS = 30
N_ACTIONS = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(OUTPUT_HEIGHT, OUTPUT_WIDTH, N_ACTIONS).to(device)
frame_buffer = np.empty((OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS), dtype=np.uint8)
input_tensor = torch.empty((1, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH), device=device)


# ========================
# GLOBALS
# ========================
frame_queue = Queue(maxsize=2)


def on_message(bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err.message}")
        print(f"Debug: {debug}")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err.message}")
    return True


input_frame_count = 0


def on_new_sample(sink, data):
    global input_frame_count

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if success:
        input_frame_count += 1
        try:
            frame = np.ndarray(shape=(OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS), dtype=np.uint8, buffer=map_info.data)

            # Save first few frames for inspection
            if SAVE_FRAMES:
                img = Image.fromarray(frame)
                path = f"{SAVE_DIR}/frame_{input_frame_count:04d}.png"
                img.save(path)
                print(f"Saved: {path}")

            # Non-blocking put - drop frame if queue is full
            try:
                frame_queue.put_nowait(frame.copy())
            except:
                pass  # Drop frame

        finally:
            buffer.unmap(map_info)

    return Gst.FlowReturn.OK


def create_pipeline(fd: int, node_id: int) -> Gst.Pipeline:
    pipeline = Gst.Pipeline.new("screenshare-pipeline")

    # src = Gst.ElementFactory.make("videotestsrc", "src")

    # Create elements
    src = Gst.ElementFactory.make("pipewiresrc", "src")
    src.set_property("fd", fd)
    src.set_property("path", str(node_id))
    src.set_property("do-timestamp", True)

    crop = Gst.ElementFactory.make("videocrop", "crop")
    crop.set_property("top", CROP_TOP)
    crop.set_property("bottom", CROP_BOTTOM)
    crop.set_property("left", CROP_LEFT)
    crop.set_property("right", CROP_RIGHT)

    convert = Gst.ElementFactory.make("videoconvert", "convert")
    scale = Gst.ElementFactory.make("videoscale", "scale")
    rate = Gst.ElementFactory.make("videorate", "rate")

    sink = Gst.ElementFactory.make("appsink", "sink")
    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers", 2)
    sink.set_property("drop", True)
    sink.set_property("sync", True)

    # Check elements created
    if not all([src, convert, scale, rate, sink]):
        print("ERROR: Not all elements could be created")
        sys.exit(1)

    # Add to pipeline
    pipeline.add(src)
    pipeline.add(crop)
    pipeline.add(convert)
    pipeline.add(scale)
    pipeline.add(rate)
    pipeline.add(sink)

    # Link with caps filter
    caps = Gst.Caps.from_string(f"video/x-raw,format=RGB,width={OUTPUT_WIDTH},height={OUTPUT_HEIGHT},framerate={FPS}/1")

    ret = src.link(crop)
    ret = ret and crop.link(convert)
    ret = ret and convert.link(scale)
    ret = ret and scale.link(rate)
    ret = ret and rate.link_filtered(sink, caps)

    if not ret:
        print("ERROR: Elements could not be linked")
        sys.exit(1)

    # Connect sample callback
    sink.connect("new-sample", on_new_sample, None)

    return pipeline


def preprocess(frame):
    """In-place preprocessing to avoid allocations."""
    np_normalized = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor.copy_(torch.from_numpy(np_normalized).unsqueeze(0))
    return input_tensor


frame_count = 0
frame_times = []
last_time = time.perf_counter()
action_0 = 0
action_1 = 0


def process_frames(ui):
    """Called by GLib when idle - process queued frames."""
    global frame_count, frame_times, last_time, action_0, action_1

    try:
        frame = frame_queue.get_nowait()
        now = time.perf_counter()

        process_start = time.perf_counter()

        # Get action
        state = preprocess(frame)
        action = model.select_action(state, epsilon=0.1)

        match action:
            case 1:
                action_1 += 1
                ui.write(e.EV_KEY, e.KEY_UP, 1)
                ui.write(e.EV_KEY, e.KEY_UP, 0)
                ui.syn()
            case _:
                action_0 += 1
                pass

        process_time = time.perf_counter() - process_start

        frame_time = now - last_time
        last_time = now

        # Add to rolling window
        frame_times.append((now, frame_time))

        # Remove entries older than 1 second
        cutoff = now - 1.0
        while frame_times and frame_times[0][0] < cutoff:
            frame_times.pop(0)

        # Calculate averages over 1-second window
        if frame_times:
            fps = len(frame_times)  # Frames in last second = FPS
            avg_frame_time = sum(ft for _, ft in frame_times) / len(frame_times)
        else:
            fps = 0
            avg_frame_time = 0

        queue_latency_ms = frame_queue.qsize() * avg_frame_time * 1000

        frame_count += 1

        print(
            # f"Frame: {frame_count:6d} | "
            f"0: {action_0:5d} | "
            f"1: {action_1:5d} | "
            f"FPS: {fps:5.1f} | "
            f"Avg: {avg_frame_time * 1000:5.1f}ms | "
            f"Proc: {process_time * 1000:5.1f}ms | "
            f"Queue: {frame_queue.qsize()}/{frame_queue.maxsize} ({queue_latency_ms:4.1f}ms)",
            end="\r",
        )

    except Empty:
        pass
    return True  # Keep calling


def main():
    ui = UInput()

    # Init screenshare first (sets up DBus main loop integration)
    screenshare_params = init_screenshare()

    # Init GStreamer
    Gst.init(sys.argv)

    # Get the main loop created by screenshare
    loop = get_main_loop()

    # Create pipeline
    pipeline = create_pipeline(screenshare_params.fd, screenshare_params.node_id)

    # Connect bus messages
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Start
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline started, running main loop...")

    GLib.idle_add(process_frames, ui)

    # Run main loop
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping...")

    ui.close()

    # Cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
