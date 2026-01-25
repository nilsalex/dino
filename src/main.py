import gi

from game_over import check_game_over

gi.require_version("Gst", "1.0")
import os
import sys
import time
from collections import deque
from queue import Empty, Full, Queue

import numpy as np
import torch
from evdev import UInput
from evdev import ecodes as e
from gi.repository import GLib, Gst
from PIL import Image

from dqn import DQN

SAVE_FRAMES = True
SAVE_DIR = "./debug_frames"

if SAVE_FRAMES:
    os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_WIDTH = 84
OUTPUT_HEIGHT = 84
CHANNELS = 1
N_ACTIONS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(OUTPUT_HEIGHT, OUTPUT_WIDTH, N_ACTIONS).to(device)
frame_buffer: deque = deque(maxlen=4)


# ========================
# GLOBALS
# ========================
frame_queue: Queue = Queue(maxsize=2)
input_frame_count = 0

frame_times = []
last_time = time.perf_counter()


def on_message(_bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
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


def on_new_sample(sink, _data):
    global input_frame_count

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if success:
        input_frame_count += 1
        try:
            frame = np.ndarray(shape=(OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.uint8, buffer=map_info.data)

            if SAVE_FRAMES and input_frame_count < 100:
                img = Image.fromarray(frame, mode="L")
                path = f"{SAVE_DIR}/frame_{input_frame_count:04d}.png"
                img.save(path)
                print(f"Saved: {path}")

            try:
                frame_queue.put_nowait(frame.copy())
            except Full:
                print("Warning: frame queue full, could not enqueue frame")
                pass

        finally:
            buffer.unmap(map_info)

    return Gst.FlowReturn.OK


def preprocess():
    stacked = np.stack(list(frame_buffer), axis=0)
    tensor = torch.from_numpy(stacked).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def run_action(ui, action):
    match action:
        case 1:
            ui.write(e.EV_KEY, e.KEY_UP, 1)
            ui.write(e.EV_KEY, e.KEY_UP, 0)
            ui.syn()
        case _:
            pass


def run_timed(f, *args):
    global frame_times, last_time

    now = time.perf_counter()
    process_start = time.perf_counter()

    if not f(*args):
        return True

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

    print(
        f"FPS: {fps:5.1f} | "
        f"Avg: {avg_frame_time * 1000:5.1f}ms | "
        f"Proc: {process_time * 1000:5.1f}ms | "
        f"Queue: {frame_queue.qsize()}/{frame_queue.maxsize} ({queue_latency_ms:4.1f}ms)",
        end="\r",
    )

    return True


def process_frames(ui):
    try:
        frame = frame_queue.get_nowait()
        frame_buffer.append(frame.copy())

        if len(frame_buffer) < 4:
            print(f"Buffering frames... {len(frame_buffer)}/4")
            return True

        if check_game_over(frame_buffer):
            print("GAME OVER")

        state = preprocess()
        action = model.select_action(state, epsilon=0.1)

        run_action(ui, action)
        return True

    except Empty:
        return False


def main():
    # Init evdev input
    ui = UInput()

    # Init GStreamer
    Gst.init(sys.argv)

    loop = GLib.MainLoop()

    # Init pipeline
    pipeline = Gst.parse_launch(
        "v4l2src device=/dev/video0 do-timestamp=true ! "
        + "videoscale ! "
        + "video/x-raw,width=84,height=84,framerate=30/1 ! "
        + "queue max-size-buffers=1 leaky=downstream ! "
        + "videoconvert ! "
        + "video/x-raw,format=GRAY8 ! "
        + "appsink name=appsink emit-signals=true max-buffers=2 drop=true"
    )
    sink = pipeline.get_by_name("appsink")
    sink.connect("new-sample", on_new_sample, None)

    # Connect bus messages
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Start
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline started, running main loop...")

    GLib.idle_add(run_timed, process_frames, ui)

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
