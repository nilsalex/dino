import gi

gi.require_version("Gst", "1.0")
import os
import threading
import time
from queue import Queue

import numpy as np
import torch
from gi.repository import Gst

from dqn import DQN

CROP_LEFT = 600
CROP_RIGHT = 600
CROP_TOP = 100
CROP_BOTTOM = 700
OUTPUT_WIDTH = 80
OUTPUT_HEIGHT = 40
CHANNELS = 3
FPS = 30
N_ACTIONS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(OUTPUT_HEIGHT, OUTPUT_WIDTH, N_ACTIONS).to(device)
frame_buffer = np.empty((OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS), dtype=np.uint8)
input_tensor = torch.empty((1, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH), device=device)

from screenshare import init_screenshare

screenshare_params = init_screenshare()

Gst.init()

pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} ! "
    "videoconvert ! "
    "videoscale ! "
    "videorate ! "
    f"video/x-raw,format=RGB,width={OUTPUT_WIDTH},height={OUTPUT_HEIGHT},framerate={FPS}/1 ! "
    "appsink name=sink emit-signals=false sync=true drop=true"
)
appsink = pipeline.get_by_name("sink")

frame_queue = Queue(maxsize=2)
exit_event = threading.Event()


def pull_frames():
    while not exit_event.is_set():
        sample = appsink.emit("try-pull-sample", 100 * Gst.MSECOND)

        if not sample:
            print("no sample")
            continue

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)

        if success:
            try:
                frame = np.ndarray(shape=(OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS), dtype=np.uint8, buffer=map_info.data)
                frame_queue.put(frame.copy())
            except:
                print("error")
            finally:
                buffer.unmap(map_info)


def start_pipeline():
    """Start pipeline and wait for it to be ready."""
    try:
        os.fstat(screenshare_params.fd)
        print(f"FD {screenshare_params.fd} is valid")
    except OSError as e:
        print(f"FD {screenshare_params.fd} is invalid: {e}")
        raise

    ret = pipeline.set_state(Gst.State.PLAYING)

    if ret == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Failed to start pipeline")

    # Wait for state change to complete
    state_change = pipeline.get_state(5 * Gst.SECOND)
    print(f"Pipeline state: {state_change.state.value_nick}")

    if state_change.state != Gst.State.PLAYING:
        raise RuntimeError(f"Pipeline failed to reach PLAYING state: {state_change}")

    return True


def preprocess(frame):
    """In-place preprocessing to avoid allocations."""
    np_normalized = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor.copy_(torch.from_numpy(np_normalized).unsqueeze(0))
    return input_tensor


pull_thread = threading.Thread(target=pull_frames)

if __name__ == "__main__":
    try:
        start_pipeline()
        pull_thread.start()

        frame_count = 0
        frame_times = []  # (timestamp, frame_time) tuples

        last_time = time.perf_counter()

        while True:
            frame = frame_queue.get()
            now = time.perf_counter()

            process_start = time.perf_counter()

            # Get action
            state = preprocess(frame)
            action = model.select_action(state, epsilon=0.1)

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
                f"a: {action:5d} | "
                f"FPS: {fps:5.1f} | "
                f"Avg: {avg_frame_time * 1000:5.1f}ms | "
                f"Proc: {process_time * 1000:5.1f}ms | "
                f"Queue: {frame_queue.qsize()}/{frame_queue.maxsize} ({queue_latency_ms:4.1f}ms)",
                end="\r",
            )

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        exit_event.set()
        pull_thread.join(timeout=0.5)
        pipeline.set_state(Gst.State.NULL)
        screenshare_params.main_loop.quit()
