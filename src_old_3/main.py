import gi
gi.require_version("Gst", "1.0")
import sys
import time
import random
import string
from queue import Queue, Empty

import dbus
from dbus.mainloop.glib import DBusGMainLoop
import numpy as np
from gi.repository import Gst, GLib

# ========================
# CONFIGURATION
# ========================
CROP_LEFT = 800
CROP_RIGHT = 800
CROP_TOP = 100
CROP_BOTTOM = 800
OUTPUT_WIDTH = 120
OUTPUT_HEIGHT = 60
CHANNELS = 3
FPS = 30

# ========================
# GLOBALS
# ========================
frame_queue = Queue(maxsize=2)
frame_count = 0
frame_times = []
pipeline = None
loop = None


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
    return True


def on_new_sample(sink, data):
    global frame_count, frame_times

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if success:
        try:
            frame = np.ndarray(
                shape=(OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS),
                dtype=np.uint8,
                buffer=map_info.data
            )
            try:
                frame_queue.put_nowait(frame.copy())
            except:
                pass

            now = time.perf_counter()
            frame_times.append(now)
            cutoff = now - 1.0
            while frame_times and frame_times[0] < cutoff:
                frame_times.pop(0)

            fps = len(frame_times)
            frame_count += 1
            print(f"Frame: {frame_count:6d} | FPS: {fps:5.1f}", end="\r")

        finally:
            buffer.unmap(map_info)

    return Gst.FlowReturn.OK


def wait_for_response(response_dict, key, timeout_sec=30):
    """Wait for response using the SAME main loop context."""
    start = time.time()
    context = GLib.MainContext.default()
    while key not in response_dict:
        if time.time() - start > timeout_sec:
            raise RuntimeError(f"Timeout waiting for {key}")
        # Iterate the main context instead of running a separate loop
        context.iteration(False)
        time.sleep(0.01)


def main():
    global pipeline, loop

    # Initialize everything in the same context
    DBusGMainLoop(set_as_default=True)
    Gst.init(sys.argv)
    
    loop = GLib.MainLoop()
    
    # ========================
    # PORTAL SETUP
    # ========================
    bus = dbus.SessionBus()
    sender_name = bus.get_unique_name()[1:].replace(".", "_")
    token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    portal = bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
    screencast = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")

    # Create session
    session_response = {}
    def session_handler(_response, results):
        session_response["handle"] = results.get("session_handle")

    request = screencast.CreateSession({"session_handle_token": token, "handle_token": token})
    bus.add_signal_receiver(session_handler, "Response", "org.freedesktop.portal.Request", path=request)
    wait_for_response(session_response, "handle")
    session_handle = session_response["handle"]

    # Select sources
    select_response = {}
    def select_handler(response, _results):
        select_response["done"] = True
        if response != 0:
            select_response["error"] = "User cancelled"

    select_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    request = screencast.SelectSources(
        session_handle,
        {"types": dbus.UInt32(1), "multiple": dbus.Boolean(False), "handle_token": select_token}
    )
    bus.add_signal_receiver(select_handler, "Response", "org.freedesktop.portal.Request", path=request)
    wait_for_response(select_response, "done")
    if "error" in select_response:
        raise RuntimeError(select_response["error"])

    # Start
    start_response = {}
    def start_handler(response, results):
        if response == 0:
            streams = results.get("streams", [])
            if streams:
                start_response["node_id"] = streams[0][0]
        else:
            start_response["error"] = "User cancelled"
        start_response["done"] = True

    print("Waiting for screen share approval...")
    start_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    request = screencast.Start(session_handle, "", {"handle_token": start_token})
    bus.add_signal_receiver(start_handler, "Response", "org.freedesktop.portal.Request", path=request)
    wait_for_response(start_response, "done")
    if "error" in start_response:
        raise RuntimeError(start_response["error"])

    node_id = start_response["node_id"]
    fd_obj = screencast.OpenPipeWireRemote(session_handle, {})
    fd = fd_obj.take()
    print(f"✓ Screen share ready! Node: {node_id}, FD: {fd}")

    # ========================
    # PIPELINE SETUP
    # ========================
    pipeline = Gst.Pipeline.new("screenshare")

    # src = Gst.ElementFactory.make("videotestsrc", "src")

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

    pipeline.add(src)
    pipeline.add(crop)
    pipeline.add(convert)
    pipeline.add(scale)
    pipeline.add(rate)
    pipeline.add(sink)

    caps = Gst.Caps.from_string(
        f"video/x-raw,format=RGB,width={OUTPUT_WIDTH},height={OUTPUT_HEIGHT},framerate={FPS}/1"
    )

    src.link(crop)
    crop.link(convert)
    # src.link(convert)
    # convert.link(crop)
    # crop.link(scale)
    convert.link(scale)
    scale.link(rate)
    rate.link_filtered(sink, caps)

    sink.connect("new-sample", on_new_sample, None)

    # Bus
    gst_bus = pipeline.get_bus()
    gst_bus.add_signal_watch()
    gst_bus.connect("message", on_message, loop)

    # ========================
    # RUN
    # ========================
    pipeline.set_state(Gst.State.PLAYING)
    print("Running...")

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
