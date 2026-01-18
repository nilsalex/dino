import gi

gi.require_version("Gst", "1.0")
import random
import string
import threading
from dataclasses import dataclass

import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib


@dataclass
class ScreenshareParams:
    node_id: int
    fd: int
    main_loop: GLib.MainLoop  # Keep reference to running loop
    main_loop_thread: threading.Thread


def init_screenshare() -> ScreenshareParams:
    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    # Create ONE main loop that will run forever
    main_loop = GLib.MainLoop()

    sender_name = bus.get_unique_name()[1:].replace(".", "_")
    token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    portal = bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
    screencast = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")

    # Use threading events instead of quitting the loop
    session_response = {}
    session_ready = threading.Event()

    def session_response_handler(_response, results):
        session_response["handle"] = results.get("session_handle")
        session_ready.set()

    options = {"session_handle_token": token, "handle_token": token}
    request = screencast.CreateSession(options)
    bus.add_signal_receiver(
        session_response_handler,
        "Response",
        "org.freedesktop.portal.Request",
        path=request,
    )

    # Start main loop in background thread - it will run forever
    main_loop_thread = threading.Thread(target=main_loop.run, daemon=True)
    main_loop_thread.start()

    # Wait using threading Event instead of loop.quit()
    if not session_ready.wait(timeout=30):
        raise RuntimeError("Timeout creating screen share session")

    if "handle" not in session_response:
        raise RuntimeError("Failed to create session")

    session_handle = session_response["handle"]

    # Select sources
    select_response = {}
    select_ready = threading.Event()

    def select_response_handler(response, _results):
        if response == 0:
            select_response["ready"] = True
        else:
            select_response["error"] = "User cancelled source selection"
        select_ready.set()

    select_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    select_request = screencast.SelectSources(
        session_handle,
        {
            "types": dbus.UInt32(1),
            "multiple": dbus.Boolean(False),
            "handle_token": select_token,
        },
    )
    bus.add_signal_receiver(
        select_response_handler,
        "Response",
        "org.freedesktop.portal.Request",
        path=select_request,
    )

    if not select_ready.wait(timeout=30):
        raise RuntimeError("Timeout selecting source")

    if "error" in select_response:
        raise RuntimeError(f"Source selection failed: {select_response['error']}")

    # Start screen sharing
    start_response = {}
    start_ready = threading.Event()

    def start_response_handler(response, results):
        if response == 0:
            streams = results.get("streams", [])
            if streams:
                start_response["node_id"] = streams[0][0]
                if len(streams[0]) > 1:
                    stream_metadata = dict(streams[0][1])
                    if "size" in stream_metadata:
                        size = stream_metadata["size"]
                        start_response["width"] = int(size[0])
                        start_response["height"] = int(size[1])
                start_response["ready"] = True
        else:
            start_response["error"] = "User cancelled or error occurred"
        start_ready.set()

    print("Waiting for screen share approval...")
    start_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    request = screencast.Start(session_handle, "", {"handle_token": start_token})
    bus.add_signal_receiver(
        start_response_handler,
        "Response",
        "org.freedesktop.portal.Request",
        path=request,
    )

    if not start_ready.wait(timeout=30):
        raise RuntimeError("Timeout waiting for approval")

    if "error" in start_response:
        raise RuntimeError(f"Screen share failed: {start_response['error']}")

    # Open PipeWire remote
    fd_obj = screencast.OpenPipeWireRemote(session_handle, {})
    fd = fd_obj.take()

    print(f"✓ Screen share approved! PipeWire node ID: {start_response['node_id']}, FD: {fd}")

    return ScreenshareParams(
        node_id=start_response["node_id"],
        fd=fd,
        main_loop=main_loop,  # Keep it running!
        main_loop_thread=main_loop_thread,
    )
