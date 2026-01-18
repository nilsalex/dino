import gi

gi.require_version("Gst", "1.0")
import random
import string
from dataclasses import dataclass

import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib


@dataclass
class ScreenshareParams:
    node_id: int
    fd: int
    bus: object
    session_handle: str


def init_screenshare() -> ScreenshareParams:
    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    sender_name = bus.get_unique_name()[1:].replace(".", "_")
    token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    session_handle = f"/org/freedesktop/portal/desktop/session/{sender_name}/{token}"
    portal = bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
    screencast = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")

    session_response = {}

    def session_response_handler(_response, results):
        session_response["handle"] = results.get("session_handle")

    options = {"session_handle_token": token, "handle_token": token}
    request = screencast.CreateSession(options)
    bus.add_signal_receiver(
        session_response_handler,
        "Response",
        "org.freedesktop.portal.Request",
        path=request,
    )

    temp_loop = GLib.MainLoop()

    def check_session():
        if "handle" in session_response:
            temp_loop.quit()
            return False
        return True

    def session_timeout():
        if "handle" not in session_response:
            temp_loop.quit()
            return False

    GLib.timeout_add(100, check_session)
    GLib.timeout_add_seconds(30, session_timeout)
    temp_loop.run()

    if "handle" not in session_response:
        raise RuntimeError("Timeout creating screen share session")

    session_handle = session_response["handle"]

    select_response = {}

    def select_response_handler(response, _results):
        if response == 0:
            select_response["ready"] = True
        else:
            select_response["error"] = "User cancelled source selection"

    select_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    select_request = screencast.SelectSources(
        session_handle,
        {
            "types": dbus.UInt32(1),  # Monitor
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

    # Wait for source selection using a mainloop
    temp_loop = GLib.MainLoop()

    def check_select():
        if "ready" in select_response or "error" in select_response:
            temp_loop.quit()
            return False
        return True

    def select_timeout():
        if "ready" not in select_response:
            select_response["error"] = "Timeout"
            temp_loop.quit()
        return False

    GLib.timeout_add(100, check_select)
    GLib.timeout_add_seconds(5, select_timeout)
    temp_loop.run()

    if "error" in select_response:
        raise RuntimeError(f"Source selection failed: {select_response['error']}")

    # Start screen sharing
    start_response = {}

    def start_response_handler(response, results):
        global node_id
        if response == 0:  # Success
            streams = results.get("streams", [])
            if streams:
                start_response["node_id"] = streams[0][0]  # PipeWire node ID
                # Extract stream metadata (position, size, etc.)
                # streams[0][1] is a dict with optional 'position' and 'size' keys
                if len(streams[0]) > 1:
                    stream_metadata = dict(streams[0][1])
                    if "size" in stream_metadata:
                        size = stream_metadata["size"]
                        start_response["width"] = int(size[0])
                        start_response["height"] = int(size[1])

                start_response["ready"] = True
        else:
            start_response["error"] = "User cancelled or error occurred"

    start_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    request = screencast.Start(session_handle, "", {"handle_token": start_token})
    bus.add_signal_receiver(
        start_response_handler,
        "Response",
        "org.freedesktop.portal.Request",
        path=request,
    )

    # Wait for user to approve using a mainloop
    print("Waiting for screen share approval...")
    temp_loop = GLib.MainLoop()

    def check_start():
        if "ready" in start_response or "error" in start_response:
            temp_loop.quit()
            return False
        return True

    def start_timeout():
        if "ready" not in start_response:
            start_response["error"] = "Timeout"
            temp_loop.quit()
        return False

    GLib.timeout_add(100, check_start)
    GLib.timeout_add_seconds(30, start_timeout)
    temp_loop.run()

    if "error" in start_response:
        raise RuntimeError(f"Screen share failed: {start_response['error']}")

    # Open PipeWire remote
    fd_obj = screencast.OpenPipeWireRemote(session_handle, {})

    # Extract integer FD from dbus.UnixFd object
    # The UnixFd object wraps the actual file descriptor
    fd = fd_obj.take()  # take() extracts the FD and transfers ownership

    print(f"✓ Screen share approved! PipeWire node ID: {start_response['node_id']}, FD: {fd}")

    return ScreenshareParams(
        node_id=start_response["node_id"],
        fd=fd,
        bus=bus,
        session_handle=session_handle,
    )
