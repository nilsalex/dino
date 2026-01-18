# screenshare.py
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


# Module-level main loop - shared between DBus and GStreamer
main_loop = None


def init_screenshare() -> ScreenshareParams:
    global main_loop

    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    sender_name = bus.get_unique_name()[1:].replace(".", "_")
    token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
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

    GLib.timeout_add(100, check_session)
    GLib.timeout_add_seconds(30, lambda: temp_loop.quit() or False)
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

    temp_loop = GLib.MainLoop()

    def check_select():
        if "ready" in select_response or "error" in select_response:
            temp_loop.quit()
            return False
        return True

    GLib.timeout_add(100, check_select)
    GLib.timeout_add_seconds(30, lambda: temp_loop.quit() or False)
    temp_loop.run()

    if "error" in select_response:
        raise RuntimeError(f"Source selection failed: {select_response['error']}")

    start_response = {}

    def start_response_handler(response, results):
        if response == 0:
            streams = results.get("streams", [])
            if streams:
                start_response["node_id"] = streams[0][0]
                start_response["ready"] = True
        else:
            start_response["error"] = "User cancelled or error occurred"

    print("Waiting for screen share approval...")
    start_token = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    request = screencast.Start(session_handle, "", {"handle_token": start_token})
    bus.add_signal_receiver(
        start_response_handler,
        "Response",
        "org.freedesktop.portal.Request",
        path=request,
    )

    temp_loop = GLib.MainLoop()

    def check_start():
        if "ready" in start_response or "error" in start_response:
            temp_loop.quit()
            return False
        return True

    GLib.timeout_add(100, check_start)
    GLib.timeout_add_seconds(30, lambda: temp_loop.quit() or False)
    temp_loop.run()

    if "error" in start_response:
        raise RuntimeError(f"Screen share failed: {start_response['error']}")

    fd_obj = screencast.OpenPipeWireRemote(session_handle, {})
    fd = fd_obj.take()

    print(f"✓ Screen share approved! PipeWire node ID: {start_response['node_id']}, FD: {fd}")

    # Create the main loop that will be used for GStreamer
    main_loop = GLib.MainLoop()

    return ScreenshareParams(
        node_id=start_response["node_id"],
        fd=fd,
    )


def get_main_loop() -> GLib.MainLoop:
    return main_loop
