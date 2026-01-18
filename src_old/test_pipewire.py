#!/usr/bin/env python3
"""
Test PipeWire screen capture initialization.
"""

import sys


def test_pipewire() -> bool:
    """Test PipeWire dependencies and screen capture initialization."""
    print("=" * 60)
    print("PipeWire Screen Capture Test")
    print("=" * 60)
    print()

    print("1. Setting up D-Bus mainloop...")
    try:
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop

        DBusGMainLoop(set_as_default=True)
        print("   ✓ D-Bus mainloop configured")
    except Exception as e:
        print(f"   ✗ Mainloop error: {e}")
        return False

    print()
    print("2. Testing D-Bus connection...")
    try:
        bus = dbus.SessionBus()
        print(f"   ✓ D-Bus connected: {bus.get_unique_name()}")
    except Exception as e:
        print(f"   ✗ D-Bus error: {e}")
        return False

    print()
    print("3. Testing GStreamer imports...")
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        print(f"   ✓ GStreamer version: {Gst.version_string()}")
    except Exception as e:
        print(f"   ✗ GStreamer error: {e}")
        return False

    print()
    print("4. Checking XDG Desktop Portal...")
    try:
        portal = bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
        _screencast = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")
        print("   ✓ XDG Desktop Portal accessible")
    except Exception as e:
        print(f"   ✗ Portal error: {e}")
        print("   Make sure xdg-desktop-portal is running!")
        return False

    print()
    print("5. Initializing PipeWire capture...")
    print("   NOTE: You will see a screen sharing permission dialog.")
    print("   Please APPROVE it to continue the test.")
    print()

    try:
        from pipewire_capture import PipeWireCapture

        capture = PipeWireCapture()
        print("   ✓ PipeWire capture initialized!")

        print()
        print("6. Testing frame capture...")
        frame = capture.grab()
        if frame is not None:
            print(f"   ✓ Frame captured: {frame.shape}")
        else:
            print("   ✗ No frame captured")
            return False

        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        capture.stop()
        return True

    except Exception as e:
        print(f"   ✗ PipeWire error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_passed = test_pipewire()
    sys.exit(0 if test_passed else 1)
