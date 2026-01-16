"""
PipeWire screen capture for Wayland using XDG Desktop Portal.

Based on: https://github.com/columbarius/xdg-desktop-portal-testscripts
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import threading
import time
import numpy as np
from typing import Optional, Dict
import random
import string

# Initialize GStreamer
Gst.init(None)


class PipeWireCapture:
    """
    Continuous screen capture using PipeWire + GStreamer + XDG Desktop Portal.

    This provides a video stream that can be grabbed frame-by-frame without
    subprocess overhead.
    """

    def __init__(self):
        """Initialize PipeWire capture. User permission will be requested."""
        self.pipeline = None
        self.appsink = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.mainloop = None
        self.mainloop_thread = None
        self.node_id = None
        self.fd = None
        self.stream_metadata = {}  # Store monitor position, size, etc.

        print("Initializing PipeWire screen capture...")
        print("NOTE: You will need to approve screen sharing in the system dialog.")

        # Request screen sharing session
        self._request_screen_share()

        # Create GStreamer pipeline
        self._create_pipeline()

        # Start capture
        self._start_capture()

        # Wait for first frame
        print("Waiting for first frame...")
        timeout = 10.0
        start = time.time()
        while self.latest_frame is None:
            time.sleep(0.1)
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for first frame. Did you approve screen sharing?")

        print(f"✓ PipeWire capture initialized! Frame shape: {self.latest_frame.shape}")

    def _request_screen_share(self):
        """Request screen sharing session via XDG Desktop Portal."""
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
        except ImportError:
            raise ImportError(
                "dbus-python is required for PipeWire capture. "
                "Install with: pip install dbus-python"
            )

        # Setup D-Bus
        DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()

        # Generate unique session handle
        sender_name = bus.get_unique_name()[1:].replace('.', '_')
        token = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        session_handle = f"/org/freedesktop/portal/desktop/session/{sender_name}/{token}"

        # Get portal interface
        portal = bus.get_object(
            'org.freedesktop.portal.Desktop',
            '/org/freedesktop/portal/desktop'
        )
        screencast = dbus.Interface(
            portal,
            'org.freedesktop.portal.ScreenCast'
        )

        # Create session
        session_response = {}
        def session_response_handler(response, results):
            session_response['handle'] = results.get('session_handle')

        options = {
            'session_handle_token': token,
            'handle_token': token
        }
        request = screencast.CreateSession(options)
        bus.add_signal_receiver(
            session_response_handler,
            'Response',
            'org.freedesktop.portal.Request',
            path=request
        )

        # Wait for session creation using a mainloop
        temp_loop = GLib.MainLoop()
        def check_session():
            if 'handle' in session_response:
                temp_loop.quit()
                return False
            return True
        def session_timeout():
            if 'handle' not in session_response:
                temp_loop.quit()
            return False

        GLib.timeout_add(100, check_session)
        GLib.timeout_add_seconds(5, session_timeout)
        temp_loop.run()

        if 'handle' not in session_response:
            raise RuntimeError("Timeout creating screen share session")

        session_handle = session_response['handle']

        # Select sources (1 = monitor, 2 = window)
        select_response = {}
        def select_response_handler(response, results):
            if response == 0:
                select_response['ready'] = True
            else:
                select_response['error'] = 'User cancelled source selection'

        select_token = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        select_request = screencast.SelectSources(
            session_handle,
            {
                'types': dbus.UInt32(1),  # Monitor
                'multiple': dbus.Boolean(False),
                'handle_token': select_token
            }
        )
        bus.add_signal_receiver(
            select_response_handler,
            'Response',
            'org.freedesktop.portal.Request',
            path=select_request
        )

        # Wait for source selection using a mainloop
        temp_loop = GLib.MainLoop()
        def check_select():
            if 'ready' in select_response or 'error' in select_response:
                temp_loop.quit()
                return False
            return True
        def select_timeout():
            if 'ready' not in select_response:
                select_response['error'] = 'Timeout'
                temp_loop.quit()
            return False

        GLib.timeout_add(100, check_select)
        GLib.timeout_add_seconds(5, select_timeout)
        temp_loop.run()

        if 'error' in select_response:
            raise RuntimeError(f"Source selection failed: {select_response['error']}")

        # Start screen sharing
        start_response = {}
        def start_response_handler(response, results):
            if response == 0:  # Success
                streams = results.get('streams', [])
                if streams:
                    self.node_id = streams[0][0]  # PipeWire node ID
                    # Extract stream metadata (position, size, etc.)
                    # streams[0][1] is a dict with optional 'position' and 'size' keys
                    if len(streams[0]) > 1:
                        self.stream_metadata = dict(streams[0][1])
                    start_response['ready'] = True
            else:
                start_response['error'] = 'User cancelled or error occurred'

        start_token = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        request = screencast.Start(
            session_handle,
            '',
            {'handle_token': start_token}
        )
        bus.add_signal_receiver(
            start_response_handler,
            'Response',
            'org.freedesktop.portal.Request',
            path=request
        )

        # Wait for user to approve using a mainloop
        print("Waiting for screen share approval...")
        temp_loop = GLib.MainLoop()
        def check_start():
            if 'ready' in start_response or 'error' in start_response:
                temp_loop.quit()
                return False
            return True
        def start_timeout():
            if 'ready' not in start_response:
                start_response['error'] = 'Timeout'
                temp_loop.quit()
            return False

        GLib.timeout_add(100, check_start)
        GLib.timeout_add_seconds(30, start_timeout)
        temp_loop.run()

        if 'error' in start_response:
            raise RuntimeError(f"Screen share failed: {start_response['error']}")

        # Open PipeWire remote
        fd_obj = screencast.OpenPipeWireRemote(
            session_handle,
            {}
        )

        # Extract integer FD from dbus.UnixFd object
        # The UnixFd object wraps the actual file descriptor
        self.fd = fd_obj.take()  # take() extracts the FD and transfers ownership

        print(f"✓ Screen share approved! PipeWire node ID: {self.node_id}, FD: {self.fd}")

        # Display monitor metadata if available
        print(f"  Stream metadata keys: {list(self.stream_metadata.keys()) if self.stream_metadata else 'None'}")
        if self.stream_metadata:
            position = self.stream_metadata.get('position')
            size = self.stream_metadata.get('size')
            if position:
                print(f"  Monitor position: {position} (type: {type(position)})")
            if size:
                print(f"  Monitor size: {size} (type: {type(size)})")
            if not position or not size:
                print("  Note: Portal provided metadata but missing position or size")
        else:
            print("  Note: Monitor position metadata not available from portal")

    def _create_pipeline(self):
        """Create GStreamer pipeline: pipewiresrc -> videoconvert -> appsink."""
        if self.node_id is None or self.fd is None:
            raise RuntimeError("Must request screen share before creating pipeline")

        # Pipeline: pipewiresrc fd=X path=Y ! videoconvert ! video/x-raw,format=BGR ! appsink
        pipeline_str = (
            f"pipewiresrc fd={self.fd} path={self.node_id} "
            "! videoconvert "
            "! video/x-raw,format=BGR "
            "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get appsink
        self.appsink = self.pipeline.get_by_name('sink')
        if not self.appsink:
            raise RuntimeError("Could not get appsink from pipeline")

        # Connect to new-sample signal
        self.appsink.connect('new-sample', self._on_new_sample)

        # Setup bus messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_bus_message)

    def _on_new_sample(self, appsink):
        """Callback when new frame is available."""
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')

            # Extract data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array (BGR format)
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )

                # Store latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer Error: {err}, {debug}")
            if self.mainloop:
                self.mainloop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"GStreamer Warning: {warn}, {debug}")
        elif t == Gst.MessageType.EOS:
            print("End of stream")
            if self.mainloop:
                self.mainloop.quit()

        return True

    def _start_capture(self):
        """Start the GStreamer pipeline in a background thread."""
        # Set pipeline to playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set pipeline to PLAYING state")

        # Start GLib mainloop in separate thread
        def run_mainloop():
            if self.mainloop is None:
                self.mainloop = GLib.MainLoop()
            self.mainloop.run()

        self.mainloop_thread = threading.Thread(target=run_mainloop, daemon=True)
        self.mainloop_thread.start()

        print("✓ Capture pipeline started")

    def get_monitor_info(self) -> Optional[Dict]:
        """
        Get information about the captured monitor.

        Returns:
            Dict with 'position' (x, y) and 'size' (width, height) if available,
            or None if metadata is not provided by the portal.
        """
        if not self.stream_metadata:
            return None

        position = self.stream_metadata.get('position')
        size = self.stream_metadata.get('size')

        if position is None or size is None:
            return None

        # Convert dbus types to Python ints
        # position is a tuple (x, y), size is a tuple (width, height)
        return {
            'left': int(position[0]) if isinstance(position, (tuple, list)) else int(position),
            'top': int(position[1]) if isinstance(position, (tuple, list)) else 0,
            'width': int(size[0]) if isinstance(size, (tuple, list)) else int(size),
            'height': int(size[1]) if isinstance(size, (tuple, list)) else 0,
        }

    def grab(self, region: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Grab the latest frame.

        Args:
            region: Optional dict with keys 'left', 'top', 'width', 'height'.
                   If provided, crops the frame to this region.

        Returns:
            numpy array in BGR format, or None if no frame available yet
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None

            frame = self.latest_frame.copy()

        # Crop to region if specified
        if region:
            x = region.get('left', 0)
            y = region.get('top', 0)
            w = region.get('width', frame.shape[1])
            h = region.get('height', frame.shape[0])

            # Ensure bounds are valid
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            frame = frame[y:y+h, x:x+w]

        return frame

    def stop(self):
        """Stop capture and cleanup resources."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.mainloop:
            self.mainloop.quit()
        if self.mainloop_thread:
            self.mainloop_thread.join(timeout=2.0)
        print("✓ PipeWire capture stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
