"""GStreamer pipeline management for frame capture."""

from __future__ import annotations

import sys
import threading
from typing import Literal

from gi.repository import GLib, Gst

from src.core.config import Config
from src.core.gstreamer_config import GStreamerConfig


class GStreamerPipeline:
    """Manages GStreamer pipeline for video capture."""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline: Gst.Pipeline | None = None
        self.loop: GLib.MainLoop = GLib.MainLoop()
        self._loop_thread: threading.Thread | None = None

    def create_pipeline(  # type: ignore[return-value]
        self,
    ) -> Gst.Element:
        """Create and configure the GStreamer pipeline."""
        Gst.init(sys.argv)

        source_type: Literal["v4l2", "ximage"] = "ximage" if self.config.headless else "v4l2"

        gst_config = GStreamerConfig(
            source_type=source_type,
            width=self.config.output_width,
            height=self.config.output_height,
            fps=self.config.fps,
            queue_max_size=self.config.queue_max_size,
            queue_leaky=self.config.queue_leaky,
            device=self.config.video_device,
            browser_width=self.config.browser_width,
            browser_height=self.config.browser_height,
            crop_x=self.config.crop_x,
            crop_y=self.config.crop_y,
            crop_width=self.config.crop_width,
            crop_height=self.config.crop_height,
            udp_port=self.config.udp_port,
            udp_port_agent=self.config.udp_port_agent,
        )

        pipeline_str = gst_config.get_pipeline_string()
        pipeline = Gst.parse_launch(pipeline_str)

        self.pipeline = pipeline  # type: ignore[assignment]
        self._setup_bus(pipeline)  # type: ignore[arg-type]

        return pipeline  # type: ignore[return-value]

    def _setup_bus(self, pipeline: Gst.Pipeline) -> None:
        """Set up GStreamer bus for message handling."""
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message, self.loop)

    def _on_message(self, _bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop) -> bool:
        """Handle GStreamer bus messages."""
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

    def set_sample_callback(self, sink_name: str, callback: object, data: object = None) -> None:
        """Set the callback for new video samples."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not created")

        sink = self.pipeline.get_by_name(sink_name)  # type: ignore[arg-type]
        if sink is None:
            raise RuntimeError(f"Sink '{sink_name}' not found in pipeline")
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", callback, data)  # type: ignore[union-attr,arg-type]

    def start(self) -> None:
        """Start the pipeline and main loop."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not created")

        self.pipeline.set_state(Gst.State.PLAYING)

        self._loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Stop the pipeline and main loop."""
        if self.pipeline is None:
            return

        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)
        print("Pipeline stopped")

    def run(self) -> None:
        """Run the main loop."""
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
