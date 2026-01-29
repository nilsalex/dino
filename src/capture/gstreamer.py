"""GStreamer pipeline management for frame capture."""

import sys

from gi.repository import GLib, Gst

from src.core.config import Config, GStreamerConfig


class GStreamerPipeline:
    """Manages GStreamer pipeline for video capture."""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline: Gst.Pipeline | None = None
        self.loop: GLib.MainLoop = GLib.MainLoop()

    def create_pipeline(  # type: ignore[return-value]
        self,
    ) -> Gst.Element:
        """Create and configure the GStreamer pipeline."""
        Gst.init(sys.argv)

        gst_config = GStreamerConfig(
            device=self.config.video_device,
            width=self.config.output_width,
            height=self.config.output_height,
            fps=self.config.fps,
            queue_max_size=self.config.queue_max_size,
            queue_leaky=self.config.queue_leaky,
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

    def stop(self) -> None:
        """Stop the pipeline and main loop."""
        if self.pipeline is None:
            return

        self.pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")

    def run(self) -> None:
        """Run the main loop."""
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
