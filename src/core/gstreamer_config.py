"""GStreamer pipeline configuration."""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class GStreamerConfig:
    """GStreamer pipeline configuration supporting multiple capture sources."""

    source_type: Literal["v4l2", "ximage"]
    width: int
    height: int
    fps: int
    queue_max_size: int
    queue_leaky: str
    appsink_max_buffers: int = 2
    appsink_drop: bool = True

    # v4l2 source options
    device: str = "/dev/video0"

    # ximage source options
    display: str = ""
    browser_width: int = 1280
    browser_height: int = 720
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 1280
    crop_height: int = 720
    udp_port: int = 5000
    udp_port_agent: int = 0

    def get_pipeline_string(self) -> str:
        """Generate GStreamer pipeline string based on source type."""
        if self.source_type == "v4l2":
            return self._get_v4l2_pipeline()
        return self._get_ximage_pipeline()

    def _get_v4l2_pipeline(self) -> str:
        """Generate v4l2src pipeline string."""
        return (
            f"v4l2src device={self.device} do-timestamp=true ! "
            f"videoscale ! "
            f"video/x-raw,width={self.width},height={self.height},framerate=30/1 ! "
            f"queue max-size-buffers={self.queue_max_size} leaky={self.queue_leaky} ! "
            f"videoconvert ! "
            f"videorate ! "
            f"video/x-raw,format=GRAY8,framerate={self.fps}/1 ! "
            f"appsink name=appsink emit-signals=true max-buffers={self.appsink_max_buffers} drop={self.appsink_drop}"
        )

    def _get_ximage_pipeline(self) -> str:
        """Generate ximagesrc pipeline string with optional UDP observability."""
        display_name = self.display or os.getenv("DISPLAY", ":0")

        crop_bottom = self.browser_height - self.crop_y - self.crop_height
        crop_right = self.browser_width - self.crop_x - self.crop_width

        pipeline = (
            f"ximagesrc display-name={display_name} "
            f"startx=0 starty=0 endx={self.browser_width} endy={self.browser_height} "
            f"do-timestamp=true ! "
            f"video/x-raw,framerate=30/1 ! "
            f"tee name=t "
        )

        pipeline += (
            f"t. ! queue max-size-buffers={self.queue_max_size} leaky={self.queue_leaky} "
            f"! videocrop top={self.crop_y} bottom={crop_bottom} left={self.crop_x} right={crop_right} "
            f"! tee name=cropped "
        )

        pipeline += (
            f"cropped. ! queue max-size-buffers=1 leaky={self.queue_leaky} ! videoscale ! "
            f"video/x-raw,width={self.width},height={self.height} ! "
            f"videoconvert ! "
            f"videorate ! "
            f"video/x-raw,format=GRAY8,framerate={self.fps}/1 ! "
            f"appsink name=appsink emit-signals=true max-buffers={self.appsink_max_buffers} drop={self.appsink_drop} "
        )

        if self.udp_port_agent > 0:
            pipeline += (
                f"cropped. ! queue max-size-buffers=1 leaky={self.queue_leaky} ! videoscale ! "
                f"video/x-raw,width=640,height=360 ! videoconvert ! video/x-raw,format=GRAY8 ! "
                f"jpegenc quality=85 ! udpsink host=127.0.0.1 port={self.udp_port_agent} sync=false "
            )

        if self.udp_port > 0:
            pipeline += (
                f"t. ! queue max-size-buffers=1 leaky={self.queue_leaky} ! videoscale ! "
                f"video/x-raw,width=640,height=360 ! jpegenc quality=85 ! "
                f"udpsink host=127.0.0.1 port={self.udp_port} sync=false"
            )

        return pipeline
