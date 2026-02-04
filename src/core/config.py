"""Configuration for the DINO reinforcement learning system."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    output_width: int = 84
    output_height: int = 84
    channels: int = 1
    n_actions: int = 2
    frame_stack: int = 4
    video_device: str = "/dev/video0"
    fps: int = 30
    queue_max_size: int = 2
    queue_leaky: str = "downstream"
    frame_queue_maxsize: int = 2
    save_frames: bool = False
    save_dir: Path = Path("./debug_frames")
    save_max_frames: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50000
    target_update_freq: int = 1000
    replay_buffer_size: int = 100000
    min_buffer_size: int = 10000
    checkpoint_path: Path = Path("./checkpoints")
    checkpoint_freq: int = 10000
    max_episodes: int = 999999999
    device: torch.device | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[assignment]


@dataclass
class GStreamerConfig:
    """GStreamer pipeline configuration."""

    device: str
    width: int
    height: int
    fps: int
    queue_max_size: int
    queue_leaky: str

    appsink_max_buffers: int = 2
    appsink_drop: bool = True

    def get_pipeline_string(self) -> str:
        """Generate GStreamer pipeline string."""
        return (
            f"v4l2src device={self.device} do-timestamp=true ! "
            f"videoscale ! "
            f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            f"queue max-size-buffers={self.queue_max_size} leaky={self.queue_leaky} ! "
            f"videoconvert ! "
            f"video/x-raw,format=GRAY8 ! "
            f"appsink name=appsink emit-signals=true max-buffers={self.appsink_max_buffers} drop={self.appsink_drop}"
        )
