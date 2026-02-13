"""Configuration for the DINO reinforcement learning system."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    output_width: int = 84
    output_height: int = 84
    channels: int = 1
    game_name: str = os.getenv("GAME", "dino")
    video_device: str = os.getenv("VIDEO_DEVICE", "/dev/video0")
    fps: int = 10
    queue_max_size: int = 2
    queue_leaky: str = "downstream"
    frame_queue_maxsize: int = 2
    save_frames: bool = False
    save_dir: Path = Path("./debug_frames")
    save_max_frames: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 0.8
    epsilon_end: float = 0.001
    epsilon_decay: int = 10000
    target_update_freq: int = 100
    replay_buffer_size: int = 10000
    min_buffer_size: int = 1000
    checkpoint_path: Path = Path("./checkpoints")
    checkpoint_freq: int = 1000
    max_episodes: int = 999999999
    device: torch.device | None = None

    # Headless mode (ximagesrc capture)
    headless: bool = os.getenv("HEADLESS", "").lower() == "true"

    # Browser dimensions (for ximagesrc capture)
    browser_width: int = int(os.getenv("BROWSER_WIDTH", "1280"))
    browser_height: int = int(os.getenv("BROWSER_HEIGHT", "720"))

    # Game region crop
    crop_x: int = int(os.getenv("CROP_X", "0"))
    crop_y: int = int(os.getenv("CROP_Y", "0"))
    crop_width: int = int(os.getenv("CROP_WIDTH", "1280"))
    crop_height: int = int(os.getenv("CROP_HEIGHT", "720"))

    # Observability UDP stream
    udp_port: int = int(os.getenv("UDP_PORT", "5000"))
    udp_port_agent: int = int(os.getenv("UDP_PORT_AGENT", "0"))

    # X11 display for headless mode
    display_name: str = os.getenv("DISPLAY_NAME", ":99")

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[assignment]
