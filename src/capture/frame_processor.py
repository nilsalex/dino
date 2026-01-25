"""Frame buffering and preprocessing for RL."""

from collections import deque
from queue import Empty, Full, Queue

import numpy as np
import torch
from PIL import Image

from src.core.config import Config
from src.core.types import Frame, FrameBuffer, State


class FrameProcessor:
    """Handles frame buffering, preprocessing and state preparation."""

    def __init__(self, config: Config):
        self.config = config
        self.frame_buffer: FrameBuffer = deque(maxlen=config.frame_stack)
        self.frame_queue: Queue[Frame] = Queue(maxsize=config.frame_queue_maxsize)
        self.input_frame_count = 0

        if config.save_frames:
            config.save_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame: Frame) -> None:
        """Add a new frame to the processing queue."""
        try:
            self.frame_queue.put_nowait(frame.copy())
        except Full:
            print("Warning: frame queue full, could not enqueue frame")

    def get_state(self) -> State | None:
        """Get current state from frame buffer, or None if not ready."""
        try:
            frame = self.frame_queue.get_nowait()
            self._add_to_buffer(frame)

            if len(self.frame_buffer) < self.config.frame_stack:
                return None

            return self._preprocess()
        except Empty:
            return None

    def _add_to_buffer(self, frame: Frame) -> None:
        """Add frame to buffer and optionally save for debug."""
        self.frame_buffer.append(frame)

        if self.config.save_frames and self.input_frame_count < self.config.save_max_frames:
            self._save_debug_frame(frame)
            self.input_frame_count += 1

    def _save_debug_frame(self, frame: Frame) -> None:
        """Save frame to disk for debugging."""
        img = Image.fromarray(frame, mode="L")
        path = self.config.save_dir / f"frame_{self.input_frame_count:04d}.png"
        img.save(path)
        print(f"Saved: {path}")

    def _preprocess(self) -> State:
        """Convert frame buffer to tensor state."""
        stacked = np.stack(list(self.frame_buffer), axis=0)
        tensor = torch.from_numpy(stacked).float() / 255.0
        return tensor.unsqueeze(0).to(self.config.device)

    def buffer_ready(self) -> bool:
        """Check if buffer has enough frames."""
        return len(self.frame_buffer) >= self.config.frame_stack

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self.frame_queue.qsize()

    @property
    def max_queue_size(self) -> int:
        """Get maximum queue size."""
        return self.frame_queue.maxsize  # type: ignore[arg-type]
