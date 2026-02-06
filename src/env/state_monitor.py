"""State monitoring for game over detection."""

import numpy as np

from src.core.constants import GAME_OVER_VARIANCE_THRESHOLD
from src.core.types import FrameBuffer


class StateMonitor:
    """Monitor game state and detect events like game over."""

    def is_game_over(self, frame_buffer: FrameBuffer) -> bool:
        """Check if the game is over by analyzing frame variance.

        Args:
            frame_buffer: Buffer of recent frames.

        Returns:
            True if game over detected, False otherwise.
        """
        stacked = np.stack(list(frame_buffer), axis=0)
        pixel_variance = np.var(stacked, axis=0)
        return np.mean(pixel_variance) < GAME_OVER_VARIANCE_THRESHOLD
