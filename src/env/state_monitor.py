"""State monitoring for game over detection."""

from __future__ import annotations

import numpy as np

from src.core.constants import GAME_OVER_VARIANCE_THRESHOLD
from src.core.types import Frame


class StateMonitor:
    """Monitor game state and detect events like game over.

    Stateless detector that analyzes frame variance to detect static screens.
    """

    def __init__(self, game_over_window: int = 2):
        self.game_over_window = game_over_window
        self._was_game_over = False

    def check_game_over(self, frames: list[Frame]) -> tuple[bool, int]:
        """Check if the game is over by analyzing frame variance.

        Computes pairwise variance between consecutive frames in the window.
        If all pairs have low variance, the screen is static (game over).

        Args:
            frames: List of recent frames (should be game_over_window + 1 frames
                    to get game_over_window pairs).

        Returns:
            Tuple of (is_game_over, offset) where:
            - is_game_over: True if all frame pairs have low variance
            - offset: Number of frames since game ended (first low-variance pair)
        """
        if len(frames) < 2:
            return False, 0

        pairwise_variances = []
        for i in range(len(frames) - 1):
            stacked = np.stack([frames[i], frames[i + 1]], axis=0)
            variance = np.mean(np.var(stacked, axis=0)) * 10000
            pairwise_variances.append(variance)

        all_low_variance = all(v < GAME_OVER_VARIANCE_THRESHOLD for v in pairwise_variances)

        if not all_low_variance:
            self._was_game_over = False
            return False, 0

        offset = len(pairwise_variances)
        for i, v in enumerate(pairwise_variances):
            if v < GAME_OVER_VARIANCE_THRESHOLD:
                offset = len(pairwise_variances) - i
                break

        is_new_game_over = not self._was_game_over
        self._was_game_over = True

        return True, offset if is_new_game_over else 0
