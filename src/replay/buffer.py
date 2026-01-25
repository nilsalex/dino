"""Experience replay buffer for deep RL."""

import random
from collections import deque

from src.core.types import Experience


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """Push an experience to the buffer.

        Args:
            experience: Tuple of (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
