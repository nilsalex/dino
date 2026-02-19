"""Thread-safe experience replay buffer for concurrent training."""

from __future__ import annotations

import random
import threading
from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.core.config import Config


class ThreadSafeExperienceBuffer:
    """Thread-safe experience replay buffer.

    Supports concurrent addition of experiences by main thread and sampling
    by training thread using internal locking.

    Args:
        config: Configuration object with replay_buffer_size.

    Attributes:
        max_size: Maximum number of experiences to store.
        lock: Thread lock for thread-safe operations.
    """

    def __init__(self, config: Config) -> None:
        self.max_size: int = config.replay_buffer_size
        self.lock = threading.Lock()

        self._states: deque[torch.Tensor] = deque(maxlen=self.max_size)
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._next_states: deque[torch.Tensor] = deque(maxlen=self.max_size)
        self._dones: list[bool] = []
        self._episode_ids: list[int] = []
        self._transitions_added: int = 0
        self._action_history: deque[tuple[int, bool]] = deque(maxlen=1000)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        episode_id: int,
    ) -> None:
        """Add experience to buffer (thread-safe).

        Args:
            state: Current state tensor [4, 84, 84] (no batch dimension).
            action: Action taken.
            reward: Reward received.
            next_state: Next state tensor [4, 84, 84] (no batch dimension).
            done: Whether episode terminated.
            episode_id: ID of the episode this transition belongs to.

        Note:
            All tensors must be on CPU for thread safety.
        """
        with self.lock:
            self._states.append(state.detach().cpu())
            self._actions.append(action)
            self._rewards.append(reward)
            self._next_states.append(next_state.detach().cpu())
            self._dones.append(done)
            self._episode_ids.append(episode_id)
            self._transitions_added += 1
            self._action_history.append((action, False))

    def sample(
        self, batch_size: int
    ) -> tuple[
        list[torch.Tensor],
        list[int],
        list[float],
        list[torch.Tensor],
        list[bool],
    ]:
        """Sample random batch of experiences (thread-safe).

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as lists.

        Raises:
            ValueError: If buffer has fewer samples than batch_size.
        """
        with self.lock:
            buffer_len = len(self._states)
            if buffer_len < batch_size:
                raise ValueError(f"Buffer has {buffer_len} samples, need {batch_size}")

            indices = random.sample(range(buffer_len), batch_size)
            states = [self._states[i] for i in indices]
            actions = [self._actions[i] for i in indices]
            rewards = [self._rewards[i] for i in indices]
            next_states = [self._next_states[i] for i in indices]
            dones = [self._dones[i] for i in indices]

        return states, actions, rewards, next_states, dones

    def size(self) -> int:
        """Get current buffer size (thread-safe)."""

        with self.lock:
            return len(self._states)

    def get_add_count(self) -> int:
        """Get total transitions added counter (thread-safe)."""

        with self.lock:
            return self._transitions_added

    def __len__(self) -> int:
        """Get current buffer size (thread-safe)."""

        return self.size()

    def mark_last_as_terminal(self, reward: float) -> None:
        """Mark the last transition as terminal with given reward.

        Args:
            reward: The reward to set for the last transition.

        Note:
            Used when game over is detected after the transition was recorded.
        """
        with self.lock:
            if len(self._rewards) > 0:
                self._rewards[-1] = reward
                self._dones[-1] = True
                if len(self._action_history) > 0:
                    action, _ = self._action_history[-1]
                    self._action_history[-1] = (action, True)

    def get_action_reward_stats(self) -> dict[int, dict[str, int]]:
        """Get reward statistics per action from rolling window (thread-safe).

        Returns:
            Dict mapping action_id to dict with 'count' and 'terminal_count'.
        """
        with self.lock:
            stats: dict[int, dict[str, int]] = {}
            for action, is_terminal in self._action_history:
                if action not in stats:
                    stats[action] = {"count": 0, "terminal_count": 0}
                stats[action]["count"] += 1
                if is_terminal:
                    stats[action]["terminal_count"] += 1
            return stats
