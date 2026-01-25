"""Base agent interface for reinforcement learning."""


class Agent:
    """Base agent interface for RL algorithms."""

    def act(self, state: object) -> int:
        """Select an action given a state."""
        raise NotImplementedError

    def train(self, experience: object) -> dict[str, float]:
        """Train the agent on a single experience.

        Args:
            experience: A tuple of (state, action, reward, next_state, done).

        Returns:
            Dictionary of training metrics.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save agent state to disk.

        Args:
            path: File path to save to.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: File path to load from.
        """
        raise NotImplementedError
