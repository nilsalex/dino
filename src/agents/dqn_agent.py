"""DQN agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base import Agent
from src.core.config import Config
from src.core.types import Action, State
from src.models.dqn import DQN


class DQNAgent(Agent):
    """Deep Q-Network agent with epsilon-greedy exploration."""

    def __init__(self, config: Config):
        self.config = config

        self.model = DQN(
            config.output_height,
            config.output_width,
            config.n_actions,
        ).to(config.device)

        self.target_model = DQN(
            config.output_height,
            config.output_width,
            config.n_actions,
        ).to(config.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.epsilon = config.epsilon_start
        self.step_count = 0

    def act(self, state: State) -> Action:  # type: ignore[override]
        if torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.config.n_actions, (1,)).item())

        with torch.no_grad():
            q_values = self.model(state)
            return q_values.argmax(dim=1).item()

    def train(self, experience: object) -> dict[str, float]:
        """Train on a single experience.

        Args:
            experience: Tuple of (state, action, reward, next_state, done).

        Returns:
            Dictionary of training metrics.
        """
        state, action, reward, next_state, done = experience  # type: ignore[assignment]

        self.optimizer.zero_grad()

        current_q_values = self.model(state)
        current_q_value = current_q_values[0, action]

        with torch.no_grad():
            if done:
                target_q_value = reward
            else:
                next_q_values = self.target_model(next_state)
                target_q_value = reward + self.config.gamma * next_q_values.max()

        loss = self.criterion(current_q_value, target_q_value)
        loss.backward()

        self.optimizer.step()

        self._update_epsilon()
        self.step_count += 1

        if self.step_count % self.config.target_update_freq == 0:
            self._update_target_model()

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def _update_epsilon(self) -> None:
        """Decay epsilon based on step count."""
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start
                - (self.config.epsilon_start - self.config.epsilon_end) * (self.step_count / self.config.epsilon_decay),
            )

    def _update_target_model(self) -> None:
        """Update target network with current network weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path: str) -> None:
        """Save model, optimizer, and training state.

        Args:
            path: File path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model, optimizer, and training state.

        Args:
            path: File path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
        self._update_target_model()
