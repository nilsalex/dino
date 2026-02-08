"""Local DQN trainer for Chrome Dino game without Ray."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from src.core.config import Config


class LocalDQNTrainer:
    """Local DQN trainer for on-device training without Ray."""

    def __init__(self, config: Config, n_actions: int, frame_stack: int = 1):
        self.config = config
        self.n_actions = n_actions
        self.frame_stack = frame_stack
        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        print(f"Local trainer initialized on {self.device}")

    def _build_model(self) -> nn.Module:
        """Build DQN model."""

        class CNN(nn.Module):
            def __init__(self, n_actions: int, frame_stack: int):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(frame_stack, 16, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                )
                self.fc = nn.Sequential(
                    nn.Linear(32 * 9 * 9, 256),
                    nn.ReLU(),
                    nn.Linear(256, n_actions),
                )

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return CNN(self.n_actions, self.frame_stack)

    def train_step(
        self,
        state_batch: list[np.ndarray],
        action_batch: list[int],
        reward_batch: list[float],
        next_state_batch: list[np.ndarray],
        done_batch: list[bool],
    ) -> dict[str, float]:
        """Execute one training step."""
        states = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device) / 255.0
        actions = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        rewards = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)

        non_dones = [i for i, d in enumerate(done_batch) if not d]
        next_states = [next_state_batch[i] for i in non_dones]

        if next_states:
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device) / 255.0
        else:
            next_states_tensor = None

        print(f"[TRAIN_STEP] States shape={states.shape}, non_dones={len(non_dones)}/{len(done_batch)}")

        self.optimizer.zero_grad()

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        print(
            f"[TRAIN_STEP] Q_values shape={current_q_values.shape}, range=({current_q_values.min().item():.4f},{current_q_values.max().item():.4f})"
        )

        with torch.no_grad():
            target_q_values = rewards.clone()

            if next_states_tensor is not None:
                next_q_values = self.target_model(next_states_tensor)
                max_next_q = next_q_values.max(1)[0]
                print(
                    f"[TRAIN_STEP] Next Q shape={next_q_values.shape}, max_next_q range=({max_next_q.min().item():.4f},{max_next_q.max().item():.4f})"
                )
                target_q_values[non_dones] += self.config.gamma * max_next_q

        print(
            f"[TRAIN_STEP] Target Q range=({target_q_values.min().item():.4f},{target_q_values.max().item():.4f}), rewards range=({rewards.min().item():.4f},{rewards.max().item():.4f})"
        )

        loss = self.criterion(current_q_values, target_q_values)
        print(f"[TRAIN_STEP] Loss={loss.item():.6f}")

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "q_mean": current_q_values.mean().item(),
            "q_max": current_q_values.max().item(),
            "target_mean": target_q_values.mean().item(),
        }

    def update_target(self) -> None:
        """Update target network weights."""

        self.target_model.load_state_dict(self.model.state_dict())

    def get_model_state(self) -> dict[str, torch.Tensor]:
        """Get current model state dict."""

        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_action(self, states: torch.Tensor) -> int:
        """Get action from states (greedy)."""
        with torch.no_grad():
            q_values = self.model(states)
            return q_values.argmax(dim=1).item()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Get model state dict for saving."""

        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model state dict."""

        self.model.load_state_dict(state_dict)
