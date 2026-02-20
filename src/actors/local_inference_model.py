"""Local inference-only model for fast action selection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.noisy_linear import NoisyLinear


class LocalInferenceModel:
    """Lightweight model for local inference without gradient computation.

    Uses NoisyLinear layers matching the trainer architecture.
    Stays in train() mode during gameplay for exploration (noise sampling).
    Switch to eval() mode during evaluation episodes for deterministic behavior.
    """

    def __init__(
        self,
        n_actions: int,
        device: torch.device,
        frame_stack: int = 4,
        sigma_init: float = 0.5,
    ):
        self.n_actions = n_actions
        self.device = device
        self.frame_stack = frame_stack
        self.sigma_init = sigma_init
        self.model = self._build_model().to(self.device)
        self.model.train()

    def _build_model(self) -> nn.Module:
        class CNN(nn.Module):
            def __init__(self, n_actions: int, frame_stack: int, sigma_init: float):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(frame_stack, 16, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                )
                self.fc = nn.Sequential(
                    NoisyLinear(32 * 9 * 9, 256, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(256, n_actions, sigma_init),
                )

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return CNN(self.n_actions, self.frame_stack, self.sigma_init)

    def get_action(self, state: torch.Tensor) -> int:
        """Get action from a single state tensor without gradients."""
        with torch.no_grad():
            q_values = self.model(state)
            return q_values.argmax(dim=1).item()

    def update_state_dict(self, state_dict: dict | list) -> None:
        """Update model weights from trainer."""
        if isinstance(state_dict, dict):
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(torch.load(state_dict))  # type: ignore[arg-type]
        self.model.train()

    def set_train_mode(self) -> None:
        """Set model to train mode for exploration (noisy weights)."""
        self.model.train()

    def set_eval_mode(self) -> None:
        """Set model to eval mode for deterministic behavior (mean weights)."""
        self.model.eval()

    def state_dict(self) -> dict:
        """Get current model state dict."""
        return self.model.state_dict()
