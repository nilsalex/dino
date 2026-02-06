"""Local inference-only model for fast action selection."""

import torch
import torch.nn as nn


class LocalInferenceModel:
    """Lightweight model for local inference without gradient computation."""

    def __init__(self, n_actions: int, device: torch.device, frame_stack: int = 4):
        self.n_actions = n_actions
        self.device = device
        self.frame_stack = frame_stack
        self.model = self._build_model().to(self.device)
        self.model.eval()  # Always in eval mode for inference

    def _build_model(self) -> nn.Module:
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

    def get_action(self, state: torch.Tensor) -> int:
        """Get action from a single state tensor without gradients."""
        with torch.no_grad():
            q_values = self.model(state)
            return q_values.argmax(dim=1).item()

    def update_state_dict(self, state_dict: dict | list) -> None:
        """Update model weights from remote trainer."""
        if isinstance(state_dict, dict):
            self.model.load_state_dict(state_dict)
        else:
            import torch

            self.model.load_state_dict(torch.load(state_dict))  # type: ignore[arg-type]
        self.model.eval()

    def state_dict(self) -> dict:
        """Get current model state dict."""
        return self.model.state_dict()
