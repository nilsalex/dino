import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_height, input_width, n_actions=2, frame_stack=4):
        super().__init__()

        self.n_actions = n_actions
        self.frame_stack = frame_stack

        # CNN feature extractor (DQN paper architecture)
        self.conv = nn.Sequential(
            nn.Conv2d(1 * frame_stack, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # Calculate conv output size
        self._conv_out_size = self._get_conv_out((1 * frame_stack, input_height, input_width))

        # Q-value head (DQN paper architecture)
        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
            return int(o.flatten().shape[0])

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)  # Q-values for each action

    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()

        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
