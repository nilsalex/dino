import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_height, input_width, n_actions=2):
        super().__init__()

        self.n_actions = n_actions

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1 * 4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Calculate conv output size
        self._conv_out_size = self._get_conv_out((1 * 4, input_height, input_width))

        # Q-value head
        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
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
