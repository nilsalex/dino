"""Simplified remote trainer - no custom dependencies needed."""

import ray


# @ray.remote(num_gpus=1)
@ray.remote(num_gpus=0)
class SimpleRemoteTrainer:
    """Remote trainer that only accepts raw tensors."""

    def __init__(self, n_actions: int, output_shape: tuple[int, int] | None = None):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.n_actions = n_actions
        self.output_shape = output_shape or (84, 84)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.SmoothL1Loss()

        print(f"Remote trainer initialized on {self.device}")

    def _build_model(self):
        import torch.nn as nn

        class CNN(nn.Module):
            def __init__(self, n_actions: int):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(4, 16, kernel_size=8, stride=4),
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

        return CNN(self.n_actions)

    def get_action(self, states):
        import numpy as np
        import torch

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device) / 255.0
        with torch.no_grad():
            q_values = self.model(states_tensor)
            return q_values.argmax(dim=1).cpu().numpy().tolist()

    def train_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        import numpy as np
        import torch

        states = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device) / 255.0
        actions = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        rewards = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)

        non_dones = [i for i, d in enumerate(done_batch) if not d]
        next_states = [next_state_batch[i] for i in non_dones]

        if next_states:
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device) / 255.0
        else:
            next_states_tensor = None

        self.optimizer.zero_grad()

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_q_values = rewards.clone()

            if next_states_tensor is not None:
                next_q_values = self.target_model(next_states_tensor)
                target_q_values[non_dones] += 0.99 * next_q_values.max(1)[0]

        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_model_state(self):
        """Get the current model state dict for syncing to local actors."""

        state_dict = self.model.state_dict()
        return {k: v.cpu() for k, v in state_dict.items()}

    def save_model(self, path: str):
        import torch

        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        import torch

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
