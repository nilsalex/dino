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
        use_torch_compile: bool = False,
    ):
        self.n_actions = n_actions
        self.device = device
        self.frame_stack = frame_stack
        self.sigma_init = sigma_init
        self.model = self._build_model().to(self.device)
        self.model.train()
        self._inference_latencies: list[float] = []

        # Compile model for faster inference (PyTorch 2.0+)
        if use_torch_compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

            # Warmup: trigger compilation at startup
            dummy_input = torch.zeros(1, frame_stack, 84, 84, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

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
        import time

        start = time.perf_counter()
        with torch.no_grad():
            q_values = self.model(state)  # type: ignore[misc]
            action = q_values.argmax(dim=1).item()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._inference_latencies.append(elapsed_ms)

        return action

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

    def reset_noise(self) -> None:
        """Reset noise for exploration (call before action selection)."""
        for module in self.model.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_sigma_means(self) -> list[float]:
        """Get mean sigma values for each NoisyLinear layer."""
        sigma_means = []
        for module in self.model.modules():  # type: ignore[union-attr]
            if isinstance(module, NoisyLinear):
                sigma_means.append(module.get_sigma_mean())
        return sigma_means

    def get_inference_latency_stats(self) -> dict[str, float]:
        """Get inference latency statistics.

        Returns:
            Dictionary with latency_ms and latency_mean_ms.
        """
        if not self._inference_latencies:
            return {"latency_ms": 0.0, "latency_mean_ms": 0.0}

        return {
            "latency_ms": self._inference_latencies[-1],
            "latency_mean_ms": sum(self._inference_latencies[-100:]) / min(len(self._inference_latencies), 100),
        }

    def state_dict(self) -> dict:
        """Get current model state dict."""
        return self.model.state_dict()
