"""Noisy linear layer for exploration via learned parametric noise."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with learned parametric noise for exploration.

    Implements: y = (mu_w + sigma_w * epsilon_w)x + (mu_b + sigma_b * epsilon_b)

    Uses factorised Gaussian noise for computational efficiency:
    epsilon_ij = f(epsilon_i) * f(epsilon_j) where f(x) = sgn(x) * sqrt(|x|)

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        sigma_init: Initial value for sigma parameters (default: 0.5).
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable mean parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))

        # Learnable sigma (std) parameters
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters - sampled each forward pass)
        self.register_buffer("epsilon_input", torch.zeros(in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters according to paper."""
        # μ initialized uniformly: U[-1/sqrt(p), 1/sqrt(p)]
        bound = 1.0 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)

        # sigma initialized: sigma_init / sqrt(p)
        sigma_init_val = self.sigma_init / math.sqrt(self.in_features)
        self.sigma_weight.data.fill_(sigma_init_val)
        self.sigma_bias.data.fill_(sigma_init_val)

        # Initialize noise buffers
        self.epsilon_input.zero_()  # type: ignore[operator]
        self.epsilon_output.zero_()  # type: ignore[operator]

    def _sample_noise(self) -> None:
        """Sample noise from standard normal distribution."""
        self.epsilon_input.normal_()  # type: ignore[operator]
        self.epsilon_output.normal_()  # type: ignore[operator]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights during training.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        if self.training:
            self._sample_noise()
            f_input = self.epsilon_input.sign() * self.epsilon_input.abs().sqrt()  # type: ignore[operator]
            f_output = self.epsilon_output.sign() * self.epsilon_output.abs().sqrt()  # type: ignore[operator]
            weight = self.mu_weight + self.sigma_weight * f_output.outer(f_input)  # type: ignore[operator]
            bias = self.mu_bias + self.sigma_bias * f_output  # type: ignore[operator]
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(x, weight, bias)
