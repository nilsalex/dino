"""Type definitions for the DINO reinforcement learning system."""

from collections import deque, namedtuple
from typing import Any

import numpy as np
import torch

# Type aliases
Device = torch.device
Frame = np.ndarray
FrameBuffer = deque[Frame]
Action = int
State = torch.Tensor
Reward = float
Done = bool

_ExperienceImpl = namedtuple("_ExperienceImpl", ["state", "action", "reward", "next_state", "done"])


Experience = _ExperienceImpl


def sample_callback(*_args: object) -> Any:
    """Type hint for GStreamer new-sample callback."""
    ...
