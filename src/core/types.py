"""Type definitions for the DINO reinforcement learning system."""

from collections import deque
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


class Experience:
    """A single experience tuple for replay buffer."""

    state: State
    action: Action
    reward: Reward
    next_state: State | None
    done: Done


def gst_callback(_bus: Any, _message: Any, _loop: Any) -> bool:
    """Type hint for GStreamer bus callback."""
    ...


def sample_callback(*_args: object) -> Any:
    """Type hint for GStreamer new-sample callback."""
    ...
