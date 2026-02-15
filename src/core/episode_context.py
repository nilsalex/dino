"""Episode context for sharing state between components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.local_trainer import LocalDQNTrainer
    from src.buffers.thread_safe_buffer import ThreadSafeExperienceBuffer
    from src.env.game_interface import GameInterface
    from src.env.state_monitor import StateMonitor
    from src.env.xlib_interface import XlibGameInterface
    from src.utils.tensorboard_logger import TensorBoardLogger

from src.core.config import Config
from src.core.game_config import GameConfig


@dataclass
class EpisodeContext:
    """Shared context for episode management.

    Holds references to all components needed for episode lifecycle management.
    """

    config: Config
    game_config: GameConfig
    game_interface: GameInterface | XlibGameInterface
    state_monitor: StateMonitor
    experience_buffer: ThreadSafeExperienceBuffer
    local_trainer: LocalDQNTrainer
    tb_logger: TensorBoardLogger
