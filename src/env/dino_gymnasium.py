"""Gymnasium environment for Chrome Dino game."""

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.capture.frame_processor import FrameProcessor
from src.core.config import Config
from src.env.game_interface import GameInterface
from src.env.state_monitor import StateMonitor


class DinoEnvironment(gym.Env):
    """Gymnasium environment for Dino game RL."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

        self.game_interface = GameInterface()
        self.state_monitor = StateMonitor()
        self.frame_processor = FrameProcessor(self.config)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.config.frame_stack, self.config.output_height, self.config.output_width),
            dtype=np.uint8,  # type: ignore[arg-type]
        )

        self.action_space = gym.spaces.Discrete(self.config.n_actions)

        self._state_buffer: list[NDArray[np.uint8]] = []
        self._episode_step = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[NDArray[np.uint8], dict]:
        super().reset(seed=seed)

        self.game_interface.reset_game()
        self.frame_processor.frame_buffer.clear()
        while not self.frame_processor.frame_queue.empty():
            try:
                self.frame_processor.frame_queue.get_nowait()
            except Exception:
                break
        self._state_buffer = []
        self._episode_step = 0

        obs = self._get_observation()
        info: dict[str, int] = {}

        return obs, info

    def step(self, action: int) -> tuple[NDArray[np.uint8], float, bool, bool, dict]:
        self.game_interface.execute_action(action)

        obs = self._get_observation()
        reward = self._compute_reward(obs)
        terminated = self._is_terminated()
        truncated = False

        self._episode_step += 1

        info = {"episode_step": self._episode_step}

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> NDArray[np.uint8]:
        state = self.frame_processor.get_state()

        if state is None:
            return np.zeros(
                (self.config.frame_stack, self.config.output_height, self.config.output_width),
                dtype=np.uint8,
            )

        return state.detach().cpu().numpy()

    def _compute_reward(self, obs: NDArray[np.uint8]) -> float:
        if self._is_terminated():
            return -1.0

        return 0.1

    def _is_terminated(self) -> bool:
        if len(self.frame_processor.frame_buffer) < self.config.frame_stack:
            return False

        return self.state_monitor.is_game_over(self.frame_processor.frame_buffer)

    def close(self) -> None:
        self.game_interface.close()
        self.frame_processor.frame_buffer.clear()
