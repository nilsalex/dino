"""Episode lifecycle management."""

from __future__ import annotations

from enum import Enum, auto

import torch

from src.core.episode_context import EpisodeContext
from src.core.types import Frame


class EpisodeEvent(Enum):
    """Events returned by EpisodeManager.process_frame()."""

    NONE = auto()
    GAME_OVER = auto()
    RESET_GAME = auto()
    EPISODE_READY = auto()
    EVAL_TIMEOUT = auto()


class EpisodeState(Enum):
    """Internal state machine states."""

    RUNNING = auto()
    GAME_OVER_DETECTED = auto()
    RESETTING = auto()
    EVAL_TIMEOUT_DETECTED = auto()


class EpisodeManager:
    """Manages complete episode lifecycle.

    Handles:
    - Game over detection
    - Reset timing (delay before reset, cooldown after)
    - Episode counting and statistics
    - Evaluation mode tracking
    """

    def __init__(self, ctx: EpisodeContext):
        self._ctx = ctx

        self._state = EpisodeState.RUNNING
        self._delay_counter = 0
        self._cooldown_counter = 0

        self._episode_count = 0
        self._episode_steps = 0
        self._curr_reward = 0.0
        self._total_reward = 0.0

        self._is_evaluating = True
        self._eval_episode_count = 1
        self._eval_episodes_remaining = ctx.config.eval_episodes
        self._eval_step_count = 0
        self._best_eval_score = 0

    def process_frame(self, frames: list[Frame], current_state: torch.Tensor) -> EpisodeEvent:
        """Process frame and handle episode lifecycle.

        Args:
            frames: List of recent frames for game over detection.
            current_state: Current state tensor (unused, kept for API compatibility).

        Returns:
            EpisodeEvent indicating what happened.
        """
        is_game_over, offset = self._ctx.state_monitor.check_game_over(frames)

        if self._state == EpisodeState.RUNNING:
            return self._handle_running(is_game_over, offset)

        if self._state == EpisodeState.GAME_OVER_DETECTED:
            return self._handle_game_over_detected()

        if self._state == EpisodeState.RESETTING:
            return self._handle_resetting(is_game_over)

        if self._state == EpisodeState.EVAL_TIMEOUT_DETECTED:
            return self._handle_eval_timeout(is_game_over)

        return EpisodeEvent.NONE

    def _handle_running(self, is_game_over: bool, offset: int) -> EpisodeEvent:
        """Handle RUNNING state."""
        if not is_game_over:
            return EpisodeEvent.NONE

        if offset == 0:
            return EpisodeEvent.NONE

        if not self._is_evaluating:
            self._ctx.experience_buffer.mark_last_as_terminal(-1.0)
            self._curr_reward += -1.0

        self._state = EpisodeState.GAME_OVER_DETECTED
        self._delay_counter = 0

        return EpisodeEvent.GAME_OVER

    def _handle_game_over_detected(self) -> EpisodeEvent:
        """Handle GAME_OVER_DETECTED state."""
        self._delay_counter += 1

        if self._delay_counter < self._ctx.game_config.reset_delay_frames:
            return EpisodeEvent.NONE

        self._state = EpisodeState.RESETTING
        self._cooldown_counter = 0

        return EpisodeEvent.RESET_GAME

    def _handle_resetting(self, is_game_over: bool) -> EpisodeEvent:
        """Handle RESETTING state."""
        if is_game_over:
            self._cooldown_counter += 1
            return EpisodeEvent.NONE

        self._state = EpisodeState.RUNNING

        return self._complete_episode()

    def _handle_eval_timeout(self, is_game_over: bool) -> EpisodeEvent:
        """Handle EVAL_TIMEOUT_DETECTED state - waiting for game over after timeout."""
        if not is_game_over:
            return EpisodeEvent.NONE

        self._state = EpisodeState.RUNNING
        return self._complete_episode()

    def _complete_episode(self) -> EpisodeEvent:
        """Complete the current episode and prepare for the next."""
        if self._is_evaluating:
            return self._complete_eval_episode()
        return self._complete_training_episode()

    def _complete_eval_episode(self) -> EpisodeEvent:
        """Complete an evaluation episode."""
        if self._eval_step_count > 1:
            print(
                f"\n[EVAL] Episode {self._eval_episode_count} complete. Steps: {self._eval_step_count}\n",
                end="",
                flush=True,
            )
            if self._eval_step_count > self._best_eval_score:
                self._best_eval_score = self._eval_step_count
                print(f"[BEST] New evaluation score: {self._best_eval_score}\n")

            self._eval_episodes_remaining -= 1
            self._eval_episode_count += 1
            self._eval_step_count = 0

            if self._eval_episodes_remaining > 0:
                print(f"[EVAL] Starting next eval episode ({self._eval_episodes_remaining} remaining)\n")
                return EpisodeEvent.EPISODE_READY

            self._is_evaluating = False
            self._eval_episodes_remaining = self._ctx.config.eval_episodes
            self._curr_reward = 0.0
            return EpisodeEvent.EPISODE_READY

        print(f"[EVAL] Episode too short ({self._eval_step_count} steps), restarting eval...")
        self._eval_step_count = 0
        return EpisodeEvent.EPISODE_READY

    def _complete_training_episode(self) -> EpisodeEvent:
        """Complete a training episode."""
        if self._episode_steps >= self._ctx.game_config.min_episode_steps:
            self._episode_count += 1
            self._total_reward += self._curr_reward

            print(
                f"\r\x1b[K[GAME OVER] Episode {self._episode_count} complete. "
                f"Steps: {self._episode_steps}, Reward: {self._curr_reward:.2f}\n",
                end="",
                flush=True,
            )

            self._ctx.tb_logger.log_episode(self._episode_count, self._curr_reward, self._episode_steps)
            self._curr_reward = 0.0
            self._episode_steps = 0

            return EpisodeEvent.EPISODE_READY

        print(f"[WARN] Episode too short ({self._episode_steps} steps), not counting")
        self._curr_reward = 0.0
        self._episode_steps = 0

        return EpisodeEvent.EPISODE_READY

    def is_game_over(self) -> bool:
        """Check if currently in game over or eval timeout state."""
        return self._state in (
            EpisodeState.GAME_OVER_DETECTED,
            EpisodeState.RESETTING,
            EpisodeState.EVAL_TIMEOUT_DETECTED,
        )

    def is_evaluating(self) -> bool:
        """Check if currently in evaluation mode."""
        return self._is_evaluating

    def start_evaluation(self) -> None:
        """Start evaluation phase."""
        self._is_evaluating = True
        self._eval_episode_count += 1
        self._eval_episodes_remaining = self._ctx.config.eval_episodes
        print("[EVAL] Starting evaluation phase\n")

    def increment_step(self) -> EpisodeEvent:
        """Increment the appropriate step counter.

        Returns:
            EpisodeEvent.EVAL_TIMEOUT if evaluation episode exceeded max_eval_steps,
            otherwise EpisodeEvent.NONE.
        """
        if self._is_evaluating:
            self._eval_step_count += 1
            if self._eval_step_count >= self._ctx.game_config.max_eval_steps:
                self._state = EpisodeState.EVAL_TIMEOUT_DETECTED
                print(
                    f"\n[EVAL] Episode {self._eval_episode_count} reached max steps ({self._eval_step_count})\n",
                    end="",
                    flush=True,
                )
                return EpisodeEvent.EVAL_TIMEOUT
        else:
            self._episode_steps += 1
        return EpisodeEvent.NONE

    def get_stats(self) -> dict:
        """Get episode statistics.

        Returns:
            Dict with episode_count, episode_steps, curr_reward, total_reward,
            best_eval_score, is_evaluating, eval_episodes_remaining.
        """
        return {
            "episode_count": self._episode_count,
            "episode_steps": self._episode_steps,
            "curr_reward": self._curr_reward,
            "total_reward": self._total_reward,
            "best_eval_score": self._best_eval_score,
            "is_evaluating": self._is_evaluating,
            "eval_episodes_remaining": self._eval_episodes_remaining,
            "eval_step_count": self._eval_step_count,
        }
