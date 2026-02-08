"""Training thread for continuous background training on local device."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.agents.local_trainer import LocalDQNTrainer
    from src.buffers.thread_safe_buffer import ThreadSafeExperienceBuffer
    from src.core.config import Config


class LocalTrainingThread:
    """Background training thread for continuous on-device training.

    Continuously samples from experience buffer and trains on local CPU/GPU
    independently of main game loop.

    Args:
        config: Configuration object.
        buffer: Thread-safe experience replay buffer.
        trainer: Local DQN trainer.
        on_weights_updated: Callback when weights update (for local inference model).

    Attributes:
        running: Whether training thread is active.
        thread: The training thread object.
        last_loss: Most recent training loss value.
    """

    def __init__(
        self,
        config: Config,
        buffer: ThreadSafeExperienceBuffer,
        trainer: LocalDQNTrainer,
        on_weights_updated: Callable[[dict[str, torch.Tensor]], None] | None = None,
    ) -> None:
        self.config = config
        self.buffer = buffer
        self.trainer = trainer
        self.on_weights_updated = on_weights_updated

        self.running: bool = False
        self.thread: threading.Thread | None = None
        self.last_loss: float = 0.0
        self.last_q_mean: float = 0.0
        self.last_q_max: float = 0.0
        self.last_target_mean: float = 0.0
        self.training_count: int = 0
        self.weight_sync_count: int = 0
        self._pending_losses: list[dict[str, float]] = []

    def start(self) -> None:
        """Start training thread."""

        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop training thread."""

        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None

    def _training_loop(self) -> None:
        """Main training loop - runs continuously in background."""

        batch_size = 16

        while self.running:
            buffer_size = self.buffer.size()

            if buffer_size < self.config.min_buffer_size:
                time.sleep(0.1)
                continue

            try:
                if len(self._pending_losses) < batch_size:
                    self._dispatch_train_step()
                else:
                    self._collect_batch()
            except Exception as e:
                print(f"Training error: {e}")
                time.sleep(0.1)

    def _dispatch_train_step(self) -> None:
        """Dispatch training step (store in pending refs)."""

        states_list, actions_list, rewards_list, next_states_list, dones_list = self.buffer.sample(
            self.config.batch_size
        )

        state_batch = [s.detach().cpu().numpy() for s in states_list]
        action_batch = actions_list
        reward_batch = rewards_list
        next_state_batch = [ns.detach().cpu().numpy() for ns in next_states_list]
        done_batch = dones_list

        loss_info = self.trainer.train_step(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        self._pending_losses.append(loss_info)

    def _collect_batch(self) -> None:
        """Collect results from batch of training steps."""

        if not self._pending_losses:
            return

        for loss_info in self._pending_losses:
            self.last_loss = loss_info["loss"]
            self.last_q_mean = loss_info.get("q_mean", 0.0)
            self.last_q_max = loss_info.get("q_max", 0.0)
            self.last_target_mean = loss_info.get("target_mean", 0.0)
            self.training_count += 1

        self._pending_losses.clear()

        should_sync = self.training_count % (self.config.target_update_freq // self.config.batch_size) == 0

        if should_sync:
            self._sync_weights()

    def _sync_weights(self) -> None:
        """Sync weights from trainer to local inference model."""

        if self.on_weights_updated:
            self.on_weights_updated(self.trainer.get_model_state())

        self.trainer.update_target()
        self.weight_sync_count += 1

    def get_stats(self) -> dict[str, int | float]:
        """Get training statistics.

        Returns:
            Dictionary with training stats.
        """

        return {
            "last_loss": self.last_loss,
            "q_mean": self.last_q_mean,
            "q_max": self.last_q_max,
            "target_mean": self.last_target_mean,
            "training_count": self.training_count,
            "weight_sync_count": self.weight_sync_count,
        }
