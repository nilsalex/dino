"""Training thread for continuous background training on remote GPU."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import ray
import torch

if TYPE_CHECKING:
    from src.buffers.thread_safe_buffer import ThreadSafeExperienceBuffer
    from src.core.config import Config


class TrainingThread:
    """Background training thread for continuous remote GPU training.

    Continuously samples from experience buffer and trains on remote GPU
    server independently of main game loop.

    Args:
        config: Configuration object.
        buffer: Thread-safe experience replay buffer.
        remote_trainer: Remote Ray trainer actor.
        on_weights_updated: Callback when weights are synced from remote to local.

    Attributes:
        running: Whether training thread is active.
        thread: The training thread object.
        last_loss: Most recent training loss value.
    """

    def __init__(
        self,
        config: Config,
        buffer: ThreadSafeExperienceBuffer,
        remote_trainer: Any,
        on_weights_updated: Callable[[dict[str, torch.Tensor]], None],
    ) -> None:
        self.config = config
        self.buffer = buffer
        self.remote_trainer = remote_trainer
        self.on_weights_updated = on_weights_updated

        self.running: bool = False
        self.thread: threading.Thread | None = None
        self.last_loss: float = 0.0
        self.training_count: int = 0
        self.weight_sync_count: int = 0
        self._pending_refs: list[ray.ObjectRef] = []

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
                if len(self._pending_refs) < batch_size:
                    self._dispatch_train_step(self._pending_refs)
                else:
                    self._collect_batch(self._pending_refs)
            except Exception as e:
                print(f"Training error: {e}")
                time.sleep(0.1)

    def _dispatch_train_step(self, pending_refs: list[ray.ObjectRef]) -> None:
        """Dispatch training step ref without blocking."""

        states_list, actions_list, rewards_list, next_states_list, dones_list = self.buffer.sample(
            self.config.batch_size
        )

        state_batch = [s.detach().cpu().numpy() for s in states_list]
        action_batch = actions_list
        reward_batch = rewards_list
        next_state_batch = [ns.detach().cpu().numpy() for ns in next_states_list]
        done_batch = dones_list

        loss_ref = self.remote_trainer.train_step.remote(  # type: ignore[arg-type]
            state_batch=state_batch,
            action_batch=action_batch,
            reward_batch=reward_batch,
            next_state_batch=next_state_batch,
            done_batch=done_batch,
        )
        pending_refs.append(loss_ref)

    def _collect_batch(self, pending_refs: list[ray.ObjectRef]) -> None:
        """Collect results from batch of training refs."""

        if not pending_refs:
            return

        loss_results = ray.get(pending_refs)  # type: ignore[arg-type]

        last_loss = 0.0
        for loss_result in loss_results:
            last_loss = float(loss_result["loss"])  # type: ignore[index]
            self.training_count += 1

        self.last_loss = last_loss
        pending_refs.clear()

        should_sync = self.training_count % (self.config.target_update_freq // self.config.batch_size) == 0

        if should_sync:
            self._sync_weights()

    def _sync_weights(self) -> None:
        """Sync weights from remote trainer to local inference model."""

        remote_state_ref = self.remote_trainer.get_model_state.remote()  # type: ignore[arg-type]
        remote_state = ray.get(remote_state_ref)
        if isinstance(remote_state, dict):
            self.on_weights_updated(remote_state)
        self.weight_sync_count += 1

    def get_stats(self) -> dict[str, int | float]:
        """Get training statistics.

        Returns:
            Dictionary with training stats.
        """

        return {
            "last_loss": self.last_loss,
            "training_count": self.training_count,
            "weight_sync_count": self.weight_sync_count,
        }
