"""Buffer for pending transitions that can be retroactively marked terminal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class PendingTransition:
    """A transition waiting to be committed to the experience buffer."""

    state: torch.Tensor
    action: int


class PendingTransitionBuffer:
    """Buffer that holds recent transitions and can retroactively mark one as terminal.

    Used to handle delayed game-over detection where we need to mark an earlier
    transition as terminal after the fact.
    """

    def __init__(self, max_size: int = 5):
        self._buffer: deque[PendingTransition] = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, state: torch.Tensor, action: int) -> None:
        """Add a pending transition.

        Args:
            state: The state tensor when action was taken.
            action: The action taken.
        """
        self._buffer.append(PendingTransition(state, action))

    def commit(self, callback, next_state: torch.Tensor, game_over_offset: int = 0) -> None:
        """Commit all pending transitions.

        Args:
            callback: Function(state, action, reward, next_state, done) to add to experience buffer.
            next_state: The current state (next state for the last pending transition).
            game_over_offset: If > 0, mark transition at len - offset as terminal with reward -1.
                             Transitions after that one are discarded.
        """
        transitions = list(self._buffer)
        self._buffer.clear()

        if len(transitions) == 0:
            return

        if game_over_offset <= 0 or game_over_offset > len(transitions):
            for i, t in enumerate(transitions):
                ns = transitions[i + 1].state if i + 1 < len(transitions) else next_state
                callback(t.state, t.action, 0.0, ns, False)
        else:
            terminal_idx = len(transitions) - game_over_offset
            for i in range(terminal_idx):
                t = transitions[i]
                ns = transitions[i + 1].state
                callback(t.state, t.action, 0.0, ns, False)

            t = transitions[terminal_idx]
            ns = transitions[terminal_idx + 1].state if terminal_idx + 1 < len(transitions) else next_state
            callback(t.state, t.action, -1.0, ns, True)

    def clear(self) -> None:
        """Clear all pending transitions."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
