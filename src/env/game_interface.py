"""Game interface for executing actions."""

from evdev import UInput
from evdev import ecodes as e

from src.core.constants import JUMP_ACTION, NO_ACTION


class GameInterface:
    """Interface for executing actions in the game."""

    def __init__(self):
        self.ui = UInput()

    def execute_action(self, action: int) -> None:
        """Execute an action in the game.

        Args:
            action: The action to execute (0 = no action, 1 = jump).
        """
        if action == JUMP_ACTION:
            self.ui.write(e.EV_KEY, e.KEY_UP, 1)
            self.ui.write(e.EV_KEY, e.KEY_UP, 0)
            self.ui.syn()
        elif action == NO_ACTION:
            pass

    def reset_game(self) -> None:
        """Reset the game by pressing up."""
        self.ui.write(e.EV_KEY, e.KEY_UP, 1)
        self.ui.write(e.EV_KEY, e.KEY_UP, 0)
        self.ui.syn()

    def close(self) -> None:
        """Close the game interface."""
        self.ui.close()
