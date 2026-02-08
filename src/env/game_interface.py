"""Game interface for executing actions."""

from evdev import UInput
from evdev import ecodes as e


class GameInterface:
    """Interface for executing actions in the game."""

    def __init__(self, action_keys: list[int]):
        self.action_keys = action_keys
        self.ui = UInput()

    def execute_action(self, action: int) -> None:
        """Execute an action in the game.

        Args:
            action: The action index to execute (maps to action_keys).
        """
        if action >= len(self.action_keys):
            raise ValueError(f"Action {action} out of range (0-{len(self.action_keys) - 1})")

        key_code = self.action_keys[action]
        if key_code == 0:
            pass
        else:
            self.ui.write(e.EV_KEY, key_code, 1)
            self.ui.write(e.EV_KEY, key_code, 0)
            self.ui.syn()

    def reset_game(self) -> None:
        """Reset the game by pressing up."""
        self.ui.write(e.EV_KEY, e.KEY_UP, 1)
        self.ui.write(e.EV_KEY, e.KEY_UP, 0)
        self.ui.syn()

    def close(self) -> None:
        """Close the game interface."""
        self.ui.close()
