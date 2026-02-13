"""Game interface using python-xlib for X11 keyboard control."""

from __future__ import annotations

from Xlib import X
from Xlib.display import Display
from Xlib.ext import xtest


class XlibGameInterface:
    """Interface for executing actions via X11 XTest extension."""

    def __init__(self, action_keys: list[int], reset_key: int, display_name: str = ":99"):
        self.action_keys = action_keys
        self.reset_key = reset_key
        self.display = Display(display_name)

    def execute_action(self, action: int) -> None:
        """Execute an action by sending keypress via XTest.

        Args:
            action: The action index to execute (maps to action_keys).
        """
        if action >= len(self.action_keys):
            raise ValueError(f"Action {action} out of range (0-{len(self.action_keys) - 1})")

        keysym = self.action_keys[action]
        if keysym == 0:
            return

        keycode = self.display.keysym_to_keycode(keysym)
        if keycode == 0:
            return
        self._send_key(keycode)

    def reset_game(self) -> None:
        """Reset the game by pressing the reset key."""
        keycode = self.display.keysym_to_keycode(self.reset_key)
        if keycode == 0:
            return
        self._send_key(keycode)

    def _send_key(self, keycode: int) -> None:
        """Send key press and release via XTest."""
        xtest.fake_input(self.display, X.KeyPress, keycode)
        xtest.fake_input(self.display, X.KeyRelease, keycode)
        self.display.flush()

    def close(self) -> None:
        """Close the display connection."""
        self.display.close()
