"""Chrome Dino Game RL Agent with screen capture support."""

import os
import sys
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from PIL import Image
from pynput.keyboard import Controller, Key
from stable_baselines3 import DQN


class ScreenCapture:
    """Cross-platform screen capture that works on X11, Wayland, Windows, and macOS."""

    def __init__(self) -> None:
        self.backend: str = self._detect_backend()
        print(f"Using screen capture backend: {self.backend}")

        if self.backend == "pipewire":
            from pipewire_capture import PipeWireCapture

            self.sct: Any = PipeWireCapture()
        elif self.backend == "mss":
            from mss import mss

            self.sct = mss()
        elif self.backend == "wayland_fallback":
            print("\nERROR: Running on Wayland but PipeWire setup failed.")
            print("Please ensure you have:")
            print("  1. PipeWire running")
            print("  2. xdg-desktop-portal installed")
            print("  3. Your compositor's portal backend installed")
            print("\nAlternatively, run under XWayland: env -u WAYLAND_DISPLAY python main.py")
            sys.exit(1)

    def _detect_backend(self) -> str:
        """Detect the best screen capture backend for the current environment."""
        session_type: str = os.environ.get("XDG_SESSION_TYPE", "").lower()
        wayland_display: str = os.environ.get("WAYLAND_DISPLAY", "")

        if session_type == "wayland" or wayland_display:
            # On Wayland, try PipeWire (best performance)
            try:
                # Check if required modules are available
                import gi

                gi.require_version("Gst", "1.0")
                import dbus  # noqa: F401

                return "pipewire"
            except (ImportError, ValueError) as e:
                print(f"PipeWire backend not available: {e}")
                return "wayland_fallback"
        # Use mss for X11, Windows, macOS
        return "mss"

    def grab(self, region: dict[str, int]) -> NDArray[np.uint8] | None:
        """Capture a screen region. Returns a numpy array in BGR format."""
        if self.backend == "pipewire":
            # PipeWireCapture.grab returns BGR numpy array
            return self.sct.grab(region)
        if self.backend == "mss":
            img = self.sct.grab(region)
            # mss returns BGRA, convert to BGR
            return np.array(img)[:, :, :3]
        return None

    def select_region_interactive(self) -> dict[str, int] | None:
        """
        Let user select a screen region by clicking two points on the screen.

        Returns:
            dict: Region with keys 'top', 'left', 'width', 'height', or None if cancelled
        """
        import tkinter as tk

        from PIL import ImageTk

        print("\n" + "=" * 60)
        print("INTERACTIVE REGION SELECTION")
        print("=" * 60)
        print("Capturing screen...")

        # Get monitor info for positioning the overlay
        monitor_info = None

        if self.backend == "mss":
            # Show available monitors
            print("\nAvailable monitors:")
            from mss.base import MSSBase

            assert isinstance(self.sct, MSSBase)
            for i, mon in enumerate(self.sct.monitors):
                print(f"  Monitor {i}: {mon}")

            # Use monitor 1 (primary monitor) instead of 0 (all monitors combined)
            # Monitor 0 in mss is a virtual screen of all monitors combined
            monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
            print(f"\nUsing monitor: {monitor}")
            full_screen = self.grab(monitor)

            # Store monitor offset for coordinate adjustment
            monitor_left = monitor["left"]
            monitor_top = monitor["top"]
            monitor_info = monitor
        elif self.backend == "pipewire":
            full_screen = self.sct.grab(None)

            # Try to get monitor position from portal metadata
            monitor_info = self.sct.get_monitor_info()
            print(f"\nDebug: monitor_info = {monitor_info}")
            if monitor_info:
                monitor_left = monitor_info["left"]
                monitor_top = monitor_info["top"]
                print(
                    f"Using monitor at position ({monitor_left}, {monitor_top}), "
                    f"size {monitor_info['width']}x{monitor_info['height']}"
                )
            else:
                monitor_left = 0
                monitor_top = 0
                print("Monitor position unknown (portal didn't provide metadata)")
                print("Overlay will appear on default monitor")

        if full_screen is None:
            raise RuntimeError("Failed to capture screen")

        # Convert BGR to RGB for PIL
        screen_rgb = cv2.cvtColor(full_screen, cv2.COLOR_BGR2RGB)
        capture_height, capture_width = screen_rgb.shape[:2]

        print(f"Captured screen: {capture_width}x{capture_height}")

        # Convert to PIL Image
        pil_image = Image.fromarray(screen_rgb)

        # Determine display dimensions
        # For PipeWire with monitor metadata, use the reported monitor size
        # For mss or PipeWire without metadata, use tkinter's screen dimensions
        root = tk.Tk()
        root.update_idletasks()

        if monitor_info and self.backend == "pipewire":
            # Use the monitor dimensions from portal metadata
            display_width = monitor_info["width"]
            display_height = monitor_info["height"]
            print(f"Using portal monitor dimensions: {display_width}x{display_height}")
        else:
            # Fall back to tkinter screen dimensions (multi-monitor setup will use total size)
            display_width = root.winfo_screenwidth()
            display_height = root.winfo_screenheight()
            print(f"Using tkinter screen dimensions: {display_width}x{display_height}")

        # Scale image to display dimensions
        display_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # State for drag-to-draw rectangle selection
        start_pos = None  # Starting position in display coordinates
        current_rect = None  # Canvas rectangle ID for live preview
        current_text = None  # Canvas text ID for dimensions
        cancelled = [False]
        final_region = [None]  # Store final selected region

        def on_button_press(event):
            nonlocal start_pos, current_rect, current_text
            # Record starting position
            start_pos = (event.x, event.y)

        def on_mouse_drag(event):
            nonlocal current_rect, current_text
            if start_pos is None:
                return

            # Delete previous rectangle and text if they exist
            if current_rect:
                canvas.delete(current_rect)
            if current_text:
                canvas.delete(current_text)

            # Draw new rectangle from start to current position
            x1, y1 = start_pos
            x2, y2 = event.x, event.y

            current_rect = canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=4, tags="selection")

            # Calculate and display dimensions in actual capture coordinates
            scale_x = capture_width / display_width
            scale_y = capture_height / display_height

            actual_x1 = int(x1 * scale_x)
            actual_y1 = int(y1 * scale_y)
            actual_x2 = int(x2 * scale_x)
            actual_y2 = int(y2 * scale_y)

            w = abs(actual_x2 - actual_x1)
            h = abs(actual_y2 - actual_y1)

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            current_text = canvas.create_text(
                mid_x,
                mid_y,
                text=f"{w}x{h}",
                fill="yellow",
                font=("Arial", 18, "bold"),
                tags="selection",
            )

        def on_button_release(event):
            nonlocal start_pos, final_region
            if start_pos is None:
                return

            # Calculate final region in actual screen coordinates
            x1, y1 = start_pos
            x2, y2 = event.x, event.y

            scale_x = capture_width / display_width
            scale_y = capture_height / display_height

            actual_x1 = int(x1 * scale_x) + monitor_left
            actual_y1 = int(y1 * scale_y) + monitor_top
            actual_x2 = int(x2 * scale_x) + monitor_left
            actual_y2 = int(y2 * scale_y) + monitor_top

            left = min(actual_x1, actual_x2)
            top = min(actual_y1, actual_y2)
            right = max(actual_x1, actual_x2)
            bottom = max(actual_y1, actual_y2)

            width = right - left
            height = bottom - top

            if width > 0 and height > 0:
                final_region[0] = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
                print(f"\nRegion selected: {width}x{height} at ({left}, {top})")
                root.after(500, root.quit)
            else:
                print("\nInvalid selection (too small)")
                start_pos = None

        def on_escape(_event):
            cancelled[0] = True
            root.quit()

        # Position window on the correct monitor
        # On Wayland, fullscreen mode ignores geometry hints, so we use a borderless window instead
        geometry_str = f"{display_width}x{display_height}+{monitor_left}+{monitor_top}"
        print(f"Setting window geometry: {geometry_str}")
        root.geometry(geometry_str)

        # Remove window decorations for a clean fullscreen-like experience
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        # Force window manager to process the geometry
        root.update_idletasks()

        canvas = tk.Canvas(root, width=display_width, height=display_height, highlightthickness=0)
        canvas.pack()

        # Display the scaled image
        photo = ImageTk.PhotoImage(display_image)
        canvas.create_image(0, 0, image=photo, anchor="nw")

        # Keep reference to prevent garbage collection
        root.photo = photo  # type: ignore[attr-defined]

        # Draw instructions overlay at top
        canvas.create_rectangle(0, 0, display_width, 120, fill="black", stipple="gray50")
        canvas.create_text(
            display_width // 2,
            40,
            text="Click and drag to select region",
            fill="yellow",
            font=("Arial", 24, "bold"),
        )
        canvas.create_text(
            display_width // 2,
            80,
            text="Press ESC to cancel",
            fill="yellow",
            font=("Arial", 16),
        )

        # Bind mouse events for drag-to-draw interaction
        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_button_release)
        root.bind("<Escape>", on_escape)

        print("\nWaiting for selection...")
        root.mainloop()
        root.destroy()

        if cancelled[0] or final_region[0] is None:
            print("\n✗ Selection cancelled")
            return None

        region = final_region[0]
        print(f"\n✓ Region selected: {region['width']}x{region['height']} at ({region['left']}, {region['top']})")
        return region


class DinoEnv(gym.Env):  # type: ignore[misc]
    """Chrome Dino Gymnasium environment."""

    def __init__(self, select_region: bool = False, game_over_threshold: float = 0.99) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(3)  # nothing, jump, duck
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.sct: ScreenCapture = ScreenCapture()
        self.keyboard: Controller = Controller()

        # Select or use default game region
        if select_region:
            region = self.sct.select_region_interactive()
            if region:
                self.game_region: dict[str, int] = region
            else:
                print("Selection cancelled, using default region")
                self.game_region = {
                    "top": 150,
                    "left": 100,
                    "width": 600,
                    "height": 150,
                }
        else:
            self.game_region = {"top": 150, "left": 100, "width": 600, "height": 150}

        self.frame_stack: list[NDArray[Any]] = []

        # Game over detection
        self.game_over_threshold: float = game_over_threshold
        self.previous_frames: list[NDArray[Any]] = []  # Store last few frames for comparison
        self.num_frames_to_compare: int = 3  # Check if last 3 frames are identical
        self.game_over_check_region: float = 0.6  # Use left 60% of frame to avoid reload button

    def _get_obs(self) -> NDArray[np.uint8]:
        img = self.sct.grab(self.game_region)
        assert img is not None
        gray: NDArray[Any] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized: NDArray[Any] = cv2.resize(gray, (84, 84))

        self.frame_stack.append(resized)
        if len(self.frame_stack) > 4:
            self.frame_stack.pop(0)
        while len(self.frame_stack) < 4:
            self.frame_stack.append(resized)

        return np.array(self.frame_stack)

    def _is_game_over(self) -> bool:
        """
        Detect game over by checking if the screen has stopped changing.

        When the game ends, the screen freezes (except for the reload button in center).
        We compare the left portion of recent frames to detect this.

        Returns:
            bool: True if game over detected, False otherwise
        """
        # Get current frame (before processing)
        current_frame = self.sct.grab(self.game_region)
        if current_frame is None:
            return False

        # Convert to grayscale for comparison
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Only check the left portion to avoid the reload button in center
        check_width = int(gray.shape[1] * self.game_over_check_region)
        gray_left = gray[:, :check_width]

        # Store frame for comparison
        self.previous_frames.append(gray_left)

        # Keep only the last N frames
        if len(self.previous_frames) > self.num_frames_to_compare:
            self.previous_frames.pop(0)

        # Need at least num_frames_to_compare frames to compare
        if len(self.previous_frames) < self.num_frames_to_compare:
            return False

        # Check if all recent frames are nearly identical
        # Compare each frame with the previous one
        for i in range(1, len(self.previous_frames)):
            frame1 = self.previous_frames[i - 1]
            frame2 = self.previous_frames[i]

            # Calculate similarity (correlation coefficient)
            # Returns value between 0 (completely different) and 1 (identical)
            correlation = np.corrcoef(frame1.flatten(), frame2.flatten())[0, 1]

            # If any pair is not similar enough, game is still running
            if correlation < self.game_over_threshold:
                return False

        # All consecutive frames are nearly identical -> game over
        return True

    def step(self, action: int) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        if action == 1:
            self.keyboard.press(Key.space)
            self.keyboard.release(Key.space)
        elif action == 2:
            self.keyboard.press(Key.down)
            self.keyboard.release(Key.down)

        obs = self._get_obs()
        done = self._is_game_over()
        reward = 1.0 if not done else -100.0  # survival reward

        return obs, reward, done, False, {}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        # Clear game over detection state
        self.previous_frames = []

        # Restart game
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

        # Wait a bit for game to restart
        import time

        time.sleep(0.5)

        self.frame_stack = []
        return self._get_obs(), {}


def main() -> None:
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Chrome Dino RL Agent")
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Interactively select the game region before training",
    )
    args = parser.parse_args()

    env = DinoEnv(select_region=args.select_region)

    print("\nStarting RL training...")
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=50000)
    model.learn(total_timesteps=100000)


if __name__ == "__main__":
    main()
