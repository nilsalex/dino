import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from pynput.keyboard import Controller, Key
from stable_baselines3 import DQN
import sys
import os
import subprocess
from PIL import Image
import io

class ScreenCapture:
    """Cross-platform screen capture that works on X11, Wayland, Windows, and macOS."""
    def __init__(self):
        self.backend = self._detect_backend()
        print(f"Using screen capture backend: {self.backend}")

        if self.backend == "pipewire":
            from pipewire_capture import PipeWireCapture
            self.sct = PipeWireCapture()
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

    def _detect_backend(self):
        """Detect the best screen capture backend for the current environment."""
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        wayland_display = os.environ.get("WAYLAND_DISPLAY", "")

        if session_type == "wayland" or wayland_display:
            # On Wayland, try PipeWire (best performance)
            try:
                # Check if required modules are available
                import gi
                gi.require_version('Gst', '1.0')
                import dbus
                return "pipewire"
            except (ImportError, ValueError) as e:
                print(f"PipeWire backend not available: {e}")
                return "wayland_fallback"
        else:
            # Use mss for X11, Windows, macOS
            return "mss"

    def grab(self, region):
        """Capture a screen region. Returns a numpy array in BGR format."""
        if self.backend == "pipewire":
            # PipeWireCapture.grab returns BGR numpy array
            return self.sct.grab(region)
        elif self.backend == "mss":
            img = self.sct.grab(region)
            # mss returns BGRA, convert to BGR
            return np.array(img)[:, :, :3]

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)  # nothing, jump, duck
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.sct = ScreenCapture()
        self.keyboard = Controller()
        self.game_region = {"top": 150, "left": 100, "width": 600, "height": 150}
        self.frame_stack = []

    def _get_obs(self):
        img = self.sct.grab(self.game_region)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        
        self.frame_stack.append(resized)
        if len(self.frame_stack) > 4:
            self.frame_stack.pop(0)
        while len(self.frame_stack) < 4:
            self.frame_stack.append(resized)
        
        return np.array(self.frame_stack)
    
    def _is_game_over(self):
        # Detect "game over" state (e.g., check for specific pixels or template)
        pass
    
    def step(self, action):
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
    
    def reset(self, seed=None):
        self.keyboard.press(Key.space)  # restart game
        self.keyboard.release(Key.space)
        self.frame_stack = []
        return self._get_obs(), {}

def main():
    """Main training function."""
    env = DinoEnv()
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=50000)
    model.learn(total_timesteps=100000)

if __name__ == "__main__":
    main()
