import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mss import mss
import cv2
from pynput.keyboard import Controller, Key
from stable_baselines3 import DQN

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)  # nothing, jump, duck
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.sct = mss()
        self.keyboard = Controller()
        self.game_region = {"top": 150, "left": 100, "width": 600, "height": 150}
        self.frame_stack = []
    
    def _get_obs(self):
        img = np.array(self.sct.grab(self.game_region))
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

# Train
env = DinoEnv()
model = DQN("CnnPolicy", env, verbose=1, buffer_size=50000)
model.learn(total_timesteps=100000)
