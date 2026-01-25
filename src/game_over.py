import numpy as np


def check_game_over(frame_buffer):
    stacked = np.stack(list(frame_buffer), axis=0)
    pixel_variance = np.var(stacked, axis=0)
    return np.mean(pixel_variance) < 0.1
