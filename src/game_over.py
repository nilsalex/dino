import numpy as np


def check_game_over(frame_buffer):
    stacked = np.stack(list(frame_buffer), axis=0)
    pixel_variance = np.var(stacked, axis=0)
    return np.mean(pixel_variance) < 0.1
    prev = frame_buffer[-2].flatten()
    cur = frame_buffer[-1].flatten()

    return (prev==cur).all()

    # prev = frame_buffer[-2][:, :width].flatten().astype(np.float32)
    # cur = frame_buffer[-1][:, :width].flatten().astype(np.float32)
    #
    # corr_matrix = np.corrcoef(prev, cur)
    # correlation = corr_matrix[0,1]
    #
    # return correlation > 0.9999

