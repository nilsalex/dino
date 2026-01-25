"""Visualization utilities for debugging frames."""

from pathlib import Path

import numpy as np
from PIL import Image


def save_frame(frame: np.ndarray, path: Path) -> None:
    """Save a frame to disk as an image.

    Args:
        frame: The frame to save (grayscale 2D array).
        path: The file path to save to.
    """
    img = Image.fromarray(frame, mode="L")
    img.save(path)


def display_info(msg: str) -> None:
    """Display an informational message.

    Args:
        msg: The message to display.
    """
    print(msg)
