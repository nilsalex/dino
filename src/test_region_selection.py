#!/usr/bin/env python3
"""
Test the interactive region selection feature.
"""

import time

import cv2
from PIL import Image

from main import ScreenCapture


def test_region_selection() -> None:
    """Test interactive region selection and frame capture."""
    print("Testing interactive region selection...")

    capture = ScreenCapture()

    # Select region interactively
    region = capture.select_region_interactive()

    if region is None:
        print("Region selection cancelled")
        return

    print(f"\nSelected region: {region}")
    print("\nCapturing 5 frames from the selected region...")

    time.sleep(0.5)

    # Test capturing frames from the selected region
    print(f"\nAttempting to capture from region: {region}")
    frames = []
    for i in range(5):
        frame = capture.grab(region)
        assert frame is not None
        frames.append(frame)
        print(f"Frame {i + 1}: {frame.shape} (expected: ({region['height']}, {region['width']}, 3))")
        time.sleep(0.5)

    print("\n✓ Region selection test complete!")
    print(f"Final region: {region}")

    # Save frames to disk
    print("\nSaving frames to disk...")
    import os

    os.makedirs("captured_frames", exist_ok=True)

    for i, frame in enumerate(frames):
        # Convert BGR to RGB for saving
        assert frame is not None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        filename = f"captured_frames/frame_{i + 1}.png"
        pil_img.save(filename)
        print(f"  Saved: {filename}")

    print("\n✓ Frames saved to ./captured_frames/")
    print(f"Region info: {region['width']}x{region['height']} at ({region['left']}, {region['top']})")
    print("\nYou can view the frames with your image viewer.")


if __name__ == "__main__":
    test_region_selection()
