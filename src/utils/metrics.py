"""Metrics tracking and logging utilities."""

import time
from dataclasses import dataclass


@dataclass
class Metrics:
    """Frame processing metrics."""

    fps: float = 0.0
    avg_frame_time: float = 0.0
    process_time: float = 0.0
    queue_size: int = 0
    queue_max_size: int = 0
    queue_latency_ms: float = 0.0


class MetricsTracker:
    """Track and display performance metrics."""

    def __init__(self, window_size: float = 1.0):
        self.window_size = window_size
        self.frame_times: list[tuple[float, float]] = []
        self.last_time = time.perf_counter()

    def update(self, process_time: float, queue_size: int, queue_max_size: int) -> Metrics:
        """Update metrics with current frame data.

        Args:
            process_time: Time to process this frame.
            queue_size: Current queue size.
            queue_max_size: Maximum queue size.

        Returns:
            Current metrics.
        """
        now = time.perf_counter()
        frame_time = now - self.last_time
        self.last_time = now

        self.frame_times.append((now, frame_time))

        cutoff = now - self.window_size
        while self.frame_times and self.frame_times[0][0] < cutoff:
            self.frame_times.pop(0)

        if self.frame_times:
            fps = float(len(self.frame_times))
            avg_frame_time = sum(ft for _, ft in self.frame_times) / len(self.frame_times)
        else:
            fps = 0.0
            avg_frame_time = 0.0

        queue_latency_ms = queue_size * avg_frame_time * 1000

        return Metrics(
            fps=fps,
            avg_frame_time=avg_frame_time,
            process_time=process_time,
            queue_size=queue_size,
            queue_max_size=queue_max_size,
            queue_latency_ms=queue_latency_ms,
        )

    def reset(self) -> None:
        """Reset metrics tracking."""
        self.frame_times.clear()
        self.last_time = time.perf_counter()
