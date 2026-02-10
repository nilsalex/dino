from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str = "runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_episode(self, episode_idx: int, reward: float, length: int):
        self.writer.add_scalar("episode/reward", reward, episode_idx)
        self.writer.add_scalar("episode/length", length, episode_idx)

    def log_training_metrics(
        self, step: int, loss: float | None = None, q_mean: float | None = None, q_max: float | None = None
    ):
        if loss is not None:
            self.writer.add_scalar("train/loss", loss, step)
        if q_mean is not None:
            self.writer.add_scalar("train/q_mean", q_mean, step)
        if q_max is not None:
            self.writer.add_scalar("train/q_max", q_max, step)

    def log_system_metrics(self, step: int, epsilon: float, fps: float, buffer_size: int):
        self.writer.add_scalar("system/epsilon", epsilon, step)
        self.writer.add_scalar("system/fps", fps, step)
        self.writer.add_scalar("system/buffer_size", buffer_size, step)

    def log_custom_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
