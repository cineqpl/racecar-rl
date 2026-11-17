"""Metrics logging (CSV + TensorBoard) and checkpoints."""

import csv
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from racecar.nets import GaussianPolicy


class MetricsLogger:
    """Appends one row per `log` call to `run_dir/<csv_name>` and writes the same
    scalars to TensorBoard in `run_dir/tb`. The columns are fixed by the first call."""

    def __init__(self, run_dir: Path, csv_name: str = "metrics.csv") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.run_dir / csv_name, "w", newline="")
        self._writer: csv.DictWriter | None = None
        self._tb = SummaryWriter(str(self.run_dir / "tb"))

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=["step", *metrics])
            self._writer.writeheader()
        unknown = set(metrics) - set(self._writer.fieldnames)
        if unknown:
            raise ValueError(f"metrics not in the CSV columns fixed by the first call: {sorted(unknown)}")
        self._writer.writerow({"step": step, **metrics})  # missing metrics become blank cells
        self._file.flush()  # a crashed run keeps its rows
        for name, value in metrics.items():
            self._tb.add_scalar(name, value, step)

    def close(self) -> None:
        self._file.close()
        self._tb.close()


def save_checkpoint(run_dir: Path, policy: GaussianPolicy, iteration: int, seed: int, config: dict) -> Path:
    """Write `run_dir/checkpoints/ckpt_<iteration>.pt` and return its path."""
    path = Path(run_dir) / "checkpoints" / f"ckpt_{iteration:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"policy_state": policy.state_dict(), "iteration": iteration, "seed": seed, "config": config},
        path,
    )
    return path


def load_checkpoint(path: Path) -> dict:
    """Keys: policy_state, iteration, seed, config. Load the weights with `policy.load_state_dict`."""
    return torch.load(path, weights_only=True)
