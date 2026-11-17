import csv

import pytest
import torch

from racecar.logging_utils import MetricsLogger, load_checkpoint, save_checkpoint
from racecar.nets import GaussianPolicy


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_csv_has_header_and_one_row_per_call(tmp_path):
    logger = MetricsLogger(tmp_path / "run")
    logger.log(0, {"mean_return": 1.5, "crash_rate": 1.0})
    logger.log(1, {"mean_return": 2.5, "crash_rate": 0.5})
    logger.close()
    rows = read_rows(tmp_path / "run" / "metrics.csv")
    assert len(rows) == 2
    assert list(rows[0]) == ["step", "mean_return", "crash_rate"]
    assert rows[1]["step"] == "1" and float(rows[1]["mean_return"]) == 2.5


def test_rows_are_on_disk_before_close(tmp_path):
    logger = MetricsLogger(tmp_path / "run")
    logger.log(0, {"a": 1.0})
    assert len(read_rows(tmp_path / "run" / "metrics.csv")) == 1  # a crashed run keeps its rows
    logger.close()


def test_columns_are_fixed_by_the_first_call(tmp_path):
    logger = MetricsLogger(tmp_path / "run")
    logger.log(0, {"a": 1.0, "b": 2.0})
    logger.log(1, {"a": 3.0})  # missing metric: blank cell
    with pytest.raises(ValueError, match="c"):
        logger.log(2, {"a": 1.0, "c": 5.0})  # unknown metric
    logger.close()
    rows = read_rows(tmp_path / "run" / "metrics.csv")
    assert len(rows) == 2 and rows[1]["b"] == ""


def test_tensorboard_events_are_written(tmp_path):
    logger = MetricsLogger(tmp_path / "run")
    logger.log(0, {"a": 1.0})
    logger.close()
    assert any((tmp_path / "run" / "tb").glob("events.out.tfevents.*"))


def test_checkpoint_round_trip(tmp_path):
    policy = GaussianPolicy(10)
    path = save_checkpoint(tmp_path, policy, iteration=7, seed=42, config={"lr": 0.003, "algo": "reinforce"})
    assert path.exists()
    data = load_checkpoint(path)
    assert data["iteration"] == 7 and data["seed"] == 42
    assert data["config"] == {"lr": 0.003, "algo": "reinforce"}
    restored = GaussianPolicy(10)
    restored.load_state_dict(data["policy_state"])
    obs = torch.rand(3, 10)
    assert torch.equal(policy(obs).mean, restored(obs).mean)
    assert torch.equal(policy.log_std, restored.log_std)


def test_checkpoints_of_different_iterations_do_not_overwrite(tmp_path):
    policy = GaussianPolicy(10)
    a = save_checkpoint(tmp_path, policy, 1, 0, {})
    b = save_checkpoint(tmp_path, policy, 2, 0, {})
    assert a != b and a.exists() and b.exists()
