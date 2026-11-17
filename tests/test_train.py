import csv

import numpy as np
import pytest

from racecar.env import RaceEnv
from racecar.logging_utils import load_checkpoint
from racecar.nets import GaussianPolicy
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image
from racecar.train import TrainConfig, seed_everything, train


class FakeLearner:
    """Does not learn: counts updates and reports one metric."""

    def __init__(self, obs_dim):
        self.policy = GaussianPolicy(obs_dim)
        self.updates = []

    def update(self, episodes):
        self.updates.append(len(episodes))
        return {"fake_metric": 1.0}


@pytest.fixture
def track(tmp_path) -> Track:
    path = tmp_path / "corridor.bmp"
    corridor_image(length=200).save(path)
    return Track.load(path, **SMALL_LIMITS)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_training(track, run_dir, **overrides):
    cfg = TrainConfig(iterations=4, episodes_per_iter=3, eval_every=2, max_steps=60, **overrides)
    seed_everything(cfg.seed)  # before building the learner: its initial weights are random
    env, eval_env = RaceEnv(track, max_steps=60), RaceEnv(track, max_steps=60)
    learner = FakeLearner(env.obs_dim)
    train(env, eval_env, learner, cfg, run_dir)
    return cfg, learner


def test_one_update_per_iteration_with_the_configured_batch(track, tmp_path):
    _, learner = run_training(track, tmp_path / "run")
    assert learner.updates == [3, 3, 3, 3]


def test_training_metrics_are_logged_every_iteration(track, tmp_path):
    run_training(track, tmp_path / "run")
    r = rows(tmp_path / "run" / "metrics.csv")
    assert [row["step"] for row in r] == ["1", "2", "3", "4"]
    for name in ("mean_return", "mean_length", "policy_std", "policy_entropy", "fake_metric"):
        assert name in r[0]


def test_initial_policy_std_and_entropy_are_reported(track, tmp_path):
    run_training(track, tmp_path / "run")
    first = rows(tmp_path / "run" / "metrics.csv")[0]
    assert float(first["policy_std"]) == pytest.approx(0.6, abs=1e-3)
    # Entropy of two independent Normals with std 0.6: 2 * 0.5 * ln(2 pi e 0.36).
    assert float(first["policy_entropy"]) == pytest.approx(np.log(2 * np.pi * np.e * 0.36), abs=1e-3)


def test_evaluation_and_checkpoints_every_eval_every(track, tmp_path):
    cfg, _ = run_training(track, tmp_path / "run", seed=5)
    ev = rows(tmp_path / "run" / "eval.csv")
    assert [row["step"] for row in ev] == ["2", "4"]
    for name in ("eval_mean_return", "eval_crash_rate", "eval_finish_rate", "eval_lap_fraction"):
        assert name in ev[0]
    ckpts = sorted((tmp_path / "run" / "checkpoints").glob("ckpt_*.pt"))
    assert [c.name for c in ckpts] == ["ckpt_000002.pt", "ckpt_000004.pt"]
    data = load_checkpoint(ckpts[-1])
    assert data["iteration"] == 4 and data["seed"] == 5 and data["config"]["episodes_per_iter"] == 3


def test_same_seed_gives_identical_training_metrics(track, tmp_path):
    run_training(track, tmp_path / "a", seed=1)
    run_training(track, tmp_path / "b", seed=1)
    assert rows(tmp_path / "a" / "metrics.csv") == rows(tmp_path / "b" / "metrics.csv")


def test_different_seeds_differ(track, tmp_path):
    run_training(track, tmp_path / "a", seed=1)
    run_training(track, tmp_path / "b", seed=2)
    assert rows(tmp_path / "a" / "metrics.csv") != rows(tmp_path / "b" / "metrics.csv")
