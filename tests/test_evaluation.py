import numpy as np
import pytest
import torch

from racecar.env import RaceEnv
from racecar.evaluation import evaluate
from racecar.nets import GaussianPolicy
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image


@pytest.fixture
def track(tmp_path) -> Track:
    path = tmp_path / "corridor.bmp"
    corridor_image(length=400).save(path)
    return Track.load(path, **SMALL_LIMITS)


def driving_policy(obs_dim: int) -> GaussianPolicy:
    """A policy whose mean is full throttle, no steering, whatever it sees."""
    policy = GaussianPolicy(obs_dim)
    with torch.no_grad():
        policy.net[-1].weight.zero_()
        policy.net[-1].bias.copy_(torch.tensor([1.0, 0.0]))
    return policy


def test_evaluation_is_deterministic(track):
    env = RaceEnv(track, max_steps=200)
    policy = GaussianPolicy(env.obs_dim)
    a = evaluate(env, policy, n_episodes=2)
    b = evaluate(env, policy, n_episodes=2)
    assert a.mean_return == b.mean_return
    assert a.mean_length == b.mean_length


def test_random_policy_result_is_well_formed(track):
    env = RaceEnv(track, max_steps=200)
    result = evaluate(env, GaussianPolicy(env.obs_dim), n_episodes=3)
    assert 0.0 <= result.crash_rate <= 1.0
    assert 0.0 <= result.finish_rate <= 1.0
    assert len(result.trajectories) == 3
    assert all(t.ndim == 2 and t.shape[1] == 3 for t in result.trajectories)
    assert len(result.crash_progress) == round(result.crash_rate * 3)


def test_driving_straight_finishes_the_corridor(track):
    env = RaceEnv(track)
    result = evaluate(env, driving_policy(env.obs_dim))
    assert result.finish_rate == 1.0 and result.crash_rate == 0.0
    assert result.lap_fraction > 0.85  # the finish area is the last 15% of this track
    assert result.crash_progress == []
    traj = result.trajectories[0]
    assert traj[0, 2] == 0.0 and traj[-1, 2] > 50  # speed column: from rest to fast
    assert traj[-1, 0] > traj[0, 0]  # x column: moved along the corridor


def test_crashing_policy_is_counted(track):
    env = RaceEnv(track)
    policy = GaussianPolicy(env.obs_dim)
    with torch.no_grad():
        policy.net[-1].weight.zero_()
        policy.net[-1].bias.copy_(torch.tensor([1.0, 1.0]))  # throttle and steer hard
    result = evaluate(env, policy, n_episodes=2)
    assert result.crash_rate == 1.0 and result.finish_rate == 0.0
    assert len(result.crash_progress) == 2
    assert result.mean_length == pytest.approx(np.mean([len(t) - 1 for t in result.trajectories]))
