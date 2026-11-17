import math

import numpy as np
import pytest
import torch

from racecar.env import RaceEnv
from racecar.evaluation import evaluate
from racecar.logging_utils import save_checkpoint
from racecar.nets import GaussianPolicy
from racecar.rays import cast_rays
from racecar.replay import Recording, car_polygon, load_policy, record_episode
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image


@pytest.fixture
def track(tmp_path) -> Track:
    path = tmp_path / "corridor.bmp"
    corridor_image(length=400).save(path)
    return Track.load(path, **SMALL_LIMITS)


def constant_policy(mean_throttle: float, mean_steer: float, obs_dim: int = 10) -> GaussianPolicy:
    policy = GaussianPolicy(obs_dim)
    with torch.no_grad():
        policy.net[-1].weight.zero_()
        policy.net[-1].bias.copy_(torch.tensor([mean_throttle, mean_steer]))
    return policy


def driving_policy() -> GaussianPolicy:
    return constant_policy(1.0, 0.0)


def crashing_policy() -> GaussianPolicy:
    return constant_policy(1.0, 1.0)


def test_recording_is_deterministic(track):
    env = RaceEnv(track)
    a = record_episode(env, driving_policy())
    b = record_episode(env, driving_policy())
    for name in ("x", "y", "heading", "speed", "progress", "rays", "actions", "rewards"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_frame_counts(track):
    rec = record_episode(RaceEnv(track), driving_policy())
    assert rec.n_frames == len(rec.rewards) + 1 == len(rec.actions) + 1
    for name in ("x", "y", "heading", "speed", "progress", "rays"):
        assert len(getattr(rec, name)) == rec.n_frames


def test_first_frame_is_the_start_state(track):
    rec = record_episode(RaceEnv(track), driving_policy())
    assert (rec.x[0], rec.y[0], rec.heading[0], rec.speed[0]) == (*track.start_pos, track.start_heading, 0.0)
    assert rec.progress[0] == 0.0


def test_return_matches_evaluate(track):
    env = RaceEnv(track, max_steps=200)
    policy = GaussianPolicy(env.obs_dim)
    assert record_episode(env, policy).rewards.sum() == pytest.approx(evaluate(env, policy).mean_return, rel=1e-5)


def test_driving_policy_finishes(track):
    rec = record_episode(RaceEnv(track), driving_policy())
    assert rec.finished and not rec.crashed


def test_crashing_policy_crashes_with_zero_rays_on_the_last_frame(track):
    rec = record_episode(RaceEnv(track), crashing_policy())
    assert rec.crashed and not rec.finished
    assert np.all(rec.rays[-1] == 0.0)
    assert np.any(rec.rays[0] != 0.0)


def test_rays_are_in_px(track):
    env = RaceEnv(track)
    rec = record_episode(env, driving_policy())
    expected = cast_rays(track, rec.x[0], rec.y[0], rec.heading[0], env.ray_angles, env.max_ray_len)
    np.testing.assert_allclose(rec.rays[0], expected, rtol=1e-5)
    assert rec.rays.max() <= env.max_ray_len


def test_ray_ends_straight_ahead_and_to_the_left(track):
    rec = record_episode(RaceEnv(track), driving_policy())
    ends = rec.ray_ends(0)
    assert ends.shape == (rec.rays.shape[1], 2)
    mid = rec.rays.shape[1] // 2
    assert ends[mid] == pytest.approx((rec.x[0] + rec.rays[0, mid], rec.y[0]), abs=0.1)  # heading is ~0
    left = ends[0]  # leftmost ray: angle -90 deg, which points toward -y when heading 0
    assert left[1] < rec.y[0] and left[0] == pytest.approx(rec.x[0], abs=0.1)


def test_car_polygon_is_centred_and_rotates():
    corners = np.array(car_polygon(100.0, 50.0, 0.0))
    assert corners.shape == (4, 2)
    np.testing.assert_allclose(corners.mean(axis=0), (100.0, 50.0), atol=1e-9)
    assert np.ptp(corners[:, 0]) == pytest.approx(16.0) and np.ptp(corners[:, 1]) == pytest.approx(8.0)
    turned = np.array(car_polygon(100.0, 50.0, math.pi / 2))
    assert np.ptp(turned[:, 0]) == pytest.approx(8.0) and np.ptp(turned[:, 1]) == pytest.approx(16.0)


def test_load_policy_round_trip_and_latest(tmp_path):
    policy = GaussianPolicy(10)
    save_checkpoint(tmp_path, policy, iteration=3, seed=1, config={"max_steps": 99})
    other = GaussianPolicy(10)
    save_checkpoint(tmp_path, other, iteration=10, seed=1, config={"max_steps": 77})
    obs = torch.rand(2, 10)
    latest, config = load_policy(tmp_path, 10)
    assert config["max_steps"] == 77
    assert torch.equal(latest(obs).mean, other(obs).mean)
    chosen, config = load_policy(tmp_path, 10, checkpoint=3)
    assert config["max_steps"] == 99
    assert torch.equal(chosen(obs).mean, policy(obs).mean)


def test_load_policy_errors_name_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path).replace("\\", "\\\\")):
        load_policy(tmp_path, 10)
    save_checkpoint(tmp_path, GaussianPolicy(10), iteration=3, seed=1, config={})
    with pytest.raises(FileNotFoundError, match=str(tmp_path).replace("\\", "\\\\")):
        load_policy(tmp_path, 10, checkpoint=7)
