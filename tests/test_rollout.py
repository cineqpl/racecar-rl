import numpy as np
import pytest
import torch

from racecar.env import RaceEnv
from racecar.nets import GaussianPolicy
from racecar.rollout import collect_episodes
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image


@pytest.fixture
def track(tmp_path) -> Track:
    path = tmp_path / "corridor.bmp"
    corridor_image(length=400).save(path)
    return Track.load(path, **SMALL_LIMITS)


def test_episode_arrays_line_up(track):
    env = RaceEnv(track)
    policy = GaussianPolicy(env.obs_dim)
    episodes = collect_episodes(env, policy, 3)
    assert len(episodes) == 3
    for ep in episodes:
        t = len(ep.rewards)
        assert t > 0
        assert ep.obs.shape == (t, env.obs_dim)
        assert ep.actions.shape == (t, 2)
        assert ep.logps.shape == (t,)
        assert isinstance(ep.terminated, bool)
        assert ep.last_obs.shape == (env.obs_dim,)


def test_timeout_is_not_terminated(track):
    env = RaceEnv(track, max_steps=5)
    (ep,) = collect_episodes(env, GaussianPolicy(env.obs_dim), 1)
    assert len(ep.rewards) == 5
    assert ep.terminated is False


def test_crash_is_terminated(track):
    env = RaceEnv(track)
    policy = GaussianPolicy(env.obs_dim)
    with torch.no_grad():  # steer hard right every step, so the car hits the wall
        policy.net[-1].bias.copy_(torch.tensor([1.0, 1.0]))
        policy.net[-1].weight.zero_()
    (ep,) = collect_episodes(env, policy, 1)
    assert ep.terminated is True
    assert len(ep.rewards) < env.max_steps


def test_stored_logps_match_the_policy(track):
    env = RaceEnv(track, max_steps=20)
    policy = GaussianPolicy(env.obs_dim)
    (ep,) = collect_episodes(env, policy, 1)
    dist = policy(torch.as_tensor(ep.obs))
    recomputed = dist.log_prob(torch.as_tensor(ep.actions)).sum(-1).detach().numpy()
    np.testing.assert_allclose(ep.logps, recomputed, atol=1e-4)


def test_stored_actions_are_not_clipped(track):
    env = RaceEnv(track, max_steps=50)
    policy = GaussianPolicy(env.obs_dim)
    with torch.no_grad():
        policy.log_std.fill_(1.0)  # std = 2.7, so many samples fall outside [-1, 1]
    (ep,) = collect_episodes(env, policy, 1)
    assert np.abs(ep.actions).max() > 1.0


def test_collection_does_not_leave_gradients(track):
    env = RaceEnv(track, max_steps=5)
    policy = GaussianPolicy(env.obs_dim)
    (ep,) = collect_episodes(env, policy, 1)
    assert all(p.grad is None for p in policy.parameters())
    assert not torch.as_tensor(ep.obs).requires_grad
