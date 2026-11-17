import numpy as np
import pytest
import torch

from racecar.algos import make_learner
from racecar.algos.reinforce import ReinforceLearner, discounted_returns
from racecar.env import RaceEnv
from racecar.evaluation import evaluate
from racecar.nets import GaussianPolicy
from racecar.rollout import Episode, collect_episodes
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image
from racecar.train import TrainConfig, seed_everything, train

OBS_DIM = 10


# --- discounted_returns --------------------------------------------------------


def test_discounted_returns_known_answer():
    np.testing.assert_allclose(discounted_returns(np.array([1.0, 1.0, 1.0]), 0.5), [1.75, 1.5, 1.0])


def test_discounted_returns_without_discount_is_a_reverse_cumsum():
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(discounted_returns(rewards, 1.0), [10.0, 9.0, 7.0, 4.0])


def test_discounted_returns_with_gamma_zero_is_the_rewards():
    rewards = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(discounted_returns(rewards, 0.0), rewards)


def test_discounted_returns_single_step_and_shape():
    out = discounted_returns(np.array([2.5]), 0.9)
    assert out.shape == (1,) and out[0] == pytest.approx(2.5)


# --- registry -------------------------------------------------------------------


def test_unknown_algo_raises():
    with pytest.raises(ValueError, match="nope"):
        make_learner("nope", OBS_DIM, TrainConfig())


def test_registry_builds_reinforce():
    assert isinstance(make_learner("reinforce", OBS_DIM, TrainConfig()), ReinforceLearner)


# --- ReinforceLearner.update ------------------------------------------------------


def make_episode(policy, obs, action, reward):
    """A one-step episode: `action` taken from `obs`, earning `reward`."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
    act_t = torch.as_tensor(action, dtype=torch.float32)[None]
    logp = policy(obs_t).log_prob(act_t).sum(-1).item()
    return Episode(
        obs=obs_t.numpy(),
        actions=act_t.numpy(),
        rewards=np.array([reward], dtype=np.float32),
        logps=np.array([logp], dtype=np.float32),
        terminated=True,
        last_obs=obs_t[0].numpy(),
    )


def logp_of(policy, obs, action):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
    return policy(obs_t).log_prob(torch.as_tensor(action, dtype=torch.float32)[None]).sum().item()


def test_update_returns_floats_and_changes_the_policy():
    learner = ReinforceLearner(OBS_DIM, TrainConfig())
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    episodes = [
        make_episode(learner.policy, obs, [0.5, 0.0], 1.0),
        make_episode(learner.policy, obs, [-0.5, 0.0], -1.0),
    ]
    before = [p.detach().clone() for p in learner.policy.parameters()]
    metrics = learner.update(episodes)
    assert metrics and all(isinstance(v, float) for v in metrics.values())
    assert any(not torch.equal(b, p) for b, p in zip(before, learner.policy.parameters()))


def test_update_makes_the_rewarded_action_more_likely():
    learner = ReinforceLearner(OBS_DIM, TrainConfig(lr=1e-2))
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    good, bad = [0.8, 0.2], [-0.8, -0.2]
    episodes = [
        make_episode(learner.policy, obs, good, 1.0),
        make_episode(learner.policy, obs, bad, -1.0),
    ]
    good_before, bad_before = logp_of(learner.policy, obs, good), logp_of(learner.policy, obs, bad)
    learner.update(episodes)
    assert logp_of(learner.policy, obs, good) > good_before
    assert logp_of(learner.policy, obs, bad) < bad_before


def test_update_with_identical_returns_stays_finite():
    # Normalising returns divides by their std, which is 0 here.
    learner = ReinforceLearner(OBS_DIM, TrainConfig())
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    episodes = [make_episode(learner.policy, obs, [0.1, 0.1], 1.0) for _ in range(3)]
    learner.update(episodes)
    assert all(torch.isfinite(p).all() for p in learner.policy.parameters())


def test_update_on_real_episodes(tmp_path):
    path = tmp_path / "corridor.bmp"
    corridor_image(length=200).save(path)
    env = RaceEnv(Track.load(path, **SMALL_LIMITS), max_steps=100)
    learner = ReinforceLearner(env.obs_dim, TrainConfig())
    episodes = collect_episodes(env, learner.policy, 4)
    metrics = learner.update(episodes)
    assert all(np.isfinite(v) for v in metrics.values())


# --- the whole loop ---------------------------------------------------------------


def test_reinforce_learns_to_drive_the_corridor(tmp_path):
    """Integration: the final bar for Task 8. If it is flaky, tune lr and iterations, not the assertion."""
    path = tmp_path / "corridor.bmp"
    corridor_image(length=200).save(path)
    track = Track.load(path, **SMALL_LIMITS)
    cfg = TrainConfig(iterations=100, episodes_per_iter=16, lr=3e-3, seed=0, eval_every=100, max_steps=150)
    seed_everything(cfg.seed)
    env = RaceEnv(track, max_steps=cfg.max_steps)
    learner = make_learner("reinforce", env.obs_dim, cfg)
    train(env, RaceEnv(track, max_steps=cfg.max_steps), learner, cfg, tmp_path / "run")
    assert evaluate(env, learner.policy).lap_fraction >= 0.5
