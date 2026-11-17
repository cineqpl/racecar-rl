import copy

import numpy as np
import pytest
import torch

from racecar.algos import make_learner
from racecar.algos.baseline import BaselineLearner, explained_variance
from racecar.algos.reinforce import discounted_returns
from racecar.env import RaceEnv
from racecar.evaluation import evaluate
from racecar.rollout import Episode, collect_episodes
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image
from racecar.train import TrainConfig, seed_everything, train

OBS_DIM = 10


# --- explained_variance ---------------------------------------------------------


def test_perfect_prediction_explains_everything():
    x = np.array([1.0, 3.0, -2.0, 5.0])
    assert explained_variance(x, x) == pytest.approx(1.0)


def test_predicting_the_mean_explains_nothing():
    x = np.array([1.0, 3.0, -2.0, 5.0])
    assert explained_variance(np.full(4, x.mean()), x) == pytest.approx(0.0, abs=1e-6)


def test_a_bad_prediction_is_negative():
    x = np.array([1.0, 3.0, -2.0, 5.0])
    assert explained_variance(-x, x) < 0.0


def test_constant_target_does_not_divide_by_zero():
    result = explained_variance(np.array([1.0, 2.0, 3.0]), np.full(3, 4.0))
    assert result == 0.0 and isinstance(result, float)


# --- registry ---------------------------------------------------------------------


def test_registry_builds_baseline():
    assert isinstance(make_learner("baseline", OBS_DIM, TrainConfig()), BaselineLearner)


# --- BaselineLearner.update ---------------------------------------------------------


def make_episode(policy, obs, action, reward):
    """A one-step episode: `action` taken from `obs`, earning `reward`."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
    act_t = torch.as_tensor(action, dtype=torch.float32)[None]
    logp = policy(obs_t).log_prob(act_t).sum(-1).item()
    return Episode(
        obs=obs_t.numpy(), actions=act_t.numpy(), rewards=np.array([reward], dtype=np.float32),
        logps=np.array([logp], dtype=np.float32), terminated=True, last_obs=obs_t[0].numpy(),
    )


def logp_of(policy, obs, action):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
    return policy(obs_t).log_prob(torch.as_tensor(action, dtype=torch.float32)[None]).sum().item()


@pytest.fixture
def corridor_env(tmp_path) -> RaceEnv:
    path = tmp_path / "corridor.bmp"
    corridor_image(length=200).save(path)
    return RaceEnv(Track.load(path, **SMALL_LIMITS), max_steps=100)


def test_update_returns_finite_floats_with_the_expected_metrics(corridor_env):
    learner = BaselineLearner(corridor_env.obs_dim, TrainConfig())
    metrics = learner.update(collect_episodes(corridor_env, learner.policy, 4))
    assert "value_loss" in metrics and "explained_variance" in metrics
    assert all(isinstance(v, float) and np.isfinite(v) for v in metrics.values())


def test_update_changes_both_networks(corridor_env):
    learner = BaselineLearner(corridor_env.obs_dim, TrainConfig())
    policy_before = [p.detach().clone() for p in learner.policy.parameters()]
    value_before = [p.detach().clone() for p in learner.value_net.parameters()]
    learner.update(collect_episodes(corridor_env, learner.policy, 4))
    assert any(not torch.equal(b, p) for b, p in zip(policy_before, learner.policy.parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(value_before, learner.value_net.parameters()))


def test_update_makes_the_better_than_expected_action_more_likely():
    learner = BaselineLearner(OBS_DIM, TrainConfig(lr=1e-2))
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    good, bad = [0.8, 0.2], [-0.8, -0.2]
    episodes = [make_episode(learner.policy, obs, good, 1.0), make_episode(learner.policy, obs, bad, -1.0)]
    good_before, bad_before = logp_of(learner.policy, obs, good), logp_of(learner.policy, obs, bad)
    learner.update(episodes)
    assert logp_of(learner.policy, obs, good) > good_before
    assert logp_of(learner.policy, obs, bad) < bad_before


def test_identical_advantages_stay_finite():
    # Every advantage can be equal (std 0 when normalised); no NaNs may reach the weights.
    learner = BaselineLearner(OBS_DIM, TrainConfig())
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    learner.update([make_episode(learner.policy, obs, [0.1, 0.1], 1.0) for _ in range(3)])
    for net in (learner.policy, learner.value_net):
        assert all(torch.isfinite(p).all() for p in net.parameters())


def test_repeated_updates_on_one_batch_fit_the_values(corridor_env):
    learner = BaselineLearner(corridor_env.obs_dim, TrainConfig())
    episodes = collect_episodes(corridor_env, learner.policy, 4)
    history = [learner.update(episodes) for _ in range(40)]
    assert history[-1]["value_loss"] < history[0]["value_loss"]
    assert history[-1]["explained_variance"] > history[0]["explained_variance"]


def test_value_net_is_trained_only_by_its_own_loss(corridor_env):
    """One update must move the value net exactly as plain regression on the returns would
    (Adam, learning rate cfg.lr). If the policy loss leaked into it (no detach), it would differ."""
    cfg = TrainConfig()
    learner = BaselineLearner(corridor_env.obs_dim, cfg)
    episodes = collect_episodes(corridor_env, learner.policy, 3)

    reference = copy.deepcopy(learner.value_net)
    optimizer = torch.optim.Adam(reference.parameters(), lr=cfg.lr)
    obs = torch.as_tensor(np.concatenate([e.obs for e in episodes]))
    returns = torch.as_tensor(
        np.concatenate([discounted_returns(e.rewards, cfg.gamma) for e in episodes]), dtype=torch.float32)
    optimizer.zero_grad()
    ((reference(obs) - returns) ** 2).mean().backward()
    optimizer.step()

    learner.update(episodes)
    for expected, actual in zip(reference.parameters(), learner.value_net.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


# --- the whole loop -----------------------------------------------------------------


def test_baseline_learns_to_drive_the_corridor(tmp_path):
    """Integration: if flaky, tune lr and iterations, not the assertion."""
    path = tmp_path / "corridor.bmp"
    corridor_image(length=200).save(path)
    track = Track.load(path, **SMALL_LIMITS)
    cfg = TrainConfig(algo="baseline", iterations=100, episodes_per_iter=16, lr=3e-3, seed=0,
                      eval_every=100, max_steps=150)
    seed_everything(cfg.seed)
    env = RaceEnv(track, max_steps=cfg.max_steps)
    learner = make_learner("baseline", env.obs_dim, cfg)
    train(env, RaceEnv(track, max_steps=cfg.max_steps), learner, cfg, tmp_path / "run")
    assert evaluate(env, learner.policy).lap_fraction >= 0.5
