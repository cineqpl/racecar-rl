import numpy as np
import torch

from racecar.nets import GaussianPolicy, ValueNet

OBS_DIM = 10


def test_forward_shapes():
    dist = GaussianPolicy(OBS_DIM)(torch.zeros(4, OBS_DIM))
    assert dist.mean.shape == (4, 2)
    assert dist.stddev.shape == (4, 2)


def test_initial_std_is_about_0_6():
    policy = GaussianPolicy(OBS_DIM)
    std = policy(torch.zeros(1, OBS_DIM)).stddev
    assert torch.allclose(std, torch.full((1, 2), 0.6), atol=1e-4)


def test_log_std_is_a_learned_parameter():
    policy = GaussianPolicy(OBS_DIM)
    assert policy.log_std.shape == (2,)
    assert any(p is policy.log_std for p in policy.parameters())


def test_gradient_reaches_every_parameter():
    policy = GaussianPolicy(OBS_DIM)
    dist = policy(torch.rand(5, OBS_DIM))
    dist.log_prob(dist.sample()).sum().backward()
    assert all(p.grad is not None for p in policy.parameters())


def test_act_returns_action_and_logp():
    action, logp = GaussianPolicy(OBS_DIM).act(np.zeros(OBS_DIM, dtype=np.float32))
    assert isinstance(action, np.ndarray) and action.shape == (2,)
    assert isinstance(logp, float)


def test_act_logp_matches_forward():
    policy = GaussianPolicy(OBS_DIM)
    obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
    action, logp = policy.act(obs)
    expected = policy(torch.as_tensor(obs)[None]).log_prob(torch.as_tensor(action)[None]).sum()
    assert logp == np.float32(expected.item()) or abs(logp - expected.item()) < 1e-4


def test_act_is_random_and_mean_action_is_not():
    policy = GaussianPolicy(OBS_DIM)
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    assert not np.array_equal(policy.act(obs)[0], policy.act(obs)[0])
    np.testing.assert_array_equal(policy.mean_action(obs), policy.mean_action(obs))


def test_mean_action_equals_forward_mean():
    policy = GaussianPolicy(OBS_DIM)
    obs = np.random.default_rng(1).random(OBS_DIM).astype(np.float32)
    mean = policy(torch.as_tensor(obs)[None]).mean[0].detach().numpy()
    result = policy.mean_action(obs)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, mean, atol=1e-6)


def test_logp_of_the_mean_beats_logp_far_from_it():
    policy = GaussianPolicy(OBS_DIM)
    obs = torch.rand(1, OBS_DIM)
    dist = policy(obs)
    assert dist.log_prob(dist.mean).sum() > dist.log_prob(dist.mean + 5).sum()


def test_value_net_shape():
    assert ValueNet(OBS_DIM)(torch.zeros(4, OBS_DIM)).shape == (4,)


def test_value_net_gradient_reaches_every_parameter():
    net = ValueNet(OBS_DIM)
    net(torch.rand(5, OBS_DIM)).sum().backward()
    assert all(p.grad is not None for p in net.parameters())
