"""Policy and value networks (Task 6). Yours to fill in: see the TODOs."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from collections import OrderedDict


class GaussianPolicy(nn.Module):
    """Maps an observation to a Normal distribution over the 2 actions.

    Architecture: obs -> Linear -> Tanh -> Linear -> Tanh -> Linear -> mean (act_dim).
    The standard deviation is a free parameter `log_std` of shape (act_dim,),
    the same for every observation, starting at std = 0.6.
    """

    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        # TODO: build `self.net` (nn.Sequential) from obs_dim, hidden and act_dim.
        layers = []
        for i in range(len(hidden)):
            if i == 0:
                layers.append((f"linear{i+1}", nn.Linear(obs_dim, hidden[i])))
            else:
                layers.append((f"linear{i+1}", nn.Linear(hidden[i-1], hidden[i])))
            layers.append((f"tanh{i+1}", nn.Tanh()))
        layers.append((f"linear{len(hidden)+1}", nn.Linear(hidden[-1], act_dim)))
        self.net = nn.Sequential(
            OrderedDict(layers)
        )
        # TODO: create `self.log_std` as an nn.Parameter of shape (act_dim,) with exp(log_std) = 0.6.
        self.log_std = nn.Parameter(torch.zeros(act_dim) + np.log(0.6))

    def forward(self, obs: torch.Tensor) -> Normal:
        """obs: (batch, obs_dim) -> Normal with mean and std of shape (batch, act_dim)."""
        # Hint: the std must have the same shape as the mean; see Tensor.expand_as.
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        res = Normal(mean, std)
        return res

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """One observation (obs_dim,) -> (sampled action (act_dim,), its log-prob as a float).

        The action is NOT clipped. The log-prob is summed over the action dimensions.
        Called while collecting data, so no gradients are needed.
        """
        res = self.forward(torch.from_numpy(obs).float().unsqueeze(0))
        action = res.sample()
        logp = res.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().numpy(), logp.item()

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        """One observation (obs_dim,) -> the mean action (act_dim,): no sampling, for evaluation."""
        dist = self.forward(torch.from_numpy(obs).float().unsqueeze(0))
        return dist.mean.squeeze(0).detach().numpy()


class ValueNet(nn.Module):
    """Estimates the value V(s) of an observation: obs -> Linear -> Tanh -> ... -> one number."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        # TODO: build `self.net`.
        layers = []
        for i in range(len(hidden)):
            if i == 0:
                layers.append((f"linear{i+1}", nn.Linear(obs_dim, hidden[i])))
            else:
                layers.append((f"linear{i+1}", nn.Linear(hidden[i-1], hidden[i])))
            layers.append((f"tanh{i+1}", nn.Tanh()))
        layers.append((f"linear{len(hidden)+1}", nn.Linear(hidden[-1], 1)))

        self.net = nn.Sequential(
            OrderedDict(layers)
        )
        # raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim) -> values of shape (batch,), not (batch, 1)."""
        return self.net(obs).squeeze(-1)
