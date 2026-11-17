"""REINFORCE (Task 8). Yours to fill in: see the TODOs."""

import numpy as np
import torch

from racecar.nets import GaussianPolicy
from racecar.rollout import Episode


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Return-to-go for every step: G[t] = rewards[t] + gamma * G[t + 1], with G[T] = 0.

    rewards: (T,) -> (T,). Example: rewards [1, 1, 1] and gamma 0.5 give [1.75, 1.5, 1.0].
    """
    # TODO: work backwards from the last step.
    G = np.zeros(len(rewards), dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        if t == len(rewards) - 1:
            G[t] = rewards[t]
        else:
            G[t] = rewards[t] + gamma * G[t + 1]
    return G


class ReinforceLearner:
    """Vanilla policy gradient: loss = -(logp * G).mean() over every step of the batch."""

    def __init__(self, obs_dim: int, cfg) -> None:
        # TODO: create `self.policy` (a GaussianPolicy), keep `self.cfg`, and create
        # `self.optimizer` (torch.optim.Adam over the policy parameters, learning rate cfg.lr).
        self.policy = GaussianPolicy(obs_dim, act_dim=2)
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        return

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """One gradient step on a batch of episodes. Returns metrics, e.g. {"policy_loss": ...}.

        Steps:
        1. Returns-to-go for each episode (`discounted_returns` with cfg.gamma); concatenate
           everything over the batch into flat arrays/tensors of length N = total steps.
        2. Normalise the returns over the whole batch: subtract the mean, divide by the std
           (add a tiny epsilon: the std can be 0). Comment why in the code.
        3. Run the policy WITH gradients on all stored observations and take the log-prob of the
           stored (unclipped) actions, summed over the action dimensions -> shape (N,).
        4. loss = -(logp * returns).mean(), then zero_grad / backward / step.
        All values in the returned dict must be Python floats.
        """
        metrics = {}

        returns_to_go = np.concatenate([discounted_returns(episode.rewards, self.cfg.gamma) for episode in episodes])

        # Adding an epsilon for edge case of zero std
        returns_to_go = (returns_to_go - np.mean(returns_to_go)) / (np.std(returns_to_go) + 1e-8)

        actions = [self.policy.forward(torch.from_numpy(episode.obs).float()).log_prob(torch.from_numpy(episode.actions).float()).sum(dim=-1) for episode in episodes]
        actions = torch.cat(actions)

        loss = -(actions * torch.from_numpy(returns_to_go).float()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics["policy_loss"] = float(loss.item())
        metrics["returns_mean"] = float(np.mean(returns_to_go))
        metrics["returns_std"] = float(np.std(returns_to_go))

        return metrics
