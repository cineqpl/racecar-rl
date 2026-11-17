"""Policy gradient with a learned value baseline (Task 9). Yours to fill in: see the TODOs."""

import numpy as np
import torch

from racecar.algos.reinforce import discounted_returns  # noqa: F401  (you will need it)
from racecar.nets import GaussianPolicy, ValueNet
from racecar.rollout import Episode


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    """How much of the variance of `target` the prediction explains: 1 - Var(target - pred) / Var(target).

    1.0 is a perfect prediction, 0.0 is no better than predicting the mean, negative is worse.
    If the target does not vary at all (Var = 0), return 0.0 instead of dividing by zero.
    """
    return 1.0 - np.var(target - pred) / np.var(target) if np.var(target) > 0 else 0.0


class BaselineLearner:
    """Policy gradient with advantages A = G - V(obs), and a value network trained on the returns."""

    def __init__(self, obs_dim: int, cfg) -> None:
        # TODO: create `self.policy` (GaussianPolicy), `self.value_net` (ValueNet), keep `self.cfg`,
        # and the optimiser(s): Adam with learning rate cfg.lr. Whether you use one optimiser over
        # both networks' parameters or one each is your call; the tests accept either.
        self.policy = GaussianPolicy(obs_dim, act_dim=2)
        self.value_net = ValueNet(obs_dim)
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value_net.parameters()), lr=cfg.lr)

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """One gradient step for the policy and one for the value net on a batch of episodes.

        Steps:
        1. Returns-to-go G for every step of the batch (as in REINFORCE), flat, length N.
           These stay RAW (not normalised): they are the value net's regression target.
        2. Values V = value_net(all stored observations), shape (N,), with gradients.
        3. Advantages A = G - V.detach(). The detach stops the policy loss from reaching the value
           net. Normalise A over the batch (subtract the mean, divide by the std plus a tiny epsilon).
        4. policy_loss = -(logp * A).mean(), with logp from the policy on the stored obs and actions.
        5. value_loss = mean((V - G) ** 2).
        6. zero_grad / backward / step for the optimiser(s).
        Returns floats: at least "value_loss" and "explained_variance" (V against G, as measured
        in step 2, before this update's step), plus "policy_loss".
        """
        metrics = {}
        returns_to_go = np.concatenate([discounted_returns(episode.rewards, self.cfg.gamma) for episode in episodes])

        V = self.value_net(torch.from_numpy(np.concatenate([episode.obs for episode in episodes])).float()).squeeze(-1)
        A = returns_to_go - V.detach().numpy()

        A = (A - np.mean(A)) / (np.std(A) + 1e-8)

        logp = self.policy.forward(torch.from_numpy(np.concatenate([episode.obs for episode in episodes])).float()).log_prob(torch.from_numpy(np.concatenate([episode.actions for episode in episodes])).float()).sum(dim=-1)
        policy_loss = -(logp * torch.from_numpy(A).float()).mean()

        value_loss = torch.mean((V - torch.from_numpy(returns_to_go).float()) ** 2)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.optimizer.step()

        metrics["policy_loss"] = float(policy_loss.item())
        metrics["value_loss"] = float(value_loss.item())
        metrics["explained_variance"] = float(explained_variance(V.detach().numpy(), returns_to_go))
        
        return metrics
