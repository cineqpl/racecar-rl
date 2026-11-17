from typing import Protocol

from racecar.nets import GaussianPolicy
from racecar.rollout import Episode


class Learner(Protocol):
    """What `train()` needs from an algorithm."""

    policy: GaussianPolicy

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """Learn from a batch of episodes; return metrics (floats) to log."""
        ...
