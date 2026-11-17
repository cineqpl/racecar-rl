"""Algorithm registry: `make_learner("reinforce", obs_dim, cfg)`."""

from racecar.algos.base import Learner
from racecar.algos.baseline import BaselineLearner
from racecar.algos.reinforce import ReinforceLearner


def make_learner(name: str, obs_dim: int, cfg) -> Learner:
    learners = {"reinforce": ReinforceLearner, "baseline": BaselineLearner}
    if name not in learners:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(learners)}")
    return learners[name](obs_dim, cfg)
