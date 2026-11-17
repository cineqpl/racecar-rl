"""The training loop shared by every algorithm."""

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import logging

from racecar.algos.base import Learner
from racecar.env import RaceEnv
from racecar.evaluation import evaluate
from racecar.logging_utils import MetricsLogger, save_checkpoint
from racecar.rollout import collect_episodes

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    algo: str = "reinforce"
    seed: int = 0
    iterations: int = 200
    episodes_per_iter: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-3
    eval_every: int = 10
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 64
    max_steps: int = 1500  # episode step limit, used when the environments are built


def seed_everything(seed: int) -> None:
    """Seed every random number generator we use. Call it BEFORE building the learner,
    so the network's initial weights are reproducible too."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(env: RaceEnv, eval_env: RaceEnv, learner: Learner, cfg: TrainConfig, run_dir: Path) -> None:
    """Collect episodes, update the learner, log, and evaluate every `cfg.eval_every` iterations.

    Writes `metrics.csv` (one row per iteration), `eval.csv` (one row per evaluation, columns
    prefixed `eval_`), TensorBoard events in `tb/`, and checkpoints in `checkpoints/`.
    """
    seed_everything(cfg.seed)  # makes the sampling reproducible from here on
    run_dir = Path(run_dir)
    train_log = MetricsLogger(run_dir)
    eval_log = MetricsLogger(run_dir, csv_name="eval.csv")
    try:
        for iteration in range(1, cfg.iterations + 1):
            episodes = collect_episodes(env, learner.policy, cfg.episodes_per_iter)
            learner_metrics = learner.update(episodes)
            std = torch.exp(learner.policy.log_std.detach())
            metrics = {
                "mean_return": float(np.mean([ep.rewards.sum() for ep in episodes])),
                "mean_length": float(np.mean([len(ep.rewards) for ep in episodes])),
                "policy_std": float(std.mean()),
                "policy_entropy": float(torch.distributions.Normal(0.0, std).entropy().sum()),
                **learner_metrics,
            }

            logger.info(f"Iteration {iteration}: {metrics}, mean return {metrics['mean_return']:.1f}, mean length {metrics['mean_length']:.1f}")
            train_log.log(iteration, metrics)
            if iteration % cfg.eval_every == 0:
                result = evaluate(eval_env, learner.policy)
                eval_log.log(iteration, {
                    "eval_mean_return": result.mean_return,
                    "eval_mean_length": result.mean_length,
                    "eval_crash_rate": result.crash_rate,
                    "eval_finish_rate": result.finish_rate,
                    "eval_lap_fraction": result.lap_fraction,
                })
                save_checkpoint(run_dir, learner.policy, iteration, cfg.seed, asdict(cfg))
    finally:
        train_log.close()
        eval_log.close()
