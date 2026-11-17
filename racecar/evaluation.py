"""Evaluation: run the policy's mean action (no sampling noise) from the fixed start."""

from dataclasses import dataclass

import numpy as np

from racecar.env import RaceEnv
from racecar.nets import GaussianPolicy


@dataclass
class EvalResult:
    mean_return: float
    mean_length: float
    crash_rate: float
    finish_rate: float
    lap_fraction: float  # mean over episodes of final progress / track length
    trajectories: list[np.ndarray]  # each (T + 1, 3): x, y, speed, starting with the start state
    crash_progress: list[float]  # final progress of each crashed episode


def evaluate(env: RaceEnv, policy: GaussianPolicy, n_episodes: int = 1) -> EvalResult:
    returns, lengths, laps = [], [], []
    crashed, finished = 0, 0
    trajectories, crash_progress = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        path = [(env.state.x, env.state.y, env.state.speed)]
        total, info = 0.0, {}
        while True:
            obs, reward, terminated, truncated, info = env.step(policy.mean_action(obs))
            total += reward
            path.append((info["x"], info["y"], info["speed"]))
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(info["steps"])
        laps.append(info["progress"] / env.track.length)
        finished += info["finished"]
        if info["crashed"]:
            crashed += 1
            crash_progress.append(info["progress"])
        trajectories.append(np.array(path, dtype=np.float32))
    return EvalResult(
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        crash_rate=crashed / n_episodes,
        finish_rate=finished / n_episodes,
        lap_fraction=float(np.mean(laps)),
        trajectories=trajectories,
        crash_progress=crash_progress,
    )
