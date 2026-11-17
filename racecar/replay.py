"""Replay a trained policy: record one deterministic episode, then watch it.

    uv run python -m racecar.replay runs/reinforce-0 [--track PATH] [--checkpoint N] [--scale S]

`record_episode` runs the real `RaceEnv`, so what is drawn is what the environment did.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from racecar.env import RaceEnv
from racecar.logging_utils import load_checkpoint
from racecar.nets import GaussianPolicy


@dataclass
class Recording:
    """One episode, frame by frame. Frame 0 is the start state; frame i > 0 is the state after step i."""

    x: np.ndarray  # (F,)
    y: np.ndarray  # (F,)
    heading: np.ndarray  # (F,) radians
    speed: np.ndarray  # (F,)
    progress: np.ndarray  # (F,) cumulative progress in px
    rays: np.ndarray  # (F, n_rays) ray lengths in px, all 0 on a crash frame
    ray_angles: np.ndarray  # (n_rays,) angle of each ray relative to the heading, left to right
    actions: np.ndarray  # (F - 1,2) as the environment received them (clipped)
    rewards: np.ndarray  # (F - 1,)
    crashed: bool
    finished: bool

    @property
    def n_frames(self) -> int:
        return len(self.x)

    def ray_ends(self, i: int) -> np.ndarray:
        """Where each ray of frame `i` ends, as an (n_rays, 2) array of (x, y)."""
        direction = self.heading[i] + self.ray_angles
        return np.stack(
            [self.x[i] + self.rays[i] * np.cos(direction), self.y[i] + self.rays[i] * np.sin(direction)],
            axis=1,
        )


def record_episode(env: RaceEnv, policy: GaussianPolicy) -> Recording:
    """Run one episode with the policy's mean action (no sampling noise) and record every frame."""
    obs = env.reset()
    xs, ys, headings, speeds, progress = [env.state.x], [env.state.y], [env.state.heading], [env.state.speed], [0.0]
    rays, actions, rewards = [obs[:-1] * env.max_ray_len], [], []
    info: dict = {}
    while True:
        action = np.clip(policy.mean_action(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        xs.append(env.state.x), ys.append(env.state.y)
        headings.append(env.state.heading), speeds.append(env.state.speed)
        progress.append(info["progress"])
        rays.append(obs[:-1] * env.max_ray_len)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return Recording(
        x=np.array(xs), y=np.array(ys), heading=np.array(headings), speed=np.array(speeds),
        progress=np.array(progress), rays=np.array(rays), ray_angles=np.array(env.ray_angles),
        actions=np.array(actions), rewards=np.array(rewards),
        crashed=bool(info["crashed"]), finished=bool(info["finished"]),
    )


def car_polygon(
    x: float, y: float, heading: float, length: float = 16.0, width: float = 8.0
) -> list[tuple[float, float]]:
    """Corners of the car rectangle centred on (x, y), its long side along the heading. Display only."""
    c, s = math.cos(heading), math.sin(heading)
    corners = [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2)]
    return [(x + fx * c - fy * s, y + fx * s + fy * c) for fx, fy in corners]


def load_policy(run_dir: Path, obs_dim: int, checkpoint: int | None = None) -> tuple[GaussianPolicy, dict]:
    """The policy and saved config from `run_dir/checkpoints`: the latest one, or iteration `checkpoint`."""
    run_dir = Path(run_dir)
    folder = run_dir / "checkpoints"
    if checkpoint is None:
        found = sorted(folder.glob("ckpt_*.pt")) if folder.is_dir() else []
        if not found:
            raise FileNotFoundError(f"no checkpoints found in {folder}")
        path = found[-1]
    else:
        path = folder / f"ckpt_{checkpoint:06d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {checkpoint} not found in {folder}")
    data = load_checkpoint(path)
    policy = GaussianPolicy(obs_dim)
    policy.load_state_dict(data["policy_state"])
    return policy, data["config"]
