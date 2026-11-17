"""Collecting episodes with the current policy (Task 6)."""

from dataclasses import dataclass

import numpy as np

from racecar.env import RaceEnv
from racecar.nets import GaussianPolicy


@dataclass
class Episode:
    obs: np.ndarray  # (T, obs_dim): the observation the action at each step was chosen from
    actions: np.ndarray  # (T, act_dim): the sampled actions, NOT clipped
    rewards: np.ndarray  # (T,)
    logps: np.ndarray  # (T,): log-probability of each action under the policy that took it
    terminated: bool  # True if the episode ended by a crash or finish, False if it timed out
    last_obs: np.ndarray  # (obs_dim,): the observation after the final step


def collect_episodes(env: RaceEnv, policy: GaussianPolicy, n_episodes: int) -> list[Episode]:
    """Run `n_episodes` full episodes (until terminated or truncated) with sampled actions."""
    # TODO: for each episode, reset, then loop: policy.act(obs), env.step(action),
    # record obs/action/reward/logp, until terminated or truncated.
    # Note: env.step clips the action itself; store the unclipped one.

    res = []
    action_unclipped = np.zeros((env.max_steps, env.act_dim), dtype=np.float32)
    for i in range(n_episodes):
        obs = env.reset()

        done, terminated = False, False

        obs_list, action_list, reward_list, logp_list = [], [], [], []

        while not (done or terminated):

            action, logp = policy.act(obs)
            next_obs, reward, terminated, done, _ = env.step(action)
            # Store the obs/action/reward/logp for this step
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            logp_list.append(logp)
            obs = next_obs

        obs_list = np.array(obs_list, dtype=np.float32)
        action_list = np.array(action_list, dtype=np.float32)
        reward_list = np.array(reward_list, dtype=np.float32)
        logp_list = np.array(logp_list, dtype=np.float32)

        res.append(Episode(obs_list, action_list, reward_list, logp_list, terminated, next_obs))
    return res
