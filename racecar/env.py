"""The racing environment: a point (or rigid set of points) driving on a Track.

The interface is `reset()` / `step()` by convention only (no RL framework).

Observation: `n_rays` wall distances / `max_ray_len`, ordered left to right as
seen from the car, then speed / `max_speed`. All in [0, 1], float32.
Action: (throttle/brake, steering), each clipped to [-1, 1].
Reward per step: `progress_scale * (progress gained) - time_penalty`.
Termination: a wall hit (crash) or reaching the finish area. Reaching
`max_steps` is truncation, not termination.
"""

import math

import numpy as np

from racecar.car import CarParams, CarState, step_car
from racecar.rays import cast_rays
from racecar.track import Track


class RaceEnv:
    def __init__(
        self,
        track: Track,
        n_rays: int = 9,
        ray_fov_deg: float = 180.0,
        max_ray_len: float = 300.0,
        max_steps: int = 1500,
        dt: float = 0.1,
        progress_scale: float = 0.1,
        time_penalty: float = 0.01,
        footprint=((0.0, 0.0),),
        params: CarParams = CarParams(),
    ) -> None:
        self.track = track
        self.params = params
        self.max_ray_len = max_ray_len
        self.max_steps = max_steps
        self.dt = dt
        self.progress_scale = progress_scale
        self.time_penalty = time_penalty
        self.obs_dim = n_rays + 1
        self.act_dim = 2
        # Negative angles point to the car's left, positive to its right (y is
        # down in the image), so this orders the rays left to right.
        half = math.radians(ray_fov_deg) / 2
        self.ray_angles = np.linspace(-half, half, n_rays)
        # Body points in the car frame (x forward, y toward the car's right, as in
        # the image). Collision uses all of them; the point car is the case of one.
        self.footprint = np.asarray(footprint, dtype=np.float64).reshape(-1, 2)
        self.state: CarState | None = None
        self._progress = 0.0
        self._steps = 0
        self._done = True
        self._started = False

    # --- public interface -----------------------------------------------------

    def reset(self) -> np.ndarray:
        x, y = self.track.start_pos
        self.set_state(x, y, self.track.start_heading, 0.0)
        self._steps = 0
        self._done = False
        self._started = True
        return self._observe()

    def set_state(self, x: float, y: float, heading: float, speed: float) -> None:
        """Place the car (for tests and debugging) and take its position as the progress reference."""
        self.state = CarState(x, y, heading, speed)
        self._progress = self.track.progress_at(x, y)
        if not math.isfinite(self._progress):
            raise ValueError(f"({x}, {y}) is not on the track")

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("the episode has ended; call reset()")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {action.shape}")
        throttle, steer = np.clip(action, -1.0, 1.0)

        old = self.state
        new = step_car(old, float(throttle), float(steer), self.dt, self.params)
        hit = self.track.sweep(self._body_points(old), self._body_points(new))
        steps_before = self._steps
        self._steps += 1

        terminated = crashed = finished = False
        if hit is None:
            self.state = new
        elif hit[0] == "wall":
            terminated = crashed = True
            self.state = new
        else:
            terminated = finished = True
            f = hit[1]  # the car is taken to be where the finish was first touched
            self.state = CarState(
                old.x + f * (new.x - old.x), old.y + f * (new.y - old.y), new.heading, new.speed
            )

        if crashed:
            # Crashing forfeits the time penalty the rest of the episode would have
            # cost, so ending early is never better than idling to the time limit.
            reward = -self.time_penalty * (self.max_steps - steps_before)
        else:
            progress = self.track.progress_at(self.state.x, self.state.y)
            reward = (progress - self._progress) * self.progress_scale - self.time_penalty
            self._progress = progress

        truncated = not terminated and self._steps >= self.max_steps
        self._done = terminated or truncated
        info = {
            "progress": self._progress,
            "crashed": crashed,
            "finished": finished,
            "x": self.state.x,
            "y": self.state.y,
            "speed": self.state.speed,
            "steps": self._steps,
        }
        return self._observe(crashed), float(reward), terminated, truncated, info

    # --- internals --------------------------------------------------------------

    def _body_points(self, state: CarState) -> np.ndarray:
        """World coordinates of the footprint points, (N, 2)."""
        c, s = math.cos(state.heading), math.sin(state.heading)
        fx, fy = self.footprint[:, 0], self.footprint[:, 1]
        return np.stack([state.x + fx * c - fy * s, state.y + fx * s + fy * c], axis=1)

    def _observe(self, crashed: bool = False) -> np.ndarray:
        s = self.state
        if crashed:
            rays = np.zeros(len(self.ray_angles), dtype=np.float32)
        else:
            rays = cast_rays(self.track, s.x, s.y, s.heading, self.ray_angles, self.max_ray_len)
        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[:-1] = rays / self.max_ray_len
        obs[-1] = s.speed / self.params.max_speed
        return np.clip(obs, 0.0, 1.0)
