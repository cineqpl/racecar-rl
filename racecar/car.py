"""Kinematic bicycle car with a grip limit.

Units: pixels and seconds. Heading follows the track convention (radians,
0 = +x, positive turns toward +y).

Speed:   dv/dt = a0 * max(u, 0) - b * max(-u, 0) - (a0 / vmax) * v      (u = throttle/brake input)
  Full throttle gives exponential approach to vmax (linear at first, then easing off);
  throttle u settles at u * vmax; letting go coasts down; the brake adds a constant
  deceleration b * |u|. Speed never goes below 0 (no reversing).
Steering: the input s in [-1, 1] is a fraction of the largest steering angle the tyres
  allow at the current speed, min(max_steer, atan(L * a_max / v^2)), which keeps the
  lateral acceleration v^2 * tan(delta) / L at or below a_max.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CarParams:
    wheelbase: float = 14.0  # px
    max_speed: float = 100.0  # px/s
    max_accel: float = 40.0  # px/s^2, throttle acceleration at zero speed
    max_brake: float = 80.0  # px/s^2, braking deceleration at full brake
    max_lateral_accel: float = 50.0  # px/s^2, grip limit
    max_steer: float = 0.5  # rad, steering lock at low speed


@dataclass(frozen=True)
class CarState:
    x: float
    y: float
    heading: float
    speed: float


def step_car(
    state: CarState, throttle: float, steer: float, dt: float, params: CarParams
) -> CarState:
    """Advance the car by `dt` seconds. `throttle` (negative = brake) and `steer` are in [-1, 1].

    Position and heading advance with the speed at the start of the step.
    """
    v = state.speed
    if v > 0.0:
        max_angle = min(params.max_steer, math.atan(params.wheelbase * params.max_lateral_accel / v**2))
    else:
        max_angle = params.max_steer
    angle = steer * max_angle

    accel = (
        params.max_accel * max(throttle, 0.0)
        - params.max_brake * max(-throttle, 0.0)
        - params.max_accel / params.max_speed * v
    )
    return CarState(
        x=state.x + v * math.cos(state.heading) * dt,
        y=state.y + v * math.sin(state.heading) * dt,
        heading=state.heading + v / params.wheelbase * math.tan(angle) * dt,
        speed=max(v + accel * dt, 0.0),
    )
