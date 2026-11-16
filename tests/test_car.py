import dataclasses
import math

import pytest

from racecar.car import CarParams, CarState, step_car

DT = 0.1
P = CarParams()  # wheelbase 14, max_speed 100, max_accel 40, max_brake 80, max_lateral_accel 50, max_steer 0.5


def at_speed(v: float) -> CarState:
    return CarState(0.0, 0.0, 0.0, v)


def dv(v: float, throttle: float) -> float:
    """Speed change over one step from speed v."""
    return step_car(at_speed(v), throttle, 0.0, DT, P).speed - v


# --- speed: throttle, drag, brake ------------------------------------------------


def test_throttle_from_rest_accelerates_linearly_at_first():
    assert dv(0.0, 1.0) == pytest.approx(P.max_accel * DT)  # 4.0


def test_acceleration_shrinks_towards_top_speed():
    assert dv(0.0, 1.0) > dv(50.0, 1.0) > dv(90.0, 1.0) > 0


def test_full_throttle_approaches_top_speed_without_exceeding_it():
    state, speeds = at_speed(0.0), []
    for _ in range(600):
        state = step_car(state, 1.0, 0.0, DT, P)
        speeds.append(state.speed)
    assert all(b >= a for a, b in zip(speeds, speeds[1:]))  # monotone
    assert max(speeds) <= P.max_speed
    assert speeds[-1] > 99.9
    assert 60 < speeds[24] < 66  # about 63% of top speed after one time constant (2.5 s)


def test_partial_throttle_settles_at_that_fraction_of_top_speed():
    state = at_speed(0.0)
    for _ in range(600):
        state = step_car(state, 0.5, 0.0, DT, P)
    assert state.speed == pytest.approx(50.0, abs=0.5)


def test_releasing_the_throttle_coasts_down_exponentially():
    assert dv(100.0, 0.0) == pytest.approx(-4.0)  # a = -(40/100) * 100 = -40 px/s^2
    assert dv(50.0, 0.0) == pytest.approx(-2.0)


def test_braking_force_is_linear_in_speed_and_input():
    # Beyond coasting, the brake adds a constant deceleration, whatever the speed.
    extra = {v: dv(v, -1.0) - dv(v, 0.0) for v in (20.0, 50.0, 90.0)}
    assert all(e == pytest.approx(-P.max_brake * DT) for e in extra.values())
    # ... and scales with the pedal.
    assert dv(50.0, -0.5) - dv(50.0, 0.0) == pytest.approx(-0.5 * P.max_brake * DT)


def test_harder_braking_always_decelerates_more():
    assert dv(60.0, -1.0) < dv(60.0, -0.5) < dv(60.0, -0.01) < dv(60.0, 0.0)


def test_speed_never_goes_negative():
    assert step_car(at_speed(1.0), -1.0, 0.0, DT, P).speed == 0.0
    assert step_car(at_speed(0.0), -1.0, 0.0, DT, P).speed == 0.0


# --- position and steering ----------------------------------------------------


def test_moves_along_the_heading_using_the_pre_update_speed():
    s = step_car(CarState(0.0, 0.0, 0.0, 10.0), 0.0, 0.0, DT, P)
    assert s.x == pytest.approx(1.0) and s.y == pytest.approx(0.0)
    assert s.speed == pytest.approx(9.6)
    s = step_car(CarState(5.0, 5.0, math.pi / 2, 10.0), 0.0, 0.0, DT, P)
    assert s.x == pytest.approx(5.0) and s.y == pytest.approx(6.0)  # heading +y


def test_steering_at_low_speed_uses_the_full_angle():
    expected = 10 / P.wheelbase * math.tan(P.max_steer) * DT
    assert step_car(at_speed(10.0), 0.0, 1.0, DT, P).heading == pytest.approx(expected)
    assert step_car(at_speed(10.0), 0.0, -1.0, DT, P).heading == pytest.approx(-expected)
    half = 10 / P.wheelbase * math.tan(0.5 * P.max_steer) * DT
    assert step_car(at_speed(10.0), 0.0, 0.5, DT, P).heading == pytest.approx(half)


def test_grip_limits_the_turn_at_top_speed():
    # At 100 px/s, full lock may only produce a_max = 50 px/s^2: heading rate a_max / v = 0.5 rad/s.
    s = step_car(at_speed(100.0), 0.0, 1.0, DT, P)
    assert s.heading == pytest.approx(P.max_lateral_accel / 100.0 * DT)
    # Steering is a fraction of the available angle, so half input gives about half the turn.
    half = step_car(at_speed(100.0), 0.0, 0.5, DT, P)
    assert half.heading == pytest.approx(s.heading / 2, rel=0.01)


@pytest.mark.parametrize("speed", [5.0, 20.0, 40.0, 60.0, 80.0, 100.0])
def test_lateral_acceleration_never_exceeds_the_grip_limit(speed):
    s = step_car(at_speed(speed), 0.0, 1.0, DT, P)
    lateral = speed * s.heading / DT  # v * (turn rate)
    assert lateral <= P.max_lateral_accel + 1e-9


def test_state_is_immutable():
    state = at_speed(10.0)
    step_car(state, 1.0, 1.0, DT, P)
    assert state == at_speed(10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.x = 1.0
