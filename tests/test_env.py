"""RaceEnv on a 400 px corridor: track x in [25, 425), y in [10, 30); start patch
x in [20, 25); the finish is the last 60 px. Known answers throughout."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageDraw

from racecar.env import RaceEnv
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, WHITE, corridor_image

RACETRACK1 = Path(__file__).parent / "data" / "racetrack1.bmp"


def make_track(tmp_path, length=400, slab=False) -> Track:
    img = corridor_image(length=length)
    if slab:
        # 3 px wall across most of the corridor (rows 10-27), two rows left open.
        ImageDraw.Draw(img).rectangle([200, 10, 202, 27], fill=WHITE)
    path = tmp_path / "track.bmp"
    img.save(path)
    return Track.load(path, **SMALL_LIMITS)


@pytest.fixture
def env(tmp_path) -> RaceEnv:
    return RaceEnv(make_track(tmp_path))


def run(env, action, n):
    """Step `n` times (stopping at the end of the episode); return rewards and the last step's output."""
    rewards, out = [], None
    for _ in range(n):
        out = env.step(np.array(action, dtype=float))
        rewards.append(out[1])
        if out[2] or out[3]:
            break
    return rewards, out


# --- observation and reset ---------------------------------------------------


def test_dimensions(env):
    assert env.obs_dim == 10 and env.act_dim == 2


def test_reset_observation(env):
    obs = env.reset()
    assert obs.shape == (10,) and obs.dtype == np.float32
    assert (obs >= 0).all() and (obs <= 1).all()
    assert obs[-1] == 0.0  # speed


def test_reset_places_car_at_the_start(env):
    env.reset()
    x, y = env.track.start_pos
    assert (env.state.x, env.state.y, env.state.heading, env.state.speed) == (
        x, y, env.track.start_heading, 0.0)


def test_ray_order_left_to_right(env):
    env.reset()
    env.set_state(200, 20, math.pi / 2, 0.0)  # facing down the image
    # Left of the car is +x (225 px to the wall), ahead is 10 px, right is -x (180 px).
    obs = env._observe()
    assert obs[0] == pytest.approx(0.75, abs=0.01)
    assert obs[4] == pytest.approx(10 / 300, abs=0.01)
    assert obs[8] == pytest.approx(0.6, abs=0.01)


def test_speed_component_is_normalised(env):
    env.reset()
    env.set_state(200, 20, 0.0, 50.0)
    assert env._observe()[-1] == pytest.approx(0.5)


# --- actions -----------------------------------------------------------------


def test_action_is_clipped(tmp_path):
    a, b = RaceEnv(make_track(tmp_path)), RaceEnv(make_track(tmp_path))
    a.reset(), b.reset()
    oa = a.step(np.array([5.0, 0.0]))[0]
    ob = b.step(np.array([1.0, 0.0]))[0]
    np.testing.assert_array_equal(oa, ob)
    a.reset(), b.reset()
    a.set_state(100, 20, 0.0, 30.0), b.set_state(100, 20, 0.0, 30.0)
    np.testing.assert_array_equal(a.step(np.array([0.0, -9.0]))[0], b.step(np.array([0.0, -1.0]))[0])


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 1)), np.float64(0.5)])
def test_wrong_action_shape_raises(env, bad):
    env.reset()
    with pytest.raises(ValueError):
        env.step(bad)


def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))


def test_step_after_episode_ended_raises(tmp_path):
    env = RaceEnv(make_track(tmp_path), max_steps=2)
    env.reset()
    run(env, [0, 0], 2)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))


# --- reward and termination ---------------------------------------------------


def test_reward_is_progress_difference_minus_time_penalty(env):
    env.reset()
    rewards, (_, _, terminated, truncated, info) = run(env, [1, 0], 30)
    assert not (terminated or truncated)
    steps = len(rewards)
    total = (sum(rewards) + steps * env.time_penalty) / env.progress_scale
    assert total == pytest.approx(info["progress"])
    assert abs(info["progress"] - (env.state.x - 24)) < 1.5  # corridor: progress is x - 24
    assert info["steps"] == steps


def test_time_penalty_when_idle(tmp_path):
    env = RaceEnv(make_track(tmp_path), max_steps=5)
    env.reset()
    rewards, _ = run(env, [0, 0], 5)
    assert sum(rewards) == pytest.approx(-5 * env.time_penalty)


def test_truncation_at_max_steps(tmp_path):
    env = RaceEnv(make_track(tmp_path), max_steps=5)
    env.reset()
    rewards, (_, _, terminated, truncated, _) = run(env, [-1, 0], 10)
    assert len(rewards) == 5 and truncated and not terminated


def test_crash(env):
    env.reset()
    rewards, (obs, reward, terminated, truncated, info) = run(env, [1, 1], 200)
    assert terminated and not truncated
    assert info["crashed"] and not info["finished"]
    assert np.all(obs[:-1] == 0.0)


def test_crash_is_never_better_than_idling(tmp_path):
    track = make_track(tmp_path)
    idle = RaceEnv(track, max_steps=50)
    idle.reset()
    idle_total = sum(run(idle, [0, 0], 50)[0])
    crash = RaceEnv(track, max_steps=50)
    crash.reset()
    crash.set_state(30, 29.9, math.pi / 2, 100.0)  # one step from the side wall
    rewards, (_, _, terminated, _, info) = run(crash, [0, 0], 50)
    assert terminated and info["crashed"] and len(rewards) == 1
    assert sum(rewards) <= idle_total + 1e-9


def test_finish(env):
    env.reset()
    _, (_, reward, terminated, _, info) = run(env, [1, 0], 400)
    assert terminated and info["finished"] and not info["crashed"]
    assert info["progress"] >= 340
    assert reward > 0


def test_backward_driving_is_negative(env):
    env.reset()
    env.set_state(200, 20, math.pi, 20.0)
    reward = env.step(np.array([0.0, 0.0]))[1]
    assert reward < 0


def test_no_tunnelling_through_a_thin_wall(tmp_path):
    env = RaceEnv(make_track(tmp_path, slab=True))
    env.reset()
    env.set_state(195, 15, 0.0, 100.0)  # a 10 px step crosses the slab at x=200-202
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0]))
    assert terminated and info["crashed"]


def test_footprint_corner_hits_a_wall_the_centre_misses(tmp_path):
    track = make_track(tmp_path)
    # Facing +x at y=28: the centre is 2 px from the wall at y=30; a corner 3 px to the side is not.
    point = RaceEnv(track)
    wide = RaceEnv(track, footprint=[[0, 0], [0, 3]])
    for e in (point, wide):
        e.reset()
        e.set_state(100, 28, 0.0, 10.0)
    assert not point.step(np.zeros(2))[2]
    assert wide.step(np.zeros(2))[2]


def test_real_track_finish_is_reached_not_skipped():
    track = Track.load(RACETRACK1)
    env = RaceEnv(track)
    env.reset()
    env.set_state(602.1, 857.1, 1.03, 100.0)  # just before the end of racetrack1
    _, _, terminated, _, info = env.step(np.zeros(2))
    assert terminated and info["finished"]
    assert not track.is_start(info["x"], info["y"])


def test_determinism(tmp_path):
    track = make_track(tmp_path)
    a, b = RaceEnv(track), RaceEnv(track)
    a.reset(), b.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):
        action = rng.uniform(-1, 1, 2)
        oa, ra, ta, _, _ = a.step(action)
        ob, rb, tb, _, _ = b.step(action)
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb
        if ta:
            break
