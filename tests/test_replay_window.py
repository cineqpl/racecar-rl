import numpy as np
import pytest
import torch

from racecar.env import RaceEnv
from racecar.nets import GaussianPolicy
from racecar.replay import Recording, ReplayWindow, car_polygon, parse_args, record_episode
from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image
from racecar.viewer import MARGIN


@pytest.fixture
def track(tmp_path) -> Track:
    path = tmp_path / "corridor.bmp"
    corridor_image(length=400).save(path)
    return Track.load(path, **SMALL_LIMITS)


def constant_policy(throttle: float, steer: float) -> GaussianPolicy:
    policy = GaussianPolicy(10)
    with torch.no_grad():
        policy.net[-1].weight.zero_()
        policy.net[-1].bias.copy_(torch.tensor([throttle, steer]))
    return policy


@pytest.fixture
def driving(track) -> Recording:
    return record_episode(RaceEnv(track), constant_policy(1.0, 0.0))


@pytest.fixture
def crashing(track) -> Recording:
    return record_episode(RaceEnv(track), constant_policy(1.0, 1.0))


@pytest.fixture
def make_window(track, tk_master):
    windows = []

    def make(recording, **kwargs) -> ReplayWindow:
        win = ReplayWindow(track, recording, master=tk_master, **kwargs)
        windows.append(win)
        return win

    yield make
    for win in windows:
        try:
            win.root.destroy()
        except Exception:
            pass  # already closed by the test


def text(win) -> str:
    return win.canvas.itemcget(win.text_id, "text")


def test_starts_paused_on_frame_zero(make_window, driving):
    win = make_window(driving)
    assert win.frame == 0 and not win.playing
    assert win.canvas.coords("car") == pytest.approx([driving.x[0], driving.y[0]])  # the sprite's centre


def test_step_forward_and_back_clamp_at_both_ends(make_window, driving):
    win = make_window(driving)
    win.step(-1)
    assert win.frame == 0
    for _ in range(driving.n_frames + 5):
        win.step(+1)
    assert win.frame == driving.n_frames - 1
    win.step(-1)
    assert win.frame == driving.n_frames - 2


def test_show_updates_slider_and_sidebar(make_window, driving):
    win = make_window(driving)
    win.show(5)
    assert win.frame == 5 and int(float(win.slider.get())) == 5
    assert "5 /" in text(win)


def test_moving_the_slider_shows_that_frame(make_window, driving):
    win = make_window(driving)
    win.slider.set(7)
    win.root.update()  # Tk runs a Scale command when it redraws, not inside set()
    assert win.frame == 7


def test_trail_grows_with_the_frame(make_window, driving):
    win = make_window(driving)
    win.show(0)  # a trail of one point must not raise
    win.show(6)
    assert len(win.canvas.coords("trail")) // 2 == 7


def test_play_advances_and_stops_at_the_end(make_window, driving):
    win = make_window(driving)
    win.toggle_play()
    assert win.playing
    for _ in range(driving.n_frames + 2):
        win.tick()
    assert win.frame == driving.n_frames - 1 and not win.playing


def test_play_at_the_end_restarts(make_window, driving):
    win = make_window(driving)
    win.show(driving.n_frames - 1)
    win.toggle_play()
    assert win.frame == 0 and win.playing


def test_pause_stops_ticking(make_window, driving):
    win = make_window(driving)
    win.toggle_play()
    win.tick()
    win.toggle_play()
    frame = win.frame
    win.tick()
    assert not win.playing and win.frame == frame


def test_restart_goes_to_frame_zero_paused(make_window, driving):
    win = make_window(driving)
    win.show(9)
    win.toggle_play()
    win.restart()
    assert win.frame == 0 and not win.playing


def test_speed_changes_the_delay(make_window, driving):
    win = make_window(driving, dt=0.1)
    assert win.delay_ms == 100
    win.set_speed(2.0)
    assert win.delay_ms == 50 and win.speed == 2.0
    win.set_speed(0.5)
    assert win.delay_ms == 200


def test_last_frame_reports_the_outcome(make_window, driving, crashing):
    win = make_window(driving)
    win.show(driving.n_frames - 1)
    assert "finished" in text(win).lower()
    win = make_window(crashing)
    win.show(crashing.n_frames - 1)
    assert "crashed" in text(win).lower()


def test_crash_frame_draws_no_rays(make_window, crashing):
    win = make_window(crashing)
    assert len(win.canvas.find_withtag("ray")) == crashing.rays.shape[1]
    win.show(crashing.n_frames - 1)  # all rays 0 here: nothing to draw, nothing raised
    assert len(win.canvas.find_withtag("ray")) == 0


def test_two_frame_recording_works(make_window):
    rec = Recording(
        x=np.array([30.0, 40.0]), y=np.array([20.0, 29.0]), heading=np.zeros(2), speed=np.array([0.0, 10.0]),
        progress=np.array([0.0, 10.0]), rays=np.array([[5.0, 6.0, 7.0], [0.0, 0.0, 0.0]]),
        ray_angles=np.array([-1.0, 0.0, 1.0]), actions=np.array([[1.0, 0.0]]), rewards=np.array([0.5]),
        crashed=True, finished=False,
    )
    win = make_window(rec)
    win.step(+1)
    assert win.frame == 1 and "crashed" in text(win).lower()
    win.toggle_play()  # at the end: restarts, then ticks to the end and stops
    win.tick()
    assert win.frame == 1 and not win.playing


def test_display_scale_scales_the_drawing(make_window, track, driving):
    full, half = make_window(driving, scale=1.0), make_window(driving, scale=0.5)
    assert half.canvas.coords("car") == pytest.approx([c / 2 for c in full.canvas.coords("car")])
    h, w = track.mask.shape
    assert int(half.canvas["width"]) == round(w * 0.5) + MARGIN
    assert int(full.canvas["width"]) == w + MARGIN


def test_closing_while_playing_cancels_the_timer(make_window, driving):
    win = make_window(driving)
    win.toggle_play()
    assert win._job is not None
    win.root.destroy()
    assert win._job is None


def test_arguments_are_parsed():
    args = parse_args(["runs/x"])
    assert str(args.run_dir) == "runs\\x" or str(args.run_dir) == "runs/x"
    assert args.scale == 1.0 and args.checkpoint is None and args.track.name == "racetrack1.bmp"
    args = parse_args(["runs/x", "--checkpoint", "5", "--scale", "0.75", "--track", "t.bmp"])
    assert args.checkpoint == 5 and args.scale == 0.75 and args.track.name == "t.bmp"


def test_car_is_an_image_not_a_polygon(make_window, driving):
    win = make_window(driving)
    win.show(5)
    assert len(win.canvas.find_withtag("car")) == 1
    assert win.canvas.type("car") == "image"


def test_car_image_is_centred_on_the_scaled_position(make_window, driving):
    win = make_window(driving, scale=0.5)
    win.show(7)
    assert win.canvas.coords("car") == pytest.approx([driving.x[7] * 0.5, driving.y[7] * 0.5])


def test_sprite_cache_reuses_images_for_nearby_headings(make_window, driving):
    win = make_window(driving)
    assert win._sprite_for(0.0) is win._sprite_for(0.01)
    assert win._sprite_for(0.0) is not win._sprite_for(1.0)
    assert win._sprite_for(-7.0) is not None and win._sprite_for(20.0) is not None


def test_window_opens_without_assets(make_window, driving, monkeypatch, tmp_path):
    monkeypatch.setattr("racecar.texture.ASSETS_DIR", tmp_path)
    win = make_window(driving)
    win.show(3)
    assert win.canvas.type("car") == "polygon"
    assert win._sprite_for(0.0) is None
