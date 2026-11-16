import tkinter as tk

import numpy as np
import pytest
from PIL import ImageTk  # noqa: F401  (fail early if Pillow's Tk support is missing)

from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image
from racecar.viewer import (
    MARGIN, RAY_RGB, build_window, describe_point, ray_endpoints, ray_lengths, rays_overlay, render,
)


@pytest.fixture
def track(corridor_track):
    return corridor_track


# --- describing a clicked point ------------------------------------------------


def test_describe_start(track):
    assert describe_point(track, 22, 20) == "Start area"


def test_describe_outside(track):
    assert describe_point(track, 70, 2) == "Outside the track"
    assert describe_point(track, -3, 4) == "Outside the track"  # off the image


def test_describe_on_track_shows_distance(track):
    text = describe_point(track, 75, 20)
    assert "On the track" in text and "51.0 px" in text


def test_describe_lists_ray_lengths_after_the_distance(track):
    text = describe_point(track, 75, 20, np.array([10.0, 50.0, 30.0]))
    assert text.index("51.0 px from the start") < text.index("Rays")
    assert text.endswith("10, 50, 30")


def test_describe_outside_ignores_ray_lengths(track):
    assert describe_point(track, 70, 2, np.array([1.0])) == "Outside the track"


def test_is_start(track):
    assert track.is_start(22, 20)
    assert not track.is_start(75, 20)
    assert not track.is_start(70, 2)


def test_render_colours(track):
    img = render(track)
    assert img.size == (track.mask.shape[1], track.mask.shape[0])
    assert img.getpixel((70, 20)) == (0, 0, 0)
    assert img.getpixel((22, 20)) == (255, 0, 0)
    assert img.getpixel((70, 2)) == (255, 255, 255)


# --- rays ---------------------------------------------------------------------


def test_ray_endpoints_heading_up(track):
    # Facing up (0 deg) from (75, 20): the cone's left ray points -x, the right ray +x.
    ends = ray_endpoints(track, 75, 20, 0, 3)
    assert ends.shape == (3, 2)
    assert abs(ends[0][0] - 20) < 2 and abs(ends[0][1] - 20) < 1e-6  # left: wall at x=20
    assert abs(ends[1][0] - 75) < 1e-6 and abs(ends[1][1] - 10) < 2  # ahead: wall at y=10
    assert abs(ends[2][0] - 125) < 2 and abs(ends[2][1] - 20) < 1e-6  # right: wall at x=125


def test_ray_endpoints_heading_right(track):
    ends = ray_endpoints(track, 75, 20, 90, 3)  # 90 deg = facing right
    assert abs(ends[1][0] - 125) < 2  # straight ahead reaches the far wall
    assert abs(ends[0][1] - 10) < 2 and abs(ends[2][1] - 30) < 2  # up and down


def test_single_ray_points_straight_ahead(track):
    ends = ray_endpoints(track, 75, 20, 90, 1)
    assert ends.shape == (1, 2) and abs(ends[0][1] - 20) < 1e-6


def test_rays_are_not_capped_at_150_px(tmp_path):
    path = tmp_path / "long.bmp"
    corridor_image(length=400).save(path)
    long_track = Track.load(path, **SMALL_LIMITS)
    d = ray_lengths(long_track, 30, 20, 90, 1)  # facing right along a 400 px corridor
    assert d[0] > 350


def test_overlay_draws_yellow_rays(track):
    overlay = rays_overlay(track, (75, 20), np.array([[125.0, 20.0]]))
    assert overlay.mode == "RGBA" and overlay.size == (track.mask.shape[1], track.mask.shape[0])
    px = overlay.getpixel((100, 20))
    assert px[:3] == RAY_RGB and px[3] == 255


def test_overlay_never_spills_onto_walls(track):
    # A 2 px wide ray along the top edge of the corridor (row 10; row 9 is wall).
    overlay = rays_overlay(track, (75, 10.5), np.array([[125.0, 10.5]]))
    alpha = np.asarray(overlay)[..., 3]
    assert alpha.max() == 255  # something is drawn
    assert alpha[~track.mask].max() == 0  # nothing on walls


def test_overlay_follows_the_display_scale(track):
    overlay = rays_overlay(track, (75, 20), np.array([[125.0, 20.0]]), scale=0.5)
    assert overlay.size == (track.mask.shape[1] // 2, track.mask.shape[0] // 2)


# --- the window ---------------------------------------------------------------


def sidebar_ray_numbers(win) -> list[float]:
    text = win.canvas.itemcget(win.text_id, "text")
    return [float(v) for v in text.split("\n")[-1].split(", ")]

@pytest.fixture
def win(track, tk_master):
    w = build_window(track, master=tk_master)
    w.root.update()
    yield w
    w.root.destroy()


def sidebar_text(win) -> str:
    return win.canvas.itemcget(win.text_id, "text")


def test_window_click_draws_rays_and_sidebar_updates_them(win):
    assert win.canvas.find_withtag("rays") == ()  # nothing before the first click
    win.canvas.event_generate("<Button-1>", x=75, y=20)
    win.root.update()
    assert len(win.canvas.find_withtag("rays")) == 1  # one overlay image
    assert "51.0 px from the start" in sidebar_text(win)
    assert len(sidebar_ray_numbers(win)) == 9
    win.n_rays.set(5)
    win.root.update()
    assert len(sidebar_ray_numbers(win)) == 5
    win.heading.set(90)  # facing right: the middle ray runs down the corridor
    win.root.update()
    assert 49 <= sidebar_ray_numbers(win)[2] <= 52
    win.canvas.event_generate("<Button-1>", x=70, y=2)  # outside the track
    win.root.update()
    assert win.canvas.find_withtag("rays") == ()
    assert sidebar_text(win) == "Outside the track"


def test_window_dragging_follows_the_mouse(win):
    win.canvas.event_generate("<Button-1>", x=75, y=20)
    win.canvas.event_generate("<B1-Motion>", x=100, y=20)  # mouse moves, button held
    win.root.update()
    assert "76.0 px from the start" in sidebar_text(win)
    win.canvas.event_generate("<B1-Motion>", x=70, y=2)  # drag across the wall
    win.root.update()
    assert sidebar_text(win) == "Outside the track"
    win.canvas.event_generate("<B1-Motion>", x=50, y=20)  # and back onto the track
    win.root.update()
    assert "26.0 px from the start" in sidebar_text(win)
    assert len(win.canvas.find_withtag("rays")) == 1
    win.canvas.event_generate("<Motion>", x=60, y=20)  # no button: nothing changes
    win.root.update()
    assert "26.0 px from the start" in sidebar_text(win)


def test_window_layout_and_sidebar_clicks(win, track):
    assert int(win.canvas.cget("width")) == track.mask.shape[1] + MARGIN
    win.canvas.event_generate("<Button-1>", x=70, y=2)
    win.root.update()
    assert sidebar_text(win) == "Outside the track"
    win.canvas.event_generate("<Button-1>", x=track.mask.shape[1] + 50, y=20)  # sidebar click
    win.root.update()
    assert sidebar_text(win) == "Outside the track"  # unchanged
