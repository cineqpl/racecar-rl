import tkinter as tk

import pytest

from racecar.track import Track
from racecar.track_gen import corridor_image
from racecar.viewer import MARGIN, build_window, describe_point, render


@pytest.fixture
def track(tmp_path):
    path = tmp_path / "t.bmp"
    corridor_image().save(path)
    return Track.load(path, min_size=(1, 1), max_size=(10_000, 10_000))


def test_describe_start(track):
    assert describe_point(track, 22, 20) == "Start area"


def test_describe_outside(track):
    assert describe_point(track, 70, 2) == "Outside the track"
    assert describe_point(track, -3, 4) == "Outside the track"  # off the image


def test_describe_on_track_shows_distance(track):
    text = describe_point(track, 75, 20)
    assert "On the track" in text and "51.0 px" in text


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


def test_window_click_updates_margin(track):
    try:
        root, canvas, text_id = build_window(track)
    except tk.TclError:
        pytest.skip("no display available")
    try:
        root.update()
        assert int(canvas.cget("width")) == track.mask.shape[1] + MARGIN
        canvas.event_generate("<Button-1>", x=75, y=20)
        root.update()
        assert "51.0 px" in canvas.itemcget(text_id, "text")
        canvas.event_generate("<Button-1>", x=70, y=2)
        root.update()
        assert canvas.itemcget(text_id, "text") == "Outside the track"
        canvas.event_generate("<Button-1>", x=track.mask.shape[1] + 50, y=20)  # margin click
        root.update()
        assert canvas.itemcget(text_id, "text") == "Outside the track"  # unchanged
    finally:
        root.destroy()
