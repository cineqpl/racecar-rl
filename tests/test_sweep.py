"""Track.sweep: what does a step run into? Uses the corridor: track x in [25, 125),
y in [10, 30); the finish area starts at about x = 65 (see test_track.py)."""

import numpy as np
import pytest


def pts(*xy):
    return np.array(xy, dtype=float).reshape(-1, 2)


def test_clear_path_returns_none(corridor_track):
    assert corridor_track.sweep(pts(30, 20), pts(50, 20)) is None


def test_wall_hit_reports_fraction(corridor_track):
    # Straight down from y=20 to y=40: the wall starts at y=30, halfway.
    kind, fraction = corridor_track.sweep(pts(30, 20), pts(30, 40))
    assert kind == "wall"
    assert fraction == pytest.approx(0.5, abs=0.06)


def test_finish_hit_reports_fraction(corridor_track):
    # From x=50 to x=100 along the corridor; finish begins near x=65 (30% along).
    kind, fraction = corridor_track.sweep(pts(50, 20), pts(100, 20))
    assert kind == "finish"
    assert fraction == pytest.approx(0.3, abs=0.04)


def test_finish_before_wall_wins(corridor_track):
    kind, _ = corridor_track.sweep(pts(60, 20), pts(140, 20))
    assert kind == "finish"


def test_wall_before_finish_wins(corridor_track):
    # Leaves through the side wall at x=30 before anything else.
    kind, _ = corridor_track.sweep(pts(30, 20), pts(30, 40))
    assert kind == "wall"


def test_off_image_counts_as_wall(corridor_track):
    kind, fraction = corridor_track.sweep(pts(2, 20), pts(-5, 20))
    assert kind == "wall"
    assert 0 < fraction <= 1


def test_fraction_is_in_zero_one(corridor_track):
    _, fraction = corridor_track.sweep(pts(30, 20), pts(30, 29.5 + 10))
    assert 0 < fraction <= 1


def test_ten_pixel_step_cannot_jump_a_thin_wall(tmp_path):
    from PIL import ImageDraw

    from racecar.track import Track
    from racecar.track_gen import SMALL_LIMITS, WHITE, corridor_image

    img = corridor_image(length=100, width=20)
    # A 1 px wall slab across the corridor at x=45, leaving 2 rows open at the
    # bottom so the track stays connected.
    ImageDraw.Draw(img).rectangle([45, 10, 45, 27], fill=WHITE)
    path = tmp_path / "slab.bmp"
    img.save(path)
    track = Track.load(path, **SMALL_LIMITS)
    kind, _ = track.sweep(pts(40, 15), pts(50, 15))
    assert kind == "wall"


def test_several_points_earliest_hit_wins(corridor_track):
    # Both points move +4 in y. The first (y=20 -> 24) stays clear; the second
    # (y=28 -> 32) crosses the wall at y=30, halfway.
    kind, fraction = corridor_track.sweep(pts(30, 20, 30, 28), pts(30, 24, 30, 32))
    assert kind == "wall"
    assert fraction == pytest.approx(0.5, abs=0.3)


def test_several_points_all_clear(corridor_track):
    assert corridor_track.sweep(pts(30, 15, 30, 25), pts(40, 15, 40, 25)) is None


def test_wall_beats_finish_at_the_same_sample(corridor_track):
    # One point is in the finish area, the other in the wall, at every sample.
    kind, _ = corridor_track.sweep(pts(100, 20, 100, 40), pts(101, 20, 101, 40))
    assert kind == "wall"


def test_mismatched_shapes_raise(corridor_track):
    with pytest.raises(ValueError):
        corridor_track.sweep(pts(30, 20), pts(30, 20, 40, 20))
