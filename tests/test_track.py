import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from racecar.track import Track, TrackValidationError
from racecar.track_gen import BLACK, RED, WHITE, corridor_image

# Test images are small, so relax the size limits that real tracks must meet.
SMALL = dict(min_size=(1, 1), max_size=(10_000, 10_000))
RACETRACK1 = Path(__file__).parent / "data" / "racetrack1.bmp"


def load(img: Image.Image, tmp_path: Path, **kwargs) -> Track:
    path = tmp_path / "track.bmp"
    img.save(path)
    return Track.load(path, **{**SMALL, **kwargs})


# --- happy path -------------------------------------------------------------


def test_classifies_pixels(tmp_path):
    t = load(corridor_image(), tmp_path)
    assert t.is_drivable(70, 20)  # black track
    assert t.is_drivable(22, 20)  # red start patch
    assert not t.is_drivable(70, 2)  # white wall


def test_out_of_image_is_not_drivable(tmp_path):
    t = load(corridor_image(), tmp_path)
    assert not t.is_drivable(-5, 10)
    assert not t.is_drivable(1000, 1000)


def test_distance_field_corridor(tmp_path):
    t = load(corridor_image(), tmp_path)
    assert abs(t.progress_at(75, 20) - 51) < 1.5
    assert 99 <= t.length <= 101
    assert t.progress_at(22, 20) == 0.0


def test_start_position_and_heading(tmp_path):
    t = load(corridor_image(), tmp_path)
    x, y = t.start_pos
    assert abs(x - 22.5) < 0.1 and abs(y - 20.0) < 0.1  # pixel centres
    assert abs(t.start_heading) < 0.05  # pointing along +x


def test_rgba_image_is_accepted(tmp_path):
    t = load(corridor_image().convert("RGBA"), tmp_path)
    assert t.is_drivable(70, 20)


def test_antialiased_image_loads_like_the_clean_one(tmp_path):
    clean = load(corridor_image(), tmp_path)
    smooth = load(corridor_image(scale=4), tmp_path)
    assert (clean.mask == smooth.mask).mean() > 0.99
    assert abs(smooth.length - clean.length) < 2


def test_track_running_off_the_image_edge_is_not_a_loop(tmp_path):
    img = corridor_image()
    ImageDraw.Draw(img).rectangle([25, 10, img.width - 1, 29], fill=BLACK)
    load(img, tmp_path)  # must not raise


def test_real_track_loads():
    t = Track.load(RACETRACK1)  # default size limits
    assert t.mask.shape == (1024, 1536)
    assert t.length > 3000
    assert abs(t.start_heading) < 0.3  # the start straight points right


# --- validation -------------------------------------------------------------


def test_size_out_of_bounds_raises(tmp_path):
    path = tmp_path / "t.bmp"
    corridor_image().save(path)
    with pytest.raises(TrackValidationError, match="size"):
        Track.load(path)  # default limits: 1200-1600 x 800-1200


def test_foreign_colour_raises_and_names_it(tmp_path):
    img = corridor_image()
    ImageDraw.Draw(img).rectangle([60, 12, 63, 14], fill=(0, 0, 255))
    with pytest.raises(TrackValidationError, match=r"colour.*\(0, 0, 255\)"):
        load(img, tmp_path)


def test_too_many_snapped_pixels_raises(tmp_path):
    img = corridor_image()
    ImageDraw.Draw(img).rectangle([0, 0, img.width - 1, 5], fill=(128, 128, 128))
    with pytest.raises(TrackValidationError, match="snapped"):
        load(img, tmp_path, max_snapped_fraction=0.01)


def test_missing_start_raises(tmp_path):
    img = corridor_image()
    ImageDraw.Draw(img).rectangle([20, 10, 24, 29], fill=BLACK)
    with pytest.raises(TrackValidationError, match="start"):
        load(img, tmp_path)


def test_two_start_patches_raise(tmp_path):
    img = corridor_image()
    ImageDraw.Draw(img).rectangle([120, 10, 124, 29], fill=RED)
    with pytest.raises(TrackValidationError, match="start"):
        load(img, tmp_path)


def test_loop_raises(tmp_path):
    img = Image.new("RGB", (160, 160), WHITE)
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 140, 140], fill=BLACK)
    d.ellipse([45, 45, 115, 115], fill=WHITE)  # white island inside the ring
    d.rectangle([75, 20, 80, 44], fill=RED)
    with pytest.raises(TrackValidationError, match="loop"):
        load(img, tmp_path)


def test_disconnected_track_raises(tmp_path):
    img = Image.new("RGB", (200, 60), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 10, 24, 29], fill=RED)
    d.rectangle([25, 10, 80, 29], fill=BLACK)
    d.rectangle([120, 10, 180, 29], fill=BLACK)  # unreachable strip
    with pytest.raises(TrackValidationError, match="connected"):
        load(img, tmp_path)
