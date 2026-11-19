import math

import numpy as np
import pytest
from PIL import Image, ImageChops

from racecar import viewer
from racecar.texture import load_car_sprite, render_textured, rotate_sprite

ASPHALT, GRASS = (80, 80, 80), (0, 150, 0)


@pytest.fixture
def assets(tmp_path):
    folder = tmp_path / "assets"
    folder.mkdir()
    Image.new("RGB", (16, 16), ASPHALT).save(folder / "asphalt.bmp")
    Image.new("RGB", (16, 16), GRASS).save(folder / "grass.bmp")
    car = Image.new("RGBA", (35, 18), (0, 0, 255, 255))  # blue tail, red nose on the right
    car.paste((255, 0, 0, 255), (18, 0, 35, 18))
    car.save(folder / "car.png")
    return folder


def _xy(mask: np.ndarray) -> tuple[int, int]:
    row, col = np.argwhere(mask)[0]
    return int(col), int(row)


def test_render_textured_size_matches_track(corridor_track, assets):
    h, w = corridor_track.mask.shape
    assert render_textured(corridor_track, assets).size == (w, h)


def test_asphalt_on_track_grass_outside(corridor_track, assets):
    t = corridor_track
    image = render_textured(t, assets)
    plain = t.mask & ~t.finish & (t.dist > 0)
    assert image.getpixel(_xy(plain)) == ASPHALT
    assert image.getpixel(_xy(~t.mask)) == GRASS


def test_start_and_finish_are_tinted(corridor_track, assets):
    t = corridor_track
    image = render_textured(t, assets)
    r, g, b = image.getpixel(_xy(t.mask & (t.dist == 0)))
    assert r > g and r > b
    r, g, b = image.getpixel(_xy(t.finish))
    assert g > r and g > b
    assert image.getpixel(_xy(t.finish)) != ASPHALT


def test_missing_assets_fall_back_to_plain_render(corridor_track, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    diff = ImageChops.difference(render_textured(corridor_track, empty), viewer.render(corridor_track))
    assert diff.getbbox() is None


def test_load_car_sprite_size_and_missing(assets, tmp_path):
    assert load_car_sprite(assets, length=20).size == (20, 10)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_car_sprite(empty) is None


def _red_centroid(image: Image.Image) -> tuple[float, float]:
    a = np.asarray(image.convert("RGBA")).astype(int)
    red = (a[..., 0] > 200) & (a[..., 2] < 50) & (a[..., 3] > 200)
    rows, cols = np.nonzero(red)
    return cols.mean() - (image.width - 1) / 2, rows.mean() - (image.height - 1) / 2


def test_rotate_sprite_nose_follows_heading(assets):
    sprite = load_car_sprite(assets, length=20)
    dx, dy = _red_centroid(rotate_sprite(sprite, 0.0))
    assert dx > 2 and abs(dy) < 1  # nose to the right
    dx, dy = _red_centroid(rotate_sprite(sprite, math.pi / 2))
    assert dy > 2 and abs(dx) < 1  # clockwise on screen: nose down
    diff = ImageChops.difference(rotate_sprite(sprite, math.pi / 2), rotate_sprite(sprite, -3 * math.pi / 2))
    assert diff.getbbox() is None
