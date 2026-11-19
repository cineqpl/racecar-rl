"""Textures for the replay view: a prerendered background and the rotated car sprite.

Display only. Pillow and numpy, no tkinter, so it is easy to test.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from racecar.track import Track
from racecar.viewer import render

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
START_RGB, FINISH_RGB = (255, 0, 0), (0, 200, 0)
TINT_ALPHA = 0.45


def _tiled(tile: Image.Image, h: int, w: int) -> np.ndarray:
    """The tile repeated to cover an (h, w) image, as a float array (h, w, 3)."""
    a = np.asarray(tile.convert("RGB"), dtype=np.float32)
    reps = (math.ceil(h / a.shape[0]), math.ceil(w / a.shape[1]), 1)
    return np.tile(a, reps)[:h, :w]


def render_textured(track: Track, assets_dir: Path | None = None) -> Image.Image:
    """Grass everywhere, asphalt on the track, the start patch tinted red and the finish area green.

    Falls back to the plain black/white `viewer.render` if a texture file is missing.
    """
    folder = ASSETS_DIR if assets_dir is None else Path(assets_dir)
    h, w = track.mask.shape
    try:
        with Image.open(folder / "grass.bmp") as g, Image.open(folder / "asphalt.bmp") as a:
            grass, asphalt = _tiled(g, h, w), _tiled(a, h, w)
    except FileNotFoundError:
        return render(track)
    out = np.where(track.mask[..., None], asphalt, grass)
    for area, colour in ((track.mask & (track.dist == 0), START_RGB), (track.finish, FINISH_RGB)):
        out[area] = (1 - TINT_ALPHA) * out[area] + TINT_ALPHA * np.array(colour, dtype=np.float32)
    return Image.fromarray(np.rint(out).astype(np.uint8))


def load_car_sprite(assets_dir: Path | None = None, length: int = 20) -> Image.Image | None:
    """`car.png` (nose pointing right) scaled to `length` px long, aspect ratio kept; None if missing."""
    folder = ASSETS_DIR if assets_dir is None else Path(assets_dir)
    try:
        with Image.open(folder / "car.png") as img:
            car = img.convert("RGBA")
    except FileNotFoundError:
        return None
    return car.resize((length, max(1, round(length * car.height / car.width))), Image.LANCZOS)


def rotate_sprite(sprite: Image.Image, heading: float) -> Image.Image:
    """The sprite turned so its nose points along `heading` (radians, clockwise on screen)."""
    return sprite.rotate(-math.degrees(heading), expand=True, resample=Image.BICUBIC)
