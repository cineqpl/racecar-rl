"""Synthetic track images for tests and quick experiments.

Convention (same as the real tracks): black = track, white = wall,
red = start patch at one end of the track.
"""

from pathlib import Path

from PIL import Image, ImageDraw

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def corridor_image(length: int = 100, width: int = 20, scale: int = 1) -> Image.Image:
    """Straight corridor, start patch on the left, heading +x.

    Layout (before scaling): image is (length + 40) x (width + 20); the red patch
    is x in [20, 25), the black track is x in [25, 25 + length), both on rows
    [10, 10 + width). With scale > 1 the image is drawn larger and shrunk back
    with smoothing, which produces anti-aliased edge pixels.
    """
    w, h = length + 40, width + 20
    img = Image.new("RGB", (w * scale, h * scale), WHITE)
    draw = ImageDraw.Draw(img)
    y0, y1 = 10 * scale, (10 + width) * scale - 1
    draw.rectangle([25 * scale, y0, (25 + length) * scale - 1, y1], fill=BLACK)
    draw.rectangle([20 * scale, y0, 25 * scale - 1, y1], fill=RED)
    if scale > 1:
        img = img.resize((w, h), Image.BOX)
    return img


def make_corridor(path: Path, length: int = 100, width: int = 20) -> None:
    corridor_image(length, width).save(path)


# Size limits that let small synthetic images pass `Track.load` validation.
SMALL_LIMITS = dict(min_size=(1, 1), max_size=(10_000, 10_000))
