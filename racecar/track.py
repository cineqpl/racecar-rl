"""Loading and validating a track from a bmp image.

Colour convention: black = track, white = wall, red = start patch at one end
of the track. Real images are usually anti-aliased, so colours are not matched
exactly: every pixel is snapped to the nearest of the three reference colours,
as long as it lies on a blend of two of them (see `_classify`).

Coordinates: image coordinates, x = column, y = row (down). Pixel (col, row)
covers x in [col, col + 1) and y in [row, row + 1).
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Pixel classes in the label array returned by `_classify`.
WALL, TRACK, START = 0, 1, 2

# Reference colours, indexed by class.
_REFERENCE = np.array([(255, 255, 255), (0, 0, 0), (255, 0, 0)], dtype=np.float32)

MIN_SIZE = (1200, 800)  # (width, height)
MAX_SIZE = (1600, 1200)
BLEND_TOLERANCE = 20.0  # max RGB distance from a blend of two reference colours
MAX_SNAPPED_FRACTION = 0.02  # max share of pixels outside the strict ranges
HEADING_BAND = (25.0, 35.0)  # distances from the start used to derive the heading
FINISH_LENGTH = 60.0  # the finish area is the last this-many px of the track (see test_track.py for why 60)

_SQRT2 = math.sqrt(2.0)


class TrackValidationError(ValueError):
    """The image is not a valid track."""


@dataclass
class Track:
    mask: np.ndarray  # (H, W) bool, drivable (track + start patch)
    dist: np.ndarray  # (H, W) float32, distance along the track from the start, inf on walls
    length: float  # largest finite distance
    start_pos: tuple[float, float]  # (x, y), centre of the start patch
    start_heading: float  # radians, 0 = +x, positive turns toward +y
    finish: np.ndarray = field(init=False)  # (H, W) bool, the last FINISH_LENGTH px of the track

    def __post_init__(self) -> None:
        self.finish = self.mask & (self.dist > self.length - FINISH_LENGTH)

    @classmethod
    def load(
        cls,
        path: str | Path,
        min_size: tuple[int, int] = MIN_SIZE,
        max_size: tuple[int, int] = MAX_SIZE,
        max_snapped_fraction: float = MAX_SNAPPED_FRACTION,
    ) -> "Track":
        """Load and validate a track. Raises TrackValidationError if invalid.

        `min_size` and `max_size` are (width, height) limits in pixels.
        """
        image = Image.open(path).convert("RGB")  # drops alpha, expands palettes
        w, h = image.size
        if not (min_size[0] <= w <= max_size[0] and min_size[1] <= h <= max_size[1]):
            raise TrackValidationError(
                f"image size {w}x{h} is outside the allowed range "
                f"{min_size[0]}-{max_size[0]} wide, {min_size[1]}-{max_size[1]} high"
            )
        rgb = np.asarray(image)
        labels = _classify(rgb, max_snapped_fraction)

        start = labels == START
        mask = labels != WALL
        n_start = _count_components(start, eight=True)
        if n_start != 1:
            raise TrackValidationError(
                f"found {n_start} start patches (red); expected exactly 1"
            )
        # A white island enclosed by the track means the track loops. The image
        # border counts as wall, so a track running off the edge is not a loop.
        if _count_components(np.pad(~mask, 1, constant_values=True), eight=False) != 1:
            raise TrackValidationError(
                "the track loops: the area outside it is not one connected piece"
            )

        dist = _geodesic_distance(mask, start)
        if np.isinf(dist[mask]).any():
            raise TrackValidationError(
                "the track is not connected: some black pixels cannot be reached "
                "from the start patch"
            )
        length = float(dist[mask].max())

        ys, xs = np.nonzero(start)
        start_pos = (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)
        band = (dist >= HEADING_BAND[0]) & (dist <= HEADING_BAND[1])
        if not band.any():
            raise TrackValidationError(
                f"track is too short to derive a start heading (needs a point "
                f"{HEADING_BAND[0]:.0f}-{HEADING_BAND[1]:.0f} px from the start patch)"
            )
        by, bx = np.nonzero(band)
        heading = math.atan2(by.mean() + 0.5 - start_pos[1], bx.mean() + 0.5 - start_pos[0])
        return cls(mask, dist, length, start_pos, heading)

    def is_drivable(self, x: float, y: float) -> bool:
        """True if (x, y) is on the track; anything outside the image is wall."""
        col, row = math.floor(x), math.floor(y)
        h, w = self.mask.shape
        return 0 <= col < w and 0 <= row < h and bool(self.mask[row, col])

    def is_start(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the red start patch (the only pixels at distance 0)."""
        return self.is_drivable(x, y) and self.progress_at(x, y) == 0.0

    def progress_at(self, x: float, y: float) -> float:
        """Distance along the track at (x, y); inf if (x, y) is not drivable."""
        if not self.is_drivable(x, y):
            return math.inf
        return float(self.dist[math.floor(y), math.floor(x)])


def _classify(rgb: np.ndarray, max_snapped_fraction: float) -> np.ndarray:
    """Label every pixel WALL / TRACK / START, or raise on foreign colours.

    A pixel is accepted if it lies within BLEND_TOLERANCE of a straight line
    between two reference colours (an anti-aliased edge is such a blend). It is
    then snapped to the reference colour that dominates the blend.
    """
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # Distance to the nearest of the three blend segments (white-black,
    # white-red, black-red).
    off = np.full(len(pixels), np.inf, dtype=np.float32)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pa, ab = _REFERENCE[a], _REFERENCE[b] - _REFERENCE[a]
        t = np.clip((pixels - pa) @ ab / (ab @ ab), 0.0, 1.0)
        off = np.minimum(off, np.linalg.norm(pixels - (pa + t[:, None] * ab), axis=1))
    # Where all three colours meet, a pixel can be a blend of all three: it then
    # lies inside the triangle they span, not on one of its edges.
    k, w_k, r_k = _REFERENCE[1], _REFERENCE[0] - _REFERENCE[1], _REFERENCE[2] - _REFERENCE[1]
    basis = np.stack([w_k, r_k])  # (2, 3)
    uv = np.linalg.solve(basis @ basis.T, basis @ (pixels - k).T).T  # barycentric-style coords
    inside = (uv >= 0).all(axis=1) & (uv.sum(axis=1) <= 1)
    off_plane = np.linalg.norm(pixels - k - uv @ basis, axis=1)
    off = np.where(inside, np.minimum(off, off_plane), off)
    bad = off > BLEND_TOLERANCE
    if bad.any():
        i = int(np.argmax(bad))
        y, x = divmod(i, rgb.shape[1])
        raise TrackValidationError(
            f"pixel colour {tuple(int(c) for c in rgb[y, x])} at x={x}, y={y} is not "
            f"black, white, red or a blend of them ({int(bad.sum())} such pixels)"
        )

    # Strict ranges: black 0-10, white 245-255, red (200-255, 0-30, 0-30).
    in_range = (
        (pixels <= 10).all(axis=1)
        | (pixels >= 245).all(axis=1)
        | ((pixels[:, 0] >= 200) & (pixels[:, 1] <= 30) & (pixels[:, 2] <= 30))
    )
    snapped = 1.0 - float(in_range.mean())
    if snapped > max_snapped_fraction:
        raise TrackValidationError(
            f"{snapped:.1%} of the pixels were snapped to the nearest colour "
            f"(limit {max_snapped_fraction:.1%}); the image is too blurry or noisy"
        )

    # Snap to the reference colour with the largest weight in the blend. (Plain
    # nearest-colour would send mid-greys to red, which is as far from them as
    # black and white are.) Class order is WALL, TRACK, START = white, black, red.
    weights = np.stack([uv[:, 0], 1.0 - uv.sum(axis=1), uv[:, 1]], axis=1)
    return weights.argmax(axis=1).reshape(rgb.shape[:2])


def _count_components(mask: np.ndarray, eight: bool) -> int:
    """Number of connected components of True pixels (4- or 8-connected)."""
    h, w = mask.shape
    wp = w + 2
    flat = np.pad(mask, 1).ravel().tolist()  # zero frame removes bounds checks
    offsets = [-1, 1, -wp, wp]
    if eight:
        offsets += [-wp - 1, -wp + 1, wp - 1, wp + 1]
    count = 0
    for seed in np.flatnonzero(np.pad(mask, 1).ravel()).tolist():
        if not flat[seed]:
            continue
        count += 1
        flat[seed] = False
        queue = deque([seed])
        while queue:
            i = queue.popleft()
            for o in offsets:
                j = i + o
                if flat[j]:
                    flat[j] = False
                    queue.append(j)
    return count


def _geodesic_distance(mask: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Shortest 8-connected path length (steps cost 1, diagonals sqrt 2) from the
    start pixels through drivable pixels. Unreachable and wall pixels are inf."""
    h, w = mask.shape
    wp = w + 2
    drivable = np.pad(mask, 1).ravel().tolist()
    dist = [math.inf] * len(drivable)
    steps = [(-1, 1.0), (1, 1.0), (-wp, 1.0), (wp, 1.0),
             (-wp - 1, _SQRT2), (-wp + 1, _SQRT2), (wp - 1, _SQRT2), (wp + 1, _SQRT2)]
    heap = []
    for i in np.flatnonzero(np.pad(start, 1).ravel()).tolist():
        dist[i] = 0.0
        heap.append((0.0, i))
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for o, cost in steps:
            j = i + o
            if drivable[j] and d + cost < dist[j]:
                dist[j] = d + cost
                heapq.heappush(heap, (d + cost, j))
    out = np.array(dist, dtype=np.float32).reshape(h + 2, wp)
    return out[1:-1, 1:-1]
