"""Ray casting against a track's drivable mask."""

import numpy as np

from racecar.track import Track


def cast_rays(
    track: Track,
    x: float,
    y: float,
    heading: float,
    angles: np.ndarray,
    max_len: float,
    step: float = 1.0,
) -> np.ndarray:
    """Distance from (x, y) to the nearest wall along each ray.

    Ray i points in direction `heading + angles[i]` (radians). The ray is
    sampled every `step` pixels; the result is the distance of the first sample
    that is not drivable, so it overestimates the true distance by less than
    `step`. It is capped at `max_len`. Outside the image counts as wall. If the
    start point itself is not drivable, every distance is 0.
    """
    if not track.is_drivable(x, y):
        return np.zeros(len(angles), dtype=np.float32)

    n_samples = int(np.ceil(max_len / step))
    d = step * np.arange(1, n_samples + 1)  # (S,)
    direction = heading + np.asarray(angles, dtype=np.float64)  # (R,)
    px = x + np.cos(direction)[:, None] * d[None, :]  # (R, S)
    py = y + np.sin(direction)[:, None] * d[None, :]
    col, row = np.floor(px).astype(int), np.floor(py).astype(int)

    h, w = track.mask.shape
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    free = inside & track.mask[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)]

    blocked = ~free
    first = blocked.argmax(axis=1)  # index of the first blocked sample (0 if none)
    hit = blocked.any(axis=1)
    return np.minimum(np.where(hit, d[first], max_len), max_len).astype(np.float32)
