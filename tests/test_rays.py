import math

import numpy as np
from PIL import ImageDraw

from racecar.rays import cast_rays
from racecar.track import Track
from racecar.track_gen import BLACK, SMALL_LIMITS, corridor_image

# Corridor: track x in [25, 125), y in [10, 30). Car in the middle at (75, 20).
# A ray's distance is within one step (1 px) above the true distance.


def test_ray_along_corridor(corridor_track):
    d = cast_rays(corridor_track, 75, 20, 0.0, np.array([0.0]), max_len=150)
    assert 50 <= d[0] < 51.5


def test_rays_sideways(corridor_track):
    d = cast_rays(corridor_track, 75, 20, 0.0, np.array([math.pi / 2, -math.pi / 2]), max_len=150)
    assert np.all((10 <= d) & (d < 11.5))


def test_angles_are_relative_to_heading(corridor_track):
    # Facing +y, a ray 90 degrees to the left (-pi/2) points along +x.
    d = cast_rays(corridor_track, 75, 20, math.pi / 2, np.array([-math.pi / 2]), max_len=150)
    assert 50 <= d[0] < 51.5


def test_max_len_caps_the_distance(corridor_track):
    d = cast_rays(corridor_track, 75, 20, 0.0, np.array([0.0]), max_len=20)
    assert d[0] == 20.0


def test_returns_one_distance_per_angle(corridor_track):
    d = cast_rays(corridor_track, 75, 20, 0.0, np.linspace(-1, 1, 7), max_len=150)
    assert d.shape == (7,)


def test_start_in_wall_or_off_image_gives_zeros(corridor_track):
    angles = np.array([0.0, 1.0])
    assert np.all(cast_rays(corridor_track, 5, 5, 0.0, angles, max_len=150) == 0)
    assert np.all(cast_rays(corridor_track, -5, 20, 0.0, angles, max_len=150) == 0)


def test_ray_leaving_the_image_counts_as_wall(tmp_path):
    img = corridor_image()
    ImageDraw.Draw(img).rectangle([25, 10, img.width - 1, 29], fill=BLACK)  # track meets the edge
    path = tmp_path / "edge.bmp"
    img.save(path)
    track = Track.load(path, **SMALL_LIMITS)
    d = cast_rays(track, 70, 20, 0.0, np.array([0.0]), max_len=500)
    assert 70 <= d[0] < 71.5  # image is 140 px wide
