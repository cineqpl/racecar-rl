"""Click-to-inspect track viewer.

    uv run python -m racecar.viewer tests/data/racetrack1.bmp [--scale 0.75]

The window shows the track plus a sidebar on the right. Clicking a point shows
whether it is the start area, outside the track, or on the track (with its
distance along the track); the point follows the mouse while the left button is
held down. A point inside the track also gets rays
spread over a 180 degree cone in front of it and lists their lengths. The
sidebar sets the number of rays and the heading (in degrees, 0 = up, clockwise).
"""

import argparse
import math
import tkinter as tk
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageTk

from racecar.rays import cast_rays
from racecar.track import Track

MARGIN = 240  # width of the sidebar on the right, in screen pixels
MAX_RAY_LEN = 2000.0  # pixels; effectively uncapped on tracks of this size
DEFAULT_RAYS = 9
RAY_RGB = (255, 255, 0)  # yellow
RAY_WIDTH = 2  # screen pixels


def describe_point(
    track: Track, x: float, y: float, ray_lengths: Sequence[float] | None = None
) -> str:
    """Text for the sidebar when the user clicks track coordinates (x, y).

    If `ray_lengths` is given and the point is inside the track, they are listed
    after the distance information.
    """
    if not track.is_drivable(x, y):
        return "Outside the track"
    text = "Start area" if track.is_start(x, y) else (
        f"On the track\n{track.progress_at(x, y):.1f} px from the start"
    )
    if ray_lengths is not None and len(ray_lengths) > 0:
        text += "\n\nRays (px, left to right):\n" + ", ".join(f"{d:.0f}" for d in ray_lengths)
    return text


def _cast(track: Track, x: float, y: float, heading_deg: float, n_rays: int, max_len: float):
    """Ray lengths and absolute directions (radians, track convention)."""
    heading = math.radians(heading_deg - 90.0)  # viewer: 0 = up; track: 0 = +x. Both clockwise.
    angles = np.linspace(-math.pi / 2, math.pi / 2, n_rays) if n_rays > 1 else np.array([0.0])
    return cast_rays(track, x, y, heading, angles, max_len), heading + angles


def ray_lengths(
    track: Track, x: float, y: float, heading_deg: float, n_rays: int,
    max_len: float = MAX_RAY_LEN,
) -> np.ndarray:
    """Length of each ray from (x, y), ordered left to right as seen facing the heading.

    `heading_deg` uses the viewer's convention: 0 = up, clockwise. The rays are
    spread equally over the 180 degree cone around that heading; one ray points
    straight ahead.
    """
    return _cast(track, x, y, heading_deg, n_rays, max_len)[0]


def ray_endpoints(
    track: Track, x: float, y: float, heading_deg: float, n_rays: int,
    max_len: float = MAX_RAY_LEN,
) -> np.ndarray:
    """Where each ray from (x, y) ends, as an (n_rays, 2) array of (x, y)."""
    d, direction = _cast(track, x, y, heading_deg, n_rays, max_len)
    return np.stack([x + d * np.cos(direction), y + d * np.sin(direction)], axis=1)


def render(track: Track) -> Image.Image:
    """The track as an image: black track, white walls, red start patch."""
    rgb = np.full(track.mask.shape + (3,), 255, dtype=np.uint8)
    rgb[track.mask] = (0, 0, 0)
    rgb[track.mask & (track.dist == 0)] = (255, 0, 0)
    return Image.fromarray(rgb)


def rays_overlay(
    track: Track, origin: tuple[float, float], endpoints: np.ndarray, scale: float = 1.0
) -> Image.Image:
    """Transparent RGBA image (the size of the displayed track) with the rays drawn on it.

    Rays are `RAY_WIDTH` pixels wide but only visible on drivable pixels, so the
    width never spills over a wall.
    """
    h, w = track.mask.shape
    size = (round(w * scale), round(h * scale))
    layer = Image.new("L", size, 0)
    draw = ImageDraw.Draw(layer)
    ox, oy = origin[0] * scale, origin[1] * scale
    for ex, ey in endpoints:
        draw.line([(ox, oy), (ex * scale, ey * scale)], fill=255, width=RAY_WIDTH)
    allowed = Image.fromarray(track.mask.astype(np.uint8) * 255)
    if scale != 1.0:
        allowed = allowed.resize(size, Image.NEAREST)
    overlay = Image.new("RGBA", size, RAY_RGB + (0,))
    overlay.putalpha(ImageChops.multiply(layer, allowed))
    return overlay


@dataclass
class ViewerWindow:
    root: tk.Tk | tk.Toplevel
    canvas: tk.Canvas
    text_id: int
    n_rays: tk.IntVar
    heading: tk.IntVar


def build_window(
    track: Track, scale: float = 1.0, margin: int = MARGIN, master: tk.Misc | None = None
) -> ViewerWindow:
    """Create the window (a Toplevel of `master` if given). The caller runs `root.mainloop()`."""
    image = render(track)
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    w, h = image.size

    root = tk.Tk() if master is None else tk.Toplevel(master)
    root.title("Track viewer")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=w + margin, height=h, bg="white", highlightthickness=0)
    canvas.pack()
    root._photo = ImageTk.PhotoImage(image)  # keep references or Tk drops the images
    canvas.create_image(0, 0, image=root._photo, anchor="nw")
    canvas.create_line(w, 0, w, h, fill="#888888")

    n_rays = tk.IntVar(root, DEFAULT_RAYS)
    heading = tk.IntVar(root, 0)
    controls = tk.Frame(root, bg="white")
    tk.Label(controls, text="Rays", bg="white").grid(row=0, column=0, sticky="w", pady=2)
    tk.Spinbox(controls, from_=1, to=64, width=5, textvariable=n_rays).grid(row=0, column=1, padx=8)
    tk.Label(controls, text="Heading (°)", bg="white").grid(row=1, column=0, sticky="w", pady=2)
    tk.Spinbox(controls, from_=0, to=359, width=5, wrap=True, textvariable=heading).grid(
        row=1, column=1, padx=8)
    canvas.create_window(w + 12, 12, anchor="nw", window=controls)
    text_id = canvas.create_text(
        w + 12, 90, anchor="nw", width=margin - 24, font=("Segoe UI", 12),
        text="Click a point on the image",
    )

    selected: list[tuple[float, float]] = []  # last clicked point in track coordinates

    def refresh(*_) -> None:
        """Redraw the rays and the sidebar text for the selected point."""
        canvas.delete("rays")
        if not selected:
            return
        x, y = selected[0]
        if not track.is_drivable(x, y):
            canvas.itemconfig(text_id, text=describe_point(track, x, y))
            return
        try:
            n, deg = n_rays.get(), heading.get()  # raises while a box is empty or invalid
        except tk.TclError:
            return
        if n < 1:
            return
        d, direction = _cast(track, x, y, deg, n, MAX_RAY_LEN)
        ends = np.stack([x + d * np.cos(direction), y + d * np.sin(direction)], axis=1)
        root._rays_photo = ImageTk.PhotoImage(rays_overlay(track, (x, y), ends, scale))
        canvas.create_image(0, 0, image=root._rays_photo, anchor="nw", tags="rays")
        canvas.tag_raise("marker")
        canvas.itemconfig(text_id, text=describe_point(track, x, y, d))

    def on_click(event) -> None:
        if event.x >= w or event.y >= h:
            return  # click in the sidebar
        selected[:] = [(event.x / scale, event.y / scale)]
        canvas.delete("marker")
        canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                           outline="#00a0ff", width=2, tags="marker")
        canvas.itemconfig(text_id, text=describe_point(track, *selected[0]))
        refresh()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<B1-Motion>", on_click)  # keep updating while the left button is held down
    n_rays.trace_add("write", refresh)
    heading.trace_add("write", refresh)
    return ViewerWindow(root, canvas, text_id, n_rays, heading)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a track bmp.")
    parser.add_argument("path")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="display scale, e.g. 0.75 if the window is taller than your screen")
    args = parser.parse_args()
    build_window(Track.load(args.path), args.scale).root.mainloop()


if __name__ == "__main__":
    main()
