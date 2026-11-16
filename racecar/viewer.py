"""Click-to-inspect track viewer.

    uv run python -m racecar.viewer tests/data/racetrack1.bmp [--scale 0.75]

The window shows the track plus a margin on the right. Clicking a point shows
whether it is the start area, outside the track, or on the track (with its
distance along the track).
"""

import argparse
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from racecar.track import Track

MARGIN = 240  # width of the info strip on the right, in screen pixels


def describe_point(track: Track, x: float, y: float) -> str:
    """Text for the margin when the user clicks track coordinates (x, y)."""
    if track.is_start(x, y):
        return "Start area"
    if not track.is_drivable(x, y):
        return "Outside the track"
    return f"On the track\n{track.progress_at(x, y):.1f} px from the start"


def render(track: Track) -> Image.Image:
    """The track as an image: black track, white walls, red start patch."""
    rgb = np.full(track.mask.shape + (3,), 255, dtype=np.uint8)
    rgb[track.mask] = (0, 0, 0)
    rgb[track.mask & (track.dist == 0)] = (255, 0, 0)
    return Image.fromarray(rgb)


def build_window(track: Track, scale: float = 1.0, margin: int = MARGIN):
    """Create the window. Returns (root, canvas, text_id); the caller runs the loop."""
    image = render(track)
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    w, h = image.size

    root = tk.Tk()
    root.title("Track viewer")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=w + margin, height=h, bg="white", highlightthickness=0)
    canvas.pack()
    root._photo = ImageTk.PhotoImage(image)  # keep a reference or Tk drops the image
    canvas.create_image(0, 0, image=root._photo, anchor="nw")
    canvas.create_line(w, 0, w, h, fill="#888888")
    text_id = canvas.create_text(
        w + 12, 12, anchor="nw", width=margin - 24, font=("Segoe UI", 12),
        text="Click a point on the image",
    )

    def on_click(event):
        if event.x >= w or event.y >= h:
            return  # click in the margin
        canvas.delete("marker")
        canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                           outline="#00a0ff", width=2, tags="marker")
        canvas.itemconfig(text_id, text=describe_point(track, event.x / scale, event.y / scale))

    canvas.bind("<Button-1>", on_click)
    return root, canvas, text_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a track bmp.")
    parser.add_argument("path")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="display scale, e.g. 0.75 if the window is taller than your screen")
    args = parser.parse_args()
    root, _, _ = build_window(Track.load(args.path), args.scale)
    root.mainloop()


if __name__ == "__main__":
    main()
