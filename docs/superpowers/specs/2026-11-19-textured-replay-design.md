# Textured replay view: design

Approved by the user on 2026-11-19 (short design, small change).

## Intent
The replay window should look good: grass outside the track, asphalt on it, and the car
sprite instead of the red rectangle. Display only; physics, rays and training do not change.

## Goals
- Grass tiled over the whole image, asphalt tiled on the drivable area, composed by alpha
  using the track mask (and its inverse) as the mask. The result is one prerendered image.
- Start patch and finish area keep a translucent red / green tint so they stay recognisable.
- The car is `racecar/assets/car.png` (points right = heading 0), scaled to about 20x10 px,
  rotated to the heading, centred on (x, y).

## Non-goals
- The click-to-inspect viewer (`racecar/viewer.py`) keeps its plain rendering.
- No physics change: the car is still a point for collisions; the sprite is cosmetic.
- No seam-blending of the tiles unless the real render shows a visible seam (the tiles were
  measured: grass wraps cleanly, asphalt is slightly rougher at the edge).

## Approaches
1. **Prerender one background image** (chosen): Pillow/numpy composite once at window
   creation; the canvas shows a single image as it does now. Cheap redraws.
2. Draw tiles as individual canvas items: thousands of objects, slow redraws. Rejected.

## Design
- New `racecar/texture.py` (Pillow + numpy, no tkinter):
  - `render_textured(track, assets_dir=ASSETS_DIR) -> Image`: tile grass, tile asphalt, paste
    asphalt where `track.mask`, tint start (`mask & dist == 0`) and finish (`track.finish`).
    Falls back to `viewer.render(track)` if the asset files are missing.
  - `load_car_sprite(assets_dir=ASSETS_DIR, length=20) -> Image` (RGBA, aspect ratio kept).
  - `rotate_sprite(sprite, heading) -> Image`: rotated so the nose points along `heading`
    (radians, image convention: clockwise on screen), expanded to fit, transparent corners.
- `replay.py`: `ReplayWindow` uses `render_textured` and a canvas image for the car, with
  rotated sprites cached by heading rounded to 2 degrees. Display `scale` applies to the
  background and the sprite. Trail, rays, sidebar and controls are unchanged.
  `car_polygon` stays (still tested) but is no longer drawn.

## Left to the builder
Exact tint alpha, the cache size, and whether to blend tile seams after looking at a render.
