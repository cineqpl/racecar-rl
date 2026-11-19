# Textured Replay View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The replay window shows grass and asphalt textures and the car sprite instead of the black/white map and the red rectangle.

**Architecture:** A new Pillow/numpy module `racecar/texture.py` prerenders one background image (tiled grass, tiled asphalt pasted through the track mask, translucent start/finish tints) and prepares rotated car sprites. `ReplayWindow` shows that image and a canvas image for the car. Physics, rays, training and `racecar/viewer.py` are untouched.

**Tech Stack:** Pillow, numpy, tkinter, pytest. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-11-19-textured-replay-design.md`

## Global Constraints

- Display only: no change to `env.py`, `car.py`, `track.py`, rays or training.
- `car.png` points right (heading 0); sprite scaled to length 20 px (35x18 source), aspect ratio kept.
- Heading convention: radians, clockwise on screen (y down), as `Recording.heading`.
- Missing asset files fall back to `racecar.viewer.render(track)` and a plain polygon car; the window must still open.
- Commit straight to `main`; commit messages end with the `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` line.

## Review Focus

- Asset folder missing or a tile file absent: window still opens (plain rendering). Pinned in Task 1 (`render_textured`) and Task 2 (window).
- Track image not a multiple of the tile size (1536x1024 vs 300 px tiles): output size equals the track's exactly. Task 1.
- `scale != 1.0`: background and sprite shrink together and the car stays centred on the scaled position. Task 2.
- Heading outside [-pi, pi] or negative (the car turns many times): rotation and the cache key must still work. Task 1 and 2.
- Car sprite at the image border: pasting/drawing near an edge must not raise (canvas images clip on their own). Covered by the existing window tests running with the sprite.

---

### Task 1: `racecar/texture.py`

**Files:**
- Create: `racecar/texture.py`
- Test: `tests/test_texture.py`

**Interfaces:**
- Consumes: `Track` (`mask`, `dist`, `finish` boolean/float arrays, shape (H, W)); `racecar.viewer.render(track) -> Image` for the fallback.
- Produces:
  - `ASSETS_DIR: Path` = `racecar/assets`
  - `render_textured(track: Track, assets_dir: Path | None = None) -> Image.Image` (RGB, size `(W, H)` of the track; `None` means `ASSETS_DIR`, read at call time)
  - `load_car_sprite(assets_dir: Path | None = None, length: int = 20) -> Image.Image | None` (RGBA, width `length`, height `round(length * 18 / 35)`; `None` if `car.png` is missing)
  - `rotate_sprite(sprite: Image.Image, heading: float) -> Image.Image` (RGBA, expanded to fit, nose along `heading`)

- [ ] **Step 1: Write the failing tests** in `tests/test_texture.py`, using `corridor_track` (conftest) and a `tmp_path` assets folder holding synthetic tiles: `asphalt.bmp` solid (80, 80, 80) 16x16, `grass.bmp` solid (0, 150, 0) 16x16, `car.png` a 35x18 RGBA, red on the right half and blue on the left half, fully opaque.
  - `test_render_textured_size_matches_track`: `render_textured(corridor_track, assets).size == (W, H)` where `W, H = corridor_track.mask.shape[::-1]`.
  - `test_asphalt_on_track_grass_outside`: pixel at a track point away from the start patch (e.g. `(75, 20)`) is (80, 80, 80); pixel outside the corridor (e.g. `(75, 2)`) is (0, 150, 0).
  - `test_start_and_finish_are_tinted`: a start-patch pixel has red > green and blue (`px[0] > px[1]` and `px[0] > px[2]`); a finish-area pixel (pick one from `np.argwhere(track.finish)`) has green > red and blue; neither equals the plain asphalt colour.
  - `test_missing_assets_fall_back_to_plain_render`: with an empty folder, the result equals `viewer.render(corridor_track)` pixel for pixel (`ImageChops.difference(...).getbbox() is None`).
  - `test_load_car_sprite_size_and_missing`: sprite size is `(20, 10)` for `length=20`; empty folder returns `None`.
  - `test_rotate_sprite_nose_follows_heading`: heading 0 keeps red (nose, right half) on the right of the centre; heading `pi/2` (clockwise on screen, nose down) puts red below the centre; heading `-3*pi/2` gives the same pixels as `pi/2` (`ImageChops.difference(...).getbbox() is None` after rotating both). Compare mean red column/row of opaque red pixels against the image centre.
- [ ] **Step 2: Run** `uv run pytest tests/test_texture.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement the module.** Tile with a numpy `np.tile` of the tile array cropped to `(H, W)` for each texture; composite with `Image.composite(asphalt, grass, mask_image)` where `mask_image` is the `track.mask` as an "L" image (255 on track); tint start (`mask & dist == 0`) and finish (`track.finish`) by blending the tile colour with red/green at alpha ~0.45 (builder's call on exact alpha). `render_textured` catches a missing `asphalt.bmp` or `grass.bmp` (`FileNotFoundError`) and returns `viewer.render(track)`. `rotate_sprite`: PIL rotates counter-clockwise by degrees, so use `sprite.rotate(-math.degrees(heading), expand=True, resample=Image.BICUBIC)`.
- [ ] **Step 4: Run** `uv run pytest tests/test_texture.py -v`. Expected: all PASS.
- [ ] **Step 5: Commit**

```bash
git add racecar/texture.py tests/test_texture.py racecar/assets docs/superpowers/specs/2026-11-19-textured-replay-design.md docs/superpowers/plans/2026-11-19-textured-replay.md
git commit -m "feat: texture rendering and car sprite helpers"
```

### Task 2: Use the textures in `ReplayWindow`

**Files:**
- Modify: `racecar/replay.py` (imports; `ReplayWindow.__init__` lines ~127-142; `ReplayWindow.show` lines ~170-171)
- Test: `tests/test_replay_window.py`

**Interfaces:**
- Consumes: `render_textured`, `load_car_sprite`, `rotate_sprite` from Task 1.
- Produces: `ReplayWindow` keeps its public surface (`show`, `step`, `toggle_play`, `tick`, `restart`, `set_speed`, `frame`, `canvas`, `slider`, `root`). Adds `ReplayWindow._sprite_for(heading: float) -> ImageTk.PhotoImage | None`, cached by `round(heading_degrees / 2) % 180` (a 2-degree key; rotate by the rounded angle), `None` when there is no sprite.

- [ ] **Step 1: Write the failing tests** in `tests/test_replay_window.py` (reuse its `track` fixture and `tk_master`):
  - `test_car_is_an_image_not_a_polygon`: after construction and `show(5)`, `canvas.find_withtag("car")` has one item and `canvas.type(...)` is `"image"`.
  - `test_car_image_is_centred_on_the_scaled_position`: with `scale=0.5`, `canvas.coords("car")` equals `(x * 0.5, y * 0.5)` of the shown frame (within 1e-6).
  - `test_sprite_cache_reuses_images_for_nearby_headings`: `_sprite_for(0.0) is _sprite_for(0.01)`, and `_sprite_for(0.0) is not _sprite_for(1.0)`; negative and above-2pi headings (`-7.0`, `20.0`) return an image without raising.
  - `test_window_opens_without_assets`: monkeypatch `racecar.replay.ASSETS_DIR`-equivalent (the assets dir argument the window passes; if the window calls the helpers with defaults, monkeypatch `racecar.texture.ASSETS_DIR` to an empty `tmp_path`) and check the window constructs, `show(3)` works, and the car item is a polygon.
- [ ] **Step 2: Run** `uv run pytest tests/test_replay_window.py -v`. Expected: the four new tests FAIL, existing ones still pass.
- [ ] **Step 3: Implement.** In `__init__`: build the background with `render_textured(track)` (replacing `render(track)`), load the sprite once with `load_car_sprite()` scaled by `scale` (resize it by `scale` before caching), and create the car item as `canvas.create_image(0, 0, tags="car")` when a sprite exists, else the existing polygon. In `show`: if the car item is an image, `itemconfig("car", image=self._sprite_for(rec.heading[i]))` and `coords("car", x * s, y * s)` (the image anchor defaults to centre); else keep the polygon branch. The helpers resolve `ASSETS_DIR` at call time (Task 1), so monkeypatching `racecar.texture.ASSETS_DIR` works. Keep PhotoImage references on `self` so Tk does not drop them.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all tests PASS (177 existing plus the new ones).
- [ ] **Step 5: Look at it.** Make a run if none exists (`uv run python scripts/train.py --algo baseline --iterations 20 --seed 0`, check `--help` for the real flags) and run `uv run python -m racecar.replay runs/<run>`. Confirm the textures tile without a visible seam, the car faces its direction of travel and the start/finish tints are visible. If the asphalt seam shows, add an edge cross-fade to the tile in `texture.py` (separate commit).
- [ ] **Step 6: Commit** `feat: textured replay view with car sprite`.
