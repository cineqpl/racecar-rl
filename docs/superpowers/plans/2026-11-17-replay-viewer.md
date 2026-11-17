# Replay Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the final checkpoint of a run as one deterministic episode in a tkinter window: the car (a small rectangle), its trail and its rays on the track, with play / pause / step controls in a bar below the track.

**Architecture:** `racecar/replay.py` holds a display-free part (`Recording`, `record_episode`, `car_polygon`, `load_policy`) and the tkinter `ReplayWindow`. The recording is taken from `RaceEnv` itself, so the picture cannot disagree with what the environment did. The existing `viewer.build_window` is left untouched; only `viewer.render` is reused.

**Tech Stack:** Python 3.12, tkinter, Pillow (via `viewer.render`), PyTorch (loading weights), pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-replay-viewer-design.md`

## Global Constraints

- Run Python only through `uv run` (`uv run pytest`, `uv run python -m racecar.replay ...`).
- Conventions: image coordinates (x = column, y = row, y down); heading in radians, 0 = +x, positive turns toward +y; distances in px.
- Replay is one deterministic episode: fixed start, `policy.mean_action`, no sampling.
- Collision stays a point; the 16 x 8 px car rectangle is display only.
- Tests that need Tk use the shared `tk_master` fixture in `tests/conftest.py` (one hidden root, windows are `Toplevel`s).
- Commit straight to `main`. Every commit message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A run directory without checkpoints, or `--checkpoint N` that does not exist, gives a clear error naming the directory, not a bare traceback. (Task 1: `load_policy`)
2. An episode that ends on its first step (2 frames) still shows and steps correctly. (Task 2)
3. The crash frame has all rays at 0: nothing is drawn and nothing raises. (Task 2)
4. A display `scale` other than 1 scales the car, trail and rays together with the track. (Task 2)
5. Closing the window while playing leaves no pending timer callback that fires into a destroyed window. (Task 2)

---

### Task 1: Recording, car shape, loading a run

**Files:**
- Modify: `racecar/env.py` (rename `self._angles` to the public `self.ray_angles`, 3 places)
- Create: `racecar/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `RaceEnv` (`reset`, `step`, `state`, `max_ray_len`, `ray_angles`, `track`), `GaussianPolicy.mean_action`, `load_checkpoint`.
- Produces, in `replay.py`:
  - `@dataclass Recording`: `x, y, heading, speed, progress: np.ndarray (F,)`; `rays: np.ndarray (F, n_rays)` in px; `ray_angles: np.ndarray (n_rays,)`; `actions: np.ndarray (F - 1, 2)` (as the env received them, clipped to [-1, 1]); `rewards: np.ndarray (F - 1,)`; `crashed: bool`; `finished: bool`. `F = steps + 1` frames; frame 0 is the start state. Property `n_frames -> int`. Method `ray_ends(i: int) -> np.ndarray (n_rays, 2)`: where each ray of frame `i` ends, `(x + d cos(heading + angle), y + d sin(heading + angle))`.
  - `record_episode(env: RaceEnv, policy: GaussianPolicy) -> Recording`.
  - `car_polygon(x: float, y: float, heading: float, length: float = 16.0, width: float = 8.0) -> list[tuple[float, float]]`: four corners of the rectangle centred on (x, y), its long side along the heading.
  - `load_policy(run_dir: Path, obs_dim: int, checkpoint: int | None = None) -> tuple[GaussianPolicy, dict]`: the policy and the checkpoint's saved `config`; the highest-numbered `run_dir/checkpoints/ckpt_*.pt` unless `checkpoint` names an iteration (`ckpt_{N:06d}.pt`).

- [ ] **Step 1: Write the failing tests** in `tests/test_replay.py`, on a 400 px corridor like `tests/test_env.py` (a `driving_policy` with mean (1, 0) and a `crashing_policy` with mean (1, 1), built as in `tests/test_evaluation.py`):
  - `test_recording_is_deterministic`: two recordings of the same policy have equal arrays.
  - `test_frame_counts`: `n_frames == len(rewards) + 1 == len(actions) + 1`; every per-frame array has `n_frames` rows.
  - `test_first_frame_is_the_start_state`: `(x[0], y[0], heading[0], speed[0]) == (*track.start_pos, track.start_heading, 0.0)`, `progress[0] == 0`.
  - `test_return_matches_evaluate`: `rewards.sum()` equals `evaluate(env, policy).mean_return` (`pytest.approx`).
  - `test_driving_policy_finishes`: `finished and not crashed`; `crashing_policy`: `crashed and not finished`, `rays[-1]` all 0, `rays[0]` not all 0.
  - `test_rays_are_in_px`: at frame 0, `rays[0]` equals `cast_rays(track, x0, y0, heading0, env.ray_angles, env.max_ray_len)` (`pytest.approx`), and `rays.max() <= env.max_ray_len`.
  - `test_ray_ends_straight_ahead`: with heading 0 the middle ray's end is `(x + d, y)`; with the leftmost ray (angle -90 deg) it points toward -y.
  - `test_car_polygon_is_centred_and_rotates`: heading 0: corners within x ± 8, y ± 4, mean of corners `== (x, y)`; heading pi/2: within x ± 4, y ± 8.
  - `test_load_policy_round_trip`: a policy saved with `save_checkpoint(tmp, policy, iteration=3, seed=1, config={"max_steps": 99})` loads with equal outputs and returns that config; with checkpoints at iterations 3 and 10 the default loads 10 and `checkpoint=3` loads 3.
  - `test_load_policy_errors_name_the_directory` (Review Focus 1): no `checkpoints/` folder, or `checkpoint=7` missing, raises `FileNotFoundError` whose message contains the run directory.
- [ ] **Step 2:** Run `uv run pytest tests/test_replay.py -q` — Expected: FAIL (module `racecar.replay` missing).
- [ ] **Step 3:** Rename `_angles` to `ray_angles` in `racecar/env.py`; implement the items above in `racecar/replay.py`. `record_episode` reads the ray distances of each frame from the observation (`obs[:-1] * env.max_ray_len`) and the pose from `env.state` after each step.
- [ ] **Step 4:** Run `uv run pytest -q` — Expected: whole suite PASS.
- [ ] **Step 5: Commit** `feat: episode recording and run loading for the replay viewer`.

### Task 2: Replay window and CLI

**Files:**
- Modify: `racecar/replay.py`
- Modify: `docs/superpowers/plans/2026-11-16-racecar-rl.md` (Task 12, Step 3: replace its text with "Done in `docs/superpowers/plans/2026-11-17-replay-viewer.md`.")
- Test: `tests/test_replay_window.py`

**Interfaces:**
- Consumes: `Recording`, `car_polygon`, `load_policy`, `viewer.render(track) -> PIL.Image`, `RaceEnv`, `Track.load`.
- Produces, in `replay.py`:
  - `class ReplayWindow` built by `ReplayWindow(track: Track, recording: Recording, dt: float = 0.1, scale: float = 1.0, master: tk.Misc | None = None)` (a `Toplevel` of `master` if given; the caller runs `window.root.mainloop()`). Attributes: `root`, `canvas`, `frame: int` (current frame, 0-based), `playing: bool`, `speed: float` (default 1.0), `delay_ms: int` (`round(1000 * dt / speed)`). Methods: `show(i: int) -> None` (clamped to `[0, n_frames - 1]`, redraws, updates the slider and the sidebar), `step(delta: int) -> None`, `toggle_play() -> None` (at the last frame, starts again from frame 0), `restart() -> None` (frame 0, paused), `set_speed(factor: float) -> None`, `tick() -> None` (advance one frame and schedule the next while playing; stop at the last frame).
  - Canvas items by tag: `"trail"` (polyline of the positions up to the current frame), `"car"` (polygon), `"ray"` (one line per ray with length > 0). The layout is the track on the left, a text sidebar on the right (frame `i / n-1`, x/y, speed, last action, last reward, progress; on the last frame "crashed" or "finished"), and a control bar below: Restart, step back, Play/Pause, step forward, a slider over the frames, a speed spinbox with the values 0.25, 0.5, 1, 2, 4.
  - `main()` for `uv run python -m racecar.replay RUN_DIR [--track PATH] [--checkpoint N] [--scale S]`: builds the track and an env with `max_steps` from the checkpoint config, loads the policy, records, and opens the window. `--track` defaults to `tests/data/racetrack1.bmp`; `--scale` to 1.0.

- [ ] **Step 1: Write the failing tests** in `tests/test_replay_window.py` (corridor track, recordings from `record_episode`, the `tk_master` fixture, a `win` fixture that builds a `ReplayWindow(master=tk_master)` and destroys it afterwards). Nothing waits on real timers: tests call `tick()` directly.
  - `test_starts_paused_on_frame_zero`: `frame == 0`, `not playing`; the car polygon sits at the start position (canvas coords of tag `"car"` are the corners from `car_polygon`).
  - `test_step_forward_and_back_clamp_at_both_ends`: `step(-1)` at 0 stays 0; many `step(+1)` end at `n_frames - 1` and stay there.
  - `test_show_updates_slider_and_sidebar`: `show(5)` sets `frame == 5`, the slider value 5, and the sidebar text contains `5`.
  - `test_trail_grows_with_the_frame`: the trail has `frame + 1` points after `show(i)`; after `show(0)` the trail does not raise (one point).
  - `test_play_advances_and_stops_at_the_end`: `toggle_play()` sets `playing`; repeated `tick()` reaches the last frame, then `playing` is false.
  - `test_play_at_the_end_restarts`: at the last frame `toggle_play()` gives `frame == 0` and `playing`.
  - `test_pause_stops_ticking`: `toggle_play()` twice leaves `playing` false and `tick()` does not advance.
  - `test_restart_goes_to_frame_zero_paused`.
  - `test_speed_changes_the_delay`: `set_speed(2.0)` gives `delay_ms == 50` with `dt = 0.1`; `set_speed(0.5)` gives 200.
  - `test_last_frame_reports_the_outcome`: crashing recording ends with "crashed" in the sidebar, driving recording with "finished".
  - `test_crash_frame_draws_no_rays` (Review Focus 3): on the crash frame there are no `"ray"` items and no exception; on frame 0 there are `n_rays` of them.
  - `test_two_frame_recording_works` (Review Focus 2): a recording that ends on its first step builds, steps forward once, and stops.
  - `test_display_scale_scales_the_drawing` (Review Focus 4): with `scale=0.5` the car polygon's canvas coordinates are half those at scale 1, and the canvas is half as wide as the track image plus the sidebar.
  - `test_closing_while_playing_cancels_the_timer` (Review Focus 5): after `toggle_play()` and `root.destroy()`, the window's pending-timer handle is cleared (no callback left to fire).
  - `test_main_arguments_are_parsed`: `parse_args(["runs/x"])` gives the default track and scale 1.0; `--checkpoint 5` gives `5` (factor the argument parser into `parse_args(argv)`).
- [ ] **Step 2:** Run `uv run pytest tests/test_replay_window.py -q` — Expected: FAIL (`ReplayWindow` missing).
- [ ] **Step 3:** Implement `ReplayWindow`, `parse_args` and `main`. Rays are drawn as plain canvas lines (2 px, yellow, from the car centre to `ray_ends(i)`), not with `viewer.rays_overlay`: they end at the first wall sample, so no clipping is needed, and a full-size image per frame would make playback slow. Use `root.after` for playback and `after_cancel` on pause and on `<Destroy>`.
- [ ] **Step 4:** Run `uv run pytest -q` — Expected: whole suite PASS. Then, with the run from `scripts/train.py` in `runs/reinforce-0`, run `uv run python -m racecar.replay runs/reinforce-0` — Expected: a window opens, Play drives the car along the track with its rays, and the last frame says "finished" (or "crashed" with no rays). Report what you see; this step is not automated.
- [ ] **Step 5: Commit** `feat: replay viewer window and CLI`.
