# Replay viewer: design (short)

Status: awaiting approval, 2026-11-17. Size: a small change (one module, one window).

## Intent and the why

The owner wants to *see* what a trained policy does, not only read curves: a car moving on the track
view, with its rays, so that behaviour such as where it brakes, how it takes a bend and where it
crashes becomes visible. This also replaces the "viewer replay mode" step of Task 12 in the main plan.

## Scope (first version)

Replay of the **final checkpoint of a run**, as one deterministic evaluation episode (fixed start,
mean action). Nothing needs to be logged: the episode is re-simulated from the saved weights.

Non-goals for now: training episodes (sampled actions), several episodes at once, picking an
iteration from a list, GIF export, a changed collision body (collision stays a point).

Anti-goal: a viewer whose picture can differ from what the environment really did. The frames are
recorded from `RaceEnv` itself, never re-derived.

## Approaches considered

- **A. Interactive tkinter window (chosen).** Same toolkit and track rendering as `viewer.py`; play,
  pause, single step, a slider; rays and speed shown live. GIF export can later reuse the frames.
- B. Render straight to a GIF: no pausing or scrubbing, large files for a 500-frame lap.
- C. A static trajectory picture: cheap, but cannot show rays or timing.

## Design

**`racecar/replay.py`**

- `record_episode(env, policy) -> Recording`: runs one episode with `policy.mean_action` and stores,
  per frame (frame 0 = the start state, then one per step): `x, y, heading, speed`, the ray
  distances in px (from the observation, all 0 on the crash frame), and per step the action and
  reward; plus `crashed`, `finished`, and cumulative progress. Pure data, no tkinter, fully testable.
- `car_polygon(x, y, heading, length=16, width=8) -> list[tuple]`: the corners of the car rectangle
  for drawing (display only; the physics car is still a point at its centre).
- `ReplayWindow`: the track image (reusing `viewer.render`), the trail driven so far, the car
  rectangle, and the rays as yellow lines (as in `viewer.py`).
- A **control bar below the track**, next to the existing right-hand sidebar idea: Restart, step
  back, Play/Pause, step forward, a slider over the frames, a speed choice (1x = real time, 10
  frames per second, since `dt` is 0.1 s; also 0.25x, 0.5x, 2x, 4x). The sidebar shows: frame,
  x/y, speed, last action, last reward, progress and, on the last frame, "crashed" or "finished".
- Playback stops on the last frame; Play there restarts from the beginning.
- CLI: `uv run python -m racecar.replay runs/reinforce-0 [--track PATH] [--checkpoint N] [--scale S]`.
  It loads the highest-numbered checkpoint in `run_dir/checkpoints/` unless `--checkpoint` names one.
  The track is not stored in checkpoints, so `--track` (default `tests/data/racetrack1.bmp`) must
  match the run; the step limit comes from the checkpoint's saved config.

## My calls (for the owner to check)

- The car is drawn as a 16 x 8 px rectangle; rays use the 9-ray, 300 px cap the environment uses.
- 1x playback speed equals real time.
- The window code is written by Claude (viewer plumbing, not RL); the recording logic is separate
  from the window so it can be tested without a display.

## Tests

- `record_episode`: deterministic (two recordings equal); frame count = steps + 1; the recorded
  return matches `evaluate()` on the same policy; the first frame is the start state; a policy that
  drives straight ends `finished`, one that steers hard ends `crashed` with zero rays on the last frame.
- `car_polygon`: four corners, centred on the car, rotated with the heading.
- Window (shared hidden Tk root as in `tests/test_viewer.py`): stepping forward and back, clamping at
  both ends, Play/Pause state, restart after the end, and the slider showing the right frame.
  Playback timing itself is not tested.

## Left to the builder

Colours, fonts and exact widget layout; the trail style; how the speed choice is drawn.
