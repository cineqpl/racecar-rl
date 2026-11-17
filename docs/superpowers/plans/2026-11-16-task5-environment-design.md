# Task 5: Environment, detailed design (draft, awaiting confirmation)

Status: discussed 2026-11-16, updated 2026-11-17 (all decisions confirmed); **built 2026-11-17** (`Track.sweep` in `racecar/track.py`, `RaceEnv` in `racecar/env.py`, tests in `tests/test_sweep.py` and `tests/test_env.py`). Replaces the stacked "REVISED" notes on Task 5 in `2026-11-16-racecar-rl.md` once confirmed.

Already built: `Track` (bmp loading, geodesic distance, `finish` mask), `cast_rays`, `step_car` / `CarParams` / `CarState`, the viewer.

## 1. `Track.sweep`: what does a step run into?

`Track.sweep(old_pts, new_pts, step=1.0) -> tuple[str, float] | None`

- `old_pts` and `new_pts` are `(N, 2)` arrays: the same N body points of the car, in world coordinates, before and after the step. A point car is the special case N = 1. Each point's straight path is sampled every `step` px (the end point is always sampled, the start point is not), and the earliest hit over all points decides the result.
- The car's body is a `footprint` (an `(N, 2)` array of body-frame points, default `[[0, 0]]`), turned into world points by rotating with the heading and translating. A future convex-hull car supplies its vertices plus points densified along its edges (vertices alone can miss a thin wall poking into the interior).
- Returns `("wall", fraction)` or `("finish", fraction)` for the **first** wall or finish pixel hit, where `fraction` in (0, 1] is how far along the step it happened, or `None` if the whole path is clear.
- Off-image counts as wall. Whichever comes first, wall or finish, decides the outcome. A 10 px step (top speed 100 px/s, `dt` 0.1) cannot jump a thin wall.

## 2. `RaceEnv`

`RaceEnv(track, n_rays=9, ray_fov_deg=180.0, max_ray_len=300.0, max_steps=1500, dt=0.1, progress_scale=0.1, time_penalty=0.01, footprint=[[0, 0]], params=CarParams())`

Attributes: `obs_dim` (= `n_rays + 1` = 10), `act_dim` (= 2), `track`, `params`, `state`. `time_penalty` (subtracted every step, default 0.01 = 1% of the top-speed progress reward) only exists to stop the car idling; it breaks even at about 1 px/s.

**Observation** (float32, all in [0, 1]): the ray distances divided by `max_ray_len`, ordered left to right as seen from the car (rays spread evenly over ±90° around the heading), then speed divided by `max_speed`.

**`reset() -> obs`**: car at the centre of the start patch, with the track's derived heading and speed 0; step count and progress cleared. Fully deterministic (fixed start).

**`step(action) -> (obs, reward, terminated, truncated, info)`**

1. Clip the action to [-1, 1] (throttle/brake, steering). Anything not of shape `(2,)` raises `ValueError`.
2. Advance the car with `step_car`.
3. `track.sweep(old position, new position)`.
4. Resolve the outcome:

| Outcome | terminated | reward | notes |
|---|---|---|---|
| Path clear | no | `Δprogress * progress_scale - time_penalty` | plain difference of `progress_at`; backward driving is negative |
| Hits a wall first | yes `-time_penalty * (max_steps - steps_before)` | observation rays all 0; `info["crashed"] = True` |
| Reaches finish first | yes | progress up to that point * scale - time_penalty | `info["finished"] = True` |
| Step count reaches `max_steps` | no, `truncated = True` | as for a clear path | a time-out, not a terminal state |

`info`: `progress` (cumulative, in px), `crashed`, `finished`, `x`, `y`, `speed`, `steps`. `step` before `reset`, or after the episode ended, raises `RuntimeError`.

**`set_state(x, y, heading, speed)`**: for tests and debugging; also resets the progress reference.

## 3. Tests (all with known answers)

`Track.sweep`, on the corridor: clear path returns `None`; wall found at the right fraction; finish found at the right fraction; wall before finish wins and vice versa; off-image counts as wall.

`RaceEnv`, on a 400 px corridor (`corridor_image(length=400)`; finish = last 60 px):

- Observation shape `(10,)`, float32, all in [0, 1]; speed component 0 after reset.
- Ray order: `set_state(200, 20, pi/2, 0)` (facing down): leftmost ray about 0.75, middle about 0.033, rightmost about 0.6 (within 0.01).
- Action `[5, 0]` behaves exactly like `[1, 0]`; `[0, -9]` like `[0, -1]`; wrong shape raises `ValueError`.
- Full throttle straight: `(sum(reward) + steps * time_penalty) / progress_scale == info["progress"]`, and `abs(progress - (x - 24)) < 1.5` once past the start patch.
- Step before reset and step after done raise `RuntimeError`.
- `max_steps=5`, brake only: step 5 is `truncated and not terminated`.
- Crash: throttle 1 with steering 1 terminates within 200 steps, `crashed` set, reward on that step as decided in section 4, rays 0 in the observation.
- Time penalty: brake-only for `max_steps=5` gives total reward `-5 * time_penalty`.
- Footprint: a 3-point footprint (nose and two rear corners) crashes when a rear corner reaches a wall that the centre would miss.
- Finish: straight full throttle ends with `finished`, progress at least 340 px, no crash.
- Backward driving: `set_state(200, 20, pi, 20)` gives negative reward.
- No tunnelling: a 3 px wall slab across most of the corridor (leave 2 rows open so the track stays connected), hit at speed 100 from 5 px away, is a crash, not a jump over it.
- Real track: `set_state` just before the end wall of `racetrack1.bmp` at speed 100 gives `finished`, and the car never ends up in the start patch.
- Determinism: two environments given the same actions return identical observations.

## 4. Decisions

Confirmed 2026-11-17:
- Start is fixed and deterministic. Later option: a `start_jitter` for training only, evaluation stays fixed.
- The car is a point for now, but collision is written against a set of body points (`footprint`), so a convex-hull car is a later change of data only. Rendering it as a small rectangle comes with the viewer work.
- A minimal time penalty (0.01 per step) so the car does not idle.

Also confirmed 2026-11-17:
- Crash step reward is `-time_penalty * (max_steps - steps_before)`, the penalty forfeited by ending early, so a crash is never better than idling (test: `test_crash_is_never_better_than_idling`).
- Reward scale 0.1 and ray cap 300 px.

## 5. Other things we agreed to do or raised

- Rewrite Tasks 5-12 of `2026-11-16-racecar-rl.md` cleanly (the "REVISED" notes stacked on old text are confusing).
- Install PyTorch at Task 6; GPU design (batched environment on the CUDA GPU) deferred until REINFORCE works on the CPU. Keep the environment's physics as plain functions of state so a batched port stays easy. The default Windows PyPI wheel of PyTorch is CPU-only; a CUDA build needs PyTorch's own package index.
- Visualization: drop matplotlib; draw trajectories and replays with Pillow on top of the viewer's track image; learning curves via TensorBoard. Add a replay mode to the viewer (load a checkpoint, watch the car drive with its rays).
- Training-speed risk: roughly 0.1 ms per environment step, so a REINFORCE iteration of 16 episodes may take a few seconds. Start with a shorter step limit and fewer episodes.
- Evaluation also reports a finish rate (from `info["finished"]`).
