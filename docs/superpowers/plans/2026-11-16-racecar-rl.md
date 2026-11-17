# Racecar RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch RL pipeline (bmp track environment, REINFORCE → baseline → GAE → PPO, evaluation, logging, visualization) in which a simulated car learns to drive along a track.

**Architecture:** A NumPy environment (`Track` from a bmp, bicycle-model car, ray casting, per-step progress reward from a geodesic distance field) with a `reset()/step()` shape. A PyTorch Gaussian policy and value net. Each algorithm is a `Learner` with `.policy` and `.update(episodes) -> metrics`, all driven by one shared `train()` loop so runs are comparable on the same track and seed.

**Tech Stack:** Python 3.12, uv, NumPy, Pillow, PyTorch, TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-racecar-rl-design.md`

## Global Constraints

- Python `>=3.12`, managed with `uv`; Windows 11, VS Code. Always run Python through `uv run` (e.g. `uv run python ...`, `uv run pytest`); never call bare `python`, it is not on the PATH. The project venv is Python 3.12.
- No RL framework (no Gymnasium, Stable-Baselines3, CleanRL). PyTorch only for networks, autograd, optimizers.
- Allowed libraries: PyTorch, NumPy, Pillow (bmp and drawing), TensorBoard (logging only), pytest.
- Real code in plain `.py` modules in the package `racecar/`; notebooks are not part of this plan.
- Observation: N ray distances plus speed. Action: continuous 2-vector in [-1, 1]² (throttle/brake, steering), clipped by the environment.
- Reward: per-step progress from a geodesic distance field, minus a small time penalty; a crash ends the episode and pays the forfeited time penalty (see the Task 5 design); time-out is truncation, not terminal; single track first.
- Evaluation uses the policy's mean action, fixed start, no sampling noise.
- Conventions: image coordinates (x = column, y = row, y down); heading in radians, 0 = +x, positive turns toward +y. Distances in pixels, time in seconds.
- Every commit message ends with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A bmp with no start line (or a start line that cuts nothing) → `ValueError` with a clear message, not a crash later. (Task 2)
2. Positions or rays outside the image → treated as wall, never an `IndexError`. (Tasks 2, 3)
3. Crossing the start line on a loop gives no ±track-length reward spike; driving backward gives negative reward. (Tasks 2, 5)
4. Bmps with imperfect colours (grey 250 vs 255, palette-mode bmp) still classify correctly. (Task 2)
5. Time-out truncation bootstraps from V(last_obs), while a crash bootstraps from 0. (Task 10)

---

### Task 1: Project setup

**Files:**
- Modify: `pyproject.toml`
- Create: `racecar/__init__.py`, `tests/__init__.py`, `.gitignore`
- Delete: `main.py`

**Interfaces:**
- Produces: importable package `racecar`; `uv run pytest` works.

- [ ] **Step 1:** `uv add numpy pillow torch matplotlib tensorboard` and `uv add --dev pytest`.
- [ ] **Step 2:** In `pyproject.toml` add `[tool.pytest.ini_options]` with `pythonpath = ["."]` and `testpaths = ["tests"]`. Create the empty package and test package. `.gitignore`: `.venv/`, `__pycache__/`, `runs/`, `*.pyc`. Remove `main.py` (the placeholder).
- [ ] **Step 3:** Run `uv run python -c "import racecar, torch, numpy, PIL, matplotlib"` — Expected: no output, exit 0. Run `uv run pytest` — Expected: "no tests ran".
- [ ] **Step 4: Commit** (`pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, package, docs) with message `chore: project setup`.

### Task 2: Track (bmp loading and geodesic distance field)

> **REVISED 2026-11-16 (done; see `racecar/track.py`, `tests/test_track.py`).** Tracks are now **black = track, white = wall, red = start patch** at one end of a **non-looping** track (the white area must be one connected piece; the image border counts as wall). Sizes 1200–1600 × 800–1200. Colours are validated by blend-snapping (anti-aliasing allowed, foreign colours rejected, at most 2% of pixels outside the strict ranges). The start heading is derived from the distance field, so there is no `start_heading_deg` parameter, no green start line, no ring tracks and no `progress_delta` wrap. `Track.load(path, min_size, max_size, max_snapped_fraction)` raises `TrackValidationError`. The synthetic helper is `track_gen.corridor_image(length, width, scale)`. **Tasks 3, 5 and 8 must be read with these changes:** drop the ring-track and loop-closure tests (Task 5 progress delta is a plain difference), use `corridor_image`-based tracks in tests, and drop `--start-heading-deg` and the ring from the CLI (default track `tests/data/racetrack1.bmp` or a copy in `tracks/`). The text below is the original, superseded Task 2.

**Files:**
- Create: `racecar/track.py`, `racecar/track_gen.py`
- Test: `tests/test_track.py`

**Interfaces:**
- Produces, in `track_gen.py`: `make_corridor(path, length=100, width=20) -> None` (writes a 140×40 bmp: white drivable rectangle x∈[20,120), y∈[10,30), black walls, green start line = column x=20 over the drivable rows; image size = `(length+40, width+20)`) and `make_ring(path, r_in=40, r_out=60, size=160) -> None` (white annulus centred at (size/2, size/2); green start line on the row y=size/2 across the ring on the +x side).
- Produces, in `track.py`: `class Track` with attributes `mask: np.ndarray[bool]` (H,W, drivable incl. start line), `dist: np.ndarray[float32]` (H,W, distance along track from the start line, 0 at the start line, `inf` on walls), `length: float` (max finite dist), `start_pos: tuple[float, float]` (centre of start line), `start_heading: float`, and `Track.load(path: str | Path, start_heading_deg: float) -> Track`, `is_drivable(x: float, y: float) -> bool` (False outside the image), `progress_at(x: float, y: float) -> float`, `progress_delta(prev: float, cur: float) -> float` (wraps by `length` when |delta| > length/2).
- Colour classification: green = start line (g high, r and b low), else drivable if mean brightness > 128, else wall; convert image with `.convert("RGB")` first.
- Distance field: 8-connected Dijkstra (step costs 1 and √2), with start-line pixels acting as wall except as seeds. Seeds are the drivable non-line pixels adjacent to the line on the side the heading points to. Line pixels themselves get dist 0.

- [ ] **Step 1: Write failing tests** in `tests/test_track.py` (use `tmp_path`):
  - `test_classifies_pixels`: corridor track: `is_drivable(70,20)` True, `is_drivable(70,2)` False, `is_drivable(20,20)` True (line).
  - `test_distance_corridor`: `abs(progress_at(70,20) - 50) < 1.5`; `95 <= length <= 100`.
  - `test_ring_length`: `abs(length - 2*pi*50) < 10`; `abs(progress_at(30,80) - length/2) < 8`.
  - `test_progress_delta_wraps`: ring, `abs(progress_delta(length-1, 1) - 2) < 1e-6` and `abs(progress_delta(1, length-1) + 2) < 1e-6`.
  - `test_missing_start_line_raises`: a bmp with no green pixels → `pytest.raises(ValueError, match="start line")`.
  - `test_out_of_bounds_not_drivable`: `is_drivable(-5,10)` and `is_drivable(1000,1000)` both False.
  - `test_imperfect_colours`: corridor re-saved with drivable grey 250, walls grey 5, and also saved in palette mode (`"P"`) → same `is_drivable` answers.
- [ ] **Step 2:** Run `uv run pytest tests/test_track.py -v` — Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `track_gen.py` (Pillow draw; ring via per-pixel distance from centre) then `track.py` per the interface. Raise `ValueError("no start line found ...")` when no green pixels.
- [ ] **Step 4:** Run `uv run pytest tests/test_track.py -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `feat: bmp track with geodesic distance field`.

### Task 3: Ray casting

**Files:**
- Create: `racecar/rays.py`
- Test: `tests/test_rays.py`

**Interfaces:**
- Consumes: `Track.is_drivable`.
- Produces: `cast_rays(track: Track, x: float, y: float, heading: float, angles: np.ndarray, max_len: float, step: float = 1.0) -> np.ndarray` — for each relative angle (radians), the distance in pixels from (x, y) along `heading + angle` to the first non-drivable point, capped at `max_len`. Returns 0 for every ray when the start point is itself not drivable. Out-of-image counts as wall.

- [ ] **Step 1: Write failing tests** (corridor track, car at (70,20)): angle 0 heading 0 → `abs(d - 50) < 1.5`; angle π/2 → `abs(d - 10) < 1.5`; `max_len=20` with angle 0 → exactly 20; start at (5,5) (wall) → all zeros; a ray towards the image edge from a drivable point near the edge never raises (`Track` where drivable touches the edge: use a ring with `size=130`, `r_out=65`).
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement vectorised marching over `np.arange(step, max_len + step, step)`; do the mask lookup with bounds checks in NumPy (not a Python loop per ray step).
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: ray casting against track mask`.

### Task 4: Car physics

> **REVISED 2026-11-16 (done; see `racecar/car.py`, `tests/test_car.py`).** Speed follows `dv/dt = a₀·max(u,0) − b·max(−u,0) − (a₀/vmax)·v` (exponential approach to the top speed, constant-force braking, no separate drag, no clamp except `speed >= 0`). Steering input `s` is a fraction of the grip-limited angle `min(max_steer, atan(wheelbase·max_lateral_accel / v²))`. Defaults: `wheelbase=14`, `max_speed=100`, `max_accel=40`, `max_brake=80`, `max_lateral_accel=50`, `max_steer=0.5`; `CarParams` and `CarState` are frozen dataclasses. The original Task 4 text below is superseded.

**Files:**
- Create: `racecar/car.py`
- Test: `tests/test_car.py`

**Interfaces:**
- Produces: `@dataclass CarParams(wheelbase=20.0, max_speed=60.0, max_accel=30.0, max_brake=60.0, drag=0.2, max_steer=0.5)`; `@dataclass CarState(x: float, y: float, heading: float, speed: float)`; `step_car(state: CarState, throttle: float, steer: float, dt: float, params: CarParams) -> CarState`.
- Rules: position and heading use the pre-update speed: `x += v cosθ dt`, `y += v sinθ dt`, `θ += v / wheelbase * tan(max_steer * steer) dt`. Then `v += ((max_accel*throttle if throttle>=0 else max_brake*throttle) - drag*v) dt`, clamped to `[0, max_speed]` (no reversing). `throttle` and `steer` are assumed already clipped.

- [ ] **Step 1: Write failing tests** (`dt=0.1`): from rest, throttle 1 → speed `== 3.0`; state `(0,0,0,10)`, throttle 0, steer 0 → `x == 1.0` and `speed == 9.8`; steer 1 at speed 10 → `heading == 10/20*tan(0.5)*0.1`; steer -1 gives the negative; full brake at speed 1 → speed `== 0.0` (clamped); throttle 1 at `max_speed` → speed `<= 60.0`.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement `step_car` per the rules.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: kinematic bicycle car model`.

### Task 5: Environment (DONE)

Built 2026-11-17: `Track.sweep` (swept collision over a set of body points) and `RaceEnv` (`racecar/env.py`). Design and decisions: `docs/superpowers/plans/2026-11-16-task5-environment-design.md`. Tests: `tests/test_sweep.py`, `tests/test_env.py`.

Interface the later tasks rely on: `RaceEnv(track, n_rays=9, ray_fov_deg=180, max_ray_len=300, max_steps=1500, dt=0.1, progress_scale=0.1, time_penalty=0.01, footprint=[[0, 0]], params=CarParams())`; `obs_dim == 10`, `act_dim == 2`; `reset() -> obs`; `step(action) -> (obs, reward, terminated, truncated, info)`; `info` has `progress`, `crashed`, `finished`, `x`, `y`, `speed`, `steps`; `set_state(x, y, heading, speed)`.

## Working mode for Tasks 6-12

The goal is learning, so the core RL and PyTorch pieces are written **by the owner**, in small steps. Each task marks:

- **[YOU]**: the owner writes it. Claude supplies a stub with the signature and docstring, the failing test, and hints on request, and reviews the result. Claude does not write the solution unless asked.
- **[CLAUDE]**: plumbing (logging, file formats, plotting, CLI) that teaches little about RL; Claude writes it, test first.

Every task is still test-first: the test is written and seen failing before the implementation. A short concept note comes before each task (what the piece is and why), not after.

## Notes that apply from here on

- Tests use `corridor_image(...)`-based tracks (see `tests/conftest.py`, `tests/test_env.py`); the default real track is `tests/data/racetrack1.bmp`. There is no ring track.
- Visualization uses Pillow, not matplotlib (drop matplotlib from the dependencies if it was installed). Learning curves go to TensorBoard.
- Speed: roughly 0.1 ms per environment step, so one REINFORCE iteration of 16 episodes of up to 1500 steps can take seconds. Start with a smaller `max_steps` and fewer episodes. Batching the environment on the GPU is deferred until REINFORCE works on the CPU; keep the physics as plain functions of state so it stays possible.
- The default Windows PyPI wheel of PyTorch is CPU-only. That is enough for now; a CUDA build needs PyTorch's own package index.

### Task 6: PyTorch, networks and rollouts (DONE 2026-11-17)

Built: `racecar/nets.py` and `racecar/rollout.py` (written by the owner), tests `tests/test_nets.py`, `tests/test_rollout.py`.

**Concept note first:** a short PyTorch introduction (tensors, `requires_grad` and autograd, `nn.Module` and `nn.Linear`, optimizers, `torch.no_grad()`, and what `torch.distributions.Normal` gives us: `log_prob`, `sample`, `entropy`).

**Files:**
- Create: `racecar/nets.py`, `racecar/rollout.py`
- Test: `tests/test_nets.py`, `tests/test_rollout.py`

**Interfaces:**
- `nets.py`: `GaussianPolicy(obs_dim, act_dim=2, hidden=(64, 64))` with a learned state-independent `log_std` (initial std about 0.6); `forward(obs: Tensor) -> Normal`; `act(obs: np.ndarray) -> (np.ndarray (2,), float)` (sampled, unclipped action and its log-prob summed over action dims); `mean_action(obs: np.ndarray) -> np.ndarray`. `ValueNet(obs_dim, hidden=(64, 64))` with `forward(obs) -> Tensor (batch,)`.
- `rollout.py`: `@dataclass Episode(obs (T, D), actions (T, 2) unclipped, rewards (T,), logps (T,), terminated: bool, last_obs (D,))`; `collect_episodes(env, policy, n_episodes) -> list[Episode]`, under `torch.no_grad()`. `last_obs` is the observation after the final step, needed to bootstrap a truncated episode (Task 10).

- [x] **Step 1: [CLAUDE]** Give the concept note, install PyTorch and TensorBoard (`uv add torch tensorboard`), write the failing tests and the stubs.
  - Policy `forward` on a `(4, 10)` batch has `.mean.shape == (4, 2)`; `act` returns shapes `(2,)` and a Python float; `mean_action` equals `forward(...).mean`; the log-prob of the mean action is greater than that of `mean + 5`; `ValueNet` output shape `(4,)`.
  - Rollout on a corridor env: per episode `len(obs) == len(actions) == len(rewards) == len(logps)`, `terminated` is a bool, `last_obs.shape == (10,)`; an env with `max_steps=5` gives `len == 5` and `terminated is False`.
- [x] **Step 2: [YOU]** `GaussianPolicy` and `ValueNet`: the layers, `forward`, `act`, `mean_action`. Run the tests until green.
- [x] **Step 3: [YOU]** `collect_episodes`. Run the tests until green.
- [x] **Step 4: Commit** `feat: policy/value networks and rollout collection`.

### Task 7: Evaluation, metrics logging, checkpoints (DONE 2026-11-17)

Built: `racecar/evaluation.py` (trajectories include the start state, so have `T + 1` rows), `racecar/logging_utils.py` (checkpoints go to `run_dir/checkpoints/ckpt_<iteration>.pt`; a metric missing from a row is a blank cell, an unknown one raises `ValueError`), tests `tests/test_evaluation.py`, `tests/test_logging.py`.

**Files:**
- Create: `racecar/evaluation.py`, `racecar/logging_utils.py`
- Test: `tests/test_evaluation.py`, `tests/test_logging.py`

**Interfaces:**
- `evaluation.py`: `@dataclass EvalResult(mean_return, mean_length, crash_rate, finish_rate, lap_fraction, trajectories: list[np.ndarray] (each (T, 3): x, y, speed), crash_progress: list[float])`; `evaluate(env, policy, n_episodes=1) -> EvalResult`. Uses `mean_action` (no sampling noise); `lap_fraction` = mean of final `info["progress"] / track.length`; `finish_rate` from `info["finished"]`; `crash_progress` holds the final progress of each crashed episode. The start is fixed, so repeated runs are identical.
- `logging_utils.py`: `MetricsLogger(run_dir)` with `log(step, metrics)` (appends a row to `run_dir/metrics.csv`, header on the first call, and writes TensorBoard scalars to `run_dir/tb`) and `close()`; `save_checkpoint(run_dir, policy, iteration, seed, config) -> Path`; `load_checkpoint(path) -> dict` (keys `policy_state`, `iteration`, `seed`, `config`).

- [x] **Step 1: [CLAUDE]** Tests: `evaluate` twice on one policy gives identical `mean_return`; a random policy has `0 <= crash_rate <= 1` and 3-column trajectories; `MetricsLogger` writes a CSV with a header and one row per call; a checkpoint round-trip restores identical policy outputs and the same seed.
- [x] **Step 2: [CLAUDE]** Implement both modules.
- [x] **Step 3: Commit** `feat: evaluation, metrics logging, checkpoints`.

### Task 8: REINFORCE and the training loop (NEXT)

**Concept note first:** the policy gradient (why `grad J = E[sum_t grad log pi(a_t|s_t) * G_t]`), returns-to-go, why we normalise returns per batch.

**Files:**
- Create: `racecar/train.py`, `racecar/algos/__init__.py`, `racecar/algos/base.py`, `racecar/algos/reinforce.py`, `scripts/train.py`
- Test: `tests/test_reinforce.py`

**Interfaces:**
- `train.py`: `@dataclass TrainConfig(algo="reinforce", seed=0, iterations=200, episodes_per_iter=16, gamma=0.99, gae_lambda=0.95, lr=3e-3, eval_every=10, clip_eps=0.2, ppo_epochs=10, minibatch_size=64, max_steps=1500)`; `train(env, eval_env, learner, cfg, run_dir)`: seeds `numpy` and `torch`; each iteration collects episodes, calls `learner.update`, logs `mean_return`, `mean_length`, `policy_std`, `policy_entropy` and the learner's own metrics; every `eval_every` iterations runs `evaluate`, logs `eval_*` and saves a checkpoint.
- `algos/base.py`: `Learner` protocol with `policy` and `update(episodes) -> dict[str, float]`. `algos/__init__.py`: `make_learner(name, obs_dim, cfg)` (unknown name raises `ValueError`).
- `algos/reinforce.py`: `discounted_returns(rewards, gamma) -> np.ndarray`; `ReinforceLearner(obs_dim, cfg)` with loss `-(logp * G).mean()` over every step of the batch, returns normalised per batch (comment why).
- `scripts/train.py`: flags `--track` (default `tests/data/racetrack1.bmp`), `--algo`, `--seed`, `--iterations`, `--run-dir` (default `runs/<algo>-<seed>`), and a VS Code launch configuration.

- [ ] **Step 1: [CLAUDE]** Concept note and failing tests: `discounted_returns([1, 1, 1], 0.5) == [1.75, 1.5, 1.0]`; `make_learner("nope", ...)` raises `ValueError`; `update` on a batch of corridor episodes changes the policy parameters and returns a dict of floats; integration: on a corridor (short, `max_steps` reduced), 100 iterations, seed 0, `evaluate(...).lap_fraction >= 0.5` (tune `lr` and iterations, not the assertion, if flaky).
- [ ] **Step 2: [YOU]** `discounted_returns`.
- [ ] **Step 3: [YOU]** `ReinforceLearner.update`: returns, normalisation, the loss, `optimizer.zero_grad()` / `backward()` / `step()`.
- [ ] **Step 4: [CLAUDE]** The registry, `train()`, the script and launch configuration.
- [ ] **Step 5:** Run the corridor integration test, then `uv run python scripts/train.py --iterations 20`; expect `runs/reinforce-0/metrics.csv` with rows. **Commit** `feat: training loop, REINFORCE`.

### Task 9: Value baseline

**Concept note first:** why subtracting a state-dependent baseline keeps the gradient unbiased but lowers its variance; what the value net is trained on.

**Files:** create `racecar/algos/baseline.py`; modify `racecar/algos/__init__.py`; test `tests/test_baseline.py`.

**Interfaces:** `explained_variance(pred, target) -> float` (`1 - Var(target - pred) / Var(target)`); `BaselineLearner(obs_dim, cfg)`: policy loss `-(logp * (G - V(obs).detach())).mean()`, value loss MSE to `G`; metrics include `value_loss` and `explained_variance`.

- [ ] **Step 1: [CLAUDE]** Tests: `explained_variance(x, x) == 1.0`; predicting the mean gives `0.0` (within 1e-6); repeated `update` on one fixed batch lowers `value_loss`; corridor integration like Task 8.
- [ ] **Step 2: [YOU]** `explained_variance` and `BaselineLearner`. Compare its learning curve with REINFORCE on the same seed.
- [ ] **Step 3: Commit** `feat: actor-critic with value baseline`.

### Task 10: GAE

**Concept note first:** the bias-variance trade-off between one-step TD and Monte Carlo, and how `lambda` interpolates; why a time-out bootstraps from `V(last_obs)` and a crash or finish from 0 (the Review Focus item).

**Files:** create `racecar/algos/gae.py`; modify `racecar/algos/__init__.py`; test `tests/test_gae.py`.

**Interfaces:** `compute_gae(rewards, values, last_value, gamma, lam) -> (advantages, returns)` with `returns = advantages + values`; `GaeLearner(obs_dim, cfg)`: like `BaselineLearner`, advantages per episode, `last_value = 0.0` if `episode.terminated` else `V(episode.last_obs)`.

- [ ] **Step 1: [CLAUDE]** Tests: with `lam=1`, `last_value=0`, advantages equal `discounted_returns - values`; with `lam=0`, `advantages[t] == rewards[t] + gamma*values[t+1] - values[t]` (`last_value` in place of `values[T]`); rewards `[1, 1]`, values `[0, 0]`, `gamma=0.5`, `lam=1`, `last_value=5` vs `0` differ in `returns[-1]` by exactly `0.5 * 5`; a truncated episode uses a nonzero last value and a terminated one 0 (stub value net returning 5); corridor integration.
- [ ] **Step 2: [YOU]** `compute_gae` (a backward loop), then `GaeLearner`.
- [ ] **Step 3: Commit** `feat: generalised advantage estimation`.

### Task 11: PPO

**Concept note first:** why one gradient step per batch wastes data, why a big step is dangerous, and how the probability ratio and clipping limit it.

**Files:** create `racecar/algos/ppo.py`; modify `racecar/algos/__init__.py`; test `tests/test_ppo.py`.

**Interfaces:** `ppo_clipped_loss(new_logp, old_logp, adv, clip_eps) -> Tensor` (`-min(r*A, clip(r, 1-eps, 1+eps)*A).mean()`, `r = exp(new_logp - old_logp)`); `PpoLearner(obs_dim, cfg)`: GAE advantages normalised per batch, `cfg.ppo_epochs` epochs of shuffled minibatches of `cfg.minibatch_size`; metrics include `approx_kl` and `clip_fraction`.

- [ ] **Step 1: [CLAUDE]** Tests: ratio 1 gives loss `== -adv.mean()`; adv > 0 with ratio above `1+eps` gives zero gradient w.r.t. `new_logp`; adv < 0 with ratio below `1-eps` gives zero gradient; adv < 0 with ratio above `1+eps` gives nonzero gradient; `update` returns `clip_fraction` in [0, 1]; corridor integration.
- [ ] **Step 2: [YOU]** `ppo_clipped_loss`, then the `PpoLearner` epochs and minibatch loop.
- [ ] **Step 3: Commit** `feat: PPO`.

### Task 12: Visualization and replay

**Files:** create `racecar/viz.py`; modify `racecar/train.py` and `racecar/viewer.py`; test `tests/test_viz.py`.

**Interfaces:** `plot_trajectories(track, trajectories, path)` (Pillow: the bmp as background, path coloured by speed); `plot_crash_histogram(crash_progress, track_length, path)` (Pillow); `save_replay_gif(track, trajectory, path, every=2)`. `train()` writes `traj_{iter}.png`, `crash_hist_{iter}.png`, `replay_{iter}.gif` into `run_dir/plots/` at each evaluation. The viewer gets a replay mode: load a checkpoint and watch the car drive with its rays (and later the small-rectangle car body).

- [ ] **Step 1: [CLAUDE]** Tests (corridor, a 10-point trajectory): each function writes a non-empty file; GIF frame count `== ceil(10 / every)`; an empty crash list and a one-point trajectory do not raise.
- [ ] **Step 2: [CLAUDE]** Implement and hook into `train()` (only the first evaluation trajectory goes to the GIF).
- [x] **Step 3: [CLAUDE]** The viewer replay mode. Done separately: `docs/superpowers/plans/2026-11-17-replay-viewer.md` (`racecar/replay.py`, `uv run python -m racecar.replay RUN_DIR`).
- [ ] **Step 4:** Run `uv run pytest -v`, then a short PPO run; expect `runs/ppo-0/plots/` with PNG and GIF files. **Commit** `feat: trajectory plots, crash histogram, replay GIF`.
