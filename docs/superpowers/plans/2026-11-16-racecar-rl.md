# Racecar RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch RL pipeline (bmp track environment, REINFORCE → baseline → GAE → PPO, evaluation, logging, visualization) in which a simulated car learns to drive along a track.

**Architecture:** A NumPy environment (`Track` from a bmp, bicycle-model car, ray casting, per-step progress reward from a geodesic distance field) with a `reset()/step()` shape. A PyTorch Gaussian policy and value net. Each algorithm is a `Learner` with `.policy` and `.update(episodes) -> metrics`, all driven by one shared `train()` loop so runs are comparable on the same track and seed.

**Tech Stack:** Python 3.12, uv, NumPy, Pillow, PyTorch, matplotlib, TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-racecar-rl-design.md`

## Global Constraints

- Python `>=3.12`, managed with `uv`; Windows 11, VS Code.
- No RL framework (no Gymnasium, Stable-Baselines3, CleanRL). PyTorch only for networks, autograd, optimizers.
- Allowed libraries: PyTorch, NumPy, Pillow (bmp), matplotlib (display only), TensorBoard (logging only), pytest.
- Real code in plain `.py` modules in the package `racecar/`; notebooks are not part of this plan.
- Observation: N ray distances plus speed. Action: continuous 2-vector in [-1, 1]² (throttle/brake, steering), clipped by the environment.
- Reward: per-step progress from a geodesic distance field; a crash ends the episode; time-out is truncation, not terminal; no explicit crash penalty; single track first.
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

### Task 5: Environment

> **REVISED 2026-11-16 (not yet built); read the text below with these changes.**
> - **Prerequisite, in `Track`:** a finish band = drivable pixels with `dist > length − FINISH_LENGTH` (`FINISH_LENGTH = 20` px), exposed as `Track.finish: np.ndarray[bool]` and `Track.is_finish(x, y) -> bool`; tested on the corridor and `racetrack1.bmp`. The viewer later shows it in green and says "Finish area".
> - **Swept check:** each step's path (old → new position) is sampled at most every 1 px. The first sample that is a wall means a crash (`terminated=True`, reward 0); the first sample in the finish band means a finish (`terminated=True`, `info["finished"] = True`, reward = progress at that sample). Tests: a 10 px step across the 5 px gap at the end of `racetrack1.bmp` crashes or finishes and never lands in the start patch; a step that would cross the finish band and then hit the wall counts as finished.
> - **Progress:** the reward is the plain difference of `track.progress_at` values (no loop wrap); drop the ring and loop-closure tests; `info` also has `finished: bool`.
> - **Defaults:** `max_steps=1500`; car parameters as in Task 4.

**Files:**
- Create: `racecar/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Track`, `cast_rays`, `CarParams`, `CarState`, `step_car`.
- Produces: `class RaceEnv` with `__init__(track: Track, n_rays: int = 9, ray_fov_deg: float = 180.0, max_ray_len: float = 150.0, max_steps: int = 1000, dt: float = 0.1, time_penalty: float = 0.0, params: CarParams = CarParams())`; attribute `obs_dim == n_rays + 1`, `act_dim == 2`, `track`, `state: CarState`; `reset() -> np.ndarray`; `step(action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]` returning `(obs, reward, terminated, truncated, info)`; `set_state(x, y, heading, speed) -> None` (for tests and debugging; also resets cumulative progress reference to the current position).
- Observation: ray distances / `max_ray_len` (rays spread evenly over `[-fov/2, +fov/2]` relative to heading), then `speed / params.max_speed`; float32, all in [0, 1].
- Step: clip action to [-1, 1]; `step_car`; if the new position is not drivable → `terminated=True`, reward 0 (progress not credited) and the observation computed with all rays 0; else reward = `track.progress_delta(prev_progress, new_progress) - time_penalty`. `truncated = steps >= max_steps` (and not terminated). `info` has `progress` (cumulative progress this episode), `x`, `y`, `speed`.

- [ ] **Step 1: Write failing tests** (corridor, defaults unless stated): `obs_dim == 10` and `reset()` obs shape `(10,)`, all in [0,1], speed component 0; reset state equals `track.start_pos` and `start_heading`; action `[5, 0]` gives the same next state as `[1, 0]`; throttle 1, steer 0 for 5 steps: sum of rewards `== info["progress"]` and `abs(info["progress"] - (state.x - 20)) < 1.5`; throttle 1, steer 1 until `terminated` within 200 steps; `max_steps=5` with brake-only → on step 5 `truncated and not terminated`, total reward `0`; `time_penalty=0.1` brake-only → total reward `-0.5`; ring: `set_state(130, 77, pi/2, 20)` then one step with `[0,0]` → `abs(reward) < 5` (crossing the line, no spike), and `set_state(130, 83, -pi/2, 20)` → `-5 < reward < 0`; two envs given identical actions produce identical observations.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement `RaceEnv` per the rules.
- [ ] **Step 4:** Run `uv run pytest tests/test_env.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: race environment with progress reward`.

### Task 6: Networks and rollouts

**Files:**
- Create: `racecar/nets.py`, `racecar/rollout.py`
- Test: `tests/test_nets.py`, `tests/test_rollout.py`

**Interfaces:**
- Produces, in `nets.py`: `class GaussianPolicy(nn.Module)`: `__init__(obs_dim: int, act_dim: int = 2, hidden: tuple[int, ...] = (64, 64))`, learned `log_std` parameter (state-independent, initial std ≈ 0.6); `forward(obs: Tensor) -> torch.distributions.Normal`; `act(obs: np.ndarray) -> tuple[np.ndarray, float]` (sampled action unclipped, its log-prob summed over action dims); `mean_action(obs: np.ndarray) -> np.ndarray`. `class ValueNet(nn.Module)`: `__init__(obs_dim: int, hidden: tuple[int, ...] = (64, 64))`, `forward(obs: Tensor) -> Tensor` of shape `(batch,)`.
- Produces, in `rollout.py`: `@dataclass Episode(obs: np.ndarray (T,D), actions: np.ndarray (T,2) unclipped, rewards: np.ndarray (T,), logps: np.ndarray (T,), terminated: bool, last_obs: np.ndarray (D,))`; `collect_episodes(env: RaceEnv, policy: GaussianPolicy, n_episodes: int) -> list[Episode]`, runs under `torch.no_grad()`.

- [ ] **Step 1: Write failing tests**: policy `forward` on a `(4, 10)` batch has `.mean.shape == (4, 2)`; `act` returns shapes `(2,)` and a Python float; `mean_action` equals `forward(...).mean`; log-prob of the mean action is greater than log-prob of `mean + 5`; `ValueNet` output shape `(4,)`; rollout on the corridor: for each episode `len(obs) == len(actions) == len(rewards) == len(logps)`, `terminated` is a bool, `last_obs.shape == (10,)`, and an env with `max_steps=5` yields `len == 5` and `terminated is False`.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: policy/value networks and rollout collection`.

### Task 7: Evaluation, metrics logging, checkpoints

**Files:**
- Create: `racecar/evaluation.py`, `racecar/logging_utils.py`
- Test: `tests/test_evaluation.py`, `tests/test_logging.py`

**Interfaces:**
- Consumes: `RaceEnv`, `GaussianPolicy.mean_action`.
- Produces, in `evaluation.py`: `@dataclass EvalResult(mean_return: float, mean_length: float, crash_rate: float, lap_fraction: float, trajectories: list[np.ndarray] (each (T,3): x, y, speed), crash_progress: list[float])`; `evaluate(env: RaceEnv, policy: GaussianPolicy, n_episodes: int = 1) -> EvalResult`. Uses `mean_action`; `lap_fraction` = mean of final `info["progress"] / track.length`; `crash_progress` holds the final progress of each crashed episode.
- Produces, in `logging_utils.py`: `class MetricsLogger(run_dir: Path)` with `log(step: int, metrics: dict[str, float]) -> None` (appends a row to `run_dir/metrics.csv`, writing the header on the first call, and writes TensorBoard scalars to `run_dir/tb`) and `close()`; `save_checkpoint(run_dir: Path, policy: GaussianPolicy, iteration: int, seed: int, config: dict) -> Path`; `load_checkpoint(path: Path) -> dict` (keys `policy_state`, `iteration`, `seed`, `config`).

- [ ] **Step 1: Write failing tests**: `evaluate` twice on the same policy gives identical `mean_return`; a random untrained policy on the corridor has `0 <= crash_rate <= 1` and each trajectory has 3 columns; `MetricsLogger` writes a CSV with a header and one row per `log` call; checkpoint round-trip restores identical policy outputs and the same `seed`.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement both modules (`torch.utils.tensorboard.SummaryWriter`; metrics columns fixed from the first call).
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: evaluation, metrics logging, checkpoints`.

### Task 8: Training loop, REINFORCE, CLI

**Files:**
- Create: `racecar/train.py`, `racecar/algos/__init__.py`, `racecar/algos/base.py`, `racecar/algos/reinforce.py`, `scripts/train.py`, `scripts/make_tracks.py`, `.vscode/launch.json`, `tracks/corridor.bmp`, `tracks/ring.bmp`
- Test: `tests/test_reinforce.py`

**Interfaces:**
- Consumes: everything from Tasks 2-7.
- Produces, in `train.py`: `@dataclass TrainConfig(algo: str = "reinforce", seed: int = 0, iterations: int = 200, episodes_per_iter: int = 16, gamma: float = 0.99, gae_lambda: float = 0.95, lr: float = 3e-3, eval_every: int = 10, clip_eps: float = 0.2, ppo_epochs: int = 10, minibatch_size: int = 64)`; `train(env: RaceEnv, eval_env: RaceEnv, learner: Learner, cfg: TrainConfig, run_dir: Path) -> None`: seeds `numpy` and `torch`; each iteration collects episodes, calls `learner.update`, logs training metrics (`mean_return`, `mean_length`, `policy_std`, `policy_entropy`, plus the learner's own metrics); every `eval_every` iterations runs `evaluate`, logs `eval_*` metrics and saves a checkpoint (plot/GIF hooks are added in Task 12).
- Produces, in `algos/base.py`: `class Learner(Protocol)`: `policy: GaussianPolicy`; `update(episodes: list[Episode]) -> dict[str, float]`. In `algos/__init__.py`: `make_learner(name: str, obs_dim: int, cfg: TrainConfig) -> Learner` (registry; unknown name → `ValueError`). In `algos/reinforce.py`: `discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray`; `class ReinforceLearner(obs_dim: int, cfg: TrainConfig)` — loss `-(logp * G).mean()` over all steps in the batch, with returns normalised per batch (document this in a comment).
- Scripts: `scripts/make_tracks.py` writes `tracks/corridor.bmp` and `tracks/ring.bmp` via `track_gen`. `scripts/train.py` flags: `--track` (default `tracks/ring.bmp`), `--start-heading-deg` (default `90`), `--algo`, `--seed`, `--iterations`, `--run-dir` (default `runs/<algo>-<seed>`).

- [ ] **Step 1: Write failing tests**: `discounted_returns(np.array([1.,1.,1.]), 0.5)` equals `[1.75, 1.5, 1.0]`; `make_learner("nope", 10, TrainConfig())` raises `ValueError`; `ReinforceLearner.update` on a batch of corridor episodes changes the policy parameters and returns a dict of floats; integration: on the corridor (generated in `tmp_path`, start heading 0) with `iterations=100`, `lr=3e-3`, seed 0, the final `evaluate(...).lap_fraction >= 0.5`.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement the modules and scripts; add a VS Code launch configuration that runs `scripts/train.py` with the ring track. Run `uv run python scripts/make_tracks.py` and commit the generated bmps.
- [ ] **Step 4:** Run `uv run pytest tests/test_reinforce.py -v` — Expected: PASS (if the integration test is flaky, tune `lr`/`iterations` in the test, not the assertion). Then run `uv run python scripts/train.py --track tracks/corridor.bmp --start-heading-deg 0 --iterations 20` — Expected: `runs/reinforce-0/metrics.csv` exists with rows.
- [ ] **Step 5: Commit** `feat: training loop, REINFORCE, CLI`.

### Task 9: Value baseline

**Files:**
- Create: `racecar/algos/baseline.py`
- Modify: `racecar/algos/__init__.py` (register `"baseline"`)
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `ValueNet`, `discounted_returns`.
- Produces: `explained_variance(pred: np.ndarray, target: np.ndarray) -> float` (`1 - Var(target - pred) / Var(target)`); `class BaselineLearner(obs_dim: int, cfg: TrainConfig)` — policy loss `-(logp * (G - V(obs).detach())).mean()`, value loss MSE to `G`; metrics include `value_loss` and `explained_variance`.

- [ ] **Step 1: Write failing tests**: `explained_variance(x, x) == 1.0`; `explained_variance(np.full(5, x.mean()), x)` is `0.0` (within 1e-6); repeated `update` on one fixed batch lowers `value_loss`; integration like Task 8 (corridor, `lap_fraction >= 0.5`).
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement and register.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: actor-critic with value baseline`.

### Task 10: GAE

**Files:**
- Create: `racecar/algos/gae.py`
- Modify: `racecar/algos/__init__.py` (register `"gae"`)
- Test: `tests/test_gae.py`

**Interfaces:**
- Produces: `compute_gae(rewards: np.ndarray, values: np.ndarray, last_value: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]` returning `(advantages, returns)` with `returns = advantages + values`; `class GaeLearner(obs_dim: int, cfg: TrainConfig)` — like `BaselineLearner` but advantages from `compute_gae` per episode with `last_value = 0.0` if `episode.terminated` else `V(episode.last_obs)`.

- [ ] **Step 1: Write failing tests**: with `lam=1`, `last_value=0`: `advantages == discounted_returns(rewards, gamma) - values`; with `lam=0`: `advantages[t] == rewards[t] + gamma*values[t+1] - values[t]` (and `last_value` in place of `values[T]`); rewards `[1,1]`, values `[0,0]`, `gamma=0.5`, `lam=1`, `last_value=5` vs `last_value=0` differ in `returns[-1]` by exactly `0.5 * 5` (truncation bootstraps); `GaeLearner` on a truncated episode uses a nonzero last value while a terminated one uses 0 (test via a stub value net returning 5); corridor integration `lap_fraction >= 0.5`.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement and register.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: generalised advantage estimation`.

### Task 11: PPO

**Files:**
- Create: `racecar/algos/ppo.py`
- Modify: `racecar/algos/__init__.py` (register `"ppo"`)
- Test: `tests/test_ppo.py`

**Interfaces:**
- Consumes: `compute_gae`.
- Produces: `ppo_clipped_loss(new_logp: Tensor, old_logp: Tensor, adv: Tensor, clip_eps: float) -> Tensor` (`-min(r·A, clip(r, 1±eps)·A).mean()`, `r = exp(new_logp - old_logp)`); `class PpoLearner(obs_dim: int, cfg: TrainConfig)` — GAE advantages (normalised per batch), `cfg.ppo_epochs` epochs of shuffled minibatches of `cfg.minibatch_size`; metrics include `approx_kl` and `clip_fraction`.

- [ ] **Step 1: Write failing tests**: ratio 1 (`new_logp == old_logp`) → loss `== -adv.mean()`; adv > 0 with ratio above `1+eps` → zero gradient w.r.t. `new_logp`; adv < 0 with ratio below `1-eps` → zero gradient; adv < 0 with ratio above `1+eps` → nonzero gradient; `PpoLearner.update` returns `clip_fraction` in [0,1]; corridor integration `lap_fraction >= 0.5`.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement and register.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5: Commit** `feat: PPO`.

### Task 12: Visualization

**Files:**
- Create: `racecar/viz.py`
- Modify: `racecar/train.py` (call the plotters at each evaluation)
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `Track`, `EvalResult`.
- Produces: `plot_trajectories(track: Track, trajectories: list[np.ndarray], path: Path) -> None` (matplotlib, Agg backend; the bmp as background, path coloured by speed); `plot_crash_histogram(crash_progress: list[float], track_length: float, path: Path) -> None`; `save_replay_gif(track: Track, trajectory: np.ndarray, path: Path, every: int = 2) -> None` (Pillow frames, one per `every` steps). Files written by `train()` at each evaluation into `run_dir/plots/`: `traj_{iter}.png`, `crash_hist_{iter}.png`, `replay_{iter}.gif`.

- [ ] **Step 1: Write failing tests** (corridor, hand-made trajectory of 10 points): each function creates a non-empty file; GIF frame count == `ceil(10 / every)` (`PIL.Image.n_frames`); an empty `crash_progress` list and a one-point trajectory both produce files without raising.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement `viz.py` and hook it into `train()` (only the first evaluation trajectory is used for the GIF).
- [ ] **Step 4:** Run `uv run pytest -v` — Expected: whole suite PASS. Then run `uv run python scripts/train.py --algo ppo --iterations 30` — Expected: `runs/ppo-0/plots/` contains PNG and GIF files.
- [ ] **Step 5: Commit** `feat: trajectory plots, crash histogram, replay GIF`.
