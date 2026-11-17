# racecar-rl

Starting my adventure with reinforcement learning: a simulated car learns to drive along a track
loaded from a bmp image. The goal is to learn the concepts, so the algorithms are written by hand;
only the neural-network computation (tensors, autograd, optimisers) comes from PyTorch.

## What it does

- **Environment** (`racecar/`): a track from a bmp (black = track, white = wall, red = start patch),
  a geodesic distance field for progress, ray casting, a kinematic car with a grip limit, and
  `RaceEnv` with a per-step progress reward and swept collision.
- **Algorithms** (`racecar/algos/`), a ladder where each step fixes a weakness of the one before:
  REINFORCE, then a learned value baseline (done); GAE and PPO (next).
- **Tooling**: evaluation and checkpoints, CSV and TensorBoard logging, a click-to-inspect track
  viewer and a replay viewer for trained policies.

## Usage

Everything runs through [uv](https://docs.astral.sh/uv/) (Python 3.12).

```
uv run pytest                                              # the test suite
uv run python -m racecar.viewer tests/data/racetrack1.bmp  # inspect a track: click for rays and distance
uv run python scripts/train.py --algo reinforce --iterations 200 --max-steps 600
uv run python scripts/train.py --algo baseline --iterations 200 --max-steps 600
uv run tensorboard --logdir runs                           # learning curves
uv run python -m racecar.replay runs/reinforce-0           # watch the final policy drive
```

A run writes to `runs/<algo>-<seed>`: `metrics.csv` (per iteration), `eval.csv` (deterministic
evaluation), TensorBoard events and checkpoints. `scripts/train.py --help` lists every setting.

## Documentation

- Design: `docs/superpowers/specs/2026-11-16-racecar-rl-design.md`
- Plan and status of every task: `docs/superpowers/plans/2026-11-16-racecar-rl.md`
- Environment design: `docs/superpowers/plans/2026-11-16-task5-environment-design.md`
- Replay viewer: `docs/superpowers/specs/2026-11-17-replay-viewer-design.md` and
  `docs/superpowers/plans/2026-11-17-replay-viewer.md`

## Status

Done: track, rays, car, environment, networks, rollouts, evaluation, logging, REINFORCE, value
baseline, replay viewer. Next: GAE, then PPO, then plots and GIF export.
