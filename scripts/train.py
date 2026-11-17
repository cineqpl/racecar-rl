"""Train an agent: `uv run python scripts/train.py --algo reinforce --iterations 100`.

Outputs go to `runs/<algo>-<seed>` (override with --run-dir): metrics.csv, eval.csv,
TensorBoard events (`uv run tensorboard --logdir runs`) and checkpoints.
"""

import argparse
import sys
import logging
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `racecar` imports when run as a script

from racecar.algos import make_learner  # noqa: E402
from racecar.env import RaceEnv  # noqa: E402
from racecar.track import Track  # noqa: E402
from racecar.train import TrainConfig, seed_everything, train  # noqa: E402

DEFAULT_TRACK = Path(__file__).resolve().parent.parent / "tests" / "data" / "racetrack1.bmp"


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--run-dir", type=Path, default=None, help="default: runs/<algo>-<seed>")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    for f in fields(TrainConfig):  # one flag per config field: --algo, --seed, --episodes-per-iter, ...
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(getattr(defaults, f.name)), default=getattr(defaults, f.name))
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    run_dir = args.run_dir or Path("runs") / f"{cfg.algo}-{cfg.seed}"

    track = Track.load(args.track)
    seed_everything(cfg.seed)  # before the learner is built: its initial weights are random
    env, eval_env = RaceEnv(track, max_steps=cfg.max_steps), RaceEnv(track, max_steps=cfg.max_steps)
    learner = make_learner(cfg.algo, env.obs_dim, cfg)
    print(f"track {args.track.name} ({track.length:.0f} px long), {cfg.algo}, seed {cfg.seed}, writing to {run_dir}")
    train(env, eval_env, learner, cfg, run_dir)
    print(f"done; look at {run_dir / 'metrics.csv'} and {run_dir / 'eval.csv'}")


if __name__ == "__main__":
    main()
