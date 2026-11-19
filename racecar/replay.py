"""Replay a trained policy: record one deterministic episode, then watch it.

    uv run python -m racecar.replay runs/reinforce-0 [--track PATH] [--checkpoint N] [--scale S]

`record_episode` runs the real `RaceEnv`, so what is drawn is what the environment did.
"""

import argparse
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ImageTk

from racecar.env import RaceEnv
from racecar.logging_utils import load_checkpoint
from racecar.nets import GaussianPolicy
from racecar.track import Track
from racecar.texture import load_car_sprite, render_textured, rotate_sprite
from racecar.viewer import MARGIN


@dataclass
class Recording:
    """One episode, frame by frame. Frame 0 is the start state; frame i > 0 is the state after step i."""

    x: np.ndarray  # (F,)
    y: np.ndarray  # (F,)
    heading: np.ndarray  # (F,) radians
    speed: np.ndarray  # (F,)
    progress: np.ndarray  # (F,) cumulative progress in px
    rays: np.ndarray  # (F, n_rays) ray lengths in px, all 0 on a crash frame
    ray_angles: np.ndarray  # (n_rays,) angle of each ray relative to the heading, left to right
    actions: np.ndarray  # (F - 1,2) as the environment received them (clipped)
    rewards: np.ndarray  # (F - 1,)
    crashed: bool
    finished: bool

    @property
    def n_frames(self) -> int:
        return len(self.x)

    def ray_ends(self, i: int) -> np.ndarray:
        """Where each ray of frame `i` ends, as an (n_rays, 2) array of (x, y)."""
        direction = self.heading[i] + self.ray_angles
        return np.stack(
            [self.x[i] + self.rays[i] * np.cos(direction), self.y[i] + self.rays[i] * np.sin(direction)],
            axis=1,
        )


def record_episode(env: RaceEnv, policy: GaussianPolicy) -> Recording:
    """Run one episode with the policy's mean action (no sampling noise) and record every frame."""
    obs = env.reset()
    xs, ys, headings, speeds, progress = [env.state.x], [env.state.y], [env.state.heading], [env.state.speed], [0.0]
    rays, actions, rewards = [obs[:-1] * env.max_ray_len], [], []
    info: dict = {}
    while True:
        action = np.clip(policy.mean_action(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        xs.append(env.state.x), ys.append(env.state.y)
        headings.append(env.state.heading), speeds.append(env.state.speed)
        progress.append(info["progress"])
        rays.append(obs[:-1] * env.max_ray_len)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return Recording(
        x=np.array(xs), y=np.array(ys), heading=np.array(headings), speed=np.array(speeds),
        progress=np.array(progress), rays=np.array(rays), ray_angles=np.array(env.ray_angles),
        actions=np.array(actions), rewards=np.array(rewards),
        crashed=bool(info["crashed"]), finished=bool(info["finished"]),
    )


def car_polygon(
    x: float, y: float, heading: float, length: float = 16.0, width: float = 8.0
) -> list[tuple[float, float]]:
    """Corners of the car rectangle centred on (x, y), its long side along the heading. Display only."""
    c, s = math.cos(heading), math.sin(heading)
    corners = [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2)]
    return [(x + fx * c - fy * s, y + fx * s + fy * c) for fx, fy in corners]


def load_policy(run_dir: Path, obs_dim: int, checkpoint: int | None = None) -> tuple[GaussianPolicy, dict]:
    """The policy and saved config from `run_dir/checkpoints`: the latest one, or iteration `checkpoint`."""
    run_dir = Path(run_dir)
    folder = run_dir / "checkpoints"
    if checkpoint is None:
        found = sorted(folder.glob("ckpt_*.pt")) if folder.is_dir() else []
        if not found:
            raise FileNotFoundError(f"no checkpoints found in {folder}")
        path = found[-1]
    else:
        path = folder / f"ckpt_{checkpoint:06d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {checkpoint} not found in {folder}")
    data = load_checkpoint(path)
    policy = GaussianPolicy(obs_dim)
    policy.load_state_dict(data["policy_state"])
    return policy, data["config"]


DEFAULT_TRACK = Path(__file__).resolve().parent.parent / "tests" / "data" / "racetrack1.bmp"
SPEEDS = ("0.25", "0.5", "1", "2", "4")
RAY_COLOUR, TRAIL_COLOUR, CAR_COLOUR = "#e6c800", "#3399ff", "#d02020"


class ReplayWindow:
    """The track with a car, its trail and rays, a text sidebar on the right and a control bar below.

    The caller runs `window.root.mainloop()`. Tests drive it through `show`, `step`, `toggle_play`
    and `tick` without waiting for timers.
    """

    def __init__(self, track: Track, recording: Recording, dt: float = 0.1, scale: float = 1.0,
                 master: tk.Misc | None = None) -> None:
        self.recording, self.dt, self.scale = recording, dt, scale
        self.frame = 0
        self.playing = False
        self.speed = 1.0
        self.delay_ms = round(1000 * dt / self.speed)
        self._job: str | None = None

        image = render_textured(track)
        if scale != 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        self._sprite = load_car_sprite()  # None when car.png is missing: a polygon is drawn instead
        if self._sprite is not None and scale != 1.0:
            self._sprite = self._sprite.resize(
                (max(1, round(self._sprite.width * scale)), max(1, round(self._sprite.height * scale))))
        self._sprites: dict[int, ImageTk.PhotoImage] = {}  # rotated sprites by heading in 2-degree steps
        w, h = image.size
        self.root = tk.Tk() if master is None else tk.Toplevel(master)
        self.root.title("Replay")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=w + MARGIN, height=h, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.root._photo = ImageTk.PhotoImage(image)  # keep a reference or Tk drops the image
        self.canvas.create_image(0, 0, image=self.root._photo, anchor="nw")
        self.canvas.create_line(w, 0, w, h, fill="#888888")
        self.text_id = self.canvas.create_text(
            w + 12, 12, anchor="nw", width=MARGIN - 24, font=("Segoe UI", 12), text="")
        self.canvas.create_line(0, 0, 0, 0, fill=TRAIL_COLOUR, width=2, tags="trail")
        if self._sprite is not None:
            self.canvas.create_image(0, 0, tags="car")  # anchored at its centre
        else:
            self.canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=CAR_COLOUR, outline="white", tags="car")

        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)
        tk.Button(bar, text="|<", width=3, command=self.restart).pack(side="left")
        tk.Button(bar, text="<", width=3, command=lambda: self.step(-1)).pack(side="left")
        self.play_button = tk.Button(bar, text="Play", width=6, command=self.toggle_play)
        self.play_button.pack(side="left", padx=4)
        tk.Button(bar, text=">", width=3, command=lambda: self.step(+1)).pack(side="left")
        self.slider = tk.Scale(bar, from_=0, to=recording.n_frames - 1, orient="horizontal",
                               showvalue=False, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        tk.Label(bar, text="Speed").pack(side="left")
        self._speed_var = tk.StringVar(self.root, "1")
        tk.Spinbox(bar, values=SPEEDS, width=5, textvariable=self._speed_var, wrap=True,
                   command=lambda: self.set_speed(float(self._speed_var.get()))).pack(side="left", padx=4)
        self.root.bind("<Destroy>", self._on_destroy)
        self.show(0)

    # --- control ---------------------------------------------------------------

    def show(self, i: int) -> None:
        """Show frame `i` (clamped to the recording)."""
        rec = self.recording
        self.frame = i = max(0, min(i, rec.n_frames - 1))
        s = self.scale
        trail = np.stack([rec.x[: i + 1] * s, rec.y[: i + 1] * s], axis=1).ravel().tolist()
        self.canvas.coords("trail", *(trail if i > 0 else trail * 2))  # a line needs two points
        if self._sprite is not None:
            self.canvas.itemconfig("car", image=self._sprite_for(rec.heading[i]))
            self.canvas.coords("car", rec.x[i] * s, rec.y[i] * s)
        else:
            corners = car_polygon(rec.x[i], rec.y[i], rec.heading[i])
            self.canvas.coords("car", *[c * s for xy in corners for c in xy])
        self.canvas.delete("ray")
        for length, (ex, ey) in zip(rec.rays[i], rec.ray_ends(i)):
            if length > 0:
                self.canvas.create_line(rec.x[i] * s, rec.y[i] * s, ex * s, ey * s,
                                        fill=RAY_COLOUR, width=2, tags="ray")
        self.canvas.tag_raise("car")
        if int(float(self.slider.get())) != i:
            self.slider.set(i)
        self.canvas.itemconfig(self.text_id, text=self._sidebar_text())

    def _sprite_for(self, heading: float) -> "ImageTk.PhotoImage | None":
        """The car sprite turned to `heading`, cached per 2 degrees; None without a sprite."""
        if self._sprite is None:
            return None
        key = round(math.degrees(heading) / 2) % 180
        if key not in self._sprites:
            self._sprites[key] = ImageTk.PhotoImage(rotate_sprite(self._sprite, math.radians(key * 2)))
        return self._sprites[key]

    def step(self, delta: int) -> None:
        self.show(self.frame + delta)

    def toggle_play(self) -> None:
        if self.playing:
            self._pause()
            return
        if self.frame >= self.recording.n_frames - 1:
            self.show(0)
        self.playing = True
        self.play_button.config(text="Pause")
        self._schedule()

    def restart(self) -> None:
        self._pause()
        self.show(0)

    def set_speed(self, factor: float) -> None:
        self.speed = factor
        self.delay_ms = round(1000 * self.dt / factor)

    def tick(self) -> None:
        """Advance one frame; schedule the next while playing, stop at the last frame."""
        self._job = None
        if not self.playing:
            return
        self.step(+1)
        if self.frame >= self.recording.n_frames - 1:
            self._pause()
        else:
            self._schedule()

    # --- internals ---------------------------------------------------------------

    def _schedule(self) -> None:
        self._cancel()
        self._job = self.root.after(self.delay_ms, self.tick)

    def _cancel(self) -> None:
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except tk.TclError:
                pass  # the window is already gone
            self._job = None

    def _pause(self) -> None:
        self.playing = False
        self._cancel()
        try:
            self.play_button.config(text="Play")
        except tk.TclError:
            pass

    def _on_slider(self, value: str) -> None:
        if int(float(value)) != self.frame:
            self.show(int(float(value)))

    def _on_destroy(self, event) -> None:
        if event.widget is self.root:
            self.playing = False
            self._cancel()

    def _sidebar_text(self) -> str:
        rec, i = self.recording, self.frame
        lines = [f"Frame {i} / {rec.n_frames - 1}", f"x {rec.x[i]:.1f}   y {rec.y[i]:.1f}",
                 f"speed {rec.speed[i]:.1f} px/s"]
        if i > 0:
            a, r = rec.actions[i - 1], rec.rewards[i - 1]
            lines += [f"throttle {a[0]:+.2f}   steer {a[1]:+.2f}", f"reward {r:+.3f}"]
        lines.append(f"progress {rec.progress[i]:.0f} px")
        if i == rec.n_frames - 1:
            lines.append("")
            lines.append("Crashed" if rec.crashed else "Finished" if rec.finished else "Time out")
        return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the final policy of a run on the track.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK, help="must be the track the run trained on")
    parser.add_argument("--checkpoint", type=int, default=None, help="iteration to load; default: the latest")
    parser.add_argument("--scale", type=float, default=1.0, help="display scale, e.g. 0.75 on a small screen")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    track = Track.load(args.track)
    policy, config = load_policy(args.run_dir, RaceEnv(track).obs_dim, args.checkpoint)
    env = RaceEnv(track, max_steps=config.get("max_steps", 1500))
    recording = record_episode(env, policy)
    outcome = "crashed" if recording.crashed else "finished" if recording.finished else "timed out"
    print(f"{recording.n_frames - 1} steps, {recording.progress[-1]:.0f} px, {outcome}")
    ReplayWindow(track, recording, dt=env.dt, scale=args.scale).root.mainloop()


if __name__ == "__main__":
    main()
