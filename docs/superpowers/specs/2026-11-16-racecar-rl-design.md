# Racecar RL: Design

## Intent and the why

The owner wants to **learn reinforcement learning concepts**, first and
foremost, by teaching a simulated car to drive along a track. The car is the
vehicle for the learning; a good lap is secondary. The owner has a
mathematics and data-science background but only superficial RL knowledge.

Because the goal is learning, the algorithms are written by hand. Only the
neural-network computation (tensors, autograd, optimizers) comes from a
library.

## Goals

- Build the RL pipeline from scratch: environment, policy-gradient algorithms,
  training loop, evaluation, logging.
- Progress through an algorithm ladder where each step fixes a visible weakness
  of the previous one: REINFORCE, then a learned value baseline, then GAE, then
  PPO.
- Tracks are loaded from bmp images, so new tracks need no code changes.
- Make learning observable: metrics, trajectory plots, replays.
- Make the code easy to step through and debug in VS Code.

## Non-goals

- Using Gymnasium, Stable-Baselines3, CleanRL or any RL framework.
- Beating benchmarks or building the fastest-possible simulator.
- Realistic vehicle physics.
- Generalisation to unseen tracks is **not required at first**. Train and
  evaluate on one track, then widen it later.

## Anti-goals

- A black-box pipeline that works but that the owner cannot explain.
- Notebook-only code that hides state and cannot be tested.
- Debugging an agent when the environment itself is wrong (hence the
  environment tests).

## Constraints

- Python 3.12+, managed with `uv` (project already exists:
  `pyproject.toml`, `uv.lock`, `main.py`). Windows 11, VS Code.
- Allowed libraries: PyTorch (networks, autograd, optimizers); NumPy; Pillow
  (reading bmp); a drawing library (pygame or matplotlib) for display only;
  TensorBoard for logging only; pytest.

## Problem formalisation (MDP)

- **Track:** a bmp where drivable pixels and wall pixels differ by colour.
  A start line and a driving direction must be defined (see "Left to the
  builder").
- **Observation:** N ray distances to the track edge at fixed angles relative
  to the car's heading, plus speed. Angular velocity or slip may be added if
  the physics has them. Normalised to a sensible range.
- **Action:** continuous 2-vector in [-1, 1]²: throttle/brake and steering.
- **Dynamics:** a simple kinematic bicycle model with fixed timestep.
- **Policy:** a neural network outputting the mean of a Gaussian over the two
  actions, with a learned state-independent log standard deviation. Actions
  are clipped to range when sent to the environment.
- **Objective:** maximise the expected discounted return
  J(θ) = E[Σ γᵗ rₜ].
- **Termination:** a crash (car leaves the drivable area) ends the episode.
  A step limit also ends it (time-out, handled as truncation, not as a true
  terminal state, when bootstrapping a value).

### Reward

The owner's original idea, "distance along the track before a crash", is kept
as the goal. The change is that it is paid **per step** as the progress made
in that step, rather than once at the end. The summed return is the same, but
the learning signal is much denser.

Progress is measured from a **geodesic distance field** computed once when a
track loads: flood-fill the drivable area from the start line, so every
drivable pixel stores its distance along the track. Per-step progress is the
change in this value at the car's position. This works for any track shape
without a hand-made centreline. Backward driving gives negative reward. The
loop closure at the start line needs handling (see "Left to the builder").

Speed is encouraged through the episode step limit, with a small per-step
time penalty available as an option. Crawling forever must not score as well
as driving fast.

## Approaches

Three ways to learn the algorithms were considered:

- **A. Policy-gradient ladder (chosen).** REINFORCE, then a value baseline,
  then GAE, then PPO. Teaches the core theory in order, and each step motivates
  the next. Cost: PPO has many small details that are easy to get subtly wrong.
- **B. Evolution strategies / neuroevolution.** Perturb weights, keep the better
  ones. Very little code and works on a small network, but skips most RL theory
  (credit assignment, bootstrapping). Kept as an optional comparison at the
  end, not the main route.
- **C. Off-policy (DDPG/SAC).** Replay buffer and learned Q-functions. Most
  moving parts and a hard first step. Possible later, after A.

The owner chose **A**.

## Algorithm ladder

1. **REINFORCE** with the Gaussian policy and Monte Carlo returns.
2. **Baseline:** learn a value function V(s) and subtract it, so the
   gradient estimate has lower variance.
3. **Generalised advantage estimation (GAE):** trade bias against variance
   with λ.
4. **PPO:** clipped surrogate objective, several epochs per batch.

Each step should be runnable on its own and comparable against the previous
step on the same track and seeds, so the owner can see what each change buys.

## Code organisation

- Real code lives in plain `.py` modules (environment, track loading,
  policy and value networks, each algorithm, training loop, evaluation,
  logging/plotting). Notebooks are for exploring and visualising only.
- pytest tests for the environment: ray distances, progress values, crash
  detection and loop closure on small hand-made bmps with known answers.
- Algorithm sanity checks: for example, a trivial toy track that a correct
  implementation must solve quickly.

## Observability

**Per iteration (CSV and TensorBoard):**

- mean episode return, mean episode length, crash rate, fraction of lap
  completed;
- policy standard deviation and entropy;
- policy loss and value loss; later PPO's KL divergence and clip fraction;
- value-function explained variance.

**Behaviour (essential, uses the bmp):**

- trajectory overlay on the bmp, coloured by speed, for a fixed evaluation
  episode, periodically;
- a short GIF replay of that episode every N iterations;
- a crash-location map / histogram of progress at crash.

**Later, in notebooks:** value over track positions, action histograms,
ray debug view. A live window during training is deliberately skipped at
first.

**Evaluation:** every N iterations, a separate run with fixed starts and
tracks and the policy's mean action (no sampling noise). This is the
learning curve to trust, not the noisy training curve. Checkpoints and the
random seed are saved with each run so runs are repeatable and comparable.

## Debugging and workflow

VS Code with the Python and Jupyter extensions, interpreter set to the
`uv` virtual environment, using the debugger with breakpoints. Claude Code
can also run inside VS Code.

## Decided by the owner

- Learning is the main goal.
- Everything hand-written except the neural-network computation.
- Algorithm route A.
- Tracks as bmp files; observation of ray distances plus speed; continuous
  throttle/brake and steering actions.

## My calls (for the owner to check)

- NumPy environment with a `reset()` / `step()` shape by convention only.
- Kinematic bicycle model for the physics.
- Per-step progress reward from a geodesic distance field, not a
  terminal-only reward.
- Time pressure through a step limit and an optional small time penalty.
- No explicit crash penalty at first: a crash ends the episode, which already
  forfeits all future progress reward. Add a penalty only if the agent
  learns to crash deliberately (for example, with a negative time penalty).
- A single track first.
- Tooling: Pillow, pygame or matplotlib, TensorBoard, pytest.

## Left to the builder

- Exact ray count, angles and maximum ray length.
- Physics constants, timestep, step limit.
- How a track bmp marks the start line and direction (colour convention or
  a separate parameter), and how loop closure is handled when the distance
  field wraps around the start.
- Network sizes, learning rates, batch sizes, discount γ and λ.
- Module and file layout within the structure above.
- Plot and GIF styling.
