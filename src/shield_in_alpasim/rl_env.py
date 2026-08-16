"""A cheap RL environment for the Tier 2 feasibility probe — safe RL via shielding.

AlpaSim exposes **no** RL interface (it is a gRPC closed-loop *eval* harness — no reward,
no reset, no per-step action injection; each rollout is a NuRec render + docker bring-up, so
training against it is infeasible). But the shield already carries its own fast, pure-numpy
kinematic world in `kitti_nav.vehicle`: `step_state` (transition), `safety_shield` (the veto),
`clearance` / `can_stop_safely` (reward + termination). This module turns those into a
Gym-style env so a low-dim policy can be trained *under the shield* at ~1e4 steps/s on one CPU
core — millions of steps in minutes, no GPU. The learned policy is then *evaluated* in AlpaSim
(its actual strength). See docs/TIER2_KICKOFF.md and the `tier2-rl-feasibility` memory.

The whole point of Tier 2 rides on one flag here: `shield=True` routes every action through
`safety_shield` before it hits the world, so unsafe exploration is vetoed and — from a
certified start — the agent provably never collides *while learning*. `shield=False` is the
unshielded control arm that crashes during exploration. Everything else is held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kitti_nav.vehicle import (
    CircleField,
    VehicleConfig,
    VehicleState,
    can_stop_safely,
    clearance,
    safety_shield,
    step_state,
)

# Discrete action grid the policy chooses from. A shield is a *filter*, so the policy still
# needs to propose an action; a small discrete set keeps the probe's policy-gradient stable
# (continuous control is a tuning rabbit-hole not worth it for a feasibility check). Three
# accel levels (brake / coast / go) × five steer angles = 15 actions.
ACCEL_CHOICES = (-4.5, 0.0, 2.0)
STEER_CHOICES = (-0.52, -0.26, 0.0, 0.26, 0.52)
ACTION_GRID: tuple[tuple[float, float], ...] = tuple(
    (a, s) for a in ACCEL_CHOICES for s in STEER_CHOICES
)
N_ACTIONS = len(ACTION_GRID)


@dataclass
class EnvConfig:
    """Scenario geometry + reward shaping for the corridor-navigation probe.

    A straight corridor along +x with discs scattered ahead; the agent must make forward
    progress to a goal line without leaving the corridor or hitting a disc. The task is
    *non-trivial under the shield*: the shield alone only brakes/swerves — it never makes
    progress — so a car that defers entirely to it times out with near-zero return. The
    agent has to learn to thread the obstacles to score, which is exactly what makes the
    shielded-vs-unshielded learning comparison meaningful.
    """

    corridor_half_width: float = 5.0     # |y| beyond this = offroad (terminate)
    goal_x: float = 32.0                 # reach this x = success
    n_obstacles: int = 4
    obstacle_x_range: tuple[float, float] = (10.0, 30.0)   # start clear so the init is certified
    obstacle_y_range: tuple[float, float] = (-3.0, 3.0)
    obstacle_radius_range: tuple[float, float] = (0.5, 1.0)
    start_speed: float = 2.0             # slow start -> initial state is shield-certifiable
    horizon: int = 160                   # max control steps per episode (dt=0.1 -> 16 s)
    n_nearest: int = 4                   # obstacles exposed in the observation

    # Reward shaping.
    progress_scale: float = 1.0          # reward per metre of +x progress
    collision_penalty: float = 20.0
    offroad_penalty: float = 10.0
    goal_bonus: float = 20.0
    step_penalty: float = 0.02           # small per-step cost -> prefer finishing, not dawdling


@dataclass
class ShieldNavEnv:
    """Gym-style corridor-navigation env over the shield's own kinematic model.

    `reset()` -> obs; `step(action_idx)` -> (obs, reward, done, info). `info` carries the
    shield's per-step flags (`intervened`, `ics`) and a `collision` flag, so the training
    loop can plot both the learning curve and the *safety during exploration* — the number
    the safe-RL claim actually rests on. Pure numpy, no AlpaSim, no torch.
    """

    cfg: EnvConfig = field(default_factory=EnvConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    shield: bool = True
    seed: int | None = None
    # Optional bank of pre-built obstacle layouts, each an (N, 3) array of (x, y, r) discs. When
    # given, `reset()` samples a layout from it instead of generating fresh random discs — the
    # drop-in seam for **real NuRec-scene fields** (SceneObstacleSource sampled at a set of ego
    # poses) so the scaled run can train on the eval distribution. None -> procedural (the probe).
    layouts: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state: VehicleState
        self._obstacles: CircleField
        self._t = 0
        self.reset()

    # --- observation --------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        # ego (v, steer, dist-to-goal) + n_nearest × (dx, dy, r)
        return 3 + 3 * self.cfg.n_nearest

    def _obstacle_discs_rig(self) -> np.ndarray:
        """The obstacle discs in the ego's rig frame (origin at rear axle, heading +x)."""
        circles = np.asarray(self._obstacles.circles, float).reshape(-1, 3)
        if len(circles) == 0:
            return np.zeros((0, 3))
        s = self._state
        dx = circles[:, 0] - s.x
        dy = circles[:, 1] - s.y
        c, sn = np.cos(-s.yaw), np.sin(-s.yaw)
        rx = c * dx - sn * dy
        ry = sn * dx + c * dy
        return np.column_stack([rx, ry, circles[:, 2]])

    def _observe(self) -> np.ndarray:
        s = self._state
        rig = self._obstacle_discs_rig()
        k = self.cfg.n_nearest
        feat = np.zeros((k, 3), float)
        if len(rig):
            # nearest-k by surface gap; pad with a far sentinel so absent slots read "clear".
            gap = np.linalg.norm(rig[:, :2], axis=1) - rig[:, 2]
            order = np.argsort(gap)[:k]
            feat[: len(order)] = rig[order]
            for j in range(len(order), k):
                feat[j] = (100.0, 0.0, 0.0)
        else:
            feat[:] = (100.0, 0.0, 0.0)
        ego = np.array([s.v, s.steer, self.cfg.goal_x - s.x], float)
        return np.concatenate([ego, feat.reshape(-1)])

    # --- episode lifecycle --------------------------------------------------------------
    def reset(self) -> np.ndarray:
        c = self.cfg
        if self.layouts:
            # Sample a pre-built layout (real-scene drop-in seam). We still assert the start is
            # certifiable below, so a layout with a disc inside the stopping envelope is rejected
            # loudly rather than silently making the guarantee vacuous.
            layout = self.layouts[int(self._rng.integers(len(self.layouts)))]
            self._obstacles = CircleField(np.asarray(layout, float).reshape(-1, 3))
        else:
            xs = self._rng.uniform(*c.obstacle_x_range, size=c.n_obstacles)
            ys = self._rng.uniform(*c.obstacle_y_range, size=c.n_obstacles)
            rs = self._rng.uniform(*c.obstacle_radius_range, size=c.n_obstacles)
            self._obstacles = CircleField(np.column_stack([xs, ys, rs]))
        self._state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=c.start_speed, steer=0.0)
        self._t = 0
        # Sanity: the start must be shield-certifiable, or the "provably no collisions" claim
        # is vacuous. Obstacles start >= 10 m ahead at a 2 m/s start, so this holds by design;
        # assert it rather than trust it.
        assert can_stop_safely(self._state, self._obstacles, self.vehicle), (
            "start state is not shield-certifiable — tighten obstacle_x_range / start_speed"
        )
        return self._observe()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        c, s = self.cfg, self._state
        accel_cmd, steer_cmd = ACTION_GRID[int(action_idx)]

        intervened = ics = False
        if self.shield:
            res = safety_shield(accel_cmd, steer_cmd, s, self._obstacles, self.vehicle)
            accel, steer = res.accel, res.steer
            intervened, ics = res.intervened, res.ics
        else:
            accel, steer = accel_cmd, steer_cmd

        nxt = step_state(s, float(accel), float(steer), self.vehicle)
        self._state = nxt
        self._t += 1

        reward = c.progress_scale * (nxt.x - s.x) - c.step_penalty
        done = False
        info = {"intervened": intervened, "ics": ics, "collision": False,
                "offroad": False, "goal": False}

        if clearance(nxt, self._obstacles, self.vehicle) < 0.0:
            reward -= c.collision_penalty
            done = True
            info["collision"] = True
        elif abs(nxt.y) > c.corridor_half_width:
            reward -= c.offroad_penalty
            done = True
            info["offroad"] = True
        elif nxt.x >= c.goal_x:
            reward += c.goal_bonus
            done = True
            info["goal"] = True
        elif self._t >= c.horizon:
            done = True

        return self._observe(), float(reward), done, info

    # --- introspection (tests / debugging) ----------------------------------------------
    @property
    def state(self) -> VehicleState:
        return self._state

    @property
    def obstacles(self) -> CircleField:
        return self._obstacles
