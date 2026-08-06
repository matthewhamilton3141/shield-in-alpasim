"""AlpaSim driver plugin wrapping kitti-nav's hard safety shield.

SCAFFOLD — not functional yet. See ../../README.md "The actual gap" before extending this.

AlpaSim's `BaseTrajectoryModel.predict()` gets camera frames in and must return a `(T, 2)`
trajectory out; kitti-nav's shield takes a per-step `(accel_cmd, steer_cmd)` in and an
`ObstacleField` to check against, and returns a certified-safe `(accel, steer)` out. Neither
side of that gap is bridged here yet — `predict()` below runs the shield against a stub
`ObstacleField` that reports no obstacles, so the plugin loads and returns a straight-line
trajectory. That is enough to prove the AlpaSim harness plumbing (entry point, camera_ids,
context_length) without either open problem being solved.

Provenance: the shield itself is not implemented here — it's imported from `kitti_nav`,
see ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from alpasim_driver.models.base import (
        BaseTrajectoryModel,
        DriveCommand,
        ModelPrediction,
        PredictionInput,
    )

    _HAS_ALPASIM = True
except ImportError:  # AlpaSim isn't installed (e.g. on a Mac dev box, no GPU stack) —
    # the class below still imports and its non-AlpaSim logic is still testable.
    _HAS_ALPASIM = False
    BaseTrajectoryModel = object  # type: ignore[assignment,misc]

from kitti_nav.vehicle import VehicleConfig, VehicleState, safety_shield, step_state


class NoObstacles:
    """Stub `ObstacleField`: reports every point as clear.

    Placeholder until problem 1 (README: "where does the shield's obstacle field come
    from?") is solved — a real field has to be synthesized from AlpaSim's camera frames
    (or sourced from privileged scene geometry, if AlpaSim exposes any).
    """

    def distance_to_obstacles(self, points: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(points).reshape(-1, 2)), np.inf)


class ShieldedDriver(BaseTrajectoryModel if _HAS_ALPASIM else object):
    """Registered as `shielded` under the `alpasim.models` entry point (pyproject.toml)."""

    def __init__(self, cfg: VehicleConfig, output_frequency_hz: int = 2, horizon_steps: int = 6):
        self._cfg = cfg
        self._output_frequency_hz = output_frequency_hz
        self._horizon_steps = horizon_steps
        self._obstacles = NoObstacles()

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        return cls(cfg=VehicleConfig(), output_frequency_hz=output_frequency_hz)

    def _encode_command(self, command) -> int:
        return int(command)

    def _rollout(self, initial_speed: float) -> np.ndarray:
        """Roll the shield forward open-loop for `horizon_steps`, return `(T, 2)` xy.

        Placeholder policy: commands "go straight, hold speed." Real use replaces this
        commanded (accel, steer) with whatever upstream policy is being validated — the
        shield's job is to filter it, not propose it. Kept free of AlpaSim types so it's
        testable without AlpaSim/torch installed.
        """
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=initial_speed, steer=0.0)
        xy = np.zeros((self._horizon_steps, 2))
        # cfg.dt is the shield's control step; sub-step up to each output waypoint so the
        # shield (and its braking lookahead) always runs at its own tested integration rate,
        # regardless of what output_frequency_hz AlpaSim asks for.
        substeps_per_waypoint = max(1, round(1.0 / self._output_frequency_hz / self._cfg.dt))
        for t in range(self._horizon_steps):
            for _ in range(substeps_per_waypoint):
                result = safety_shield(0.0, 0.0, state, self._obstacles, self._cfg)
                state = step_state(state, result.accel, result.steer, self._cfg)
            xy[t] = (state.x, state.y)
        return xy

    @staticmethod
    def _headings_from_xy(xy: np.ndarray) -> np.ndarray:
        prev = np.zeros_like(xy)
        prev[1:] = xy[:-1]
        deltas = xy - prev
        return np.arctan2(deltas[:, 1], deltas[:, 0])

    def predict(self, prediction_input: "PredictionInput") -> "ModelPrediction":
        xy = self._rollout(prediction_input.speed)
        return ModelPrediction(trajectory_xy=xy, headings=self._headings_from_xy(xy))

    @property
    def camera_ids(self) -> list[str]:
        return ["front_wide"]

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz
