"""AlpaSim driver plugin wrapping kitti-nav's hard safety shield.

AlpaSim's `BaseTrajectoryModel.predict()` gets camera frames in and must return waypoint
poses out; kitti-nav's shield takes a per-step `(accel_cmd, steer_cmd)` in and an
`ObstacleField` to check against, and returns a certified-safe `(accel, steer)` out.

Obstacles are real now: with `$SHIELD_SCENE_USDZ` set, `predict()` samples the scene's
actors at the current ego pose and timestamp (`scene.py`, `obstacles.py`) and the shield
brakes for them. Unset, the field is empty and the car coasts — the old scaffold behaviour,
kept as the fallback because a driver that refuses to start is worse on a metered box.

**Still missing a policy.** `_rollout` commands `(0.0, 0.0)`, so this drives straight and
holds speed until the shield vetoes. The shield is a filter: it certifies a proposed action
and never proposes one, so nothing here self-drives until a policy is wired upstream. See
HANDOFF.md, "Open decision".

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

from kitti_nav.vehicle import CircleField, VehicleConfig, VehicleState, shielded_rollout
from shield_in_alpasim.scene import SceneObstacleSource, ego_pose_from_history

# A field with no obstacles reports every point infinitely clear, so the shield never
# intervenes. Used when no scene artifact is configured, and as the per-call fallback when
# the ego pose history has not arrived yet.
EMPTY_FIELD = CircleField(None)


class ShieldedDriver(BaseTrajectoryModel if _HAS_ALPASIM else object):
    """Registered as `shielded` under the `alpasim.models` entry point (pyproject.toml).

    CPU-only: the shield is pure numpy, so unlike AlpaSim's stock drivers this one needs
    no GPU and no checkpoint (see `configs/driver/shielded.yaml`).
    """

    def __init__(
        self,
        cfg: VehicleConfig,
        camera_ids: list[str],
        context_length: int = 1,
        output_frequency_hz: int = 2,
        horizon_steps: int = 6,
        obstacles=None,
        scene_source: SceneObstacleSource | None = None,
    ):
        self._cfg = cfg
        self._camera_ids = list(camera_ids)
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._horizon_steps = horizon_steps
        # A fixed field, for tests and for `scripts/preview_trajectory.py`. Ignored when a
        # `scene_source` is present, since that resamples per call as actors move.
        self._obstacles = EMPTY_FIELD if obstacles is None else obstacles
        self._scene_source = scene_source

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        # `device` is ignored on purpose — the shield is numpy, there is nothing to move
        # onto a GPU. `context_length` is None when the config leaves it to the model.
        #
        # The scene path comes from the environment, not `model_cfg`: the driver merges its
        # YAML onto a structured `DriverConfig` in struct mode, so an extra key here would
        # raise. See `scene.py` for the full reasoning.
        scene_source = SceneObstacleSource.from_env()

        # Take the ego's real dimensions from the scene when we have them. kitti-nav's
        # defaults are a VW Passat; AlpaSim's ego is a much larger S223, and shielding the
        # wrong footprint is optimistic in exactly the direction that matters.
        cfg = VehicleConfig()
        if scene_source is not None and scene_source.ego_vehicle_config is not None:
            cfg = scene_source.ego_vehicle_config

        return cls(
            cfg=cfg,
            camera_ids=camera_ids,
            context_length=context_length or 1,
            output_frequency_hz=output_frequency_hz,
            scene_source=scene_source,
        )

    def _encode_command(self, command: "DriveCommand") -> int:
        # The shield filters a command, it does not propose one, so LEFT/STRAIGHT/RIGHT has
        # nothing to act on until a real upstream policy is wired in (plan step 4).
        # DriveCommand is an IntEnum, so this is its own encoding.
        return int(command)

    def _obstacles_for(self, prediction_input) -> "CircleField":
        """Obstacle field for this inference: scene actors if configured, else the fixed one.

        Falls back to the fixed field when the ego pose history is empty, which happens on
        the first query of a session before any pose has been reported. An empty field is
        the safe direction to fail here only because the car is still under ground-truth
        replay at that point (`force_gt_duration_us`); it is not yet driving on the shield.
        """
        if self._scene_source is None:
            return self._obstacles

        ego = ego_pose_from_history(prediction_input.ego_pose_history)
        if ego is None:
            return self._obstacles

        ego_xy, ego_yaw, timestamp_us = ego
        return self._scene_source.field_at(ego_xy, ego_yaw, timestamp_us)

    def _rollout(self, initial_speed: float, obstacles=None) -> np.ndarray:
        """Roll the shield forward open-loop for `horizon_steps`, return `(T, 2)` xy.

        Placeholder policy: commands "go straight, hold speed." Real use replaces this
        commanded (accel, steer) with whatever upstream policy is being validated — the
        shield's job is to filter it, not propose it. Kept free of AlpaSim types so it's
        testable without AlpaSim/torch installed.
        """
        obstacles = self._obstacles if obstacles is None else obstacles
        # Waypoints are ego-relative in the rig frame, so the rollout always starts at the
        # origin. Initial steer is assumed centred: AlpaSim's `PredictionInput` carries no
        # road-wheel angle (`ego_pose_history` could be differenced for it — refinement,
        # not needed while the commanded steer is zero anyway).
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=initial_speed, steer=0.0)
        # cfg.dt is the shield's control step; sub-step up to each output waypoint so the
        # shield (and its braking lookahead) always runs at its own tested integration rate,
        # regardless of what output_frequency_hz AlpaSim asks for.
        substeps_per_waypoint = max(1, round(1.0 / self._output_frequency_hz / self._cfg.dt))
        states, _stats = shielded_rollout(
            lambda _state: (0.0, 0.0),
            state,
            obstacles,
            self._cfg,
            n_steps=self._horizon_steps * substeps_per_waypoint,
        )
        xy = np.zeros((self._horizon_steps, 2))
        for t in range(self._horizon_steps):
            # states[0] is the initial state, so waypoint t lands substeps*(t+1) steps in.
            # shielded_rollout truncates on collision; clamping holds the last pose so the
            # trajectory stays the fixed length AlpaSim expects.
            xy[t] = states[min((t + 1) * substeps_per_waypoint, len(states) - 1)].xy
        return xy

    def predict(self, prediction_input: "PredictionInput") -> "ModelPrediction":
        self._validate_cameras(prediction_input.camera_images)
        xy = self._rollout(prediction_input.speed, self._obstacles_for(prediction_input))
        # The shield plans in the ground plane; `from_planar` lifts (T, 2) + headings into
        # the (K, T, 3) / (K, T, 3, 3) pose pair `ModelPrediction` actually holds.
        return ModelPrediction.from_planar(xy, self._compute_headings_from_trajectory(xy))

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz
