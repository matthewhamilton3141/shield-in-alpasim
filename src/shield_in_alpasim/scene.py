"""Getting scene geometry to the driver, given that AlpaSim will not hand it over.

`obstacles.py` turns actors into an obstacle field. This module answers the prior question:
*which* actors, and where is the ego right now.

**Why an environment variable.** The driver cannot ask what scene it is in. `PredictionInput`
carries no scene id, `BaseTrajectoryModel` has no session hook, and the servicer keeps
`debug_scene_id` to itself (`driver/main.py:174,218`). The obvious fix — a path in
`configs/driver/shielded.yaml` — does not work either: the driver merges its YAML onto a
structured `DriverConfig` (`main.py:1189`), so OmegaConf runs in struct mode and an unknown
key under `model:` raises rather than passing through. Adding one means forking AlpaSim's
schema.

So the path arrives out-of-band, through the `environments` list the wizard already supports
on every service (see `trafficsim/catk.yaml` upstream for the same mechanism). Unset means
no ground-truth geometry and an empty field — the pre-existing inert behaviour, not a crash,
because a driver that refuses to start is worse on a metered box than one that coasts.

This is the privileged channel, and it is supposed to look like one. See HANDOFF.md: the
scene id is withheld from real benchmark runs by design, so this arm is a baseline, never a
score.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os

import numpy as np

from kitti_nav.vehicle import VehicleConfig
from shield_in_alpasim.obstacles import DEFAULT_MAX_RANGE_M, field_from_traffic_objects

logger = logging.getLogger(__name__)

# Absolute path to the scene's `.usdz`, inside the driver container. With the wizard's
# standard mount that is `/mnt/nre-data/<sceneset>/<scene>.usdz`.
SCENE_ENV_VAR = "SHIELD_SCENE_USDZ"


def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rotation about z) from a gRPC-order quaternion.

    Matches `alpasim_utils.geometry.quat_to_yaw`, which routes through the compiled
    `Pose.yaw()`. Reimplemented in three lines of numpy so the ego-pose path stays testable
    on a box without AlpaSim — the same reason `obstacles.py` keeps its core array-only.
    """
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ego_pose_from_history(ego_pose_history) -> tuple[np.ndarray, float, int] | None:
    """Latest ego pose as `(xy, yaw, timestamp_us)`, or None if the history is empty.

    `ego_pose_history` is a list of gRPC `PoseAtTime` — `.pose.vec.{x,y,z}`,
    `.pose.quat.{w,x,y,z}`, `.timestamp_us` — kept sorted by the servicer
    (`main.py:335`), so the last entry is the newest.

    Note the frame: these are `rig_est` poses in the `local` frame, i.e. deliberately
    noised. Scene actors are in the true `local` frame. Putting them in one frame therefore
    inherits that localization error — the "perfect geometry, imperfect localization"
    caveat in HANDOFF.md, and the one place this arm is not actually ground truth.
    """
    if not ego_pose_history:
        return None

    latest = ego_pose_history[-1]
    vec, quat = latest.pose.vec, latest.pose.quat
    return (
        np.array([vec.x, vec.y], dtype=float),
        yaw_from_quat(quat.w, quat.x, quat.y, quat.z),
        int(latest.timestamp_us),
    )


def ego_config_from_rig(rig_vehicle_config, base: VehicleConfig | None = None) -> VehicleConfig:
    """Retag kitti-nav's vehicle geometry with the ego AlpaSim is actually simulating.

    **This matters more than it looks.** kitti-nav's defaults describe the KITTI recording
    car, a VW Passat B6: 4.77 x 1.82 m. AlpaSim's default ego is a Mercedes S223 —
    5.393 x 2.109 m, with the rig origin 1.3 m ahead of the rear bumper plane rather than
    0.97. Left alone, the shield would certify a footprint ~0.6 m short and ~0.3 m narrow of
    the body the simulator collides with, and quietly hand back clearances the car does not
    have. A safety envelope computed for the wrong car is not a safety envelope.

    Runtime resolves the ego the same way (`unbound_rollout.py:294`): the scene's
    `rig.vehicle_config`, or a config override. Only *geometry* is taken from it —
    wheelbase, actuation limits and shield parameters stay kitti-nav's, because AlpaSim's
    `VehicleConfig` carries no steering or brake model to copy. Wheelbase in particular is
    still the Passat's 2.71 m against a real ~3.11 m, so the bicycle model turns slightly
    tighter than the S223 does; it affects the swept path, not the straight-line braking
    distance the certificate mostly rests on.
    """
    base = base or VehicleConfig()
    if rig_vehicle_config is None:
        return base

    return dataclasses.replace(
        base,
        length=float(rig_vehicle_config.aabb_x_m),
        width=float(rig_vehicle_config.aabb_y_m),
        # `aabb_x_offset_m` runs from the rig origin *back* to the bumper plane, so it is
        # negative where kitti-nav's rear overhang is positive.
        rear_overhang=float(-rig_vehicle_config.aabb_x_offset_m),
    )


class SceneObstacleSource:
    """Scene actors, sampled into a rig-frame obstacle field on demand.

    Holds AlpaSim's `TrafficObjects` for one scene. `Artifact` caches its own parse, and the
    per-call work is a few hundred rows of numpy, so this is re-sampled every `predict()`
    rather than cached — actors move, and a stale field is the failure this whole component
    exists to avoid.
    """

    def __init__(self, traffic_objects, n_discs: int = 5,
                 max_range_m: float = DEFAULT_MAX_RANGE_M,
                 ego_vehicle_config: VehicleConfig | None = None):
        self._traffic_objects = traffic_objects
        self._n_discs = n_discs
        self._max_range_m = max_range_m
        # The ego the scene actually simulates, in kitti-nav terms. None when the artifact
        # carries no rig vehicle config, in which case the caller keeps its own default.
        self.ego_vehicle_config = ego_vehicle_config

    @classmethod
    def from_usdz(cls, usdz_path: str, **kwargs) -> "SceneObstacleSource":
        """Load scene geometry, and the ego's dimensions, from a `.usdz` artifact.

        `alpasim_utils` is imported here rather than at module scope so this file stays
        importable without AlpaSim installed.
        """
        from alpasim_utils.artifact import Artifact

        artifact = Artifact(source=usdz_path)
        traffic_objects = artifact.traffic_objects

        ego_vehicle_config = None
        try:
            ego_vehicle_config = ego_config_from_rig(artifact.rig.vehicle_config)
        except Exception:  # noqa: BLE001 — geometry is the point; the rig is a bonus
            logger.warning(
                "Could not read the rig's vehicle config from %s; the shield will use "
                "kitti-nav's KITTI-car footprint, which is smaller than AlpaSim's default "
                "ego and therefore optimistic. Check this before trusting a result.",
                usdz_path, exc_info=True,
            )

        logger.info("Loaded %d scene actors from %s", len(traffic_objects), usdz_path)
        return cls(traffic_objects, ego_vehicle_config=ego_vehicle_config, **kwargs)

    @classmethod
    def from_env(cls, **kwargs) -> "SceneObstacleSource | None":
        """Build from `$SHIELD_SCENE_USDZ`, or None when it is unset.

        A bad path is worth failing loudly on — it means the run was *meant* to have
        ground-truth geometry and silently would not have, which would look like a shield
        that never fires rather than like a misconfiguration.
        """
        usdz_path = os.environ.get(SCENE_ENV_VAR)
        if not usdz_path:
            logger.warning(
                "%s is unset: running with an empty obstacle field, so the shield will "
                "never intervene. Set it to the scene's .usdz to enable the "
                "ground-truth arm.", SCENE_ENV_VAR,
            )
            return None

        if not os.path.exists(usdz_path):
            raise FileNotFoundError(
                f"{SCENE_ENV_VAR}={usdz_path!r} does not exist. It must be the path to the "
                "scene's .usdz *inside the driver container* — with the wizard's standard "
                "mount, /mnt/nre-data/<sceneset>/<scene>.usdz."
            )

        return cls.from_usdz(usdz_path, **kwargs)

    def field_at(self, ego_xy, ego_yaw: float, timestamp_us: int):
        """Obstacle field in the ego's rig frame at `timestamp_us`."""
        return field_from_traffic_objects(
            self._traffic_objects, ego_xy, ego_yaw, timestamp_us,
            n_discs=self._n_discs, max_range_m=self._max_range_m,
        )

    def __len__(self) -> int:
        return len(self._traffic_objects)
