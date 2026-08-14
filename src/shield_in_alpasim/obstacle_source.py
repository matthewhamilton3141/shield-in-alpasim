"""Camera-perception obstacle field: RGB frames -> kitti-nav `CircleField`.

The *learned-perception* arm of the experiment, and the counterpart to `obstacles.py`'s
ground-truth arm. The shield's whole value — a provable no-collision certificate — is only as
sound as the geometry it certifies against. Swapping this camera source in for the ground-truth
source, holding the policy and shield fixed, is how you *measure* how much the guarantee degrades
when perception is learned instead of perfect. That degradation is the result the project exists
to produce.

Pipeline (see the README "camera-perception obstacle source" sketch):

    frame -> metric depth -> back-project (known intrinsics) -> rig frame (known extrinsics)
          -> drop ground/sky by height -> BEV occupancy grid -> one disc per occupied cell.

Everything here is pure numpy and *dependency-injects the depth model*, so the geometry is
unit-tested on a dev box with a synthetic depth map; only the depth net's forward pass needs a
GPU. Same discipline as the rest of the repo: the math is tested locally, the model runs on the
box.

FRAME CONVENTIONS — VERIFY AGAINST ALPASIM ON THE BOX BEFORE TRUSTING A NUMBER (this is exactly
the kind of interface assumption this project checks against upstream rather than guessing):
  - Back-projection assumes the OpenCV pinhole camera frame — x right, y down, z forward —
    matching the `opencv_pinhole` intrinsics AlpaSim's `extra_cameras` config carries.
  - `rig_to_camera` is taken to map a rig-frame point into the camera frame:
    `p_cam = R @ p_rig + t`, so `p_rig = Rᵀ (p_cam - t)`. If AlpaSim's stored transform is the
    inverse (camera pose in the rig), transpose R and negate accordingly.
  - Output is the rig frame `obstacles.py` and the shield use: x forward, y left, z up.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from kitti_nav.vehicle import CircleField

logger = logging.getLogger(__name__)


class ObstacleSource(Protocol):
    """Produces the shield's obstacle field, in the ego rig frame, for one inference.

    Both the ground-truth source (`scene.py`) and this camera source satisfy this shape, so the
    driver can swap them by configuration and everything downstream (tracker, shield, fixes) is
    unchanged — which is what makes the perfect-vs-learned comparison a clean, single-variable
    experiment.
    """

    def field_for(self, prediction_input) -> CircleField: ...


def quat_xyzw_to_matrix(q) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion (AlpaSim's `rotation_xyzw` order)."""
    x, y, z, w = (np.asarray(q, float) / np.linalg.norm(q)).tolist()
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def backproject_depth(depth, fx, fy, cx, cy, max_range_m, stride: int = 1) -> np.ndarray:
    """`(H, W)` metric depth -> `(N, 3)` points in the OpenCV camera frame.

    Keeps only finite, positive, in-range pixels. `stride` subsamples the image (a depth map is
    ~1e5 pixels and the shield needs coverage, not per-pixel fidelity), the cheapest way to keep
    the per-call cost bounded.
    """
    depth = np.asarray(depth, float)
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0.0) & (d <= max_range_m)
    u, v, d = us[valid], vs[valid], d[valid]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    return np.stack([x, y, d], axis=1)


def camera_to_rig(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """OpenCV camera-frame points -> rig frame, given `rig_to_camera` (`p_cam = R p_rig + t`).

    `p_rig = Rᵀ (p_cam - t)`; as row vectors that is `(pts_cam - t) @ R`.
    """
    if len(pts_cam) == 0:
        return pts_cam.reshape(-1, 3)
    return (pts_cam - np.asarray(t, float).reshape(3)) @ np.asarray(R, float)


def height_band_mask(pts_rig: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Keep points whose rig-frame height (z) is in `[z_min, z_max]` — crude ground/sky removal.

    Cheap and wrong on slopes/curbs (a v2 uses a semantic-segmentation mask); good enough to
    stand up the learned arm and start measuring.
    """
    z = pts_rig[:, 2]
    return (z >= z_min) & (z <= z_max)


def occupancy_to_circles(xy: np.ndarray, cell_m: float, max_range_m: float,
                         min_pts: int) -> np.ndarray:
    """BEV points `(N, 2)` -> `(M, 3)` circles: one disc per occupied cell.

    Rasterize to a `cell_m` grid, keep cells with at least `min_pts` points, and circumscribe
    each with a disc (radius = cell half-diagonal). Circumscribing *over-covers* the cell, which
    is the conservative direction for a safety field — it can invent clearance the world does not
    have only by reading a phantom obstacle, never by missing a real one within an occupied cell.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) == 0:
        return np.zeros((0, 3))
    xy = xy[np.linalg.norm(xy, axis=1) <= max_range_m]
    if len(xy) == 0:
        return np.zeros((0, 3))
    cells = np.floor(xy / cell_m).astype(np.int64)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    occupied = uniq[counts >= min_pts]
    if len(occupied) == 0:
        return np.zeros((0, 3))
    centres = (occupied + 0.5) * cell_m
    radius = float(np.hypot(cell_m, cell_m) / 2.0)
    return np.column_stack([centres, np.full(len(centres), radius)])


class CameraObstacleSource:
    """`ObstacleSource` backed by a monocular metric-depth net over one camera.

    `depth_model` is any callable `HWC image -> (H, W) metric depth in metres`; inject a
    pretrained net (Depth Anything V2 metric / Metric3D / UniDepth) on the box, or a synthetic
    map in tests. `intrinsics` is `(fx, fy, cx, cy)` and `rig_to_camera` is `(R, t)` — both come
    straight from AlpaSim's `extra_cameras` config block (`from_config` builds them), so there is
    no calibration step.
    """

    def __init__(self, depth_model, camera_id: str, intrinsics, rig_to_camera,
                 ground_band=(0.3, 2.5), max_range_m: float = 40.0,
                 cell_m: float = 0.5, min_pts: int = 5, stride: int = 2):
        self._depth = depth_model
        self._camera_id = camera_id
        self._fx, self._fy, self._cx, self._cy = intrinsics
        R, t = rig_to_camera
        self._R = np.asarray(R, float).reshape(3, 3)
        self._t = np.asarray(t, float).reshape(3)
        self._ground_band = ground_band
        self._max_range_m = max_range_m
        self._cell_m = cell_m
        self._min_pts = min_pts
        self._stride = stride

    @classmethod
    def from_config(cls, depth_model, camera_id: str, opencv_pinhole: dict,
                    rig_to_camera: dict, **kwargs) -> "CameraObstacleSource":
        """Build from AlpaSim's `extra_cameras` entry (its `intrinsics.opencv_pinhole` +
        `rig_to_camera`). See the frame-convention caveat in the module docstring."""
        fx, fy = opencv_pinhole["focal_length"]
        cx, cy = opencv_pinhole["principal_point"]
        R = quat_xyzw_to_matrix(rig_to_camera["rotation_xyzw"])
        t = np.asarray(rig_to_camera["translation_m"], float)
        return cls(depth_model, camera_id, (fx, fy, cx, cy), (R, t), **kwargs)

    def field_for(self, prediction_input) -> CircleField:
        frames = prediction_input.camera_images[self._camera_id]
        depth = np.asarray(self._depth(frames[-1].image), float)
        pts_cam = backproject_depth(
            depth, self._fx, self._fy, self._cx, self._cy, self._max_range_m, self._stride
        )
        pts_rig = camera_to_rig(pts_cam, self._R, self._t)
        obstacles = pts_rig[height_band_mask(pts_rig, *self._ground_band)]
        circles = occupancy_to_circles(
            obstacles[:, :2], self._cell_m, self._max_range_m, self._min_pts
        )
        logger.info(
            "camera obstacle source: %d in-band points -> %d discs", len(obstacles), len(circles)
        )
        return CircleField(circles if len(circles) else None)
