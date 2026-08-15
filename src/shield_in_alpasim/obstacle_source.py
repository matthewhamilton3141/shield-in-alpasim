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
  - `rig_to_camera` in AlpaSim's config stores the **camera's pose in the rig** (its
    `translation_m` is the camera's location — ~1.66 m forward, 1.5 m up), i.e. `(R, t)` is
    camera→rig and `p_rig = R @ p_cam + t`. Verified on the box: the inverse put obstacles behind
    the ego; this puts a forward pixel ahead at camera height.
  - Output is the rig frame `obstacles.py` and the shield use: x forward, y left, z up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kitti_nav.vehicle import CircleField

logger = logging.getLogger(__name__)


# Calibration of `camera_front_wide_120fov`, copied from the `extra_cameras` block the renderer
# uses (vavam_configs.yaml / shielded_vavam_configs.yaml). The driver's `from_config` does not
# receive `extra_cameras`, so for the single-camera VaVAM setup we carry the known calibration
# here; `field_for` rescales the intrinsics to whatever resolution the frame actually arrives at.
FRONT_WIDE_REF_HW = (1080, 1920)
FRONT_WIDE_FOCAL = (1545.0, 1545.0)          # fx, fy at the reference resolution
FRONT_WIDE_PRINCIPAL = (960.0, 560.0)        # cx, cy
FRONT_WIDE_RIG_TO_CAMERA = {
    "translation_m": [1.65897811, -0.01443456, 1.51539499],
    "rotation_xyzw": [-0.49929397355810856, 0.5039939168301356,
                      -0.4972939976976715, 0.49939397235113037],
}

# 4-camera surround calibration for the multi-camera (surround) perception arm. These are the
# `extra_cameras` poses inlined into `shielded_vavam_surround_configs.yaml` (themselves from
# AlpaSim's transfuser_configs.yaml), so the perception calibration matches exactly what the
# renderer uses for that config. Every camera shares the same `opencv_pinhole` intrinsics; only
# the rig pose differs. NOTE the front-wide pose here is transfuser's, ~1 cm off the box-verified
# single-cam `FRONT_WIDE_*` above — each perception calib is kept consistent with the extra_cameras
# of the config it runs under (single-cam VaVAM uses FRONT_WIDE_*, surround uses these), rather
# than forced to one number. Re-verify each camera on the box via the BEV: side discs should land
# to the sides, the rear camera's obstacles behind the ego (its translation x is negative).
SURROUND_REF_HW = (1080, 1920)
SURROUND_FOCAL = (1545.0, 1545.0)
SURROUND_PRINCIPAL = (960.0, 560.0)
SURROUND_RIG_TO_CAMERA = {
    "camera_cross_left_120fov": {
        "translation_m": [1.646354, 0.143369, 1.521469],
        "rotation_xyzw": [0.679354, -0.207915, 0.215233, -0.670018]},
    "camera_front_wide_120fov": {
        "translation_m": [1.670100, -0.025875, 1.522623],
        "rotation_xyzw": [0.509222, -0.503331, 0.495086, -0.492180]},
    "camera_cross_right_120fov": {
        "translation_m": [1.626168, -0.161517, 1.526269],
        "rotation_xyzw": [0.205424, -0.674057, 0.676355, -0.214458]},
    "camera_rear_left_70fov": {
        "translation_m": [-0.486641, -0.000595, 1.486321],
        "rotation_xyzw": [0.503851, 0.497823, -0.499723, -0.498582]},
}


def _frame_image(frame):
    """The HWC image out of a frame, whether it's a `CameraFrame(timestamp, image)` namedtuple, a
    plain `(timestamp, image)` tuple (what the servicer actually sends), or a bare array."""
    if hasattr(frame, "image"):
        return frame.image
    if isinstance(frame, tuple):
        return frame[1]  # (timestamp_us, image)
    return frame


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
    """OpenCV camera-frame points -> rig frame.

    AlpaSim's `rig_to_camera` stores the **camera's pose in the rig** (its `translation_m` is the
    camera's location in the rig — ~1.66 m forward, 1.5 m up), i.e. `(R, t)` is camera→rig:
    `p_rig = R @ p_cam + t`. (Verified on the box: with this, a forward pixel lands ahead at
    camera height; the inverse put obstacles behind the ego.) As row vectors: `pts_cam @ Rᵀ + t`.
    """
    if len(pts_cam) == 0:
        return pts_cam.reshape(-1, 3)
    return pts_cam @ np.asarray(R, float).T + np.asarray(t, float).reshape(3)


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


@dataclass
class CorridorGate:
    """Geometric relevance gate on rig-frame points — the *filter seam* for cleaning perception.

    Dense monocular depth over the surround rig reconstructs the WHOLE street (building facades,
    curbs, parked cars on both sides) — ~2500 discs/cycle, of which only a couple dozen are ever in
    the ego's driving path. The rest is static roadside clutter the forward-braking shield never
    acts on, but it buries the real threats, looks like chaos, and risks a phantom brake from a
    stray in-path point. This keeps only points inside a corridor around the forward path:

        x in [x_min, x_max]   (rig x is forward; a little behind, out to braking range ahead)
        |y| <= half_width     (a few lanes wide — keeps lead + adjacent-lane + cross traffic)
        and within max_range_m of the ego.

    Pure geometry (numpy), so it's unit-tested off the box. It is one pluggable stage: a semantic
    filter (keep only vehicles/pedestrians) can stack in the same seam later — see the obstacle
    sources' `point_filter` and docs/MULTICAM_HANDOFF.md.
    """

    x_min: float = -3.0
    x_max: float = 25.0
    half_width: float = 4.0
    max_range_m: float = 20.0

    def __call__(self, pts_rig) -> np.ndarray:
        pts = np.asarray(pts_rig, float).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, bool)
        x, y = pts[:, 0], pts[:, 1]
        keep = (x >= self.x_min) & (x <= self.x_max) & (np.abs(y) <= self.half_width)
        if self.max_range_m is not None:
            keep &= np.hypot(x, y) <= self.max_range_m
        return keep


class CameraObstacleSource:
    """`ObstacleSource` backed by a monocular metric-depth net over one camera.

    `depth_model` is any callable `HWC image -> (H, W) metric depth in metres`; inject a
    pretrained net (Depth Anything V2 metric / Metric3D / UniDepth) on the box, or a synthetic
    map in tests. `intrinsics` is `(fx, fy, cx, cy)` and `rig_to_camera` is `(R, t)` — both come
    straight from AlpaSim's `extra_cameras` config block (`from_config` builds them), so there is
    no calibration step.
    """

    def __init__(self, depth_model, camera_id: str, intrinsics, rig_to_camera,
                 ref_hw=FRONT_WIDE_REF_HW, ground_band=(0.3, 2.5), max_range_m: float = 40.0,
                 cell_m: float = 0.5, min_pts: int = 5, stride: int = 2, point_filter=None):
        self._depth = depth_model
        self._camera_id = camera_id
        self._fx, self._fy, self._cx, self._cy = intrinsics
        self._ref_h, self._ref_w = ref_hw  # resolution the intrinsics were calibrated at
        R, t = rig_to_camera
        self._R = np.asarray(R, float).reshape(3, 3)
        self._t = np.asarray(t, float).reshape(3)
        self._ground_band = ground_band
        self._max_range_m = max_range_m
        self._cell_m = cell_m
        self._min_pts = min_pts
        self._stride = stride
        # Optional relevance filter on rig-frame points (e.g. CorridorGate) — the pluggable seam
        # that drops roadside clutter; None = keep everything the height band passes.
        self._point_filter = point_filter

    def _intrinsics_for(self, h: int, w: int) -> tuple[float, float, float, float]:
        """Intrinsics rescaled from the reference resolution to the actual frame `(h, w)`.

        The renderer may deliver, or the model may resize to, a resolution other than the one the
        intrinsics were calibrated at; a pinhole's focal length and principal point scale linearly
        with resolution, so rescale rather than trust the frame is full-res.
        """
        sw, sh = w / self._ref_w, h / self._ref_h
        return self._fx * sw, self._fy * sh, self._cx * sw, self._cy * sh

    @classmethod
    def from_config(cls, depth_model, camera_id: str, opencv_pinhole: dict,
                    rig_to_camera: dict, ref_hw=FRONT_WIDE_REF_HW, **kwargs) -> "CameraObstacleSource":
        """Build from an AlpaSim `extra_cameras` entry (`intrinsics.opencv_pinhole` +
        `rig_to_camera`). See the frame-convention caveat in the module docstring."""
        fx, fy = opencv_pinhole["focal_length"]
        cx, cy = opencv_pinhole["principal_point"]
        R = quat_xyzw_to_matrix(rig_to_camera["rotation_xyzw"])
        t = np.asarray(rig_to_camera["translation_m"], float)
        return cls(depth_model, camera_id, (fx, fy, cx, cy), (R, t), ref_hw=ref_hw, **kwargs)

    @classmethod
    def front_wide(cls, depth_model, camera_id: str = "camera_front_wide_120fov", **kwargs):
        """The single-camera VaVAM setup: `camera_front_wide_120fov` with its known calibration."""
        return cls.from_config(
            depth_model, camera_id,
            {"focal_length": FRONT_WIDE_FOCAL, "principal_point": FRONT_WIDE_PRINCIPAL},
            FRONT_WIDE_RIG_TO_CAMERA, ref_hw=FRONT_WIDE_REF_HW, **kwargs,
        )

    _frame_image = staticmethod(_frame_image)

    def field_for(self, prediction_input) -> CircleField:
        frames = prediction_input.camera_images[self._camera_id]
        depth = np.asarray(self._depth(self._frame_image(frames[-1])), float)
        fx, fy, cx, cy = self._intrinsics_for(*depth.shape[:2])
        pts_cam = backproject_depth(depth, fx, fy, cx, cy, self._max_range_m, self._stride)
        pts_rig = camera_to_rig(pts_cam, self._R, self._t)
        keep = height_band_mask(pts_rig, *self._ground_band)
        if self._point_filter is not None:
            keep = keep & self._point_filter(pts_rig)
        obstacles = pts_rig[keep]
        circles = occupancy_to_circles(
            obstacles[:, :2], self._cell_m, self._max_range_m, self._min_pts
        )
        # Rich per-cycle log: on the box this is how the frame conventions get verified (are the
        # discs ahead, at plausible ranges?) and the depth scale sanity-checked.
        d = depth[np.isfinite(depth) & (depth > 0)]
        logger.info(
            "camera field: frame=%s depth[m] med=%.1f p90=%.1f | %d pts -> %d in-band -> %d discs"
            "%s", depth.shape, float(np.median(d)) if len(d) else -1.0,
            float(np.percentile(d, 90)) if len(d) else -1.0, len(pts_cam), len(obstacles),
            len(circles),
            "" if not len(circles) else " | nearest x=%.1f y=%.1f" % (
                tuple(circles[np.argmin(np.linalg.norm(circles[:, :2], axis=1)), :2])),
        )
        return CircleField(circles if len(circles) else None)


@dataclass
class CameraCalib:
    """One camera's calibration for the multi-camera source: intrinsics at a reference resolution
    plus the camera→rig transform `(R, t)` (see the module docstring's frame conventions).

    `intrinsics` is `(fx, fy, cx, cy)` at `ref_hw`; `intrinsics_for` rescales to the frame's actual
    resolution, exactly as the single-camera source does.
    """

    camera_id: str
    intrinsics: tuple            # (fx, fy, cx, cy) at ref_hw
    R: np.ndarray                # (3, 3) camera->rig rotation
    t: np.ndarray                # (3,) camera position in the rig
    ref_hw: tuple = SURROUND_REF_HW

    def __post_init__(self):
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.t = np.asarray(self.t, float).reshape(3)

    def intrinsics_for(self, h: int, w: int) -> tuple[float, float, float, float]:
        fx, fy, cx, cy = self.intrinsics
        sw, sh = w / self.ref_hw[1], h / self.ref_hw[0]
        return fx * sw, fy * sh, cx * sw, cy * sh

    def points_rig(self, depth, max_range_m: float, stride: int) -> np.ndarray:
        """`(H,W)` depth → `(N,3)` rig-frame points, via the pinhole model + this camera's pose."""
        depth = np.asarray(depth, float)
        fx, fy, cx, cy = self.intrinsics_for(*depth.shape[:2])
        pts_cam = backproject_depth(depth, fx, fy, cx, cy, max_range_m, stride)
        return camera_to_rig(pts_cam, self.R, self.t)

    @classmethod
    def from_config(cls, camera_id: str, opencv_pinhole: dict, rig_to_camera: dict,
                    ref_hw=SURROUND_REF_HW) -> "CameraCalib":
        """Build from an AlpaSim `extra_cameras` entry (`intrinsics.opencv_pinhole` + `rig_to_camera`)."""
        fx, fy = opencv_pinhole["focal_length"]
        cx, cy = opencv_pinhole["principal_point"]
        R = quat_xyzw_to_matrix(rig_to_camera["rotation_xyzw"])
        t = np.asarray(rig_to_camera["translation_m"], float)
        return cls(camera_id, (fx, fy, cx, cy), R, t, ref_hw=ref_hw)


@dataclass
class FthetaCamera:
    """One fisheye (ftheta) camera: real per-scene calibration → rig-frame points.

    The correct model for AlpaSim's actual cameras (see `ftheta.py`): intrinsics are the
    angle→pixeldist polynomial at `native_hw`, and `R, t` is the camera→rig pose in the **sensor
    FLU** frame (x forward/optical, y left, z up) — `p_rig = R @ p_flu + t`. Built from the USDZ
    calibration (`from_params`, fed by `parse_cameras_from_usdz` on the box), so unlike the pinhole
    `CameraCalib` there is nothing hardcoded and the geometry matches what the renderer produced.
    """

    camera_id: str
    cx: float
    cy: float
    poly: tuple            # ftheta polynomial (c0 first) at native_hw
    native_hw: tuple       # (H, W) the intrinsics are calibrated at
    R: np.ndarray          # (3,3) camera(FLU)->rig rotation
    t: np.ndarray          # (3,) camera position in the rig
    linear_cde: tuple = (1.0, 0.0, 0.0)
    poly_kind: str = "angle_to_pixeldist"  # or "pixeldist_to_angle" (both occur across scenes)

    def __post_init__(self):
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.t = np.asarray(self.t, float).reshape(3)
        self.poly = tuple(float(c) for c in self.poly)

    def points_rig(self, depth, max_range_m: float, stride: int) -> np.ndarray:
        """`(H,W)` depth → `(N,3)` rig-frame points, via the ftheta model + this camera's pose."""
        from shield_in_alpasim.ftheta import backproject_ftheta

        pts_flu = backproject_ftheta(depth, self.cx, self.cy, self.poly, self.native_hw,
                                     linear_cde=self.linear_cde, max_range_m=max_range_m,
                                     stride=stride, poly_kind=self.poly_kind)
        if len(pts_flu) == 0:
            return pts_flu.reshape(-1, 3)
        return pts_flu @ self.R.T + self.t

    @classmethod
    def from_params(cls, camera_id, translation_m, rotation_xyzw, cx, cy, poly, native_hw,
                    linear_cde=(1.0, 0.0, 0.0),
                    poly_kind="angle_to_pixeldist") -> "FthetaCamera":
        """Build from raw calibration numbers (pure — the box-only USDZ read is done by the caller).

        `rotation_xyzw` is the sensor-FLU→rig quaternion from `nominalSensor2Rig_FLU`; the optical
        axis is +x in that frame (verified: this makes the front camera look forward and the rear
        cameras look to the rear quarters, docs/MULTICAM_HANDOFF.md). `poly_kind` selects the
        polynomial direction (`angle_to_pixeldist` or `pixeldist_to_angle`); both occur per scene.
        """
        R = quat_xyzw_to_matrix(rotation_xyzw)
        t = np.asarray(translation_m, float)
        return cls(camera_id, float(cx), float(cy), tuple(poly), tuple(native_hw), R, t,
                   linear_cde=tuple(linear_cde), poly_kind=poly_kind)


def _pose_vec(v):
    """A `Pose.vec3`/`.quat` as a plain list, whether it's an attribute array or a method."""
    v = v() if callable(v) else v
    return [float(x) for x in list(v)]


def load_ftheta_cameras(usdz_path: str, camera_ids) -> list[FthetaCamera]:
    """Real per-scene ftheta cameras from a scene `.usdz` (box-only; needs `alpasim_runtime`).

    Uses AlpaSim's own `parse_cameras_from_usdz` — the same calibration `_register_scene_cameras`
    feeds the renderer — so perception geometry matches what was rendered, and the rear cameras
    carry their true rear-quarter angles (no blind wedge, unlike the old hardcoded pinhole rig).
    The `alpasim_runtime` import is deferred so this module still imports on a Mac.
    """
    from alpasim_runtime.video_model.usdz_calibration import parse_cameras_from_usdz

    defs = parse_cameras_from_usdz(usdz_path)
    cams = []
    for cid in camera_ids:
        if cid not in defs:
            raise KeyError(f"camera {cid!r} not in the scene calibration ({sorted(defs)})")
        cd = defs[cid]
        pose = cd.rig_to_camera
        ft = cd.intrinsics.ftheta_param
        # Cameras store the fisheye polynomial in EITHER direction depending on the scene
        # (both occur in the sample set). Prefer angle->pixeldist (invert on use); otherwise take
        # pixeldist->angle (evaluated directly). The earlier code assumed the former and crashed on
        # scenes that ship the latter (e.g. 01d503d4's cross_left).
        a2p = list(ft.angle_to_pixeldist_poly)
        p2a = list(ft.pixeldist_to_angle_poly)
        if a2p:
            poly, kind = a2p, "angle_to_pixeldist"
        elif p2a:
            poly, kind = p2a, "pixeldist_to_angle"
        else:
            raise ValueError(f"{cid}: camera has neither ftheta polynomial direction populated")
        lc = ft.linear_cde
        cams.append(FthetaCamera.from_params(
            cid, _pose_vec(pose.vec3), _pose_vec(pose.quat),
            float(ft.principal_point_x), float(ft.principal_point_y), poly,
            (int(cd.intrinsics.resolution_h), int(cd.intrinsics.resolution_w)),
            linear_cde=(lc.linear_c, lc.linear_d, lc.linear_e), poly_kind=kind,
        ))
    logger.info("Loaded ftheta calibration for %d cameras from %s: %s",
                len(cams), usdz_path, [c.camera_id for c in cams])
    return cams


class MultiCameraObstacleSource:
    """`ObstacleSource` fusing N cameras' metric depth into one rig-frame occupancy field.

    The surround counterpart to `CameraObstacleSource`: same pipeline, but the back-project → rig
    step runs *per camera* with that camera's own intrinsics/extrinsics, and all cameras' rig-frame
    points are unioned before the single height-band + occupancy step. That gives the shield a
    field that covers the sides and rear, not just the forward cone a single front camera sees —
    the whole point of the surround arm (see docs/MULTICAM_HANDOFF.md).

    `depth_model` is the same callable the single-cam source takes (`HWC image -> (H, W) metres`).
    With `batched=True` it is instead called once with a *list* of the N frames and must return a
    list of depth maps (`HFDepthModel` supports this) — one forward pass instead of N, the cost
    lever for the 4× depth budget.
    """

    def __init__(self, depth_model, cameras, ground_band=(0.3, 2.5),
                 max_range_m: float = 40.0, cell_m: float = 0.5, min_pts: int = 5,
                 stride: int = 2, batched: bool = False, point_filter=None):
        # `cameras` is any list of camera models exposing `.camera_id` and
        # `.points_rig(depth, max_range_m, stride)` — pinhole `CameraCalib` or fisheye
        # `FthetaCamera`. The fusion loop is identical; only the per-camera un-projection differs.
        if not cameras:
            raise ValueError("MultiCameraObstacleSource needs at least one camera")
        self._depth = depth_model
        self._cameras = list(cameras)
        self._ground_band = ground_band
        self._max_range_m = max_range_m
        self._cell_m = cell_m
        self._min_pts = min_pts
        self._stride = stride
        self._batched = batched
        # Optional relevance filter on the fused rig-frame points (e.g. CorridorGate) — the same
        # pluggable seam the single-cam source has; None = keep everything the height band passes.
        self._point_filter = point_filter

    @classmethod
    def surround(cls, depth_model, camera_ids: list[str], **kwargs) -> "MultiCameraObstacleSource":
        """Build the standard surround rig from `SURROUND_*` for the given advertised cameras.

        Each `camera_id` must be one of the known surround cameras (`SURROUND_RIG_TO_CAMERA`); they
        all share the surround intrinsics and differ only in rig pose.
        """
        cams = []
        for cid in camera_ids:
            if cid not in SURROUND_RIG_TO_CAMERA:
                raise KeyError(
                    f"no surround calibration for {cid!r}; known: {sorted(SURROUND_RIG_TO_CAMERA)}")
            cams.append(CameraCalib.from_config(
                cid, {"focal_length": SURROUND_FOCAL, "principal_point": SURROUND_PRINCIPAL},
                SURROUND_RIG_TO_CAMERA[cid], ref_hw=SURROUND_REF_HW))
        return cls(depth_model, cams, **kwargs)

    def _depths(self, images: list) -> list[np.ndarray]:
        """Metric depth for each frame — one batched call when `batched`, else per-frame."""
        if self._batched:
            return [np.asarray(d, float) for d in self._depth(images)]
        return [np.asarray(self._depth(im), float) for im in images]

    def field_for(self, prediction_input) -> CircleField:
        images = [_frame_image(prediction_input.camera_images[c.camera_id][-1])
                  for c in self._cameras]
        depths = self._depths(images)

        chunks, per_cam = [], []
        for cam, depth in zip(self._cameras, depths):
            rig_pts = cam.points_rig(depth, self._max_range_m, self._stride)
            chunks.append(rig_pts)
            per_cam.append(len(rig_pts))
        pts_rig = np.concatenate(chunks) if chunks else np.zeros((0, 3))
        keep = height_band_mask(pts_rig, *self._ground_band)
        if self._point_filter is not None:
            keep = keep & self._point_filter(pts_rig)
        obstacles = pts_rig[keep]
        circles = occupancy_to_circles(
            obstacles[:, :2], self._cell_m, self._max_range_m, self._min_pts)

        # Per-camera point counts + nearest disc bearing: on the box this is how each camera's
        # frame convention gets checked (side cams contribute lateral discs, the rear cam discs
        # behind), the way the single-cam log verifies "are the discs ahead?".
        logger.info(
            "surround field: %s | %d pts -> %d in-band -> %d discs%s",
            {c.camera_id: n for c, n in zip(self._cameras, per_cam)},
            int(sum(per_cam)), len(obstacles), len(circles),
            "" if not len(circles) else " | nearest x=%.1f y=%.1f" % (
                tuple(circles[np.argmin(np.linalg.norm(circles[:, :2], axis=1)), :2])),
        )
        return CircleField(circles if len(circles) else None)
