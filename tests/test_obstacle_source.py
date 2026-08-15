"""Tests for the camera-perception obstacle source (`obstacle_source.py`).

Pure numpy, no AlpaSim and no depth net: a synthetic depth map stands in for the model, so the
back-projection / height-filter / occupancy-to-circles geometry is verified on a dev box. Only
the real depth net's forward pass is box-only. The frame conventions themselves still get
confirmed on the box (see the module docstring) — these tests pin the maths, not the AlpaSim
interface.
"""

import json
import pathlib

import numpy as np

from shield_in_alpasim.obstacle_source import (
    SURROUND_RIG_TO_CAMERA,
    CameraCalib,
    CameraObstacleSource,
    FthetaCamera,
    MultiCameraObstacleSource,
    backproject_depth,
    camera_to_rig,
    height_band_mask,
    occupancy_to_circles,
    quat_xyzw_to_matrix,
)

# Real per-scene ftheta calibration dumped from the 02eadd92 USDZ (parse_cameras_from_usdz).
_REAL_CALIB = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "docs/real_rig_calib_02eadd92.json").read_text()
)


def _real_ftheta(cid: str) -> FthetaCamera:
    c = _REAL_CALIB[cid]
    return FthetaCamera.from_params(
        cid, c["translation_m"], c["rotation_xyzw"], c["cx"], c["cy"],
        c["angle_to_pixeldist_poly"], c["resolution_hw"], linear_cde=c["linear_cde"],
    )

# OpenCV camera (x right, y down, z forward) -> rig (x forward, y left, z up): p_rig = M p_cam.
# camera_to_rig applies (R, t) as camera->rig (p_rig = R p_cam + t), so pass R = M directly.
_M_CAM_TO_RIG = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
# A camera facing backward: cam-forward (z) -> rig -x (behind the ego), cam-right (x) -> rig +y.
_M_CAM_TO_RIG_REAR = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)


def test_quat_identity_is_identity():
    assert np.allclose(quat_xyzw_to_matrix([0.0, 0.0, 0.0, 1.0]), np.eye(3))


def test_backproject_center_pixel_sits_on_the_optical_axis():
    depth = np.full((11, 11), 5.0)
    pts = backproject_depth(depth, fx=100, fy=100, cx=5, cy=5, max_range_m=40)
    # The centre pixel (u=cx, v=cy) back-projects to (0, 0, depth).
    centre = pts[np.argmin(np.linalg.norm(pts[:, :2], axis=1))]
    assert np.allclose(centre, [0.0, 0.0, 5.0])


def test_backproject_drops_invalid_and_out_of_range():
    depth = np.array([[0.0, -1.0], [10.0, 999.0]])  # 0, negative, valid, too-far
    pts = backproject_depth(depth, fx=1, fy=1, cx=0, cy=0, max_range_m=40)
    assert len(pts) == 1 and pts[0, 2] == 10.0


def test_camera_to_rig_maps_forward_depth_to_rig_x():
    # A point straight ahead in the camera (0, 0, d) should land at rig (d, 0, 0).
    p = camera_to_rig(np.array([[0.0, 0.0, 8.0]]), _M_CAM_TO_RIG, np.zeros(3))
    assert np.allclose(p[0], [8.0, 0.0, 0.0])


def test_height_band_mask():
    pts = np.array([[1, 0, 0.0], [1, 0, 1.0], [1, 0, 3.0]])
    keep = height_band_mask(pts, 0.3, 2.5)
    assert keep.tolist() == [False, True, False]


def test_occupancy_to_circles_clusters_and_thresholds():
    # 5 points in one cell -> one disc; 1 stray point -> dropped (min_pts=3).
    xy = np.array([[10.1, 0.1], [10.2, 0.2], [10.05, 0.05], [10.3, 0.15], [10.15, 0.25],
                   [30.0, -5.0]])
    circles = occupancy_to_circles(xy, cell_m=1.0, max_range_m=40, min_pts=3)
    assert len(circles) == 1
    cx, cy, r = circles[0]
    assert (10.0 <= cx <= 11.0) and (0.0 <= cy <= 1.0)
    assert np.isclose(r, np.hypot(1.0, 1.0) / 2.0)  # circumscribes the cell (conservative)


def test_occupancy_range_filter_and_empty():
    assert len(occupancy_to_circles(np.zeros((0, 2)), 0.5, 40, 5)) == 0
    far = np.tile([100.0, 0.0], (10, 1))
    assert len(occupancy_to_circles(far, 0.5, 40, 5)) == 0  # all beyond max_range


def _fake_prediction_input(depth, cam="camera_front_wide_120fov", as_tuple=False):
    # The real servicer sends plain (timestamp_us, image) tuples, not objects with `.image`.
    frame = (123, depth) if as_tuple else type("F", (), {"image": depth})()
    return type("PI", (), {"camera_images": {cam: [frame]}})()


def test_frame_image_handles_tuple_and_object():
    from shield_in_alpasim.obstacle_source import CameraObstacleSource as C
    assert C._frame_image((123, "img")) == "img"
    assert C._frame_image(type("F", (), {"image": "img"})()) == "img"


def test_intrinsics_rescale_with_resolution():
    # Calibrated at 1920x1080; a half-res frame halves fx/fy/cx/cy.
    src = CameraObstacleSource.front_wide(depth_model=lambda x: x)
    fx, fy, cx, cy = src._intrinsics_for(540, 960)  # half of 1080x1920
    assert np.allclose([fx, fy, cx, cy], [1545 / 2, 1545 / 2, 960 / 2, 560 / 2])
    # At the reference resolution the intrinsics are unchanged.
    assert np.allclose(src._intrinsics_for(1080, 1920), [1545, 1545, 960, 560])


def test_camera_source_end_to_end_puts_a_blob_ahead():
    # A central patch of obstacle at 10 m, everything else invalid. With the standard frame
    # mapping the disc(s) should land ~10 m ahead (rig x) and near-zero lateral (rig y).
    depth = np.zeros((100, 100))
    depth[40:60, 40:60] = 10.0
    src = CameraObstacleSource(
        depth_model=lambda _img: depth,
        camera_id="camera_front_wide_120fov",
        intrinsics=(100.0, 100.0, 50.0, 50.0),
        rig_to_camera=(_M_CAM_TO_RIG, np.zeros(3)),
        ref_hw=(100, 100),  # intrinsics above are already for this frame size (no rescale)
        ground_band=(-100.0, 100.0),  # disable height filtering for this plumbing test
        max_range_m=40.0, cell_m=1.0, min_pts=3,
    )
    field = src.field_for(_fake_prediction_input(depth))
    assert len(field.circles) > 0
    assert 8.0 < field.circles[:, 0].max() < 12.0     # ~10 m ahead
    assert abs(np.median(field.circles[:, 1])) < 2.0  # roughly centred laterally


# --- the surround (multi-camera) source ---


def _central_patch_depth(value=10.0, size=100):
    depth = np.zeros((size, size))
    depth[40:60, 40:60] = value
    return depth


def _multi_prediction_input(frames_by_cam):
    return type("PI", (), {"camera_images": {c: [f] for c, f in frames_by_cam.items()}})()


def _calib(camera_id, R):
    # Intrinsics already at the frame size, so no rescale; disable height filtering downstream.
    return CameraCalib(camera_id, (100.0, 100.0, 50.0, 50.0), R, np.zeros(3), ref_hw=(100, 100))


def test_camera_calib_intrinsics_rescale():
    c = CameraCalib("cam", (1545.0, 1545.0, 960.0, 560.0), np.eye(3), np.zeros(3), ref_hw=(1080, 1920))
    assert np.allclose(c.intrinsics_for(540, 960), [1545 / 2, 1545 / 2, 960 / 2, 560 / 2])
    assert np.allclose(c.intrinsics_for(1080, 1920), [1545, 1545, 960, 560])


def test_multicamera_fuses_front_and_rear_into_one_field():
    # Front camera sees a blob at 10 m -> a disc ~+10 ahead; rear camera sees one at 10 m ->
    # a disc ~-10 behind. The union field must contain both, which the front-only source cannot.
    front, rear = _central_patch_depth(), _central_patch_depth()
    src = MultiCameraObstacleSource(
        depth_model=lambda img: img,  # the frame IS the depth map here
        cameras=[_calib("front", _M_CAM_TO_RIG), _calib("rear", _M_CAM_TO_RIG_REAR)],
        ground_band=(-100.0, 100.0), max_range_m=40.0, cell_m=1.0, min_pts=3,
    )
    field = src.field_for(_multi_prediction_input({"front": front, "rear": rear}))
    xs = field.circles[:, 0]
    assert xs.max() > 8.0    # a disc well ahead (front cam)
    assert xs.min() < -8.0   # and one well behind (rear cam) — the surround win


def test_multicamera_batched_calls_depth_once_with_a_list():
    calls = []

    def batch_depth(images):
        calls.append(len(images))  # one call, N images
        return [np.asarray(im, float) for im in images]

    src = MultiCameraObstacleSource(
        depth_model=batch_depth,
        cameras=[_calib("front", _M_CAM_TO_RIG), _calib("rear", _M_CAM_TO_RIG_REAR)],
        ground_band=(-100.0, 100.0), max_range_m=40.0, cell_m=1.0, min_pts=3, batched=True,
    )
    field = src.field_for(
        _multi_prediction_input({"front": _central_patch_depth(), "rear": _central_patch_depth()})
    )
    assert calls == [2]  # a single batched forward pass over both frames
    assert field.circles[:, 0].max() > 8.0 and field.circles[:, 0].min() < -8.0


def test_surround_builds_the_named_cameras_and_rejects_unknown():
    import pytest

    ids = ["camera_front_wide_120fov", "camera_cross_left_120fov",
           "camera_cross_right_120fov", "camera_rear_left_70fov"]
    src = MultiCameraObstacleSource.surround(depth_model=lambda x: x, camera_ids=ids)
    assert [c.camera_id for c in src._cameras] == ids
    assert set(ids) == set(SURROUND_RIG_TO_CAMERA)  # all four known cameras are calibrated
    with pytest.raises(KeyError):
        MultiCameraObstacleSource.surround(lambda x: x, ["camera_nonexistent"])


# --- guard the REAL surround calibration (the values we trust on the box) ---
#
# The other tests use synthetic rotations; these pin the actual SURROUND_RIG_TO_CAMERA quaternions
# so a transcription slip can't silently point a camera the wrong way and waste a metered run. This
# is the off-box half of "re-verify the frame convention per camera"; the box still has to confirm
# AlpaSim's delivered pixels actually follow this OpenCV convention, but a wrong number is caught
# here for free. Expected rig bearing of each camera's optical axis (computed from the calibration):
# front ~0deg, cross-left ~+55 (front-left), cross-right ~-55 (front-right), rear ~180 (behind).
_EXPECTED = {
    "camera_front_wide_120fov":  dict(lo=-10.0, hi=10.0,  fwd=+1, use_abs=False),
    "camera_cross_left_120fov":  dict(lo=30.0,  hi=80.0,  fwd=+1, use_abs=False),
    "camera_cross_right_120fov": dict(lo=-80.0, hi=-30.0, fwd=+1, use_abs=False),
    # Rear wraps at +/-180, so compare |bearing|.
    "camera_rear_left_70fov":    dict(lo=150.0, hi=180.0, fwd=-1, use_abs=True),
}


def test_surround_calibration_points_each_camera_the_physically_right_way():
    src = MultiCameraObstacleSource.surround(lambda x: x, list(_EXPECTED))
    for calib in src._cameras:
        exp = _EXPECTED[calib.camera_id]
        # A proper rotation (det +1), and camera 'up' (-y) maps to world up (+z), not flipped.
        assert np.isclose(np.linalg.det(calib.R), 1.0, atol=1e-3), calib.camera_id
        assert (calib.R @ np.array([0.0, -1.0, 0.0]))[2] > 0.9, calib.camera_id
        # Camera at ~1.5 m height; front cameras ahead of the rig origin, rear one behind it.
        assert 1.3 <= calib.t[2] <= 1.7, calib.camera_id
        assert np.sign(calib.t[0]) == exp["fwd"], calib.camera_id
        # Optical axis (cam +z) points the physically expected way.
        axis = calib.R @ np.array([0.0, 0.0, 1.0])
        bearing = np.degrees(np.arctan2(axis[1], axis[0]))
        b = abs(bearing) if exp["use_abs"] else bearing
        assert exp["lo"] <= b <= exp["hi"], (calib.camera_id, bearing)


def test_surround_pipeline_lands_a_patch_on_the_right_side_per_real_camera():
    # A blob dead-centre in each REAL camera, all the way through field_for, must land where that
    # camera looks: front ahead (y~0), cross-left front-left (+y), cross-right front-right (-y),
    # rear behind (-x). Calibration + back-project + camera_to_rig + occupancy, end to end.
    expected_xy_sign = {
        "camera_front_wide_120fov":  (+1, 0),   # ahead, roughly centred
        "camera_cross_left_120fov":  (+1, +1),  # front-left
        "camera_cross_right_120fov": (+1, -1),  # front-right
        "camera_rear_left_70fov":    (-1, 0),   # behind
    }
    for cid, (sx, sy) in expected_xy_sign.items():
        src = MultiCameraObstacleSource.surround(
            lambda x: x, [cid], ground_band=(-100.0, 100.0),
            max_range_m=40.0, cell_m=1.0, min_pts=3)
        field = src.field_for(_multi_prediction_input({cid: _central_patch_depth()}))
        assert field.circles is not None and len(field.circles), cid
        # Nearest disc to the ego, the one the shield reacts to first.
        c = field.circles[np.argmin(np.linalg.norm(field.circles[:, :2], axis=1))]
        assert np.sign(c[0]) == sx, (cid, "x", c)
        if sy == 0:
            assert abs(c[1]) < 4.0, (cid, "y~0", c)
        else:
            assert np.sign(c[1]) == sy, (cid, "y", c)


# --- the REAL ftheta rig (from the scene's own calibration) ---


def test_ftheta_camera_places_a_forward_patch_ahead_at_camera_height():
    # Real front-wide ftheta calib + pose: a central depth patch at 10 m lands ~ahead of the ego
    # at camera height, using the true fisheye model (not the pinhole approximation).
    cam = _real_ftheta("camera_front_wide_120fov")
    depth = np.zeros((108, 192))       # rendered res; native is 2160x3840, rescaled internally
    # Centre the patch on the rescaled principal point (s = 192/3840 = 0.05), so the rays are
    # near the optical axis -> straight ahead at camera height (front-wide cy is below image mid).
    s = 192 / 3840
    r0, c0 = int(round(cam.cy * s)), int(round(cam.cx * s))
    depth[r0 - 4:r0 + 4, c0 - 4:c0 + 4] = 10.0
    pts = cam.points_rig(depth, max_range_m=40.0, stride=1)
    assert len(pts) > 0
    assert np.all(pts[:, 0] > 0)                      # ahead of the rig origin
    assert abs(np.median(pts[:, 1])) < 3.0            # near the longitudinal axis
    assert 0.8 < np.median(pts[:, 2]) < 2.0           # ~camera height (~1.3 m)


def test_multicamera_fuses_real_ftheta_front_and_rear():
    # Front + rear-right real ftheta cameras: a patch in each -> a disc ahead AND one behind,
    # exercising the generalized MultiCameraObstacleSource over ftheta models end to end.
    front, rear = _real_ftheta("camera_front_wide_120fov"), _real_ftheta("camera_rear_right_70fov")
    src = MultiCameraObstacleSource(
        lambda img: img, [front, rear], ground_band=(-100.0, 100.0),
        max_range_m=40.0, cell_m=1.0, min_pts=3)
    field = src.field_for(_multi_prediction_input({
        front.camera_id: _central_patch_depth(), rear.camera_id: _central_patch_depth()}))
    xs = field.circles[:, 0]
    assert xs.max() > 5.0     # a disc ahead (front cam)
    assert xs.min() < -5.0    # and one behind (rear-right cam)


def test_real_rig_covers_360_degrees_no_blind_wedge():
    # THE fix's off-box proof: the real 5-camera rig (front + 2 cross + 2 rear, angled to the
    # quarters) covers the full circle, unlike the hardcoded pinhole rig whose rear cams pointed
    # straight back and left ~30deg blind wedges at each rear quarter (docs/MULTICAM_HANDOFF.md).
    half_fov = {"120fov": 60.0, "70fov": 35.0}
    sectors = []
    for cid in ("camera_front_wide_120fov", "camera_cross_left_120fov",
                "camera_cross_right_120fov", "camera_rear_left_70fov", "camera_rear_right_70fov"):
        cam = _real_ftheta(cid)
        axis = cam.R @ np.array([1.0, 0.0, 0.0])       # FLU optical axis is +x
        b = np.degrees(np.arctan2(axis[1], axis[0]))
        hf = half_fov[cid.rsplit("_", 1)[1]]
        sectors.append((b, hf))
    # Every bearing on the circle must be inside at least one camera's [axis-hf, axis+hf] sector.
    for deg in range(-180, 180):
        covered = any(abs((deg - b + 180) % 360 - 180) <= hf for b, hf in sectors)
        assert covered, f"blind spot at bearing {deg} deg"
