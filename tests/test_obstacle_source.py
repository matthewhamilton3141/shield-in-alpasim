"""Tests for the camera-perception obstacle source (`obstacle_source.py`).

Pure numpy, no AlpaSim and no depth net: a synthetic depth map stands in for the model, so the
back-projection / height-filter / occupancy-to-circles geometry is verified on a dev box. Only
the real depth net's forward pass is box-only. The frame conventions themselves still get
confirmed on the box (see the module docstring) — these tests pin the maths, not the AlpaSim
interface.
"""

import numpy as np

from shield_in_alpasim.obstacle_source import (
    SURROUND_RIG_TO_CAMERA,
    CameraCalib,
    CameraObstacleSource,
    MultiCameraObstacleSource,
    backproject_depth,
    camera_to_rig,
    height_band_mask,
    occupancy_to_circles,
    quat_xyzw_to_matrix,
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
