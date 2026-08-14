"""Tests for the camera-perception obstacle source (`obstacle_source.py`).

Pure numpy, no AlpaSim and no depth net: a synthetic depth map stands in for the model, so the
back-projection / height-filter / occupancy-to-circles geometry is verified on a dev box. Only
the real depth net's forward pass is box-only. The frame conventions themselves still get
confirmed on the box (see the module docstring) — these tests pin the maths, not the AlpaSim
interface.
"""

import numpy as np

from shield_in_alpasim.obstacle_source import (
    CameraObstacleSource,
    backproject_depth,
    camera_to_rig,
    height_band_mask,
    occupancy_to_circles,
    quat_xyzw_to_matrix,
)

# OpenCV camera (x right, y down, z forward) -> rig (x forward, y left, z up): p_rig = M p_cam.
# camera_to_rig applies (R, t) as camera->rig (p_rig = R p_cam + t), so pass R = M directly.
_M_CAM_TO_RIG = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


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
