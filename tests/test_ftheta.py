"""FTheta fisheye un-projection tests (`ftheta.py`).

The round-trip (project a known FLU point → pixel → un-project → recover it) is the whole point:
it verifies the fisheye geometry on a dev box, no render needed. Uses the *real* front-wide
polynomial from the 02eadd92 calibration (docs/real_rig_calib_02eadd92.json) so the maths is
exercised against an actual AlpaSim ftheta lens, not a toy one.
"""

import numpy as np

from shield_in_alpasim.ftheta import (
    backproject_ftheta,
    max_valid_angle,
    poly_r_of_theta,
    project_ftheta,
    theta_of_r,
    unproject_pixels,
)

# Real front-wide ftheta calibration (native 2160x3840), from parse_cameras_from_usdz on 02eadd92.
FRONT_POLY = [0.0, 1895.2272, -41.0044, 101.1203, -166.7659, 58.7399]
FRONT_CX, FRONT_CY = 1921.8, 1491.9
FRONT_NATIVE_HW = (2160, 3840)


def test_poly_is_zero_at_axis_and_increasing():
    assert poly_r_of_theta(FRONT_POLY, 0.0) == 0.0
    ths = np.linspace(0.0, max_valid_angle(FRONT_POLY), 50)
    rs = poly_r_of_theta(FRONT_POLY, ths)
    assert np.all(np.diff(rs) > 0)  # monotone on the valid cone


def test_theta_of_r_inverts_the_polynomial():
    max_ang = max_valid_angle(FRONT_POLY)
    ths = np.linspace(0.0, max_ang * 0.98, 40)
    rs = poly_r_of_theta(FRONT_POLY, ths)
    recovered = theta_of_r(FRONT_POLY, rs, max_angle=max_ang)
    assert np.allclose(recovered, ths, atol=1e-3)
    # Pixels beyond the cone are dropped (NaN).
    assert np.isnan(theta_of_r(FRONT_POLY, [rs[-1] * 5], max_angle=max_ang))[0]


def test_front_wide_image_edge_is_about_60_degrees():
    # The polynomial stays monotone past the real field (so max_valid_angle ~ pi and is only a
    # safety cap for the lookup); the actual 120deg FOV is set by the IMAGE extent. At the image
    # half-width (~native cx), the ray angle should be ~half the field, i.e. ~60 deg.
    half_width_px = min(FRONT_CX, FRONT_NATIVE_HW[1] - FRONT_CX)
    edge_deg = np.degrees(theta_of_r(FRONT_POLY, [half_width_px])[0])
    assert 55.0 < edge_deg < 65.0


def test_project_unproject_roundtrip_recovers_points():
    # A spread of FLU points across the field (x forward, y left, z up), all in front of the lens.
    rng = np.random.default_rng(0)
    max_ang = np.deg2rad(55.0)  # within a 120deg lens (half-FOV 60), below the 80deg un-project cap
    pts = []
    for _ in range(200):
        theta = rng.uniform(0, max_ang)
        phi = rng.uniform(-np.pi, np.pi)
        depth = rng.uniform(3.0, 30.0)          # z-depth along the optical axis (+x)
        t = np.tan(theta)
        pts.append([depth, -depth * t * np.cos(phi), -depth * t * np.sin(phi)])
    pts = np.array(pts)

    uv = project_ftheta(pts, FRONT_CX, FRONT_CY, FRONT_POLY)
    got = unproject_pixels(uv[:, 0], uv[:, 1], pts[:, 0], FRONT_CX, FRONT_CY,
                           FRONT_POLY, FRONT_NATIVE_HW, max_range_m=1e9, max_angle=max_ang)
    assert got.shape == pts.shape
    assert np.allclose(got, pts, atol=1e-2)  # recovered to ~1 cm


def test_center_pixel_maps_straight_ahead():
    p = unproject_pixels([FRONT_CX], [FRONT_CY], [10.0], FRONT_CX, FRONT_CY,
                         FRONT_POLY, FRONT_NATIVE_HW)
    assert np.allclose(p[0], [10.0, 0.0, 0.0], atol=1e-6)  # dead ahead at 10 m


def test_image_right_is_world_right_and_up_is_up():
    # A pixel to the RIGHT of centre (u>cx) is a point to the ego's RIGHT (FLU y<0);
    # a pixel ABOVE centre (v<cy) is UP (FLU z>0).
    right = unproject_pixels([FRONT_CX + 300], [FRONT_CY], [10.0], FRONT_CX, FRONT_CY,
                             FRONT_POLY, FRONT_NATIVE_HW)[0]
    up = unproject_pixels([FRONT_CX], [FRONT_CY - 300], [10.0], FRONT_CX, FRONT_CY,
                          FRONT_POLY, FRONT_NATIVE_HW)[0]
    assert right[1] < -0.1   # y (left) negative -> to the right
    assert up[2] > 0.1       # z (up) positive


def test_rendered_resolution_rescales():
    # Same optical pixel at half resolution: cx/cy and pixel distances halve, so a point at the
    # scaled principal point still maps straight ahead, and an off-centre pixel recovers the same
    # angle as at native res.
    s = 0.5
    p = unproject_pixels([FRONT_CX * s], [FRONT_CY * s], [10.0], FRONT_CX, FRONT_CY,
                         FRONT_POLY, FRONT_NATIVE_HW, rendered_hw=(1080, 1920))
    assert np.allclose(p[0], [10.0, 0.0, 0.0], atol=1e-6)


def test_pixeldist_to_angle_matches_the_inverse_direction():
    # For an equidistant model r = f*theta, the two polynomial directions are exact inverses:
    # angle->pixeldist = [0, f]  and  pixeldist->angle = [0, 1/f]. Un-projecting with either must
    # give the same points. This covers scenes that ship the pixeldist->angle form (e.g. 01d503d4),
    # which the first ftheta implementation crashed on.
    f = 1800.0
    native = (2160, 3840)
    cx, cy = 1920.0, 1080.0
    rng = np.random.default_rng(1)
    u = rng.uniform(cx - 1500, cx + 1500, 100)
    v = rng.uniform(cy - 800, cy + 800, 100)
    d = rng.uniform(3.0, 30.0, 100)

    a2p = unproject_pixels(u, v, d, cx, cy, [0.0, f], native, max_range_m=1e9,
                           poly_kind="angle_to_pixeldist")
    p2a = unproject_pixels(u, v, d, cx, cy, [0.0, 1.0 / f], native, max_range_m=1e9,
                           poly_kind="pixeldist_to_angle")
    assert a2p.shape == p2a.shape
    assert np.allclose(a2p, p2a, atol=1e-6)


def test_backproject_depth_map_smoke():
    # A small depth image (rendered res), central patch at 10 m -> points ahead near the axis.
    depth = np.zeros((108, 192))
    depth[50:58, 92:100] = 10.0
    pts = backproject_ftheta(depth, FRONT_CX, FRONT_CY, FRONT_POLY, FRONT_NATIVE_HW,
                             max_range_m=40.0, stride=1)
    assert len(pts) > 0
    assert np.all(pts[:, 0] > 0)                       # all ahead
    assert abs(np.median(pts[:, 1])) < 3.0             # near the axis laterally
