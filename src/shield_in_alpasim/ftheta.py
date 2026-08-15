"""FTheta (fisheye) camera un-projection — the correct geometry for AlpaSim's real cameras.

AlpaSim renders every camera from the scene's *real* per-clip calibration
(`_register_scene_cameras` → `parse_cameras_from_usdz`), and those cameras are **ftheta**
(fisheye), not pinhole: the mapping from a ray's angle-off-axis `θ` to its pixel distance `r` from
the principal point is a 5th-order polynomial `r(θ) = c1·θ + c2·θ² + … + c5·θ⁵` (c0 = 0), not the
pinhole `r = f·tanθ`. Our first surround pass back-projected with a *pinhole* model and hardcoded
poses, which is only ~right near each camera's centre and wrong toward the periphery of a 120° lens
(and misplaced the side/rear cameras entirely). This module does the real thing.

Everything here is pure numpy and unit-tested by round-tripping (project a known point → pixel →
un-project → recover it), so the fisheye maths is verified on a dev box without a render. Only the
depth net and the USDZ calibration read need the box.

Frames (all right-handed):
  - **Sensor FLU** (AlpaSim's `nominalSensor2Rig_FLU`): x = optical axis / forward, y = left,
    z = up. This module's un-projection returns points in *this* frame; the caller applies the
    camera→rig pose `R,t` (which is FLU→rig) to reach the rig frame.
  - **Image**: u = right, v = down, origin top-left. In the sensor FLU frame the image u-axis
    (right) is −y and the v-axis (down) is −z (verified by the round-trip test).
"""

from __future__ import annotations

import numpy as np


def poly_r_of_theta(poly, theta):
    """`r(θ)` — the angle→pixeldist polynomial (c0 first), pixel distance at the native resolution."""
    theta = np.asarray(theta, float)
    r = np.zeros_like(theta)
    for c in reversed(np.asarray(poly, float)):
        r = r * theta + c
    return r


def max_valid_angle(poly, ceiling: float = np.pi) -> float:
    """Largest `θ` for which `r(θ)` is still increasing — the edge of the usable cone.

    The polynomials have negative higher-order terms and eventually turn over; past the turning
    point `r(θ)` is no longer invertible (two angles map to one pixel distance). We only ever
    un-project pixels inside this cone.
    """
    ths = np.linspace(0.0, ceiling, 2000)
    rs = poly_r_of_theta(poly, ths)
    turn = np.argmax(rs)  # first max; rs increases then decreases for these polys
    return float(ths[turn])


def theta_of_r(poly, r, max_angle: float | None = None):
    """Invert `r(θ)` → `θ` for pixel distances `r` (native pixels), via a monotone lookup.

    Returns NaN where `r` exceeds the cone's max radius (pixels outside the lens's valid field),
    so the caller can drop them.
    """
    if max_angle is None:
        max_angle = max_valid_angle(poly)
    grid = np.linspace(0.0, max_angle, 4096)
    r_grid = poly_r_of_theta(poly, grid)  # monotone increasing on [0, max_angle]
    r = np.asarray(r, float)
    theta = np.interp(r, r_grid, grid, right=np.nan)
    theta[r > r_grid[-1]] = np.nan
    return theta


def _linear_undistort(du, dv, linear_cde):
    """Undo the small `linear_cde` affine `[[c, d],[e, 1]]` applied to the distorted image offset.

    c≈1, d,e≈0 for these cameras (a sub-0.1% correction), but apply it exactly:
    `[u',v'] = A [du,dv]` with `A=[[c,d],[e,1]]`, so `[du,dv] = A⁻¹ [u',v']`.
    """
    c, d, e = linear_cde
    det = c * 1.0 - d * e
    xu = (1.0 * du - d * dv) / det
    yu = (-e * du + c * dv) / det
    return xu, yu


def project_ftheta(pts_flu, cx, cy, poly, linear_cde=(1.0, 0.0, 0.0)):
    """Sensor-FLU points `(N,3)` → pixel coords `(N,2)` at the native resolution (the forward model).

    Only used to *test* the un-projection (round-trip). `θ` is the angle off the optical axis (+x);
    the transverse direction maps to image (u,v) = (−y, −z).
    """
    p = np.asarray(pts_flu, float).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rho = np.hypot(y, z)
    theta = np.arctan2(rho, x)
    r = poly_r_of_theta(poly, theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        iu = np.where(rho > 0, -y / rho, 0.0)  # image u-direction (right) = -y
        iv = np.where(rho > 0, -z / rho, 0.0)  # image v-direction (down) = -z
    du, dv = r * iu, r * iv
    c, d, e = linear_cde
    u = cx + c * du + d * dv
    v = cy + e * du + 1.0 * dv
    return np.stack([u, v], axis=1)


def unproject_pixels(u, v, depth, cx, cy, poly, native_hw, linear_cde=(1.0, 0.0, 0.0),
                     rendered_hw=None, max_range_m: float = 40.0,
                     max_angle: float | None = None) -> np.ndarray:
    """Flat pixel arrays `(u, v, depth)` → `(N,3)` points in the **sensor FLU** frame.

    `cx, cy, poly` are the ftheta intrinsics at `native_hw`; `u, v` are at `rendered_hw` (defaults to
    native), so pixel distances are rescaled to native before inverting the polynomial. Depth is
    z-depth along the optical axis, so a point is `depth · [1, -tanθ cosφ, -tanθ sinφ]`. Pixels
    outside the lens's valid cone (NaN θ) or out of range are dropped.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    d = np.asarray(depth, float)
    nH, nW = native_hw
    s = 1.0 if rendered_hw is None else rendered_hw[1] / nW  # rendered→native scale

    du = u - cx * s
    dv = v - cy * s
    du, dv = _linear_undistort(du, dv, linear_cde)
    r_native = np.hypot(du, dv) / s

    theta = theta_of_r(poly, r_native, max_angle=max_angle)
    phi = np.arctan2(dv, du)

    valid = np.isfinite(theta) & np.isfinite(d) & (d > 0.0) & (d <= max_range_m)
    theta, phi, d = theta[valid], phi[valid], d[valid]
    t = np.tan(theta)
    return np.stack([d, -d * t * np.cos(phi), -d * t * np.sin(phi)], axis=1)


def backproject_ftheta(depth, cx, cy, poly, native_hw, linear_cde=(1.0, 0.0, 0.0),
                       max_range_m: float = 40.0, stride: int = 1,
                       max_angle: float | None = None) -> np.ndarray:
    """`(H,W)` metric depth → `(N,3)` points in the **sensor FLU** frame (x fwd, y left, z up).

    Subsamples by `stride` (a depth map is ~1e5 px and the shield needs coverage, not per-pixel
    fidelity), then un-projects through the ftheta model (see `unproject_pixels`).
    """
    depth = np.asarray(depth, float)
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    return unproject_pixels(
        us.ravel(), vs.ravel(), depth[::stride, ::stride].ravel(),
        cx, cy, poly, native_hw, linear_cde=linear_cde, rendered_hw=(h, w),
        max_range_m=max_range_m, max_angle=max_angle,
    )
