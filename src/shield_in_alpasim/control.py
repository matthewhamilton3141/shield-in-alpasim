"""Turn a proposed waypoint trajectory into the per-step `(accel, steer)` the shield filters.

The shield is a *filter*: `safety_shield` certifies a commanded `(accel, steer)` and never
proposes one. To put a real driving policy underneath it, its **waypoints** have to become
the per-step commands the shield reasons about. This module is that converter — a
pure-pursuit lateral controller plus a speed-profile longitudinal controller — packaged as a
`policy(state) -> (accel, steer)` closure that drops straight into `kitti_nav`'s
`shielded_rollout`.

**Why a tracker, not an analytic inversion.** A learned model's waypoints need not be
feasible under the shield's rate-limited kinematic bicycle (bounded `max_steer`,
`max_steer_rate`, `max_accel`/`max_decel`). A tracker follows them as closely as the bicycle
allows and leaves the shield to veto the rest — which is exactly the division of labour the
experiment measures: the policy proposes, the shield disposes, and `n_interventions` counts
how often they disagreed.

Pure numpy, and touches only `VehicleConfig`/`VehicleState` from kitti-nav, so it stays
testable on a box with neither AlpaSim nor a GPU. Frame convention matches the shield's
rollout: waypoints and the vehicle state live in the rig frame (x forward, y left), and the
rollout starts at the origin with zero yaw, so the tracker works in that same frame.
"""

from __future__ import annotations

import numpy as np

from kitti_nav.vehicle import VehicleConfig, VehicleState

# Pure-pursuit lookahead grows with speed (a faster car aims farther ahead so it does not
# saw at the wheel), clamped to a sane band. These are metres; the band brackets the horizon
# a 2 Hz / ~6-step plan covers at city speeds.
DEFAULT_LOOKAHEAD_GAIN_S = 1.0   # seconds of travel to look ahead: lookahead = gain * speed
DEFAULT_LOOKAHEAD_MIN_M = 3.0
DEFAULT_LOOKAHEAD_MAX_M = 12.0


def pure_pursuit_steer(state: VehicleState, target_xy: np.ndarray, wheelbase: float) -> float:
    """Road-wheel angle that arcs the rear-axle bicycle toward `target_xy`.

    Standard pure pursuit: transform the target into the vehicle frame, then command the
    curvature of the circle through the origin tangent to the heading that also passes
    through the target. Returned unclamped — `step_state` and the shield enforce
    `max_steer`/`max_steer_rate`, and clamping here as well would hide infeasible asks the
    shield is meant to see.
    """
    d = np.asarray(target_xy, dtype=float) - state.xy
    cos_y, sin_y = np.cos(state.yaw), np.sin(state.yaw)
    # World -> vehicle frame: x forward along heading, y to the left.
    forward = d[0] * cos_y + d[1] * sin_y
    left = -d[0] * sin_y + d[1] * cos_y
    dist_sq = forward * forward + left * left
    if dist_sq < 1e-9:
        return 0.0
    # Curvature of the pure-pursuit arc is 2*lateral / L_d^2; steer = atan(wheelbase * kappa).
    # A target to the left (left > 0) yields positive steer, which the bicycle turns toward
    # +y — the same sign convention as the shield's yaw_rate = v/L * tan(steer).
    curvature = 2.0 * left / dist_sq
    return float(np.arctan(wheelbase * curvature))


def _lookahead_index(position_xy: np.ndarray, waypoints_xy: np.ndarray, lookahead_m: float) -> int:
    """Index of the waypoint to aim at: the first one at least `lookahead_m` ahead.

    Anchored at the closest waypoint and searched forward, so a path that curves back on
    itself cannot select a waypoint the car has already passed. Falls back to the last
    waypoint when the whole remaining path is nearer than the lookahead (the plan is running
    out — aim at its end).
    """
    dists = np.linalg.norm(waypoints_xy - position_xy, axis=1)
    start = int(np.argmin(dists))
    for i in range(start, len(waypoints_xy)):
        if dists[i] >= lookahead_m:
            return i
    return len(waypoints_xy) - 1


def segment_speeds(waypoints_xy: np.ndarray, waypoint_dt: float) -> np.ndarray:
    """Speed implied by each segment of the plan: `|Δwaypoint| / waypoint_dt`.

    The plan is a sequence of positions at a fixed cadence, so its own spacing *is* the speed
    profile the policy intended. Element `i` is the average speed over the segment ending at
    waypoint `i`; the first segment runs from the rig origin, where the plan begins.
    """
    pts = np.vstack([np.zeros((1, 2)), np.asarray(waypoints_xy, dtype=float)])
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return seg_len / waypoint_dt


def make_tracking_policy(
    waypoints_xy: np.ndarray,
    waypoint_dt: float,
    cfg: VehicleConfig,
    lookahead_gain_s: float = DEFAULT_LOOKAHEAD_GAIN_S,
    lookahead_min_m: float = DEFAULT_LOOKAHEAD_MIN_M,
    lookahead_max_m: float = DEFAULT_LOOKAHEAD_MAX_M,
):
    """A `policy(state) -> (accel, steer)` that tracks `waypoints_xy` for the shield to filter.

    `waypoints_xy` is `(T, 2)` in the rig frame at `waypoint_dt` spacing (i.e.
    `1 / output_frequency_hz`). The returned closure is what `shielded_rollout` calls every
    `cfg.dt`: pure pursuit for steer, and a one-step approach to the plan's local speed for
    accel. Both are returned within the bicycle's limits, but the shield still has the final
    say — that is the point.
    """
    waypoints_xy = np.asarray(waypoints_xy, dtype=float)
    speeds = np.clip(segment_speeds(waypoints_xy, waypoint_dt), cfg.min_speed, cfg.max_speed)

    def policy(state: VehicleState) -> tuple[float, float]:
        lookahead = float(np.clip(lookahead_gain_s * state.v, lookahead_min_m, lookahead_max_m))
        idx = _lookahead_index(state.xy, waypoints_xy, lookahead)
        steer = pure_pursuit_steer(state, waypoints_xy[idx], cfg.wheelbase)
        # Desired speed is the plan's speed at the aimed-for segment. Close the gap in one
        # control step; step_state clamps to max_accel/max_decel, so an aggressive ask just
        # saturates rather than overshooting.
        v_des = float(speeds[idx])
        accel = float(np.clip((v_des - state.v) / cfg.dt, -cfg.max_decel, cfg.max_accel))
        return accel, steer

    return policy
