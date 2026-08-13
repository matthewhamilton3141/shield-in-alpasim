"""Tests for the trajectory tracker (`control.py`).

Pure numpy, no AlpaSim: the tracker converts a proposed waypoint plan into `(accel, steer)`,
and this asserts it tracks straight lines, steers the right way on curves, follows the plan's
implied speed, and — crucially — that when fed through the shield, the shield still brakes
for an obstacle the plan drives straight at. That last one is the whole reason the tracker
exists: a policy that proposes, a shield that vetoes.
"""

import numpy as np

from kitti_nav.vehicle import (
    CircleField,
    VehicleConfig,
    VehicleState,
    shielded_rollout,
)
from shield_in_alpasim.control import (
    make_tracking_policy,
    pure_pursuit_steer,
    segment_speeds,
)

CFG = VehicleConfig()


def test_target_straight_ahead_gives_zero_steer():
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0)
    assert pure_pursuit_steer(state, np.array([10.0, 0.0]), CFG.wheelbase) == 0.0


def test_target_left_steers_left_right_steers_right():
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0)
    # +y is left in the rig frame; positive steer turns toward +y (matches yaw_rate sign).
    assert pure_pursuit_steer(state, np.array([10.0, 3.0]), CFG.wheelbase) > 0.0
    assert pure_pursuit_steer(state, np.array([10.0, -3.0]), CFG.wheelbase) < 0.0


def test_steer_respects_heading_not_just_world_frame():
    # Target due north; car already pointing north -> straight. Same target with the car
    # yawed east must command a left turn back toward it.
    north = np.array([0.0, 10.0])
    facing_north = VehicleState(x=0.0, y=0.0, yaw=np.pi / 2, v=5.0)
    facing_east = VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0)
    assert abs(pure_pursuit_steer(facing_north, north, CFG.wheelbase)) < 1e-9
    assert pure_pursuit_steer(facing_east, north, CFG.wheelbase) > 0.0


def test_segment_speeds_read_the_plans_cadence():
    # Waypoints 2 m apart at dt=0.5 s imply 4 m/s; the first segment runs from the origin.
    wps = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    assert np.allclose(segment_speeds(wps, 0.5), 4.0)


def test_tracker_follows_a_straight_plan_forward():
    # A 4 m/s straight plan: no obstacles, the tracked+shielded rollout should march down +x
    # with negligible lateral drift and settle near the plan's speed.
    wps = np.array([[2.0 * (i + 1), 0.0] for i in range(6)])
    policy = make_tracking_policy(wps, waypoint_dt=0.5, cfg=CFG)
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=4.0)
    states, stats = shielded_rollout(policy, state, CircleField(None), CFG, n_steps=20)
    assert states[-1].x > states[0].x
    assert abs(states[-1].y) < 0.2
    assert stats["n_interventions"] == 0  # empty field: nothing to veto
    assert abs(states[-1].v - 4.0) < 1.0


def test_tracker_accelerates_toward_a_faster_plan():
    # Plan implies 10 m/s; starting from rest the car should speed up (bounded by max_accel).
    wps = np.array([[5.0 * (i + 1), 0.0] for i in range(6)])
    policy = make_tracking_policy(wps, waypoint_dt=0.5, cfg=CFG)
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0)
    states, _ = shielded_rollout(policy, state, CircleField(None), CFG, n_steps=10, shield=False)
    assert states[-1].v > states[0].v


def test_tracker_steers_along_a_left_curve():
    # A plan that bends to the left should produce a net-positive heading change.
    angles = np.linspace(0.0, 0.6, 7)[1:]
    wps = np.stack([np.cumsum(np.cos(angles) * 2.0), np.cumsum(np.sin(angles) * 2.0)], axis=1)
    policy = make_tracking_policy(wps, waypoint_dt=0.5, cfg=CFG)
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0)
    states, _ = shielded_rollout(policy, state, CircleField(None), CFG, n_steps=20, shield=False)
    assert states[-1].yaw > 0.05


def test_shield_vetoes_a_plan_that_drives_into_an_obstacle():
    # The plan says "keep going 8 m/s straight"; an obstacle sits dead ahead. Unshielded, the
    # tracker drives in and collides. Shielded, the shield must intervene and avoid collision.
    wps = np.array([[4.0 * (i + 1), 0.0] for i in range(6)])
    policy = make_tracking_policy(wps, waypoint_dt=0.5, cfg=CFG)
    obstacle = CircleField(np.array([[20.0, 0.0, 1.0]]))  # x, y, radius
    start = VehicleState(x=0.0, y=0.0, yaw=0.0, v=8.0)

    unshielded, un_stats = shielded_rollout(policy, start, obstacle, CFG, n_steps=40, shield=False)
    shielded, sh_stats = shielded_rollout(policy, start, obstacle, CFG, n_steps=40, shield=True)

    assert un_stats["collided"] and not sh_stats["collided"]
    assert sh_stats["n_interventions"] > 0
    # The shielded car stops short of the obstacle surface (x=20 minus its radius and margin).
    assert shielded[-1].x < 20.0
