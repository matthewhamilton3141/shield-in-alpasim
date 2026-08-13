"""Smoke tests for the scaffold. Run without AlpaSim/torch installed (Mac-friendly),
same discipline as kitti-nav/gsplat-rt: pure-Python logic tested locally, GPU/AlpaSim-only
paths verified separately on a box that actually has AlpaSim installed.

`predict()` is deliberately untested here — it is a three-line adapter over AlpaSim types
that cannot be imported on this box, and testing it against hand-rolled fakes would only
assert that the fakes match a guess at the interface. It is verified by running the driver
under AlpaSim (plan step 2), not here.
"""

import numpy as np

from kitti_nav.vehicle import VehicleConfig
from shield_in_alpasim.driver import EMPTY_FIELD, ShieldedDriver

CAMERAS = ["camera_front_wide_120fov"]


def _driver(**kwargs) -> ShieldedDriver:
    return ShieldedDriver(
        cfg=VehicleConfig(), camera_ids=CAMERAS, output_frequency_hz=2, horizon_steps=6, **kwargs
    )


def test_empty_field_reports_everything_clear():
    d = EMPTY_FIELD.distance_to_obstacles(np.zeros((5, 2)))
    assert np.all(np.isinf(d))


def test_rollout_produces_expected_shape_and_moves_forward():
    xy = _driver()._rollout(initial_speed=5.0)
    assert xy.shape == (6, 2)
    # No obstacles + forward speed + zero commanded steer -> monotonically increasing x.
    assert np.all(np.diff(xy[:, 0]) > 0)
    assert np.allclose(xy[:, 1], 0.0)


def test_rollout_waypoints_land_on_the_output_period():
    """Sub-stepping must not change *when* waypoints are sampled, only how finely the
    shield integrates between them: at 2 Hz from 5 m/s, waypoint spacing is 0.5 s * 5 m/s.
    """
    xy = _driver()._rollout(initial_speed=5.0)
    assert np.allclose(np.diff(xy[:, 0]), 2.5)


def test_forward_relevant_field_drops_only_whats_behind():
    from kitti_nav.vehicle import CircleField

    from shield_in_alpasim.driver import forward_relevant_field

    cfg = VehicleConfig()  # rear_overhang 0.97 + safety_margin 0.30 -> rear cut at -1.27 m
    circles = np.array([
        [15.0, 0.0, 1.0],   # dead ahead        -> keep
        [-5.0, 0.0, 1.0],   # well behind        -> drop (forward extent -4 < -1.27)
        [-1.0, 2.0, 1.0],   # beside-rear, reaches forward to 0 >= -1.27 -> keep (conservative)
        [-3.0, 0.0, 0.5],   # behind, small      -> drop (forward extent -2.5)
    ])
    kept = forward_relevant_field(CircleField(circles), cfg).circles
    xs = sorted(kept[:, 0].tolist())
    assert xs == [-1.0, 15.0]  # the two behind discs are gone, the rest stay


def test_forward_relevant_field_is_a_noop_when_nothing_is_behind():
    from kitti_nav.vehicle import CircleField

    from shield_in_alpasim.driver import forward_relevant_field

    f = CircleField(np.array([[10.0, 0.0, 1.0]]))
    assert forward_relevant_field(f, VehicleConfig()) is f  # same object, no copy


def test_rear_filter_flag_is_honored():
    assert _driver(rear_filter=False)._rear_filter is False
    assert _driver(rear_filter=True)._rear_filter is True


def test_default_horizon_spans_the_mpc_window():
    # No explicit horizon_steps -> derive from the rate to span DEFAULT_HORIZON_S (3.0 s),
    # which must cover the controller's 2.0 s MPC horizon or the MPC brakes to a stop at our
    # truncated endpoint. At 10 Hz that is 30 waypoints.
    from shield_in_alpasim.driver import DEFAULT_HORIZON_S

    d = ShieldedDriver(cfg=VehicleConfig(), camera_ids=CAMERAS, output_frequency_hz=10)
    assert d._horizon_steps == round(DEFAULT_HORIZON_S * 10)
    xy = d._rollout(initial_speed=5.0)
    assert xy.shape == (round(DEFAULT_HORIZON_S * 10), 2)
    # 3 s of trajectory covers the 2.0 s MPC horizon.
    assert d._horizon_steps / 10 >= 2.0


def test_rollout_honors_a_horizon_override():
    d = ShieldedDriver(cfg=VehicleConfig(), camera_ids=CAMERAS, output_frequency_hz=10)
    assert d._rollout(initial_speed=5.0, horizon_steps=25).shape == (25, 2)


def test_from_config_keeps_the_cameras_and_frequency_alpasim_asked_for():
    driver = ShieldedDriver.from_config(
        model_cfg=None, device="cpu", camera_ids=CAMERAS, context_length=None, output_frequency_hz=4
    )
    assert driver.camera_ids == CAMERAS
    assert driver.output_frequency_hz == 4
    assert driver.context_length == 1  # None in config -> the model's own default


# --- the decorator path: shielding an inner policy ---
#
# `predict()` itself stays untested here (it builds AlpaSim's ModelPrediction, uninstallable
# on this box), but everything up to that boundary — pulling the inner plan, projecting it,
# and rolling the tracked plan through the shield — is pure Python and tested below with a
# fake inner model standing in for another BaseTrajectoryModel.


class _FakePrediction:
    def __init__(self, positions_xyz):
        self.selected_positions = np.asarray(positions_xyz, dtype=float)


class _FakeInner:
    """Minimal stand-in for an inner BaseTrajectoryModel: returns a fixed plan."""

    def __init__(self, waypoints_xy):
        # Lift (T, 2) to the (T, 3) rig-frame positions a real ModelPrediction carries.
        xyz = np.zeros((len(waypoints_xy), 3))
        xyz[:, :2] = waypoints_xy
        xyz[:, 2] = 1.7  # nonzero z, to prove the driver drops it to the ground plane
        self._pred = _FakePrediction(xyz)

    def predict(self, _prediction_input):
        return self._pred


def test_proposed_waypoints_is_none_when_coasting():
    assert _driver()._proposed_waypoints(prediction_input=None) is None


def test_proposed_waypoints_projects_inner_plan_to_the_ground_plane():
    plan = np.array([[2.0, 0.0], [4.0, 1.0], [6.0, 2.0]])
    driver = _driver(inner_model=_FakeInner(plan))
    got = driver._proposed_waypoints(prediction_input=None)
    assert got.shape == (3, 2)  # z dropped
    assert np.allclose(got, plan)


def test_shield_brakes_for_an_obstacle_the_inner_plan_drives_into():
    # Inner plan: keep going straight. Obstacle dead ahead. The tracked-and-shielded rollout
    # must stop short rather than reproduce the inner plan's march into the obstacle.
    from kitti_nav.vehicle import CircleField

    from shield_in_alpasim.control import make_tracking_policy

    plan = np.array([[4.0 * (i + 1), 0.0] for i in range(6)])
    driver = _driver(inner_model=_FakeInner(plan), obstacles=CircleField(np.array([[18.0, 0.0, 1.0]])))
    proposed = driver._proposed_waypoints(prediction_input=None)
    policy = make_tracking_policy(proposed, 1.0 / driver.output_frequency_hz, VehicleConfig())

    braked = driver._rollout(initial_speed=10.0, policy=policy)
    coasting_into_it = driver._rollout(initial_speed=10.0, policy=None)  # coast also gets braked
    # Either way the shield keeps the nose short of the obstacle surface (~17 m).
    assert braked[-1, 0] < 17.0
    assert coasting_into_it[-1, 0] < 17.0


# --- diagnostics: ahead vs behind, the "over-conservative or blocked?" signal ---

def test_diagnostics_flag_an_obstacle_ahead():
    from kitti_nav.vehicle import CircleField, VehicleState

    from shield_in_alpasim.driver import rollout_diagnostics

    field = CircleField(np.array([[15.0, 0.0, 1.0]]))  # 15 m dead ahead
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=6.0)
    d = rollout_diagnostics(field, state, VehicleConfig(), policy=lambda _s: (1.0, 0.0))
    assert d["proposed_accel"] == 1.0
    assert abs(d["nearest_bearing_deg"]) < 1.0          # straight ahead
    assert d["nearest_ahead_gap_m"] == 14.0             # 15 m minus the 1 m radius
    assert d["init_clearance_m"] > 0                     # not in collision


def test_diagnostics_distinguish_a_rear_obstacle():
    # An actor close *behind* the ego (dense traffic). It dominates nearest_gap but there is
    # nothing ahead — the tell-tale of a shield that would freeze for something braking cannot
    # avoid. nearest_ahead_gap_m is None precisely because no disc is in front.
    from kitti_nav.vehicle import CircleField, VehicleState

    from shield_in_alpasim.driver import rollout_diagnostics

    field = CircleField(np.array([[-3.0, 0.0, 1.0]]))  # 3 m behind
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=6.0)
    d = rollout_diagnostics(field, state, VehicleConfig(), policy=lambda _s: (0.0, 0.0))
    assert abs(abs(d["nearest_bearing_deg"]) - 180.0) < 1.0  # directly behind
    assert d["nearest_ahead_gap_m"] is None
