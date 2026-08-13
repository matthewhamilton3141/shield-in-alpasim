"""Tests for the ground-truth obstacle adapter.

Same discipline as `test_driver.py`: everything here is pure numpy and runs without AlpaSim
installed. AlpaSim's `Pose`/`Trajectory` are compiled Rust types (`utils_rs`), so the two
tests that exercise the glue layer stand up fakes for them. Those fakes assert nothing
about AlpaSim's real behaviour — they only pin the *narrow* surface the glue is allowed to
touch, so that if `obstacles.py` starts reaching for some other attribute, the fake stops
matching and the test fails rather than the fake silently growing to cover it.
"""

import numpy as np
import pytest

from kitti_nav.vehicle import VehicleConfig, VehicleState, clearance
from shield_in_alpasim.driver import ShieldedDriver
from shield_in_alpasim.obstacles import (
    field_from_boxes,
    field_from_traffic_objects,
    rect_discs,
    to_rig_frame,
)


# --- fakes for the compiled AlpaSim types -------------------------------------------


class FakePose:
    def __init__(self, x, y, yaw):
        self.vec3 = np.array([x, y, 0.0])
        self._yaw = yaw

    def yaw(self):
        return self._yaw


class FakeTrajectory:
    """One pose held over `[start_us, end_us]` — enough for a pose lookup at a timestamp."""

    def __init__(self, pose, start_us, end_us):
        self._pose, self._start, self._end = pose, start_us, end_us

    def get_time_range_tuple(self):
        return (self._start, self._end)

    def interpolate_pose(self, at_us):
        assert self._start <= at_us <= self._end, "glue must clamp before interpolating"
        return self._pose


class FakeAABB:
    def __init__(self, x, y, z=1.5):
        self.x, self.y, self.z = x, y, z


class FakeObject:
    def __init__(self, track_id, x, y, yaw=0.0, length=4.5, width=1.9,
                 is_static=False, start_us=0, end_us=10_000_000):
        self.track_id = track_id
        self.aabb = FakeAABB(length, width)
        self.is_static = is_static
        self.trajectory = FakeTrajectory(FakePose(x, y, yaw), start_us, end_us)


def _objects(*objs):
    return {o.track_id: o for o in objs}


# --- disc cover ---------------------------------------------------------------------


def test_disc_cover_leaves_no_gap_inside_the_box():
    """The cover must be conservative: every point of the box lies inside some disc.

    This is the property the shield's soundness rests on — a gap in the cover is a place
    the certificate says is clear and the actor actually is.
    """
    length, width = 4.5, 1.9
    field = field_from_boxes([[0.0, 0.0]], [0.0], [length], [width], n_discs=5)

    u, v = np.meshgrid(np.linspace(-length / 2, length / 2, 40),
                       np.linspace(-width / 2, width / 2, 20))
    interior = np.stack([u.ravel(), v.ravel()], axis=1)

    # The box corners lie exactly *on* the circumscribing discs, so the tolerance is for
    # float epsilon at those four points only — not slack in the cover.
    assert np.all(field.distance_to_obstacles(interior) <= 1e-12)


def test_disc_cover_tightens_as_discs_are_added():
    """The cover is always outward, but its excess should shrink with more discs.

    Guards the radius formula: get the segment length wrong and the excess stays flat or
    grows, which would quietly inflate every actor in the scene.
    """
    length, width = 4.5, 1.9
    excess = [
        rect_discs([[0.0, 0.0]], [0.0], [length], [width], n)[0, 2] - width / 2
        for n in (3, 5, 9)
    ]
    assert excess[0] > excess[1] > excess[2] > 0.0


def test_disc_cover_follows_box_orientation():
    """A box rotated 90 degrees should extend along y, not x."""
    circles = rect_discs([[0.0, 0.0]], [np.pi / 2], [6.0], [2.0], n_discs=5)
    assert np.ptp(circles[:, 1]) > 4.0        # spread along y
    assert np.allclose(circles[:, 0], 0.0)    # none along x


# --- frame transform ----------------------------------------------------------------


def test_local_pose_ahead_of_a_rotated_ego_lands_on_the_rig_x_axis():
    """Ego at (5, 5) facing +y; an actor 20 m up-field sits at (20, 0) in the rig frame."""
    xy, yaws = to_rig_frame([5.0, 5.0], np.pi / 2, [[5.0, 25.0]], [np.pi / 2])
    assert np.allclose(xy[0], [20.0, 0.0], atol=1e-9)
    assert np.isclose(yaws[0], 0.0)           # actor is aligned with the ego


def test_transform_preserves_range():
    """A passive transform is a rigid motion, so distances to the ego must not change."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 50, size=(16, 2))
    ego_xy, ego_yaw = np.array([3.0, -7.0]), 0.9

    rig_xy, _ = to_rig_frame(ego_xy, ego_yaw, pts)
    assert np.allclose(np.linalg.norm(rig_xy, axis=1),
                       np.linalg.norm(pts - ego_xy, axis=1))


# --- field assembly -----------------------------------------------------------------


def _longitudinal_bulge(length, width, n_discs):
    """How far a disc cover reaches past the end of the box it covers.

    A disc circumscribing a segment sticks out beyond that segment's flat end by
    `radius - seg/2`. This is the dominant conservatism in the cover — for a 4.5 m car at
    5 discs it is ~0.60 m, five times the ~0.12 m lateral excess — and it is what makes
    the reported gap smaller than the true one.
    """
    seg = length / n_discs
    return np.hypot(seg / 2, width / 2) - seg / 2


def test_gap_to_a_box_straight_ahead_is_measured_to_its_surface():
    """Distance from the ego origin to a box 20 m ahead is to its near face, short by
    exactly the cover's longitudinal bulge — never more than the true gap."""
    length, width, n = 4.5, 1.9, 5
    field = field_from_boxes([[20.0, 0.0]], [0.0], [length], [width], n_discs=n)

    true_gap = 20.0 - length / 2
    reported = field.distance_to_obstacles(np.zeros((1, 2)))[0]

    assert reported < true_gap                       # conservative, never optimistic
    assert reported == pytest.approx(true_gap - _longitudinal_bulge(length, width, n))


def test_range_filter_keeps_a_box_that_reaches_inside_the_cutoff():
    """Filtering is on the box's nearest point, so a long box centred past the cutoff but
    overhanging it must survive."""
    far = field_from_boxes([[100.0, 0.0]], [0.0], [4.5], [1.9], max_range_m=50.0)
    assert far.circles.size == 0

    straddling = field_from_boxes([[52.0, 0.0]], [0.0], [12.0], [2.5], max_range_m=50.0)
    assert straddling.circles.size > 0


def test_empty_scene_yields_a_field_that_reports_everything_clear():
    field = field_from_boxes(np.zeros((0, 2)), [], [], [])
    assert np.all(np.isinf(field.distance_to_obstacles(np.zeros((3, 2)))))


# --- glue over the AlpaSim types ----------------------------------------------------


def test_ego_box_is_not_shielded_against_itself():
    """The runtime prepends the ego's own box to the actor list; treating it as an obstacle
    would report a permanent collision and pin the car at zero speed."""
    objs = _objects(FakeObject("EGO", 0.0, 0.0), FakeObject("car_1", 30.0, 0.0))
    field = field_from_traffic_objects(objs, [0.0, 0.0], 0.0, timestamp_us=1_000_000)

    assert np.all(field.distance_to_obstacles(np.zeros((1, 2))) > 0.0)
    assert len(field.circles) == 5          # only car_1, one box's worth of discs


def test_actor_whose_track_does_not_cover_now_is_dropped_not_extrapolated():
    """A track that has already ended says nothing about where that actor is. Holding its
    last pose would plant a phantom obstacle in front of a braking certificate."""
    expired = FakeObject("car_1", 10.0, 0.0, start_us=0, end_us=1_000_000)
    field = field_from_traffic_objects(_objects(expired), [0.0, 0.0], 0.0,
                                       timestamp_us=9_000_000)
    assert field.circles.size == 0


def test_static_actor_outside_its_track_window_is_kept():
    """A parked car's pose is constant, so clamping onto its one-pose track is exact —
    and dropping it would lose a real obstacle."""
    parked = FakeObject("barrier", 10.0, 0.0, is_static=True, start_us=0, end_us=1_000_000)
    field = field_from_traffic_objects(_objects(parked), [0.0, 0.0], 0.0,
                                       timestamp_us=9_000_000)
    assert field.circles.size > 0


def test_actors_arrive_in_the_frame_the_shield_rolls_out_in():
    """End to end: ego facing +y at (5, 5), a car 20 m up-field. The shield measures
    clearance from a rollout that starts at the rig origin, so the car has to show up
    20 m along +x for `clearance` to agree with the real gap."""
    car = FakeObject("car_1", 5.0, 25.0, yaw=np.pi / 2, length=4.5, width=1.9)
    field = field_from_traffic_objects(_objects(car), [5.0, 5.0], np.pi / 2,
                                       timestamp_us=1_000_000)

    cfg = VehicleConfig()
    gap = clearance(VehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0, steer=0.0), field, cfg)

    # `clearance` measures bumper to bumper, so the true gap is 20 m less the ego's nose
    # and the car's rear half-length. Both bodies are disc-covered, so both bulges apply.
    nose_to_axle = cfg.length - cfg.rear_overhang
    true_gap = 20.0 - nose_to_axle - 4.5 / 2
    bulges = (_longitudinal_bulge(4.5, 1.9, 5)
              + _longitudinal_bulge(cfg.length, cfg.width, cfg.n_footprint_discs))

    assert 0.0 < gap < true_gap                      # conservative on both bodies
    assert gap == pytest.approx(true_gap - bulges)


def test_shield_brakes_for_an_actor_loaded_through_this_adapter():
    """The payoff, and the whole point of the ground-truth arm: a scene actor sampled from
    the artifact makes the shield intervene, with no hand-injected obstacle anywhere.

    `preview_trajectory.py` shows the same braking behaviour, but against a `CircleField`
    written by hand. This is the first time the field comes from the geometry AlpaSim
    itself steps the simulation with.
    """
    stopped = FakeObject("car_1", 30.0, 0.0, is_static=True)
    field = field_from_traffic_objects(_objects(stopped), [0.0, 0.0], 0.0,
                                       timestamp_us=1_000_000)

    cfg = VehicleConfig()
    driver = ShieldedDriver(cfg=cfg, camera_ids=["camera_front_wide_120fov"],
                            output_frequency_hz=2, horizon_steps=12, obstacles=field)
    xy = driver._rollout(initial_speed=12.0)

    # Coasting at 12 m/s for the 6 s horizon would cover 72 m and drive straight through.
    nose = xy[:, 0] + (cfg.length - cfg.rear_overhang)
    assert nose.max() < 30.0 - 4.5 / 2, "shield let the car reach the actor"
    assert np.all(np.diff(xy[:, 0]) >= -1e-9), "braking must not reverse the car"
