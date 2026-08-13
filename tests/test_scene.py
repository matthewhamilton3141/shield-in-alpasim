"""Tests for the scene-geometry plumbing.

Covers the parts that can be checked without AlpaSim: the quaternion convention, reading
the ego pose out of gRPC-shaped history, the environment-variable contract, and the
driver's choice of which obstacle field to shield against.

`SceneObstacleSource.from_usdz` is deliberately untested — it is a two-line call into
`alpasim_utils.artifact.Artifact`, and a fake would only assert that the fake matches a
guess at the loader. It is covered by the on-box smoke check instead
(`scripts/check_scene_geometry.py`).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from kitti_nav.vehicle import VehicleConfig
from shield_in_alpasim.driver import EMPTY_FIELD, ShieldedDriver
from shield_in_alpasim.obstacles import field_from_boxes
from shield_in_alpasim.scene import (
    SCENE_ENV_VAR,
    SceneObstacleSource,
    ego_config_from_rig,
    ego_pose_from_history,
    yaw_from_quat,
)

from test_obstacles import FakeObject, _objects


def _pose_at_time(x, y, yaw, timestamp_us):
    """A stand-in for the gRPC `PoseAtTime` the servicer actually passes."""
    half = yaw / 2.0
    return SimpleNamespace(
        pose=SimpleNamespace(
            vec=SimpleNamespace(x=x, y=y, z=0.0),
            quat=SimpleNamespace(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)),
        ),
        timestamp_us=timestamp_us,
    )


# --- quaternion convention ----------------------------------------------------------


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.2, math.pi / 2, -math.pi + 0.01])
def test_yaw_survives_the_round_trip_through_a_quaternion(yaw):
    """Guards the gRPC (w, x, y, z) ordering. Swapping w and z here would put every actor
    in the wrong half of the world, which is the kind of bug that looks like a working
    shield until it silently brakes for nothing."""
    half = yaw / 2.0
    assert yaw_from_quat(math.cos(half), 0.0, 0.0, math.sin(half)) == pytest.approx(yaw)


# --- ego pose ----------------------------------------------------------------------


def test_latest_pose_is_the_one_used():
    """The servicer keeps history sorted and appends, so the newest pose is last."""
    history = [_pose_at_time(0.0, 0.0, 0.0, 1_000_000),
               _pose_at_time(12.0, -3.0, 0.4, 2_000_000)]
    xy, yaw, t = ego_pose_from_history(history)

    assert np.allclose(xy, [12.0, -3.0])
    assert yaw == pytest.approx(0.4)
    assert t == 2_000_000


def test_empty_history_is_reported_rather_than_guessed():
    """Happens on the first query of a session, before any pose has been reported."""
    assert ego_pose_from_history([]) is None


# --- environment contract ------------------------------------------------------------


def test_unset_scene_variable_yields_no_source(monkeypatch):
    """Unset must degrade to the inert driver, not crash: on a metered box, a driver that
    refuses to start costs more than one that coasts."""
    monkeypatch.delenv(SCENE_ENV_VAR, raising=False)
    assert SceneObstacleSource.from_env() is None


def test_a_scene_path_that_does_not_exist_fails_loudly(monkeypatch):
    """The opposite case: a typo'd path means the run was *meant* to have ground-truth
    geometry and silently would not have — indistinguishable from a shield that never
    fires, which is the one failure that must never be quiet."""
    monkeypatch.setenv(SCENE_ENV_VAR, "/nope/missing_scene.usdz")
    with pytest.raises(FileNotFoundError, match="missing_scene.usdz"):
        SceneObstacleSource.from_env()


# --- ego footprint -------------------------------------------------------------------


def _alpasim_s223():
    """AlpaSim's default ego (`alpasim_utils.scenario.VehicleConfig`, a Mercedes S223)."""
    return SimpleNamespace(aabb_x_m=5.393, aabb_y_m=2.109, aabb_z_m=1.503,
                           aabb_x_offset_m=-1.3)


def test_ego_footprint_follows_the_car_alpasim_is_actually_simulating():
    """kitti-nav ships a VW Passat; AlpaSim's ego is a much larger S223. Shielding the
    Passat would certify a footprint short and narrow of the body the simulator collides
    with — optimistic in the one direction a safety envelope must never be."""
    cfg = ego_config_from_rig(_alpasim_s223())

    assert cfg.length == pytest.approx(5.393)
    assert cfg.width == pytest.approx(2.109)
    assert cfg.rear_overhang == pytest.approx(1.3)   # sign flips: offset runs rig -> bumper

    kitti = VehicleConfig()
    assert cfg.length > kitti.length and cfg.width > kitti.width


def test_non_geometric_shield_parameters_are_left_alone():
    """AlpaSim's VehicleConfig has no steering or brake model, so only geometry is taken.
    Silently resetting the actuation limits would change what the shield certifies."""
    kitti = VehicleConfig()
    cfg = ego_config_from_rig(_alpasim_s223())

    assert cfg.max_decel == kitti.max_decel
    assert cfg.max_speed == kitti.max_speed
    assert cfg.wheelbase == kitti.wheelbase      # known gap: S223 is ~3.11 m, see docstring


def test_missing_rig_config_keeps_the_existing_defaults():
    assert ego_config_from_rig(None) == VehicleConfig()


# --- driver field selection ----------------------------------------------------------


def _driver(**kwargs):
    return ShieldedDriver(cfg=VehicleConfig(), camera_ids=["camera_front_wide_120fov"],
                          output_frequency_hz=2, horizon_steps=12, **kwargs)


def test_driver_shields_against_the_scene_actors_when_a_source_is_configured():
    """Ego facing +y at (5, 5), a parked car 30 m up-field. The field has to be rebuilt in
    the ego's frame each call, so the car must show up ahead and stop the rollout."""
    source = SceneObstacleSource(
        _objects(FakeObject("car_1", 5.0, 35.0, yaw=np.pi / 2, is_static=True))
    )
    driver = _driver(scene_source=source)

    prediction_input = SimpleNamespace(
        ego_pose_history=[_pose_at_time(5.0, 5.0, np.pi / 2, 1_000_000)]
    )
    xy = driver._rollout(12.0, driver._obstacles_for(prediction_input))

    assert xy[-1, 0] < 30.0, "shield did not brake for the scene actor"


def test_driver_falls_back_to_the_fixed_field_before_any_pose_arrives():
    """No ego pose yet means no frame to place actors in. Falling back is safe here only
    because the car is still under ground-truth replay at that point."""
    fixed = field_from_boxes([[10.0, 0.0]], [0.0], [4.5], [1.9])
    source = SceneObstacleSource(_objects(FakeObject("car_1", 5.0, 35.0, is_static=True)))
    driver = _driver(obstacles=fixed, scene_source=source)

    assert driver._obstacles_for(SimpleNamespace(ego_pose_history=[])) is fixed


def test_driver_without_a_scene_source_is_unchanged():
    """The inert path still has to work — it is what runs when the env var is unset."""
    driver = _driver()
    assert driver._obstacles_for(SimpleNamespace(ego_pose_history=[])) is EMPTY_FIELD
