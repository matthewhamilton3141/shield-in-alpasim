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


def test_from_config_keeps_the_cameras_and_frequency_alpasim_asked_for():
    driver = ShieldedDriver.from_config(
        model_cfg=None, device="cpu", camera_ids=CAMERAS, context_length=None, output_frequency_hz=4
    )
    assert driver.camera_ids == CAMERAS
    assert driver.output_frequency_hz == 4
    assert driver.context_length == 1  # None in config -> the model's own default
