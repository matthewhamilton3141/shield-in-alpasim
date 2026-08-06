"""Smoke tests for the scaffold. Run without AlpaSim/torch installed (Mac-friendly),
same discipline as kitti-nav/gsplat-rt: pure-Python logic tested locally, GPU/AlpaSim-only
paths verified separately on a box that actually has AlpaSim installed.
"""

import numpy as np

from kitti_nav.vehicle import VehicleConfig
from shield_in_alpasim.driver import NoObstacles, ShieldedDriver


def test_no_obstacles_reports_everything_clear():
    field = NoObstacles()
    d = field.distance_to_obstacles(np.zeros((5, 2)))
    assert np.all(np.isinf(d))


def test_rollout_produces_expected_shape_and_moves_forward():
    driver = ShieldedDriver(cfg=VehicleConfig(), output_frequency_hz=2, horizon_steps=6)
    xy = driver._rollout(initial_speed=5.0)
    assert xy.shape == (6, 2)
    # No obstacles + forward speed + zero commanded steer -> monotonically increasing x.
    assert np.all(np.diff(xy[:, 0]) > 0)
    assert np.allclose(xy[:, 1], 0.0)


def test_headings_from_straight_line_are_zero():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    headings = ShieldedDriver._headings_from_xy(xy)
    assert np.allclose(headings, 0.0)
