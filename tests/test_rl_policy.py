"""Tests for the Tier 2 RL policy deployment path (`rl_policy` + driver integration).

The load-bearing property for transfer: the observation the *driver* feeds the policy at
deployment must be byte-for-byte the one the *env* fed it during training. If that drifts, the
policy silently mis-behaves in AlpaSim and every render is wasted. These run on the Mac (torch is
present; AlpaSim is not — the RL path goes through `_rollout`, which is AlpaSim-free).
"""

from __future__ import annotations

import numpy as np

from kitti_nav.vehicle import CircleField, VehicleConfig, VehicleState
from shield_in_alpasim.driver import ShieldedDriver
from shield_in_alpasim.rl_env import EnvConfig, ShieldNavEnv, build_observation
from shield_in_alpasim.rl_policy import ActorCritic, RLPolicy, load_policy, save_policy

CAMERAS = ["camera_front_wide_120fov"]


def _tiny_policy(cfg: EnvConfig) -> RLPolicy:
    from shield_in_alpasim.rl_env import observation_dim
    net = ActorCritic(observation_dim(cfg))
    return RLPolicy(net, cfg)


def test_obs_bridge_matches_env_observation():
    # The driver builds its observation from (rollout state, this-cycle field) via the SAME
    # `build_observation` the env uses — assert they agree exactly for a non-trivial state+field.
    cfg = EnvConfig(n_nearest=6, goal_x=40.0)
    env = ShieldNavEnv(cfg=cfg, seed=3)
    env.reset()
    # Move the ego somewhere non-origin so the rig-frame transform is actually exercised.
    from dataclasses import replace
    env._state = replace(env.state, x=4.0, y=0.7, yaw=0.15, v=6.0, steer=0.1)
    circles = np.asarray(env.obstacles.circles, float)
    assert np.allclose(env._observe(), build_observation(env.state, circles, cfg))


def test_save_load_roundtrip_preserves_action(tmp_path):
    cfg = EnvConfig(n_nearest=6, goal_x=40.0)
    from shield_in_alpasim.rl_env import observation_dim
    net = ActorCritic(observation_dim(cfg))
    path = str(tmp_path / "p.pt")
    save_policy(net, path, cfg, meta={"arm": "unit"})
    loaded = load_policy(path)
    field = np.array([[12.0, 0.5, 0.8], [22.0, -1.2, 0.6]])
    st = VehicleState(x=0.0, y=0.0, yaw=0.0, v=2.0, steer=0.0)
    a_saved = RLPolicy(net, cfg).bound(field)(st)
    a_loaded = loaded.bound(field)(st)
    assert a_saved == a_loaded          # identical net + obs -> identical greedy action
    assert loaded.cfg.n_nearest == cfg.n_nearest and loaded.cfg.goal_x == cfg.goal_x


def test_bound_policy_returns_a_grid_action():
    from shield_in_alpasim.rl_env import ACTION_GRID
    pol = _tiny_policy(EnvConfig(n_nearest=6))
    a = pol.bound(np.array([[10.0, 0.0, 1.0]]))(VehicleState(v=3.0))
    assert a in ACTION_GRID


def test_driver_rollout_with_rl_policy_shape_and_shielding():
    # Wire the RL policy into a driver and roll it out; the shield must still bound it. With a disc
    # dead ahead the shielded rollout must not drive the ego into it (final x stays short of it).
    cfg = EnvConfig(n_nearest=6, goal_x=40.0)
    pol = _tiny_policy(cfg)
    d = ShieldedDriver(cfg=VehicleConfig(), camera_ids=CAMERAS, output_frequency_hz=2,
                       horizon_steps=6, rl_policy=pol)
    field = CircleField(np.array([[6.0, 0.0, 1.0]]))
    xy = d._rollout(initial_speed=8.0, obstacles=field, policy=pol.bound(field.circles))
    assert xy.shape == (6, 2)
    assert np.max(xy[:, 0]) < 6.0        # shield kept the ego short of the lead disc


def test_shield_disabled_flag_lets_the_policy_crash():
    # SHIELD_FILTER=0 (shield_enabled=False) must skip the veto: a full-throttle-straight policy
    # then drives into a lead disc (collided=True), where the shielded run brakes and does not.
    # Disc far enough that the start IS shield-certifiable (else both collide from an ICS).
    field = CircleField(np.array([[14.0, 0.0, 1.0]]))

    def flat_out(_state):
        return 2.0, 0.0

    shielded = ShieldedDriver(cfg=VehicleConfig(), camera_ids=CAMERAS, output_frequency_hz=2,
                              horizon_steps=8, shield_enabled=True)
    unshielded = ShieldedDriver(cfg=VehicleConfig(), camera_ids=CAMERAS, output_frequency_hz=2,
                                horizon_steps=8, shield_enabled=False)
    _, s_stats = shielded._rollout(6.0, field, flat_out, return_stats=True)
    _, u_stats = unshielded._rollout(6.0, field, flat_out, return_stats=True)
    assert u_stats["collided"] and not s_stats["collided"]
    assert u_stats["n_interventions"] == 0     # veto truly off in the unshielded run
