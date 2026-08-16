"""Tests for the Tier 2 RL probe env (`shield_in_alpasim.rl_env`).

The one property that matters for the safe-RL claim: with the shield ON and a certified start,
the agent never collides, no matter how adversarial its action stream. With the shield OFF the
same reckless stream *can* collide — so the env is genuinely exercising the veto, not a task
that's trivially collision-free either way.
"""

from __future__ import annotations

import numpy as np

from shield_in_alpasim.rl_env import (
    ACTION_GRID,
    N_ACTIONS,
    EnvConfig,
    ShieldNavEnv,
)


def test_reset_obs_shape_and_dim():
    env = ShieldNavEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert env.obs_dim == 3 + 3 * env.cfg.n_nearest


def test_action_grid_covers_brake_and_go_straight():
    # A shield needs a brake option and the policy a go-straight; both must exist in the grid.
    assert (-4.5, 0.0) in ACTION_GRID          # full brake, straight
    assert (2.0, 0.0) in ACTION_GRID           # full go, straight
    assert N_ACTIONS == len(ACTION_GRID)


def test_step_returns_wellformed_tuple():
    env = ShieldNavEnv(seed=1)
    env.reset()
    obs, r, done, info = env.step(0)
    assert obs.shape == (env.obs_dim,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert {"intervened", "ics", "collision", "offroad", "goal"} <= set(info)


def _run_full_throttle(shield: bool, seeds=range(40)):
    """Drive straight at full throttle toward the obstacle field on many layouts; report if any
    episode collided. Full-throttle-straight is the worst reckless policy for a corridor with
    obstacles dead ahead."""
    go_straight = ACTION_GRID.index((2.0, 0.0))
    collisions = 0
    for s in seeds:
        env = ShieldNavEnv(cfg=EnvConfig(), shield=shield, seed=s)
        env.reset()
        while True:
            _, _, done, info = env.step(go_straight)
            collisions += int(info["collision"])
            if done:
                break
    return collisions


def test_shield_guarantees_no_collision_from_certified_start():
    # The whole point: shield ON => zero collisions across many random layouts, even under the
    # most reckless (full-throttle-straight) action stream.
    assert _run_full_throttle(shield=True) == 0


def test_unshielded_reckless_policy_does_collide():
    # Control: the same reckless stream WITHOUT the shield collides on at least some layouts,
    # proving the env actually poses a collision risk (the shield isn't guarding a safe task).
    assert _run_full_throttle(shield=False) > 0


def test_start_state_is_certifiable_every_seed():
    # reset() asserts certifiability internally; make that a first-class test across seeds.
    for s in range(50):
        ShieldNavEnv(cfg=EnvConfig(), shield=True, seed=s).reset()


def test_goal_reachable_with_clear_corridor():
    # With no obstacles, driving straight should reach the goal and pay the goal bonus.
    env = ShieldNavEnv(cfg=EnvConfig(n_obstacles=0), shield=True, seed=0)
    env.reset()
    go_straight = ACTION_GRID.index((2.0, 0.0))
    reached = False
    for _ in range(env.cfg.horizon):
        _, _, done, info = env.step(go_straight)
        if done:
            reached = info["goal"]
            break
    assert reached


def test_observation_is_finite():
    env = ShieldNavEnv(seed=3)
    obs = env.reset()
    for _ in range(20):
        obs, _, done, _ = env.step(np.random.default_rng(0).integers(N_ACTIONS))
        assert np.all(np.isfinite(obs))
        if done:
            obs = env.reset()
