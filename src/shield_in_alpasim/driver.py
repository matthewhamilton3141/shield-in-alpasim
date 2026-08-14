"""AlpaSim driver plugin wrapping kitti-nav's hard safety shield.

AlpaSim's `BaseTrajectoryModel.predict()` gets camera frames in and must return waypoint
poses out; kitti-nav's shield takes a per-step `(accel_cmd, steer_cmd)` in and an
`ObstacleField` to check against, and returns a certified-safe `(accel, steer)` out.

Obstacles are real now: with `$SHIELD_SCENE_USDZ` set, `predict()` samples the scene's
actors at the current ego pose and timestamp (`scene.py`, `obstacles.py`) and the shield
brakes for them. Unset, the field is empty and the car coasts — the old scaffold behaviour,
kept as the fallback because a driver that refuses to start is worse on a metered box.

**The policy is now optional-but-supported.** The shield is a filter: it certifies a proposed
`(accel, steer)` and never proposes one. Set `$SHIELD_INNER_MODEL` to another registered
`alpasim.models` name (e.g. `transfuser`) and this becomes a *decorator*: each `predict()`
calls the inner model, tracks its waypoints into per-step commands (`control.py`), and rolls
those through the shield — so the car actually drives and the shield vetoes only what is
unsafe. Unset, `_rollout` commands `(0.0, 0.0)` — the coasting baseline, which self-drives
nowhere but keeps the ground-truth arm runnable. See HANDOFF.md, "Open decision".

Provenance: the shield itself is not implemented here — it's imported from `kitti_nav`,
see ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
import os

import numpy as np

try:
    import torch
    from alpasim_driver.models.base import (
        BaseTrajectoryModel,
        DriveCommand,
        ModelPrediction,
        PredictionInput,
    )

    _HAS_ALPASIM = True
except ImportError:  # AlpaSim isn't installed (e.g. on a Mac dev box, no GPU stack) —
    # the class below still imports and its non-AlpaSim logic is still testable.
    _HAS_ALPASIM = False
    BaseTrajectoryModel = object  # type: ignore[assignment,misc]

from kitti_nav.vehicle import (
    CircleField,
    VehicleConfig,
    VehicleState,
    clearance,
    shielded_rollout,
)
from shield_in_alpasim.control import make_tracking_policy
from shield_in_alpasim.scene import SceneObstacleSource, ego_pose_from_history

logger = logging.getLogger(__name__)

# The env var naming the inner policy to shield. Unset -> coasting baseline (see module docs).
INNER_MODEL_ENV_VAR = "SHIELD_INNER_MODEL"

# Seconds of trajectory to emit. This MUST cover the controller's MPC horizon, which is
# `n_horizon * dt_mpc = 20 * 0.1 = 2.0 s` (alpasim_controller `mpc_controller.py`,
# `controller/default.yaml`). The MPC *clamps* horizon timestamps to the reference's time
# range (`linear_mpc.py:_interpolate_reference`), so any horizon step past the end of our
# trajectory tracks our LAST waypoint — a stationary target the MPC then brakes to a stop at.
# The original 6-waypoint (~0.6 s) output undershot this and made the car halt ~2 m in; both
# the coast and VaVAM runs died at ~8 m for exactly this reason. 3.0 s clears 2.0 s with margin
# (matches the route `horizon_s: 3.0` in base_config.yaml).
DEFAULT_HORIZON_S = 3.0

# A field with no obstacles reports every point infinitely clear, so the shield never
# intervenes. Used when no scene artifact is configured, and as the per-call fallback when
# the ego pose history has not arrived yet.
EMPTY_FIELD = CircleField(None)

# Drop obstacles the shield cannot hit by moving forward — set SHIELD_REAR_FILTER=0 to disable.
REAR_FILTER_ENV_VAR = "SHIELD_REAR_FILTER"

# Emit the inner policy's plan unchanged when the shield doesn't intervene (a true filter only
# acts when it must). Set SHIELD_PASSTHROUGH=0 to always emit our tracked re-roll instead.
PASSTHROUGH_ENV_VAR = "SHIELD_PASSTHROUGH"

# Defer to the inner policy when the ego is faster than the shield's model represents (its
# kinematic model clamps speed to max_speed). Set SHIELD_OOD_GUARD=0 to shield anyway.
OOD_GUARD_ENV_VAR = "SHIELD_OOD_GUARD"

# Where the shield's obstacle field comes from: `gt` (default; ground-truth scene geometry) or
# `camera` (learned metric-depth perception over the front camera — the degradation experiment).
OBSTACLE_SOURCE_ENV_VAR = "SHIELD_OBSTACLE_SOURCE"


def forward_relevant_field(field, cfg):
    """Drop obstacle discs that sit entirely behind the ego's rear bumper.

    The shield never reverses (`min_speed=0`) and its braking rollout only moves *forward*, so a
    disc behind the rear bumper only recedes — it can never be hit. But kitti-nav's `clearance`
    is omnidirectional and `can_stop_safely` refuses to certify when the *current* clearance is
    below `safety_margin`, so a car close behind freezes the ego in place. That is both useless
    (braking cannot avoid a rear threat) and measurably costly (finding 2: it halved progress on
    one sweep scene). This removes exactly those discs and nothing else.

    Cut at the rear bumper (`x = -rear_overhang` in the rig frame, origin at the rear axle)
    minus `safety_margin`, and use each disc's own forward-most extent (`x + r`) so a disc that
    only *reaches* the rear corner is kept — conservative against the rear overhang's outward
    swing on a turn. Discs beside or ahead of the ego are untouched: those are real.
    """
    circles = np.asarray(getattr(field, "circles", np.zeros((0, 3))), float).reshape(-1, 3)
    if len(circles) == 0:
        return field
    rear_cut = -(cfg.rear_overhang + cfg.safety_margin)
    keep = (circles[:, 0] + circles[:, 2]) >= rear_cut
    if keep.all():
        return field
    return CircleField(circles[keep])


def _coast(_state) -> tuple[float, float]:
    """The no-policy fallback: command zero accel and zero steer, and let the shield decide.

    Straight, speed-holding, and inert — the baseline that keeps the ground-truth arm
    runnable when no `$SHIELD_INNER_MODEL` is configured. It self-drives nowhere; that is the
    point of the "the shield needs a policy" caveat in HANDOFF.md.
    """
    return 0.0, 0.0


def rollout_diagnostics(field, state0: VehicleState, cfg: VehicleConfig, policy) -> dict:
    """Per-cycle diagnostics: is the shield braking for something *ahead*, or *behind*?

    The whole field is in the rig frame — ego at the origin, heading +x — so the sign of an
    obstacle disc's x coordinate says ahead (+) or behind (−). That distinction is the crux
    of "over-conservative vs genuinely blocked": braking cannot avoid a threat behind the
    ego, so a shield that freezes for a rear/side disc is being over-conservative, while one
    that stops for a lead disc is doing its job. `nearest_gap_m` is the surface-to-origin gap
    (negative means a disc already overlaps the rig origin — the shield will read that as an
    unavoidable collision and command full brake).

    Pure numpy so it is unit-tested here rather than only observed on a metered run.
    """
    circles = np.asarray(getattr(field, "circles", np.zeros((0, 3))), float).reshape(-1, 3)
    proposed_accel, proposed_steer = policy(state0)
    diag: dict = {
        "speed_in": round(float(state0.v), 2),
        "proposed_accel": round(float(proposed_accel), 2),
        "proposed_steer": round(float(proposed_steer), 3),
        "init_clearance_m": round(float(clearance(state0, field, cfg)), 2),
        "n_obstacles": int(len(circles)),
    }
    if len(circles):
        gap = np.linalg.norm(circles[:, :2], axis=1) - circles[:, 2]
        i = int(np.argmin(gap))
        diag["nearest_gap_m"] = round(float(gap[i]), 2)
        diag["nearest_bearing_deg"] = round(float(np.degrees(np.arctan2(circles[i, 1], circles[i, 0]))), 1)
        ahead = circles[circles[:, 0] > 0.0]
        diag["nearest_ahead_gap_m"] = (
            round(float(np.min(np.linalg.norm(ahead[:, :2], axis=1) - ahead[:, 2])), 2)
            if len(ahead) else None
        )
    return diag


class ShieldedDriver(BaseTrajectoryModel if _HAS_ALPASIM else object):
    """Registered as `shielded` under the `alpasim.models` entry point (pyproject.toml).

    CPU-only: the shield is pure numpy, so unlike AlpaSim's stock drivers this one needs
    no GPU and no checkpoint (see `configs/driver/shielded.yaml`).
    """

    def __init__(
        self,
        cfg: VehicleConfig,
        camera_ids: list[str],
        context_length: int = 1,
        output_frequency_hz: int = 2,
        horizon_steps: int | None = None,
        horizon_s: float = DEFAULT_HORIZON_S,
        obstacles=None,
        scene_source: SceneObstacleSource | None = None,
        obstacle_source=None,
        inner_model=None,
        rear_filter: bool | None = None,
        passthrough: bool | None = None,
        ood_guard: bool | None = None,
    ):
        self._cfg = cfg
        # Ignore un-hittable behind-the-ego obstacles (see `forward_relevant_field`). On by
        # default; `SHIELD_REAR_FILTER=0` disables it for an A/B against the over-conservative
        # behaviour. Explicit constructor arg wins (tests).
        self._rear_filter = (
            os.environ.get(REAR_FILTER_ENV_VAR, "1") != "0" if rear_filter is None
            else rear_filter
        )
        # Emit the inner policy's plan verbatim when the shield doesn't intervene, rather than
        # our tracked re-roll (whose lateral error otherwise drifts the car off the logged
        # route). On by default; `SHIELD_PASSTHROUGH=0` disables it.
        self._passthrough = (
            os.environ.get(PASSTHROUGH_ENV_VAR, "1") != "0" if passthrough is None
            else passthrough
        )
        # Defer to the inner policy when the ego is out of the shield's speed domain (see
        # `_out_of_domain`). On by default; `SHIELD_OOD_GUARD=0` disables it.
        self._ood_guard = (
            os.environ.get(OOD_GUARD_ENV_VAR, "1") != "0" if ood_guard is None
            else ood_guard
        )
        self._camera_ids = list(camera_ids)
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        # Emit enough waypoints to span `horizon_s` at the output rate, so the trajectory
        # covers the MPC horizon (see DEFAULT_HORIZON_S). An explicit `horizon_steps` still
        # wins (tests, previews); None derives it from the rate.
        self._horizon_steps = (
            horizon_steps if horizon_steps is not None
            else max(2, round(horizon_s * output_frequency_hz))
        )
        # A fixed field, for tests and for `scripts/preview_trajectory.py`. Ignored when a
        # `scene_source` is present, since that resamples per call as actors move.
        self._obstacles = EMPTY_FIELD if obstacles is None else obstacles
        self._scene_source = scene_source
        # The learned-perception arm: any `ObstacleSource` (see `obstacle_source.py`) that builds
        # the field from camera frames. When set it takes precedence over the ground-truth
        # `scene_source` — this is the one seam the perfect-vs-learned experiment swaps.
        self._obstacle_source = obstacle_source
        # The policy being shielded, or None for the coasting baseline. Any object with a
        # `predict(prediction_input) -> ModelPrediction` — i.e. another `BaseTrajectoryModel`.
        self._inner_model = inner_model

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        # `device` is ignored on purpose — the shield is numpy, there is nothing to move
        # onto a GPU. `context_length` is None when the config leaves it to the model.
        #
        # The scene path comes from the environment, not `model_cfg`: the driver merges its
        # YAML onto a structured `DriverConfig` in struct mode, so an extra key here would
        # raise. See `scene.py` for the full reasoning.
        scene_source = SceneObstacleSource.from_env()

        # Take the ego's real dimensions from the scene when we have them. kitti-nav's
        # defaults are a VW Passat; AlpaSim's ego is a much larger S223, and shielding the
        # wrong footprint is optimistic in exactly the direction that matters.
        cfg = VehicleConfig()
        if scene_source is not None and scene_source.ego_vehicle_config is not None:
            cfg = scene_source.ego_vehicle_config

        inner_model = cls._build_inner_model(
            model_cfg, device, camera_ids, context_length, output_frequency_hz
        )
        obstacle_source = cls._build_obstacle_source(device, camera_ids)

        return cls(
            cfg=cfg,
            camera_ids=camera_ids,
            context_length=context_length or 1,
            output_frequency_hz=output_frequency_hz,
            scene_source=scene_source,
            obstacle_source=obstacle_source,
            inner_model=inner_model,
        )

    @staticmethod
    def _build_obstacle_source(device, camera_ids):
        """The learned-perception obstacle source when `$SHIELD_OBSTACLE_SOURCE=camera`, else None.

        None keeps the ground-truth arm (the shield reads the scene's actors). `camera` swaps in
        a monocular metric-depth net over the front camera — the same shield, a *learned* obstacle
        field, so the run measures how much the certificate degrades under real perception. The
        depth-net imports (`torch`/`transformers`) are deferred to here so the ground-truth path
        and the Mac tests never touch them.
        """
        kind = os.environ.get(OBSTACLE_SOURCE_ENV_VAR, "gt").lower()
        if kind in ("", "gt", "ground_truth", "groundtruth"):
            return None
        if kind != "camera":
            raise ValueError(f"{OBSTACLE_SOURCE_ENV_VAR}={kind!r}; expected 'gt' or 'camera'.")

        from shield_in_alpasim.depth import DEFAULT_MODEL, HFDepthModel
        from shield_in_alpasim.obstacle_source import CameraObstacleSource

        model_name = os.environ.get("SHIELD_DEPTH_MODEL", DEFAULT_MODEL)
        logger.info("Obstacle field from CAMERA perception (depth model %s)", model_name)
        depth_model = HFDepthModel(model_name, device=str(device))
        return CameraObstacleSource.front_wide(depth_model, camera_id=camera_ids[0])

    @staticmethod
    def _build_inner_model(model_cfg, device, camera_ids, context_length, output_frequency_hz):
        """The policy to shield, from `$SHIELD_INNER_MODEL`, or None for the coasting baseline.

        Env-selected for the same reason the scene path is (`scene.py`): the driver merges its
        YAML onto a struct-mode `DriverConfig`, so a new `model:` key would raise rather than
        pass through. The inner model reuses *this* driver's `model_cfg` — so the shielded
        config's `checkpoint_path`/`device` are the inner model's, and its `use_cameras` must
        be the ones the inner model expects (e.g. Transfuser's four).

        The registry import is deferred to here because it lives in AlpaSim, which is absent on
        a Mac dev box; `from_config` only ever runs inside the driver container, where it is
        present.
        """
        inner_name = os.environ.get(INNER_MODEL_ENV_VAR)
        if not inner_name:
            logger.info(
                "%s unset: coasting baseline (the shield filters, but nothing drives).",
                INNER_MODEL_ENV_VAR,
            )
            return None

        from alpasim_plugins.plugins import models as model_registry

        inner_cls = model_registry.get(inner_name)
        logger.info("Shielding inner policy %r (%s)", inner_name, inner_cls.__name__)
        return inner_cls.from_config(
            model_cfg, device, camera_ids, context_length, output_frequency_hz
        )

    def _encode_command(self, command: "DriveCommand") -> int:
        # The shield filters a command, it does not propose one, so LEFT/STRAIGHT/RIGHT has
        # nothing to act on until a real upstream policy is wired in (plan step 4).
        # DriveCommand is an IntEnum, so this is its own encoding.
        return int(command)

    def _obstacles_for(self, prediction_input) -> "CircleField":
        """Obstacle field for this inference: scene actors if configured, else the fixed one.

        Falls back to the fixed field when the ego pose history is empty, which happens on
        the first query of a session before any pose has been reported. An empty field is
        the safe direction to fail here only because the car is still under ground-truth
        replay at that point (`force_gt_duration_us`); it is not yet driving on the shield.
        """
        if self._obstacle_source is not None:
            # Learned-perception arm: build the field from camera frames.
            field = self._obstacle_source.field_for(prediction_input)
        elif self._scene_source is not None:
            # Ground-truth arm: sample the scene's actors at the current ego pose.
            ego = ego_pose_from_history(prediction_input.ego_pose_history)
            if ego is None:
                return self._obstacles
            ego_xy, ego_yaw, timestamp_us = ego
            field = self._scene_source.field_at(ego_xy, ego_yaw, timestamp_us)
        else:
            return self._obstacles

        if self._rear_filter:
            field = forward_relevant_field(field, self._cfg)
        return field

    def _rollout(self, initial_speed: float, obstacles=None, policy=None,
                 horizon_steps: int | None = None, return_stats: bool = False):
        """Roll `policy` through the shield for `horizon_steps`, return `(T, 2)` xy.

        `policy(state) -> (accel, steer)`; defaults to `_coast` (go straight, hold speed), the
        baseline the shield filters when no inner model is configured. `horizon_steps` defaults
        to `self._horizon_steps` (sized to span the MPC horizon, see DEFAULT_HORIZON_S). With
        `return_stats`, returns `(xy, stats)` so the caller can see whether the shield
        intervened. Kept free of AlpaSim types so it's testable without AlpaSim/torch installed.
        """
        obstacles = self._obstacles if obstacles is None else obstacles
        policy = _coast if policy is None else policy
        horizon_steps = self._horizon_steps if horizon_steps is None else horizon_steps
        # Waypoints are ego-relative in the rig frame, so the rollout always starts at the
        # origin. Initial steer is assumed centred: AlpaSim's `PredictionInput` carries no
        # road-wheel angle (`ego_pose_history` could be differenced for it — refinement,
        # not needed while the commanded steer is zero anyway).
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=initial_speed, steer=0.0)
        # cfg.dt is the shield's control step; sub-step up to each output waypoint so the
        # shield (and its braking lookahead) always runs at its own tested integration rate,
        # regardless of what output_frequency_hz AlpaSim asks for.
        substeps_per_waypoint = max(1, round(1.0 / self._output_frequency_hz / self._cfg.dt))
        n_steps = horizon_steps * substeps_per_waypoint

        # Snapshot the situation *before* the shield acts, so the log distinguishes "blocked
        # ahead" (shield doing its job) from "braking for a rear/side actor" (over-conservative).
        diag = rollout_diagnostics(obstacles, state, self._cfg, policy)

        states, stats = shielded_rollout(policy, state, obstacles, self._cfg, n_steps=n_steps)

        # The intervention count is the experiment's signal: how often the shield overrode the
        # inner policy. One line per cycle (runs are short) so a run's driver log is a full
        # trace of proposed-vs-shielded and what the shield was reacting to.
        diag.update(
            horizon_s=round(horizon_steps / self._output_frequency_hz, 2),
            n_interventions=stats["n_interventions"],
            final_speed=round(float(stats["final_speed"]), 2),
            collided=stats["collided"],
        )
        logger.info("shield cycle %s", diag)
        xy = np.zeros((horizon_steps, 2))
        for t in range(horizon_steps):
            # states[0] is the initial state, so waypoint t lands substeps*(t+1) steps in.
            # shielded_rollout truncates on collision; clamping holds the last pose so the
            # trajectory stays the fixed length AlpaSim expects.
            xy[t] = states[min((t + 1) * substeps_per_waypoint, len(states) - 1)].xy
        return (xy, stats) if return_stats else xy

    def _proposed_waypoints(self, prediction_input) -> np.ndarray | None:
        """Ground-plane `(T, 2)` waypoints from the inner policy, or None when coasting.

        Reads the inner model's selected trajectory and drops to the ground plane; the shield
        is a BEV kinematic model, so the z of the waypoints is discarded (same yaw-only
        projection `obstacles.py` uses on actors).
        """
        if self._inner_model is None:
            return None
        prediction = self._inner_model.predict(prediction_input)
        return np.asarray(prediction.selected_positions, dtype=float)[:, :2]

    def _out_of_domain(self, speed: float) -> bool:
        """True when the ego is faster than the shield's kinematic model represents.

        `step_state` clamps speed to `max_speed`, so above it the shield's stopping-distance
        reasoning (`v²/2a`) is computed for a *slower* car than reality — it under-estimates
        the distance needed and hard-brakes into collisions a faster policy would steer around
        (measured: it turned the highway scene 04394343 from VaVAM's pass into a shielded
        crash). Out of domain, the sound move is to abstain, not to act on a wrong model.
        """
        return self._ood_guard and float(speed) > self._cfg.max_speed

    def predict(self, prediction_input: "PredictionInput") -> "ModelPrediction":
        self._validate_cameras(prediction_input.camera_images)
        obstacles = self._obstacles_for(prediction_input)

        # Coasting baseline (no inner policy): shield-filter a go-straight command.
        if self._inner_model is None:
            xy = self._rollout(prediction_input.speed, obstacles, None)
            return ModelPrediction.from_planar(xy, self._compute_headings_from_trajectory(xy))

        # Decorator path: track the inner plan, run it through the shield.
        inner_prediction = self._inner_model.predict(prediction_input)

        # Out-of-domain guard: above the shield's modelled speed, defer to the inner policy
        # untouched rather than certify with a model that does not apply (see `_out_of_domain`).
        if self._out_of_domain(prediction_input.speed):
            logger.info(
                "shield out-of-domain: ego %.1f m/s > max_speed %.1f — passing inner plan through",
                float(prediction_input.speed), self._cfg.max_speed,
            )
            return inner_prediction
        proposed = np.asarray(inner_prediction.selected_positions, dtype=float)[:, :2]
        policy = make_tracking_policy(proposed, 1.0 / self._output_frequency_hz, self._cfg)
        # Emit at least the inner model's own horizon as well as the MPC's: a shorter output
        # would be clamped to its endpoint by the MPC and braked to a stop there.
        horizon_steps = max(self._horizon_steps, len(proposed))
        xy, stats = self._rollout(
            prediction_input.speed, obstacles, policy, horizon_steps, return_stats=True
        )

        if self._passthrough and stats["n_interventions"] == 0:
            # The shield had no objection to the inner plan — emit it *verbatim* rather than our
            # tracked re-roll, whose lateral error otherwise walks the car off the logged route
            # (measured: it truncated progress on curvier scenes). A filter only alters its
            # input when it must. Soundness note (HANDOFF, "passthrough"): the certificate is on
            # the tracked commands, and the raw plan is within tracking error of them.
            return inner_prediction

        # The shield intervened: emit the certified (braked/steered) trajectory instead.
        return ModelPrediction.from_planar(xy, self._compute_headings_from_trajectory(xy))

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz
