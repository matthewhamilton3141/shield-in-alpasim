"""Ground-truth obstacle field: AlpaSim scene geometry -> kitti-nav `CircleField`.

This is the privileged/"guarantee intact" arm of the experiment (HANDOFF.md, "Settled").
AlpaSim hands the driver no geometry in-band — neither `PredictionInput` nor the
`DriveRequest` behind it carries actor state — but the same geometry the simulator itself
steps is loadable straight off the scene artifact via
`alpasim_utils.scene_data_source.SceneDataSource.traffic_objects`.

Read the two caveats in HANDOFF.md before trusting a number produced through here:

1. The on-disk log equals the simulated state **only under `trafficsim: disabled`** (the
   wizard default). Under `trafficsim=catk` actors react and diverge, and this field goes
   quietly stale — which is the worst failure mode a safety shield can have, so the caller
   is responsible for not doing that.
2. The ego pose the driver is given is deliberately noised (`local -> rig_est`), so even
   this arm is *perfect geometry, imperfect localization*.

Layering: everything below the `field_from_traffic_objects` glue is pure numpy and takes
plain arrays. `Pose` and `Trajectory` are compiled Rust types from `utils_rs` that cannot
be imported on a dev box without AlpaSim, so the glue touches them through the narrowest
duck-typed surface that works (`vec3`, `yaw()`, `interpolate_pose`, `get_time_range_tuple`)
and the geometry itself stays testable here.
"""

from __future__ import annotations

import numpy as np

from kitti_nav.vehicle import CircleField

# The runtime prepends the ego's own box to the actor list it sends trafficsim
# (`runtime/services/traffic_service.py:69`). Shielding against your own footprint would
# wedge the car in place, so it is dropped by id wherever it shows up.
EGO_TRACK_ID = "EGO"

# Obstacles beyond this are dropped before the field is built. `distance_to_obstacles` is
# O(query points x circles) and every actor contributes `n_discs` circles, so a busy scene
# is worth trimming. The shield only ever reasons out to its braking lookahead, and the
# default is several times the ~56 m it takes to stop from `max_speed` under `max_decel`.
DEFAULT_MAX_RANGE_M = 80.0


def rect_discs(
    centres_xy: np.ndarray,
    yaws: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    n_discs: int = 5,
) -> np.ndarray:
    """Cover each oriented box with `n_discs` equal discs; returns `(N * n_discs, 3)`.

    Same construction kitti-nav uses for the ego footprint (`vehicle.footprint_discs`):
    split the box along its length into equal segments and circumscribe each, with radius
    `sqrt((L/2n)^2 + (W/2)^2)`. Coverage is complete, so the approximation is
    **conservative** — it can invent clearance the box does not have, never the reverse.

    Not a call into `footprint_discs` itself: that one is parameterised on a `VehicleState`
    and a `VehicleConfig` and measures its offsets forward from the *rear axle*, because
    that is where the ego's pose lives. An actor's pose is at its box *centre*
    (CONTRIBUTING.md, "Coordinate Systems": the `aabb` frame), so the offsets differ and
    there is no vehicle config to speak of. The shared part is the two lines of algebra
    below; if that formula ever changes upstream, this needs the same change.
    """
    n = max(int(n_discs), 1)
    centres_xy = np.asarray(centres_xy, float).reshape(-1, 2)
    yaws = np.asarray(yaws, float).reshape(-1)
    lengths = np.asarray(lengths, float).reshape(-1)
    widths = np.asarray(widths, float).reshape(-1)

    if len(centres_xy) == 0:
        return np.zeros((0, 3), float)

    seg = lengths / n                                        # (N,)
    radii = np.hypot(seg / 2.0, widths / 2.0)                # (N,)

    # Segment centres along each box axis, measured from the box centre.
    steps = np.arange(n) + 0.5                               # (n,)
    offsets = -lengths[:, None] / 2.0 + seg[:, None] * steps  # (N, n)

    cos_yaw, sin_yaw = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    cx = centres_xy[:, 0:1] + offsets * cos_yaw              # (N, n)
    cy = centres_xy[:, 1:2] + offsets * sin_yaw

    return np.stack(
        [cx.ravel(), cy.ravel(), np.repeat(radii, n)], axis=1
    )


def to_rig_frame(
    ego_xy: np.ndarray,
    ego_yaw: float,
    points_xy: np.ndarray,
    yaws: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Passive transform of `local`-frame poses into the ego's `rig` frame.

    The rig frame is x-forward, y-left, origin at the rear axle projected onto the ground
    (CONTRIBUTING.md), which is also the frame kitti-nav's shield rolls out in — it starts
    every rollout at the origin.

    Yaw-only, i.e. the boxes are projected onto the ground plane rather than transformed in
    full SE(3). That is deliberate: the shield is a bird's-eye kinematic model with no
    notion of pitch or roll, so carrying them would be discarded a line later. It costs
    accuracy on graded road, where a box's ground-plane footprint is slightly foreshortened.
    """
    points_xy = np.asarray(points_xy, float).reshape(-1, 2)
    ego_xy = np.asarray(ego_xy, float).reshape(2)

    delta = points_xy - ego_xy
    cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
    rotated = np.stack(
        [
            delta[:, 0] * cos_yaw - delta[:, 1] * sin_yaw,
            delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw,
        ],
        axis=1,
    )

    if yaws is None:
        return rotated, np.zeros(len(rotated))
    return rotated, np.asarray(yaws, float).reshape(-1) - ego_yaw


def field_from_boxes(
    centres_xy: np.ndarray,
    yaws: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    n_discs: int = 5,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
) -> CircleField:
    """Build a `CircleField` from oriented boxes already expressed in the rig frame."""
    centres_xy = np.asarray(centres_xy, float).reshape(-1, 2)
    lengths = np.asarray(lengths, float).reshape(-1)
    widths = np.asarray(widths, float).reshape(-1)
    yaws = np.asarray(yaws, float).reshape(-1)

    if len(centres_xy) == 0:
        return CircleField(None)

    # Range-filter on the nearest possible point of each box, not its centre, so trimming
    # can never drop a box that reaches inside the cutoff.
    half_diagonal = np.hypot(lengths, widths) / 2.0
    within = np.linalg.norm(centres_xy, axis=1) - half_diagonal <= max_range_m

    return CircleField(
        rect_discs(centres_xy[within], yaws[within], lengths[within], widths[within], n_discs)
    )


def field_from_traffic_objects(
    traffic_objects,
    ego_xy,
    ego_yaw: float,
    timestamp_us: int,
    n_discs: int = 5,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
) -> CircleField:
    """Sample every actor at `timestamp_us` and return the field in the ego's rig frame.

    `traffic_objects` is AlpaSim's `TrafficObjects` (a `dict[str, TrafficObject]`), but is
    only ever duck-typed: each value needs `.track_id`, `.is_static`, `.aabb.x/.y` (box
    length/width, per the `aabb` frame's x-forward convention) and a `.trajectory`
    exposing `get_time_range_tuple()` and `interpolate_pose()`.

    Actors whose track does not cover `timestamp_us` are **skipped, not extrapolated** — a
    track that has not started or has already ended says nothing about where that actor is,
    and inventing a pose for it would put a phantom obstacle in front of a braking
    certificate. Static objects are exempt: their pose is constant, so clamping a query
    onto the end of a one-pose track is exact rather than a guess.
    """
    centres, yaws, lengths, widths = [], [], [], []

    for obj in traffic_objects.values():
        if getattr(obj, "track_id", None) == EGO_TRACK_ID:
            continue

        trajectory = obj.trajectory
        start_us, end_us = trajectory.get_time_range_tuple()
        if start_us == 0 and end_us == 0:                      # empty track
            continue

        at_us = min(max(int(timestamp_us), int(start_us)), int(end_us))
        if at_us != int(timestamp_us) and not obj.is_static:
            continue

        pose = trajectory.interpolate_pose(at_us)
        centres.append(np.asarray(pose.vec3, float)[:2])
        yaws.append(float(pose.yaw()))
        lengths.append(float(obj.aabb.x))
        widths.append(float(obj.aabb.y))

    if not centres:
        return CircleField(None)

    rig_xy, rig_yaws = to_rig_frame(ego_xy, ego_yaw, np.array(centres), np.array(yaws))
    return field_from_boxes(
        rig_xy, rig_yaws, np.array(lengths), np.array(widths), n_discs, max_range_m
    )
