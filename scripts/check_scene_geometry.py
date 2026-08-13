#!/usr/bin/env python3
"""Validate the ground-truth obstacle path against a real scene. CPU only, no renderer.

Run this **before** starting a full AlpaSim rollout. Loading a `.usdz` and building
obstacle fields needs no GPU, no NuRec renderer, no physics and no gRPC — so every
frame-convention bug in `obstacles.py` / `scene.py` is catchable here, at zero metered cost,
instead of showing up as a mystifying collision an hour into a rendered run.

**The oracle.** The scene's logged ego trajectory is a real human drive that did not crash.
Replay it against the logged actors and the ego footprint must stay clear the whole way. If
clearance goes negative, the geometry is wrong — a frame confusion, a bad quaternion
convention, or the wrong ego dimensions — not the scene. That single check exercises the
transform, the disc cover, the time sampling and the ego footprint at once.

Usage:
    python3 scripts/check_scene_geometry.py /mnt/nre-data/<sceneset>/<scene>.usdz
    python3 scripts/check_scene_geometry.py          # falls back to $SHIELD_SCENE_USDZ

Exits non-zero if the oracle fails, so it can gate a run in a shell script.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usdz", nargs="?", default=os.environ.get("SHIELD_SCENE_USDZ"),
                        help="Path to the scene .usdz (default: $SHIELD_SCENE_USDZ)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sample every Nth logged ego pose (default: 5)")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Allowed negative clearance in metres before failing. The disc "
                             "cover is conservative, so the logged drive can legitimately "
                             "graze it in tight traffic; raise this only with a reason.")
    args = parser.parse_args()

    if not args.usdz:
        parser.error("no scene given and SHIELD_SCENE_USDZ is unset")

    # Imported here so --help works without AlpaSim installed.
    from alpasim_utils.artifact import Artifact
    from kitti_nav.vehicle import can_stop_safely, clearance, VehicleState
    from shield_in_alpasim.obstacles import field_from_traffic_objects
    from shield_in_alpasim.scene import ego_config_from_rig

    print(f"Scene: {args.usdz}")
    artifact = Artifact(source=args.usdz)
    rig, traffic = artifact.rig, artifact.traffic_objects

    # --- what the scene contains ---------------------------------------------------
    classes = collections.Counter(getattr(o, "label_class", "?") for o in traffic.values())
    n_static = sum(1 for o in traffic.values() if o.is_static)
    print(f"  actors:  {len(traffic)} ({n_static} static) {dict(classes)}")

    cameras = [c.logical_name for c in rig.camera_ids]
    print(f"  cameras: {cameras}")
    print("           ^ these are the names for `inference.use_cameras` and "
          "`runtime...cameras[].logical_id`")

    cfg = ego_config_from_rig(rig.vehicle_config)
    print(f"  ego:     {cfg.length:.3f} x {cfg.width:.3f} m, "
          f"rear overhang {cfg.rear_overhang:.2f} m"
          f"{'  (scene default — no rig config)' if rig.vehicle_config is None else ''}")

    traj = rig.trajectory
    timestamps = np.asarray(traj.timestamps_us)[:: args.stride]
    positions = np.asarray(traj.positions)[:: args.stride]
    yaws = np.asarray(traj.yaws)[:: args.stride]
    if len(timestamps) < 2:
        print("  FAIL: logged ego trajectory is too short to replay", file=sys.stderr)
        return 2
    print(f"  drive:   {len(timestamps)} sampled poses over "
          f"{(timestamps[-1] - timestamps[0]) / 1e6:.1f} s")

    # Speed by finite difference; the shield needs it to judge stopping distance.
    steps = np.diff(positions[:, :2], axis=0)
    dt = np.diff(timestamps) / 1e6
    speeds = np.concatenate([[0.0], np.linalg.norm(steps, axis=1) / np.maximum(dt, 1e-6)])

    # --- replay the logged drive ---------------------------------------------------
    clearances, unsafe = [], 0
    for t, pos, yaw, speed in zip(timestamps, positions, yaws, speeds):
        field = field_from_traffic_objects(traffic, pos[:2], float(yaw), int(t))
        # The ego sits at the rig origin of its own frame, so the state is always zeroed.
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=float(speed), steer=0.0)
        clearances.append(clearance(state, field, cfg))
        if not can_stop_safely(state, field, cfg):
            unsafe += 1

    clearances = np.array(clearances)
    finite = clearances[np.isfinite(clearances)]

    print(f"\n  speed:      {speeds.mean():.1f} m/s mean, {speeds.max():.1f} m/s max")
    if len(finite) == 0:
        print("  clearance:  no actor ever came within range")
    else:
        print(f"  clearance:  min {finite.min():.2f} m, "
              f"5th pct {np.percentile(finite, 5):.2f} m, median {np.median(finite):.2f} m "
              f"({len(finite)}/{len(clearances)} steps had an actor in range)")
    print(f"  shield:     would refuse to hold speed at {unsafe}/{len(clearances)} steps "
          f"({100 * unsafe / len(clearances):.0f}%)")
    print("              ^ not a bug: the logged driver brakes and turns, while this "
          "check asks whether coasting straight stays certifiable.")

    # --- the oracle ----------------------------------------------------------------
    worst = float(finite.min()) if len(finite) else float("inf")
    if worst < -abs(args.tolerance):
        print(f"\nFAIL: the logged human drive collides ({worst:.2f} m) against its own "
              "logged actors.\nThe geometry is wrong — suspect the frame transform, the "
              "quaternion convention,\nor the ego dimensions. Do not start a rendered run "
              "until this is clean.", file=sys.stderr)
        return 1

    print(f"\nPASS: the logged drive stays clear (worst {worst:.2f} m). "
          "Frames, cover, sampling and ego footprint agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
