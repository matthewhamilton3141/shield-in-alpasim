#!/usr/bin/env python3
"""Extract a scene's world-frame geometry for a top-down video: ego + actor tracks + the map.

Pulls everything a clean schematic top-down needs straight from the scene `.usdz` (no NuRec, no
GPU, no rollout parquet): the ego's logged trajectory, every other actor's box + pose over time, and
the vector map (road edges + lane centrelines). Dumps to a portable npz that `scene_topdown_video.py`
renders on any machine.

Box-only (needs `alpasim_utils`):
    uv run --project ~/alpasim python scripts/dump_scene_topdown.py <scene.usdz> <out.npz> [n_frames]
"""

from __future__ import annotations

import sys

import numpy as np

EGO_TRACK_ID = "EGO"


def _elements_by_name(elements, name):
    """The element dict for a `MapElementType` by its `.name` (avoids importing the enum)."""
    for k, v in elements.items():
        if getattr(k, "name", str(k)) == name:
            return v
    return {}


def _polylines(elements, type_name, attr):
    """Concatenated world-XY points of every element's polyline + per-polyline lengths."""
    pts, lens = [], []
    for el in _elements_by_name(elements, type_name).values():
        poly = getattr(el, attr)
        line = np.asarray(poly.xy if hasattr(poly, "xy") else poly, float)  # Polyline -> (N, 2)
        line = line.reshape(-1, line.shape[-1])[:, :2]
        if len(line) >= 2:
            pts.append(line)
            lens.append(len(line))
    return (np.concatenate(pts) if pts else np.zeros((0, 2))), np.asarray(lens, int)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    from alpasim_utils.artifact import Artifact

    usdz, out = argv[0], argv[1]
    n_frames = int(argv[2]) if len(argv) > 2 else 100
    a = Artifact(source=usdz)
    actors = a.traffic_objects

    # The ego's logged trajectory is on the RIG (not a traffic object).
    ego_traj = a.rig.trajectory
    s, e = ego_traj.get_time_range_tuple()
    times = np.linspace(s, e - 1, n_frames).astype(np.int64)

    ego_xy = np.zeros((n_frames, 2))
    ego_yaw = np.zeros(n_frames)
    for i, t in enumerate(times):
        p = ego_traj.interpolate_pose(int(t))
        ego_xy[i] = np.asarray(p.vec3, float)[:2]
        ego_yaw[i] = float(p.yaw())

    # All non-ego actor poses across the timeline -> flat rows [frame, cx, cy, yaw, len, wid, static]
    rows = []
    for o in actors.values():
        if getattr(o, "track_id", None) == EGO_TRACK_ID:
            continue
        a0, a1 = o.trajectory.get_time_range_tuple()
        length, width = float(o.aabb.x), float(o.aabb.y)
        static = 1.0 if getattr(o, "is_static", False) else 0.0
        for i, t in enumerate(times):
            if not (a0 <= t < a1):
                continue
            p = o.trajectory.interpolate_pose(int(t))
            xy = np.asarray(p.vec3, float)[:2]
            rows.append([i, xy[0], xy[1], float(p.yaw()), length, width, static])
    boxes = np.asarray(rows, float) if rows else np.zeros((0, 7))

    re_pts, re_lens = _polylines(a.map.elements, "ROAD_EDGE", "polyline")
    lane_pts, lane_lens = _polylines(a.map.elements, "ROAD_LANE", "center")

    np.savez(out, times_us=times, ego_xy=ego_xy, ego_yaw=ego_yaw, boxes=boxes,
             road_edge_pts=re_pts, road_edge_lens=re_lens,
             lane_pts=lane_pts, lane_lens=lane_lens, scene_id=str(a.scene_id))
    print(f"wrote {out}: {n_frames} frames, {len(actors)} actors, {len(boxes)} box-poses, "
          f"{len(re_lens)} road edges, {len(lane_lens)} lanes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
