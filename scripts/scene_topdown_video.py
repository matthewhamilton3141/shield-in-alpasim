#!/usr/bin/env python3
"""Clean world-frame top-down video from a `dump_scene_topdown.py` npz — no NuRec, fully vector.

The map (road edges + lane centrelines), every actor as an oriented box, and the ego driving through
with a trail — a nuScenes-style schematic. Light-themed, camera follows the ego. Encodes MP4.

    python3 scripts/scene_topdown_video.py <scene_topdown.npz> [out.mp4] [--title ...]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BG = "#ffffff"; EDGE = "#8b97a7"; LANE = "#d7dee7"
ACTOR = "#2563eb"; ACTOR_E = "#1e3a8a"; STATIC = "#9aa6b2"
EGO = "#dc2626"; EGO_E = "#7f1d1d"; FG = "#111827"; MUTED = "#6b7280"

EGO_L, EGO_W = 5.39, 2.11  # AlpaSim S223 ego footprint (scene.py)


def _split(pts, lens):
    out, i = [], 0
    for n in lens:
        out.append(pts[i:i + n]); i += n
    return out


def _box_corners(cx, cy, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = length / 2.0, width / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cy])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--title", default="Autonomous driving scene · top-down")
    ap.add_argument("--subtitle", default="ego (red) · traffic (blue) · road + lanes · world frame")
    ap.add_argument("--half-window", type=float, default=45.0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    z = np.load(args.npz)
    ego_xy, ego_yaw, boxes = z["ego_xy"], z["ego_yaw"], z["boxes"]
    edges = _split(z["road_edge_pts"], z["road_edge_lens"])
    lanes = _split(z["lane_pts"], z["lane_lens"])
    F = len(ego_xy)
    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".mp4")

    tmp = Path(tempfile.mkdtemp())
    for i in range(F):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
        cx, cy = ego_xy[i]
        hw = args.half_window
        ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)

        ax.add_collection(LineCollection(lanes, colors=LANE, linewidths=1.0, zorder=1))
        ax.add_collection(LineCollection(edges, colors=EDGE, linewidths=1.8, zorder=2))

        fr = boxes[boxes[:, 0] == i]
        polys = [_box_corners(*b[1:6]) for b in fr]
        cols = [STATIC if b[6] else ACTOR for b in fr]
        ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=ACTOR_E,
                                         linewidths=0.8, alpha=0.9, zorder=4))

        ax.plot(ego_xy[:i + 1, 0], ego_xy[:i + 1, 1], color=EGO, lw=2.0, alpha=0.5, zorder=5)
        ego_poly = _box_corners(cx, cy, ego_yaw[i], EGO_L, EGO_W)
        ax.add_collection(PolyCollection([ego_poly], facecolors=EGO, edgecolors=EGO_E,
                                         linewidths=1.4, zorder=6))
        # heading tick
        hx = cx + np.cos(ego_yaw[i]) * EGO_L * 0.8
        hy = cy + np.sin(ego_yaw[i]) * EGO_L * 0.8
        ax.plot([cx, hx], [cy, hy], color=EGO_E, lw=1.6, zorder=7)

        fig.text(0.5, 0.95, args.title, color=FG, fontsize=15, ha="center", weight="bold")
        fig.text(0.5, 0.915, args.subtitle, color="#2563eb", fontsize=10, ha="center")
        fig.text(0.5, 0.06, f"frame {i:02d}/{F - 1}  ·  {len(fr)} actors in view",
                 color=MUTED, fontsize=9, ha="center", family="monospace")
        fig.savefig(tmp / f"f{i:04d}.png", facecolor=BG); plt.close(fig)

    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {out}  ({F} frames)")


if __name__ == "__main__":
    main()
