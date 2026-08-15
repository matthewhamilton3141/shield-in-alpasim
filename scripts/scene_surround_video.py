#!/usr/bin/env python3
"""Real-scene surround video: the front + rear camera views beside the shield's-eye BEV.

Consumes a `$SHIELD_DEBUG_DIR` dump that has BOTH the per-cycle BEV npz (`cyc_XXXX.npz`) and the
per-cycle camera JPEGs (`cyc_XXXX_<camera_id>.jpg`, written when `$SHIELD_DEBUG_CAMERAS` is set).
Lays out, per cycle, the forward camera + the two rear cameras (the real rendered scene) next to the
360° radar of true (green) vs perceived (red) obstacles, and encodes an MP4.

    python3 scripts/scene_surround_video.py <dump_dir> [out.mp4] [--title ...] [--subtitle ...]

Off-meter: matplotlib + ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

RIG_FOV = [
    ("front", 0.0, 60.0), ("cross_left", 67.0, 60.0), ("cross_right", -67.0, 60.0),
    ("rear_left", 153.0, 35.0), ("rear_right", -151.0, 35.0),
]
BG = "#0a0e14"; GRID = "#1b2430"; FOVC = "#22d3ee"
GT_C = "#34d399"; CAM_C = "#fb5b5b"; FG = "#e5e7eb"; ACCENT = "#22d3ee"

FRONT = "camera_front_wide_120fov"
REAR_L = "camera_rear_left_70fov"
REAR_R = "camera_rear_right_70fov"


def _bev(ax, z):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Wedge

    ax.set_facecolor(BG)
    XLIM, YB, YT = 20.0, -15.0, 20.0
    ax.set_xlim(-XLIM, XLIM); ax.set_ylim(YB, YT); ax.set_aspect("equal"); ax.axis("off")
    for rng in (5, 10, 15, 20):
        ax.add_artist(plt.Circle((0, 0), rng, fill=False, ec=GRID, lw=0.8, zorder=1))
    for _n, b, hf in RIG_FOV:
        pc = 90.0 + b
        ax.add_patch(Wedge((0, 0), XLIM * 1.6, pc - hf, pc + hf, facecolor=FOVC, alpha=0.05,
                           ec="none", zorder=0))
    cam = np.asarray(z["camera"], float).reshape(-1, 3)
    gt = np.asarray(z["gt"], float).reshape(-1, 3)
    if len(cam):
        ax.scatter(-cam[:, 1], cam[:, 0], s=12, c=CAM_C, alpha=0.30, edgecolors="none", zorder=3)
    if len(gt):
        ax.scatter(-gt[:, 1], gt[:, 0], s=60, c=GT_C, alpha=0.65, edgecolors="none", zorder=4)
    ax.add_patch(FancyBboxPatch((-0.9, -1.1), 1.8, 3.2, boxstyle="round,pad=0.02,rounding_size=0.4",
                                fc="#334155", ec=FG, lw=1.2, zorder=5))
    ax.plot([0, 0], [0, 1.6], color=ACCENT, lw=1.4, zorder=6)
    ax.set_title("shield's-eye · 360° radar", color=ACCENT, fontsize=10, pad=6)


def _cam(ax, path, label):
    ax.set_facecolor("#05070a"); ax.axis("off")
    if path.exists():
        import matplotlib.image as mpimg
        ax.imshow(mpimg.imread(str(path)))
    else:
        ax.text(0.5, 0.5, "(no frame)", color="#64748b", ha="center", va="center", fontsize=9)
    ax.text(0.02, 0.94, label, color=FG, fontsize=10, transform=ax.transAxes, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec="none", alpha=0.7))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--title", default="Hard safety shield · surround camera perception")
    ap.add_argument("--subtitle", default="real rendered scene (front + rear cameras) → the shield's 360° obstacle field")
    ap.add_argument("--caption", default="")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dump = Path(args.dump_dir)
    npzs = sorted(dump.glob("cyc_[0-9]*.npz"))
    if not npzs:
        raise SystemExit(f"no cyc_*.npz in {dump}")
    out = Path(args.out) if args.out else dump / "scene_surround.mp4"

    tmp = Path(tempfile.mkdtemp())
    for i, np_path in enumerate(npzs):
        z = np.load(np_path)
        fig = plt.figure(figsize=(16, 9), dpi=110)
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.6], height_ratios=[1, 1],
                              left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.06, hspace=0.10)
        stem = np_path.stem  # cyc_XXXX
        _cam(fig.add_subplot(gs[0, 0:2]), dump / f"{stem}_{FRONT}.jpg", "front camera")
        _cam(fig.add_subplot(gs[1, 0]), dump / f"{stem}_{REAR_L}.jpg", "rear-left camera")
        _cam(fig.add_subplot(gs[1, 1]), dump / f"{stem}_{REAR_R}.jpg", "rear-right camera")
        _bev(fig.add_subplot(gs[:, 2]), z)

        fig.text(0.5, 0.955, args.title, color=FG, fontsize=17, ha="center", weight="bold")
        fig.text(0.5, 0.915, args.subtitle, color=ACCENT, fontsize=10.5, ha="center")
        spd = float(z["speed"]) if "speed" in z else float("nan")
        ncam = len(np.asarray(z["camera"], float).reshape(-1, 3))
        fig.text(0.30, 0.015, f"cycle {i:02d}/{len(npzs) - 1}    ego {spd:4.1f} m/s    "
                 f"perceived {ncam} discs", color=FG, fontsize=10, ha="center", family="monospace")
        if args.caption:
            fig.text(0.78, 0.015, args.caption, color="#9fb3c8", fontsize=9, ha="center", style="italic")
        fig.savefig(tmp / f"f{i:04d}.png", facecolor=BG)
        plt.close(fig)

    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {out}  ({len(npzs)} frames)")


if __name__ == "__main__":
    main()
