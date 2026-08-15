#!/usr/bin/env python3
"""A presentation-quality bird's-eye "shield's-eye view" video from the per-cycle BEV dumps.

Same data as make_bev_video.py (the `cyc_*.npz` written when `$SHIELD_DEBUG_DIR` is set), but
styled for a portfolio/demo: dark theme, the ego, range rings, the surround cameras' FOV sectors
(to show the 360deg coverage), true obstacles in green and camera-perceived obstacles in red, with a
per-frame HUD. Renders MP4 (ffmpeg) or GIF.

    python3 scripts/nice_bev_video.py <dump_dir> [out.mp4] [--title "..."] [--subtitle "..."]

Off-meter: matplotlib only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# The real surround rig (from the 02eadd92 calibration): optical-axis bearing (deg, rig frame,
# atan2(y,x)) and half-FOV. Front + 2 cross (120deg) + 2 rear (70deg) tile the full circle.
RIG_FOV = [
    ("front", 0.0, 60.0), ("cross_left", 67.0, 60.0), ("cross_right", -67.0, 60.0),
    ("rear_left", 153.0, 35.0), ("rear_right", -151.0, 35.0),
]

BG = "#0a0e14"
GRID = "#1b2430"
FOVC = "#22d3ee"
GT_C = "#34d399"
CAM_C = "#fb5b5b"
FG = "#e5e7eb"
ACCENT = "#22d3ee"


def _discs(z, key):
    return np.asarray(z[key], float).reshape(-1, 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--title", default="Hard safety shield · surround camera perception")
    ap.add_argument("--subtitle", default="5-camera ftheta 360° rig — true (green) vs perceived (red)")
    ap.add_argument("--caption", default="")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.patches import FancyBboxPatch, Wedge

    files = sorted(Path(args.dump_dir).glob("cyc_*.npz"))
    if not files:
        raise SystemExit(f"no cyc_*.npz in {args.dump_dir}")
    frames = [np.load(f) for f in files]
    out = Path(args.out) if args.out else Path(args.dump_dir) / "nice_bev.mp4"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    XLIM, YB, YT = 20.0, -15.0, 20.0
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(YB, YT)
    ax.set_aspect("equal")
    ax.axis("off")

    # Static backdrop: range rings + camera FOV sectors (plot angle = 90 + rig bearing).
    for rng in (5, 10, 15, 20):
        ax.add_artist(plt.Circle((0, 0), rng, fill=False, ec=GRID, lw=0.8, zorder=1))
        ax.text(0.4, rng - 0.2, f"{rng} m", color=GRID, fontsize=7, zorder=1)
    for _name, bearing, hf in RIG_FOV:
        pc = 90.0 + bearing
        ax.add_patch(Wedge((0, 0), XLIM * 1.6, pc - hf, pc + hf, facecolor=FOVC,
                           alpha=0.05, ec="none", zorder=0))
    # Ego (a rounded body pointing up / +forward).
    ax.add_patch(FancyBboxPatch((-0.9, -1.1), 1.8, 3.2, boxstyle="round,pad=0.02,rounding_size=0.4",
                                fc="#334155", ec=FG, lw=1.2, zorder=5))
    ax.plot([0, 0], [0, 1.6], color=ACCENT, lw=1.4, zorder=6)  # heading

    fig.text(0.5, 0.955, args.title, color=FG, fontsize=14, ha="center", weight="bold")
    fig.text(0.5, 0.925, args.subtitle, color=ACCENT, fontsize=9.5, ha="center")
    if args.caption:
        fig.text(0.5, 0.035, args.caption, color="#9fb3c8", fontsize=8.5, ha="center",
                 style="italic", wrap=True)
    hud = fig.text(0.5, 0.08, "", color=FG, fontsize=10, ha="center", family="monospace")

    # Legend chips.
    ax.scatter([], [], s=60, c=GT_C, alpha=0.9, label="true obstacle (ground truth)")
    ax.scatter([], [], s=30, c=CAM_C, alpha=0.9, label="perceived (camera depth)")
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.0, labelcolor=FG)
    for h in leg.legend_handles:
        h.set_alpha(0.9)

    cam_sc = ax.scatter([], [], s=14, c=CAM_C, alpha=0.30, edgecolors="none", zorder=3)
    gt_sc = ax.scatter([], [], s=70, c=GT_C, alpha=0.65, edgecolors="none", zorder=4)

    def update(i):
        z = frames[i]
        cam, gt = _discs(z, "camera"), _discs(z, "gt")
        cam_sc.set_offsets(np.column_stack([-cam[:, 1], cam[:, 0]]) if len(cam) else np.empty((0, 2)))
        gt_sc.set_offsets(np.column_stack([-gt[:, 1], gt[:, 0]]) if len(gt) else np.empty((0, 2)))
        spd = float(z["speed"]) if "speed" in z else float("nan")
        hud.set_text(f"cycle {i:02d}/{len(frames) - 1}    ego {spd:4.1f} m/s    "
                     f"perceived {len(cam):4d}   ·   true {len(gt):3d}")
        return cam_sc, gt_sc, hud

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    try:
        anim.save(str(out), writer=FFMpegWriter(fps=args.fps, bitrate=3000),
                  savefig_kwargs={"facecolor": BG})
    except Exception as exc:  # noqa: BLE001 — fall back to GIF if ffmpeg is unavailable
        out = out.with_suffix(".gif")
        print(f"ffmpeg failed ({exc}); writing GIF instead")
        anim.save(str(out), writer=PillowWriter(fps=args.fps), savefig_kwargs={"facecolor": BG})
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
