#!/usr/bin/env python3
"""Render the shield's-eye view: perceived (camera) vs true (ground-truth) obstacles, per cycle.

Reads the per-cycle `cyc_*.npz` dumps written by the driver when `$SHIELD_DEBUG_DIR` is set
(`_dump_debug`), and animates a bird's-eye view in the ego rig frame — ego at the origin looking
up (+x forward), true obstacles in green, the camera's *perceived* obstacles in red. That side by
side is the point of the learned-perception arm: where red misses green, the shield is blind;
where red invents obstacles green doesn't have, it brakes at ghosts.

    python3 scripts/make_bev_video.py <dump_dir> [out.gif]

Off-meter: pure matplotlib + Pillow (an animated GIF, so no ffmpeg needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _draw(ax, circles, colour, label):
    from matplotlib.patches import Circle

    for i, (x, y, r) in enumerate(np.asarray(circles, float).reshape(-1, 3)):
        # Rig frame is x-forward, y-left; plot as (left, forward) = (-y, x) so forward is up
        # and left is left, i.e. a natural top-down view.
        ax.add_patch(Circle((-y, x), r, color=colour, alpha=0.35,
                            label=label if i == 0 else None))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    dump_dir = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else dump_dir / "bev.gif"
    files = sorted(dump_dir.glob("cyc_*.npz"))
    if not files:
        print(f"no cyc_*.npz in {dump_dir}")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    R = 45.0  # metres shown each side
    frames = []
    for f in files:
        d = np.load(f)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        for ring in (10, 20, 30, 40):
            ax.add_patch(plt.Circle((0, 0), ring, fill=False, color="0.85", lw=0.8))
        _draw(ax, d["gt"], "green", "true (GT)")
        _draw(ax, d["camera"], "red", "camera")
        ax.plot(0, 0, "k^", ms=10)  # ego, pointing up (+forward)
        ax.set_xlim(-R, R); ax.set_ylim(-5, 2 * R - 5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"ego {float(d['speed']):.1f} m/s   green=true  red=camera", fontsize=9)
        if len(d["gt"]) or len(d["camera"]):
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))
        plt.close(fig)

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print(f"wrote {out}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
