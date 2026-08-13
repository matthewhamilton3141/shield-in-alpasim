#!/usr/bin/env python3
"""Draw the trajectory this driver hands AlpaSim, with and without an obstacle.

Runs on a laptop: no AlpaSim, no GPU, no scene assets. The point is to see the shield
actually doing something *through the AlpaSim-shaped output* — `_rollout` returns the
(T, 2) waypoints that `ModelPrediction.from_planar` lifts into the pose pair AlpaSim
drives. Under AlpaSim today the obstacle field is empty (README problem 1), so the real
sim video would show a car going straight and nothing else; injecting a field here shows
what the same code path looks like once problem 1 is solved.

    python3 scripts/preview_trajectory.py --out docs/preview.png

Left panel: clear road, waypoints evenly spaced at v*dt_out. Right panel: obstacle in the
lane, the shield brakes, and the waypoints bunch up and stop short of it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_kitti_nav_src = Path(__file__).resolve().parents[2] / "kitti-nav" / "src"
if _kitti_nav_src.is_dir():
    sys.path.insert(0, str(_kitti_nav_src))

from kitti_nav.vehicle import CircleField, VehicleConfig  # noqa: E402

from shield_in_alpasim.driver import ShieldedDriver  # noqa: E402

CAMERAS = ["camera_front_wide_120fov"]


def _driver(cfg: VehicleConfig, obstacles, horizon_steps: int, hz: int) -> ShieldedDriver:
    return ShieldedDriver(
        cfg=cfg,
        camera_ids=CAMERAS,
        output_frequency_hz=hz,
        horizon_steps=horizon_steps,
        obstacles=obstacles,
    )


def _panel(ax, cfg: VehicleConfig, xy: np.ndarray, obstacle, title: str) -> None:
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)

    if obstacle is not None:
        ox, oy, r = obstacle
        ax.add_patch(plt.Circle((ox, oy), r, color="#c0392b", alpha=0.35, zorder=1))
        ax.add_patch(plt.Circle((ox, oy), r + cfg.safety_margin, color="#c0392b",
                                alpha=0.12, ls="--", fill=False, lw=1.5, zorder=1))

    # Ego footprint at the origin: waypoints are ego-relative in the rig frame.
    ax.add_patch(plt.Rectangle((-cfg.rear_overhang, -cfg.width / 2), cfg.length, cfg.width,
                               color="#2c3e50", alpha=0.7, zorder=2))

    ax.plot(xy[:, 0], xy[:, 1], "-", color="#2980b9", lw=1.5, zorder=3)
    ax.scatter(xy[:, 0], xy[:, 1], s=55, color="#2980b9", zorder=4)
    # Once the shield has stopped the car the waypoints pile up on one spot; labelling
    # every one of them there is unreadable, so only label those that actually moved.
    last_labelled = -np.inf
    for i, (x, y) in enumerate(xy):
        if x - last_labelled < 1.2:
            continue
        ax.annotate(f"{i + 1}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#2980b9")
        last_labelled = x

    spacing = np.diff(xy[:, 0])
    ax.set_title(f"{title}\nreach {xy[-1, 0]:.1f} m   "
                 f"waypoint gap {spacing.min():.2f}–{spacing.max():.2f} m", fontsize=10)
    ax.set_xlabel("x forward (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speed", type=float, default=10.0, help="initial speed (m/s)")
    p.add_argument("--obstacle-x", type=float, default=22.0, help="obstacle distance (m)")
    p.add_argument("--obstacle-r", type=float, default=1.5, help="obstacle radius (m)")
    p.add_argument("--horizon-steps", type=int, default=8)
    p.add_argument("--hz", type=int, default=2, help="AlpaSim output_frequency_hz")
    p.add_argument("--out", type=Path, default=Path("docs/preview.png"))
    args = p.parse_args()

    cfg = VehicleConfig()
    obstacle = (args.obstacle_x, 0.0, args.obstacle_r)

    clear = _driver(cfg, None, args.horizon_steps, args.hz)._rollout(args.speed)
    blocked = _driver(
        cfg, CircleField(np.array([obstacle])), args.horizon_steps, args.hz
    )._rollout(args.speed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 2.9), sharey=True)
    _panel(axes[0], cfg, clear, None, "Clear road — shield never fires")
    _panel(axes[1], cfg, blocked, obstacle, "Obstacle ahead — shield brakes")
    axes[0].set_ylabel("y left (m)")
    for ax in axes:
        ax.set_xlim(-3, max(clear[-1, 0], args.obstacle_x + args.obstacle_r) + 3)
        ax.set_ylim(-3, 3)
    fig.suptitle(
        f"Trajectory handed to AlpaSim  ({args.speed:.0f} m/s, {args.hz} Hz, "
        f"{args.horizon_steps} waypoints)", fontsize=12)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    print(f"  clear:   reaches {clear[-1, 0]:.2f} m")
    print(f"  blocked: reaches {blocked[-1, 0]:.2f} m "
          f"(obstacle surface at {args.obstacle_x - args.obstacle_r:.2f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
