#!/usr/bin/env python3
"""Tier 2 preview video — the crutch, and the fix, in one shot.

A light-themed top-down (BEV) animation of the corridor task, three synced panels on the **same
obstacle layout**, so the whole Tier 2 story reads at a glance:

  1. shield-trained policy, shield ON  -> drives clean (safe exploration paid off)
  2. same policy,           shield OFF -> crashes (the crutch: safety lived in the shield)
  3. teacher policy (intervention penalty), shield OFF -> drives clean again (the fix)

Trains the two policies (or reuses a cheap budget), searches a few layouts for one that makes the
contrast crisp, renders to results/rl_tier2_preview.mp4. CPU-only, no AlpaSim.

Run:  python3 scripts/rl_video.py                 # default budget
      python3 scripts/rl_video.py --steps 150000  # faster/cruder policies
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_KN = _ROOT.parent / "kitti-nav" / "src"
if _KN.is_dir():
    sys.path.insert(0, str(_KN))

from kitti_nav.vehicle import VehicleConfig  # noqa: E402
from shield_in_alpasim.rl_env import ShieldNavEnv  # noqa: E402
from rl_scaled import SCALED_ENV, train_seed  # noqa: E402


def rollout(ac, shield: bool, layout_seed: int):
    """Greedy rollout on the layout fixed by `layout_seed`; record ego poses + collision frame."""
    env = ShieldNavEnv(cfg=SCALED_ENV, shield=shield, seed=layout_seed)
    obs = env.reset()
    obstacles = np.asarray(env.obstacles.circles, float).reshape(-1, 3)
    poses, collided_at, goal = [], None, False
    done = False
    while not done:
        s = env.state
        poses.append((s.x, s.y, s.yaw))
        with torch.no_grad():
            logits, _ = ac(torch.as_tensor(obs, dtype=torch.float32))
        obs, _, done, info = env.step(int(torch.argmax(logits)))
        if info["collision"]:
            collided_at = len(poses)
        goal |= info["goal"]
    poses.append((env.state.x, env.state.y, env.state.yaw))
    return {"poses": np.array(poses), "collided_at": collided_at, "goal": goal,
            "obstacles": obstacles}


def _ego_polygon(x, y, yaw, veh: VehicleConfig):
    """Four corners of the ego rectangle (rear-axle origin), rotated into world frame."""
    fo, ro, hw = veh.front_overhang, veh.rear_overhang, veh.width / 2
    corners = np.array([[-ro, -hw], [fo, -hw], [fo, hw], [-ro, hw]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + np.array([x, y])


def pick_layout(ac_shield, ac_teacher, seeds):
    """Find a layout where the crutch is crisp: shield-ON drives, shield-OFF crashes, teacher-OFF
    drives. Falls back to the first seed if none is perfectly clean."""
    for sd in seeds:
        on = rollout(ac_shield, True, sd)
        off = rollout(ac_shield, False, sd)
        tea = rollout(ac_teacher, False, sd)
        if on["collided_at"] is None and off["collided_at"] is not None \
                and tea["collided_at"] is None:
            return sd, (on, off, tea)
    sd = seeds[0]
    return sd, (rollout(ac_shield, True, sd), rollout(ac_shield, False, sd),
                rollout(ac_teacher, False, sd))


def animate(rolls, titles, path, veh: VehicleConfig, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.patches import Circle, Polygon

    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor("white")
    hw = cfg.corridor_half_width
    n_frames = max(len(r["poses"]) for r in rolls)

    ego_patches, trail_lines, badge = [], [], []
    for ax, roll, title in zip(axes, rolls, titles):
        ax.set_facecolor("#fbfbfd")
        ax.set_xlim(-2, cfg.goal_x + 3); ax.set_ylim(-hw - 1.5, hw + 1.5)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(hw, color="#c9ced6", lw=1.5); ax.axhline(-hw, color="#c9ced6", lw=1.5)
        ax.axvline(cfg.goal_x, color="#7fb069", lw=1.6, ls="--", alpha=0.8)  # goal line
        for ox, oy, orr in roll["obstacles"]:
            ax.add_patch(Circle((ox, oy), orr, color="#9aa0aa", alpha=0.85, zorder=2))
        (trail,) = ax.plot([], [], color="#2b6cb0", lw=1.6, alpha=0.7, zorder=3)
        poly = Polygon(_ego_polygon(*roll["poses"][0], veh), closed=True,
                       facecolor="#2b6cb0", edgecolor="#1a3f66", alpha=0.9, zorder=4)
        ax.add_patch(poly)
        txt = ax.text(0.5, 0.93, "", transform=ax.transAxes, ha="center", fontsize=12,
                      fontweight="bold")
        ego_patches.append(poly); trail_lines.append(trail); badge.append(txt)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Tier 2 — shield-trained policy: crutch (middle) and its fix (right)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def frame(i):
        arts = []
        for roll, poly, trail, txt in zip(rolls, ego_patches, trail_lines, badge):
            poses = roll["poses"]
            k = min(i, len(poses) - 1)
            crashed = roll["collided_at"] is not None and k >= roll["collided_at"]
            poly.set_xy(_ego_polygon(*poses[k], veh))
            poly.set_facecolor("#c0392b" if crashed else "#2b6cb0")
            poly.set_edgecolor("#7b241c" if crashed else "#1a3f66")
            trail.set_data(poses[:k + 1, 0], poses[:k + 1, 1])
            if crashed:
                txt.set_text("COLLISION"); txt.set_color("#c0392b")
            elif roll["goal"] and k >= len(poses) - 2:
                txt.set_text("GOAL"); txt.set_color("#2e7d32")
            else:
                txt.set_text("")
            arts += [poly, trail, txt]
        return arts

    fps = 12
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        ext = ".mp4"
    except Exception:
        writer = PillowWriter(fps=fps); ext = ".gif"
    out = str(Path(path).with_suffix(ext))
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, frame, frames=n_frames + 8, blit=True)
    anim.save(out, writer=writer)
    plt.close(fig)
    print(f"wrote {out}  ({n_frames} frames @ {fps}fps)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--penalty", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "results" / "rl_tier2_preview"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"training shield-trained policy ({args.steps} steps)...")
    _, ac_shield = train_seed(True, args.seed, args.steps, 4096, 0.02, 3e-4)
    print(f"training teacher policy (penalty {args.penalty})...")
    teacher_cfg = dataclasses.replace(SCALED_ENV, intervention_penalty=args.penalty)
    _, ac_teacher = train_seed(True, args.seed, args.steps, 4096, 0.02, 3e-4, cfg=teacher_cfg)

    layout_seed, rolls = pick_layout(ac_shield, ac_teacher, list(range(300_000, 300_040)))
    print(f"layout seed {layout_seed}: "
          f"ON crash={rolls[0]['collided_at']} OFF crash={rolls[1]['collided_at']} "
          f"teacher-OFF crash={rolls[2]['collided_at']}")
    titles = ["Shield-trained · shield ON\n(safe)",
              "Shield-trained · shield OFF\n(crutch — crashes)",
              f"Teacher (pen {args.penalty:g}) · shield OFF\n(fixed — safe without shield)"]
    animate(rolls, titles, args.out, VehicleConfig(), SCALED_ENV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
