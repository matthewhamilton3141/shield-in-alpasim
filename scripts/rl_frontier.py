#!/usr/bin/env python3
"""Tier 2 — the safety–performance frontier of the intervention penalty.

`rl_teacher.py` showed a single penalty (0.4) *partially* fixes the crutch (off-shield collision
0.94 → 0.49). This sweeps the penalty to map the whole trade-off: as the cost of leaning on the
shield rises, the deployed-without-shield collision rate falls toward the unshielded baseline —
but deployment return falls too (the policy buys safety with caution). One knob, a full frontier,
instead of one noisy point.

Every arm still trains **100% crash-free** (the shield vetoes during exploration regardless of the
penalty); the penalty only shapes what the policy learns to *propose*. CPU-only. Writes
results/rl_frontier.{csv,png}.

Run:  python3 scripts/rl_frontier.py
      python3 scripts/rl_frontier.py --penalties 0,0.4,0.8,1.2 --seeds 5 --steps 250000
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_KN = _ROOT.parent / "kitti-nav" / "src"
if _KN.is_dir():
    sys.path.insert(0, str(_KN))

from rl_scaled import SCALED_ENV, train_seed  # noqa: E402
from rl_transfer import transfer_eval  # noqa: E402

# Baseline (from rl_transfer, same task/budget family): a policy trained WITHOUT the shield learns
# caution the hard way and collides ~0.09 deployed without a shield — the frontier's target floor.
UNSHIELDED_OFF_COLL = 0.09


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="0,0.4,0.8,1.2,1.8")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    penalties = [float(p) for p in args.penalties.split(",")]
    print(f"Tier 2 frontier: penalties {penalties} × {args.seeds} seeds × {args.steps} steps")

    rows, agg = [], []
    for pen in penalties:
        cfg = dataclasses.replace(SCALED_ENV, intervention_penalty=pen)
        off_coll, off_ret, on_ret, tcols = [], [], [], []
        for sd in range(args.seeds):
            hist, ac = train_seed(True, sd, args.steps, args.rollout, args.entropy, args.lr, cfg=cfg)
            off = transfer_eval(ac, False, args.eval_n, sd)   # deployed WITHOUT the shield
            on = transfer_eval(ac, True, args.eval_n, sd)      # deployed WITH the shield
            off_coll.append(off["collision_rate"]); off_ret.append(off["return"])
            on_ret.append(on["return"]); tcols.append(hist[-1]["cum_collisions"])
            rows.append({"penalty": pen, "seed": sd, "train_collisions": hist[-1]["cum_collisions"],
                         "off_collision": off["collision_rate"], "off_return": off["return"],
                         "on_return": on["return"]})
        row = {
            "penalty": pen,
            "off_coll_mean": float(np.mean(off_coll)), "off_coll_std": float(np.std(off_coll)),
            "off_ret_mean": float(np.mean(off_ret)), "off_ret_std": float(np.std(off_ret)),
            "on_ret_mean": float(np.mean(on_ret)),
            "train_coll_mean": float(np.mean(tcols)),
        }
        agg.append(row)
        print(f"  penalty {pen:>4}: off-shield coll {row['off_coll_mean']:.2f}±{row['off_coll_std']:.2f}"
              f"  off-shield ret {row['off_ret_mean']:.1f}  on-shield ret {row['on_ret_mean']:.1f}"
              f"  (train crashes ~{row['train_coll_mean']:.0f})")

    with open(outdir / "rl_frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {outdir / 'rl_frontier.csv'}")
    _plot(agg, outdir / "rl_frontier.png")
    _report(agg)
    return 0


def _plot(agg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = [a["penalty"] for a in agg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    coll = np.array([a["off_coll_mean"] for a in agg])
    cstd = np.array([a["off_coll_std"] for a in agg])
    ax1.plot(p, coll, "-o", color="#b23a48", lw=2)
    ax1.fill_between(p, coll - cstd, coll + cstd, color="#b23a48", alpha=0.18)
    ax1.axhline(UNSHIELDED_OFF_COLL, color="#2b6cb0", ls="--", lw=1.5,
                label=f"unshielded-trained ({UNSHIELDED_OFF_COLL:.2f})")
    ax1.set_xlabel("intervention penalty"); ax1.set_ylabel("collision rate, shield OFF at deploy")
    ax1.set_title("Safety vs penalty (crutch → teacher)"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(p, [a["on_ret_mean"] for a in agg], "-o", color="#1a7f5a", lw=2, label="deploy shield ON")
    oret = np.array([a["off_ret_mean"] for a in agg])
    ax2.plot(p, oret, "-o", color="#b23a48", lw=2, label="deploy shield OFF")
    ax2.set_xlabel("intervention penalty"); ax2.set_ylabel("eval return")
    ax2.set_title("Performance cost of the penalty"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Tier 2 — intervention-penalty safety/performance frontier (5 seeds)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _report(agg):
    print("\n" + "=" * 66)
    print("FRONTIER (5-seed mean) — off-shield collision as the penalty rises")
    for a in agg:
        print(f"  penalty {a['penalty']:>4}: off-coll {a['off_coll_mean']:.2f}  "
              f"off-ret {a['off_ret_mean']:5.1f}  on-ret {a['on_ret_mean']:5.1f}")
    best = min(agg, key=lambda a: a["off_coll_mean"])
    print(f"\n  lowest off-shield collision: {best['off_coll_mean']:.2f} at penalty "
          f"{best['penalty']} (unshielded floor {UNSHIELDED_OFF_COLL:.2f}); "
          f"crutch (penalty 0) was {agg[0]['off_coll_mean']:.2f}")
    print("=" * 66)


if __name__ == "__main__":
    raise SystemExit(main())
