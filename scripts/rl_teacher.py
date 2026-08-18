#!/usr/bin/env python3
"""Tier 2 — turning the crutch into a teacher.

`rl_transfer.py` showed shielding is a *crutch*: the shield-trained policy explores safely and
learns fast, but leans on the veto — remove the shield at deployment and it collides ~0.94. The
cause is that the policy pays nothing for proposing unsafe actions; the shield silently fixes them.

The fix: **penalise shield interventions during training** (`EnvConfig.intervention_penalty`). The
policy still explores fully safely (the shield still vetoes — 0 training crashes), but now it is
rewarded for proposing actions the shield need not override, so it internalises safe behaviour that
survives the shield's removal. This runs three arms — unshielded / shielded / shielded+penalty
("teacher") — and evaluates each under shield-on and shield-off deployment, to see if the penalty
buys a policy that is safe *without* the shield while keeping the safe-exploration guarantee.

CPU-only. Writes results/rl_teacher.{csv,png}.

Run:  python3 scripts/rl_teacher.py                         # 5 seeds, penalty 0.15
      python3 scripts/rl_teacher.py --penalty 0.3 --seeds 3 --steps 200000
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

# (arm label, shield-during-training, intervention penalty) — penalty filled from CLI for teacher.
ARMS = ["unshielded", "shielded", "teacher"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", type=float, default=0.15, help="intervention penalty (teacher arm)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=350_000)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    teacher_cfg = dataclasses.replace(SCALED_ENV, intervention_penalty=args.penalty)
    arm_spec = {
        "unshielded": (False, SCALED_ENV),
        "shielded": (True, SCALED_ENV),
        "teacher": (True, teacher_cfg),
    }
    print(f"Tier 2 teacher run: {args.seeds} seeds × {args.steps} steps, "
          f"teacher intervention_penalty={args.penalty}")

    rows = []
    cells = {(arm, te): [] for arm in ARMS for te in (True, False)}
    train_cols = {arm: [] for arm in ARMS}
    for arm in ARMS:
        shield, cfg = arm_spec[arm]
        for sd in range(args.seeds):
            hist, ac = train_seed(shield, sd, args.steps, args.rollout,
                                  args.entropy, args.lr, cfg=cfg)
            train_cols[arm].append(hist[-1]["cum_collisions"])
            for te in (True, False):
                ev = transfer_eval(ac, te, args.eval_n, sd)
                cells[(arm, te)].append(ev)
                rows.append({"arm": arm, "test_shield": te, "seed": sd,
                             "train_collisions": hist[-1]["cum_collisions"], **ev})
            print(f"  [{arm:>10}] seed {sd}: train_col {hist[-1]['cum_collisions']:4d} | "
                  f"test-ON coll {cells[(arm, True)][-1]['collision_rate']:.2f} "
                  f"ret {cells[(arm, True)][-1]['return']:.1f} | "
                  f"test-OFF coll {cells[(arm, False)][-1]['collision_rate']:.2f} "
                  f"ret {cells[(arm, False)][-1]['return']:.1f}")

    with open(outdir / "rl_teacher.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {outdir / 'rl_teacher.csv'}")
    _plot(cells, train_cols, outdir / "rl_teacher.png", args.penalty)
    _report(cells, train_cols, args.penalty)
    return 0


def _agg(evals, key):
    a = np.array([e[key] for e in evals], float)
    return a.mean(), a.std()


def _plot(cells, train_cols, path, penalty):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(ARMS)); w = 0.36
    for ax, key, title, ylab in (
        (ax1, "collision_rate", "Deployment safety", "eval collision rate"),
        (ax2, "return", "Deployment performance", "eval return"),
    ):
        for j, (te, lab, col) in enumerate(((True, "test shield ON", "#1a7f5a"),
                                            (False, "test shield OFF", "#b23a48"))):
            means = [_agg(cells[(a, te)], key)[0] for a in ARMS]
            stds = [_agg(cells[(a, te)], key)[1] for a in ARMS]
            ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=4, color=col, label=lab, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([a + (f"\n(pen {penalty:g})" if a == "teacher" else "") for a in ARMS])
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.suptitle("Tier 2 — intervention penalty turns the crutch into a teacher", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _report(cells, train_cols, penalty):
    print("\n" + "=" * 70)
    print(f"THREE-ARM DEPLOYMENT (5-seed mean ± std) — teacher penalty={penalty}")
    for arm in ARMS:
        tc = int(np.mean(train_cols[arm]))
        for te in (True, False):
            c_m, c_s = _agg(cells[(arm, te)], "collision_rate")
            r_m, r_s = _agg(cells[(arm, te)], "return")
            print(f"  {arm:>10} | test {'ON ' if te else 'OFF'}: coll {c_m:.2f}±{c_s:.2f}  "
                  f"return {r_m:.1f}±{r_s:.1f}   (train collisions ~{tc})")
    off = {a: _agg(cells[(a, False)], "collision_rate")[0] for a in ARMS}
    safe_train = int(np.mean(train_cols["teacher"])) == 0
    print("\n  CRUX — collision with shield REMOVED at deployment:")
    print(f"    unshielded {off['unshielded']:.2f} | shielded(crutch) {off['shielded']:.2f} | "
          f"teacher {off['teacher']:.2f}")
    fixed = off["teacher"] < 0.5 * off["shielded"]
    print(f"    teacher trained crash-free: {safe_train} | crutch fixed (< half): {fixed}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
