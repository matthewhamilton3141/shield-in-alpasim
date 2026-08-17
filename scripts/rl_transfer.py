#!/usr/bin/env python3
"""Tier 2, the sharper question — is the shield a *teacher* or a *crutch*?

Shielded RL's central worry (Alshiekh et al. 2018): a policy trained under a shield may just
learn to lean on it — drive recklessly and let the veto clean up — so it is unsafe the moment the
shield is removed. Or it may internalise safe behaviour that survives without the shield. This
script settles it with a 2×2: train each policy with the shield on/off, then **evaluate each under
both** shield-on and shield-off deployment.

  train \\ test        shield ON            shield OFF
  ------------------   ------------------   ------------------
  trained shielded     safe by construction  <- the crux: did it LEARN to be safe?
  trained unshielded   shield rescues it     the raw learned policy

It also reports **sample efficiency** (env steps to first reach a return threshold) — the "learns
faster?" half of the Tier 2 research question — and connects to Tier 1: deploying the shield with
degraded camera perception is a *weaker* shield, of which "shield OFF" is the limiting case.

Reuses the PPO trainer from `rl_scaled.py`. CPU-only. Writes results/rl_transfer.{csv,png}.

Run:  python3 scripts/rl_transfer.py                    # 5 seeds/arm
      python3 scripts/rl_transfer.py --seeds 3 --steps 200000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_KN = _ROOT.parent / "kitti-nav" / "src"
if _KN.is_dir():
    sys.path.insert(0, str(_KN))

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts (rl_scaled)
from shield_in_alpasim.rl_env import ShieldNavEnv  # noqa: E402
from rl_scaled import SCALED_ENV, train_seed  # noqa: E402


def transfer_eval(ac, test_shield: bool, n: int, seed: int) -> dict:
    """Greedy eval of a trained policy under a chosen *deployment* shield setting (independent of
    how it was trained). Held-out seeds, disjoint from training."""
    ret = coll = goal = 0.0
    for i in range(n):
        env = ShieldNavEnv(cfg=SCALED_ENV, shield=test_shield, seed=seed + 200_000 + i)
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = ac(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, done, info = env.step(int(torch.argmax(logits)))
            ret += r; coll += info["collision"]; goal += info["goal"]
    return {"return": ret / n, "collision_rate": coll / n, "goal_rate": goal / n}


def steps_to_threshold(hist, thresh: float, window: int = 5) -> float:
    """Env steps at which the smoothed training return first reaches `thresh` (nan if never) —
    the sample-efficiency measure for the 'learns faster' claim."""
    rets = [h["mean_return"] for h in hist]
    for i in range(len(rets)):
        sm = np.nanmean(rets[max(0, i - window + 1):i + 1])
        if sm >= thresh:
            return float(hist[i]["env_steps"])
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=350_000)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Tier 2 transfer (teacher-or-crutch): {args.seeds} seeds × {args.steps} steps/arm")

    rows = []
    cells = {(tr, te): [] for tr in ("shielded", "unshielded") for te in (True, False)}
    ttt = {"shielded": [], "unshielded": []}   # steps-to-threshold per seed
    for train_shield in (True, False):
        tr = "shielded" if train_shield else "unshielded"
        for sd in range(args.seeds):
            hist, ac = train_seed(train_shield, sd, args.steps, args.rollout,
                                  args.entropy, args.lr)
            ttt[tr].append(steps_to_threshold(hist, args.threshold))
            for test_shield in (True, False):
                ev = transfer_eval(ac, test_shield, args.eval_n, sd)
                cells[(tr, test_shield)].append(ev)
                rows.append({"train": tr, "test_shield": test_shield, "seed": sd, **ev})
            print(f"  [{tr}] seed {sd}: "
                  f"test-ON coll {cells[(tr, True)][-1]['collision_rate']:.2f} "
                  f"ret {cells[(tr, True)][-1]['return']:.1f} | "
                  f"test-OFF coll {cells[(tr, False)][-1]['collision_rate']:.2f} "
                  f"ret {cells[(tr, False)][-1]['return']:.1f} | "
                  f"steps→{args.threshold:g}: {ttt[tr][-1]:.0f}")

    with open(outdir / "rl_transfer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {outdir / 'rl_transfer.csv'}")
    _plot(cells, outdir / "rl_transfer.png")
    _report(cells, ttt, args.threshold)
    return 0


def _agg(evals, key):
    a = np.array([e[key] for e in evals], float)
    return a.mean(), a.std()


def _plot(cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = ["trained\nshielded", "trained\nunshielded"]
    x = np.arange(2)
    w = 0.36
    for ax, key, title, ylab in (
        (ax1, "collision_rate", "Deployment safety", "eval collision rate"),
        (ax2, "return", "Deployment performance", "eval return"),
    ):
        for j, (te, lab, col) in enumerate(((True, "test shield ON", "#1a7f5a"),
                                            (False, "test shield OFF", "#b23a48"))):
            means = [_agg(cells[("shielded", te)], key)[0], _agg(cells[("unshielded", te)], key)[0]]
            stds = [_agg(cells[("shielded", te)], key)[1], _agg(cells[("unshielded", te)], key)[1]]
            ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=4, color=col, label=lab, alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=0.3, axis="y")
        ax.legend()
    fig.suptitle("Tier 2 — teacher or crutch? (train × test shield, 2×2)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _report(cells, ttt, thresh):
    print("\n" + "=" * 66)
    print("2×2 DEPLOYMENT (mean ± std over seeds)")
    for tr in ("shielded", "unshielded"):
        for te in (True, False):
            c_m, c_s = _agg(cells[(tr, te)], "collision_rate")
            r_m, r_s = _agg(cells[(tr, te)], "return")
            print(f"  trained {tr:>10} | test shield {'ON ' if te else 'OFF'}: "
                  f"coll {c_m:.2f}±{c_s:.2f}  return {r_m:.1f}±{r_s:.1f}")
    off_sh = _agg(cells[("shielded", False)], "collision_rate")[0]
    off_un = _agg(cells[("unshielded", False)], "collision_rate")[0]
    print("\n  CRUX — shield removed at test (the crutch test):")
    print(f"    shield-trained policy collides {off_sh:.2f}   vs unshielded-trained {off_un:.2f}")
    verdict = ("TEACHER: shield-trained policy stays safer than unshielded even with the shield "
               "removed — safe behaviour transferred, not just leaned on."
               if off_sh < off_un - 0.02 else
               "CRUTCH: without the shield the trained policy is no safer than unshielded — it "
               "learned to rely on the veto. Safety lives in the shield, not the policy.")
    print(f"    -> {verdict}")
    def mean_ttt(v):
        finite = [x for x in v if np.isfinite(x)]
        return (np.mean(finite), len(finite), len(v)) if finite else (float("nan"), 0, len(v))

    sh_t, sh_n, sh_N = mean_ttt(ttt["shielded"])
    un_t, un_n, un_N = mean_ttt(ttt["unshielded"])
    print(f"\n  SAMPLE EFFICIENCY — steps to return≥{thresh:g}: "
          f"shielded {sh_t:.0f} ({sh_n}/{sh_N} seeds reached)  "
          f"unshielded {un_t:.0f} ({un_n}/{un_N} seeds reached)")
    print("=" * 66)


if __name__ == "__main__":
    raise SystemExit(main())
