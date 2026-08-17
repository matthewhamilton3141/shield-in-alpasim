#!/usr/bin/env python3
"""Train and checkpoint the Tier 2 policies for AlpaSim eval.

Produces the two checkpoints the photoreal eval needs: a **crutch** (shield-trained, no penalty)
and a **teacher** (intervention penalty 0.6, the frontier sweet spot). Saved via
`rl_policy.save_policy` so `ShieldedDriver` can load them with `$SHIELD_RL_CKPT`. CPU-only.

Run:  python3 scripts/rl_export.py                 # both arms, results/checkpoints/*.pt
      python3 scripts/rl_export.py --steps 400000 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_KN = _ROOT.parent / "kitti-nav" / "src"
if _KN.is_dir():
    sys.path.insert(0, str(_KN))

from rl_scaled import SCALED_ENV, train_seed  # noqa: E402
from shield_in_alpasim.rl_policy import save_policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--penalty", type=float, default=0.6, help="teacher intervention penalty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(_ROOT / "results" / "checkpoints"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    arms = {
        "crutch": SCALED_ENV,
        "teacher": dataclasses.replace(SCALED_ENV, intervention_penalty=args.penalty),
    }
    for name, cfg in arms.items():
        print(f"training {name} ({args.steps} steps, penalty={cfg.intervention_penalty})...")
        hist, ac = train_seed(True, args.seed, args.steps, 4096, 0.02, 3e-4, cfg=cfg)
        path = outdir / f"{name}.pt"
        save_policy(ac, str(path), cfg, meta={
            "arm": name, "steps": args.steps, "seed": args.seed,
            "intervention_penalty": cfg.intervention_penalty,
            "final_return": hist[-1]["mean_return"], "train_collisions": hist[-1]["cum_collisions"],
        })
        print(f"  saved {path}  (return {hist[-1]['mean_return']:.1f}, "
              f"train collisions {hist[-1]['cum_collisions']})")
    print("\nEval in AlpaSim:  SHIELD_RL_CKPT=<path> [SHIELD_FILTER=0] driver=shielded ... "
          "(see docs/TIER2_EVAL_KICKOFF.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
