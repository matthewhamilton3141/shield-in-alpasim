#!/usr/bin/env python3
"""Diagnose a shielded rollout from AlpaSim's per-step `metrics.parquet`.

Purpose: answer whether the controller *executed* the trajectory the shielded driver emitted.
The instrumented driver log (`shield cycle ...`) shows what we *proposed* and what the shield
did; this shows what the vehicle actually *achieved* per step — speed, position, and (if the
eval recorded them) the commanded acceleration/steer. If we emit an accelerating plan (verified
kinematically off-box) but the achieved speed stays ~0, the gap is downstream of the driver.

Run on the box, where pandas/pyarrow are in the venv:
    uv run python ~/shield-in-alpasim/scripts/analyze_rollout_parquet.py \
        ~/alpasim/out_vdiag/rollouts/*/metrics.parquet

Robust to unknown schemas: it prints the columns first, then a compact early-time trace of
whatever speed/accel/position/collision columns it can find (matched by fuzzy name).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find(cols: list[str], *needles: str) -> list[str]:
    """Columns whose lowercased name contains any needle, in original order."""
    lc = {c: c.lower() for c in cols}
    return [c for c in cols if any(n in lc[c] for n in needles)]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"not found: {path}")
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("pandas not available; run under `uv run python` on the box.")
        return 1

    df = pd.read_parquet(path)
    print(f"# {path}")
    print(f"rows={len(df)}  cols={len(df.columns)}")
    print("\n== columns ==")
    for c in df.columns:
        print(f"  {c:40s} {str(df[c].dtype):10s} e.g. {df[c].iloc[0] if len(df) else '—'!r:.60}")

    # Fuzzy-pick the columns that answer "did it move?" — names vary across AlpaSim versions,
    # so cast a wide net and let the printed schema above disambiguate if a guess is wrong.
    time_cols = _find(df.columns, "timestamp", "time_us", "_us")
    speed_cols = _find(df.columns, "speed", "velocity", "vel_", "_vel")
    accel_cols = _find(df.columns, "accel", "acceleration")
    steer_cols = _find(df.columns, "steer", "steering", "yaw_rate", "curvature")
    pos_cols = _find(df.columns, "pos", "_x", "_y", "translation", "location")
    flag_cols = _find(df.columns, "collision", "intervention", "offroad", "clearance")

    print("\n== guessed roles ==")
    for name, cols in [("time", time_cols), ("speed", speed_cols), ("accel", accel_cols),
                       ("steer", steer_cols), ("position", pos_cols), ("flags", flag_cols)]:
        print(f"  {name:9s}: {cols}")

    show = (time_cols[:1] + speed_cols[:2] + accel_cols[:1] + steer_cols[:1]
            + pos_cols[:2] + flag_cols[:3])
    show = [c for i, c in enumerate(show) if c not in show[:i]]  # dedupe, keep order
    if show:
        print("\n== per-step trace (first 40 rows of the picked columns) ==")
        with pd.option_context("display.max_rows", 60, "display.width", 200):
            print(df[show].head(40).to_string())

    # The headline question in one number: did achieved speed ever exceed a crawl?
    if speed_cols:
        s = df[speed_cols[0]]
        print(f"\n== '{speed_cols[0]}' summary ==  max={s.max():.2f}  mean={s.mean():.2f}  "
              f"final={s.iloc[-1]:.2f}  (m/s if that's the unit)")
        print("If max stays near a crawl while the driver log proposed acceleration, the "
              "controller did not execute our plan — the bug is downstream of the driver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
