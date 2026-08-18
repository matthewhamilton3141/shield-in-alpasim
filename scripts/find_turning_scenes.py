#!/usr/bin/env python3
"""Mine the NuRec dataset for *dynamic* drives — scenes where the ego actually turns.

The curated tier1 scenes are mostly straight suburban drives, so rendered clips are static.
The full `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` dataset has **1,607** scenes, each with a
lightweight `camera_front_wide_120fov.mp4` reference video (the raw drive) alongside the 2.4 GB
USDZ. This screens those cheap reference videos and ranks scenes by how much the drive *turns*,
so we render only the dynamic ones in AlpaSim.

**Turn score** = the peak of the running sum of per-frame horizontal camera pan (FFT phase
correlation between consecutive frames), plus a **coherence** ratio (peak / total motion). A real
turn is a large *sustained, coherent* pan in one direction; camera shake / lane-drift produces a
large total motion but low coherence, so the coherence filter rejects those false positives.

Needs: `huggingface_hub`, `numpy`, `Pillow`, and `ffmpeg`/`ffprobe` on PATH. HF_TOKEN in env
(gated NuRec read token — supply inline, never persist).

    HF_TOKEN=hf_... python3 scripts/find_turning_scenes.py --n 40 --min-quality 9.2

Prints a ranked table of full clip_ids flagged TURN; render the top ones with, e.g.,
    driver=shielded_vavam runtime.simulation_config.force_gt_duration_us=30000000
    scenes.scene_ids=[clipgt-<id>]   (see scripts/box/rl_eval.sh / the GT-path batches).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile

import numpy as np

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
REF = "sample_set/26.04_release/{cid}/camera_front_wide_120fov.mp4"


def _frame_shifts(path: str, n: int = 20) -> np.ndarray:
    """Signed per-step horizontal pan (px) between `n` evenly-spaced frames, via 1-D FFT phase
    correlation over column means — cheap and rotation/shift-robust enough for ranking."""
    from PIL import Image

    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True).stdout or 20)
    prev, shifts = None, []
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "f.png")
        for i in range(n):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(dur * (i + 0.5) / n),
                            "-i", path, "-vf", "scale=160:90", "-frames:v", "1", fp],
                           capture_output=True)
            a = np.asarray(Image.open(fp).convert("L"), float)
            if prev is not None:
                fa = np.fft.rfft(prev.mean(0) - prev.mean())
                fb = np.fft.rfft(a.mean(0) - a.mean())
                r = np.fft.irfft(fa * np.conj(fb) / (np.abs(fa * np.conj(fb)) + 1e-9), n=a.shape[1])
                s = int(np.argmax(r))
                shifts.append(s - a.shape[1] if s > a.shape[1] // 2 else s)
            prev = a
    return np.array(shifts, float)


def turn_score(shifts: np.ndarray) -> tuple[float, float, float]:
    """(peak sustained pan, total motion, coherence). Coherence = |peak| / total."""
    cum = np.cumsum(shifts)
    peak = float(cum[np.argmax(np.abs(cum))]) if len(cum) else 0.0
    total = float(np.abs(shifts).sum())
    return peak, total, abs(peak) / (total + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="reference videos to screen")
    ap.add_argument("--min-quality", type=float, default=9.2)
    ap.add_argument("--min-peak", type=float, default=90.0, help="TURN threshold: |peak| pan px")
    ap.add_argument("--min-coherence", type=float, default=0.35)
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()
    tok = os.environ.get("HF_TOKEN") or ""
    if not tok:
        raise SystemExit("set HF_TOKEN (gated NuRec read token)")

    from huggingface_hub import hf_hub_download

    ratings = hf_hub_download(REPO, "clip_ratings_26.04.csv", repo_type="dataset", token=tok)
    rows = [r for r in csv.DictReader(open(ratings))
            if float(r["quality_score"]) >= args.min_quality]
    # spread across time-of-day / speed buckets for variety
    import collections
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r["time_of_day"], r["ego_speed"])].append(r)
    picks, k = [], 0
    while len(picks) < args.n and any(buckets.values()):
        for key in list(buckets):
            if buckets[key]:
                picks.append(buckets[key].pop(0))
                if len(picks) >= args.n:
                    break
        k += 1
        if k > args.n:
            break

    results = []
    for r in picks:
        cid = r["clip_id"]
        try:
            f = hf_hub_download(REPO, REF.format(cid=cid), repo_type="dataset", token=tok)
            peak, total, coh = turn_score(_frame_shifts(f, args.frames))
            results.append((cid, r["time_of_day"], r["ego_speed"], peak, total, coh))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {cid[:8]}: {str(exc)[:50]}")

    results.sort(key=lambda x: -(abs(x[3]) * (x[5] >= args.min_coherence)))
    print(f"\n{'clip_id':36s} {'tod':8s} {'speed':7s} {'peak':>6s} {'coh':>5s}  flag")
    for cid, tod, spd, peak, total, coh in results:
        flag = "TURN" if (abs(peak) >= args.min_peak and coh >= args.min_coherence) else ""
        print(f"{cid:36s} {tod:8s} {spd:7s} {peak:+6.0f} {coh:5.2f}  {flag}")
    turners = [c for c, *_ , coh in [(r[0], r[3], r[5]) for r in results]]  # noqa: F841
    n_turn = sum(1 for r in results if abs(r[3]) >= args.min_peak and r[5] >= args.min_coherence)
    print(f"\n{n_turn}/{len(results)} flagged TURN (|peak|>={args.min_peak:.0f}, "
          f"coh>={args.min_coherence}). Render those with force_gt_duration_us to follow the path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
