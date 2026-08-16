"""Tier 1 perception ladder: at-fault rate + progress across four obstacle-field sources, from
ground-truth geometry to the full learned surround stack. Reads the two sweep CSVs (GT + surround-
semantic in tier1_degradation.csv; front + surround-gated in tier1_ladder.csv), aggregates over the
10 scenes (error bars = std across scenes), and writes the light-themed figure beside them.
"""
import csv
import statistics as st

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return list(csv.DictReader(open(path)))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


data = {}
for r in load("results/tier1_degradation.csv") + load("results/tier1_ladder.csv"):
    data.setdefault(r["arm"], {})[r["scene"]] = r

# rung label -> csv arm key, in increasing perception-realism order
RUNGS = [("ground truth\n(perfect geometry)", "gt"),
         ("front camera\n(mono depth)", "front"),
         ("surround\n(gated)", "surround_gated"),
         ("surround\n(+ semantic)", "camera")]


def agg(key, field):
    vals = [f(v[field]) for v in data.get(key, {}).values() if f(v[field]) is not None]
    return st.mean(vals), st.pstdev(vals)


af = [agg(k, "at_fault") for _, k in RUNGS]
pr = [agg(k, "progress") for _, k in RUNGS]
labels = [lab for lab, _ in RUNGS]
x = np.arange(len(RUNGS))

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

# colour ramp: green (safe/GT) -> deepening orange as perception replaces geometry
COLORS = ["#4C9A6A", "#E0A458", "#D9743F", "#C25438"]
ax1.bar(x, [m for m, _ in af], yerr=[s for _, s in af], capsize=4, color=COLORS, ecolor="#999")
for i, (m, _) in enumerate(af):
    ax1.text(i, m + 0.015, "%.2f" % m, ha="center", fontsize=11, weight="bold")
ax1.set_ylabel("at-fault collision rate")
ax1.set_title("Every learned rung degrades the guarantee 3–6×\n(and it isn't monotonic)",
              fontsize=12, weight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, max(m for m, _ in af) + 0.22)
ax1.axhline(af[0][0], ls="--", lw=1, color="#4C9A6A", alpha=0.6)
ax1.text(3.35, af[0][0] + 0.005, "GT baseline", ha="right", fontsize=8, color="#3A7A52")

ax2.bar(x, [m for m, _ in pr], yerr=[s for _, s in pr], capsize=4, color=COLORS, ecolor="#999")
for i, (m, _) in enumerate(pr):
    ax2.text(i, m + 0.02, "%.2f" % m, ha="center", fontsize=11, weight="bold")
ax2.set_ylabel("route progress")
ax2.set_title("Surround arms cost mobility\n(finding-2 side-actor over-braking)", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylim(0, 1.0)

fig.suptitle("Perception ladder — hard shield over VaVAM, 10 NuRec scenes × n=5",
             fontsize=13, weight="bold", y=1.0)
fig.text(0.5, 0.005,
         "obstacle field: ground-truth actors  →  front mono depth  →  surround ftheta + corridor gate "
         " →  + SegFormer semantic filter.   front rung is ungated (config), so treat it as texture; "
         "the controlled contrast is GT vs surround-semantic.",
         ha="center", fontsize=7.8, color="#666")
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig("results/tier1_ladder.png", dpi=150)
print("wrote results/tier1_ladder.png")
