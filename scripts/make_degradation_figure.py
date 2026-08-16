"""Tier 1 headline figure: how much the hard shield's guarantee degrades when its obstacle field
comes from learned camera perception (surround ftheta + metric depth + SegFormer) instead of
ground-truth geometry. Per-scene GT-vs-camera at-fault rate (the degradation) and progress (~flat),
plus the aggregate. Light-themed. Reads results/tier1_degradation.csv, writes the PNG beside it.
"""
import csv
import statistics as st

import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open("results/tier1_degradation.csv")))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


scenes = {}
for r in rows:
    scenes.setdefault(r["scene"][7:15], {})[r["arm"]] = r
labels = list(scenes)
gt_af = [f(scenes[s]["gt"]["at_fault"]) for s in labels]
cam_af = [f(scenes[s]["camera"]["at_fault"]) for s in labels]
gt_afe = [f(scenes[s]["gt"]["at_fault_std"]) for s in labels]
cam_afe = [f(scenes[s]["camera"]["at_fault_std"]) for s in labels]
gt_pr = [f(scenes[s]["gt"]["progress"]) for s in labels]
cam_pr = [f(scenes[s]["camera"]["progress"]) for s in labels]

GT_C, CAM_C = "#4C72B0", "#DD8452"  # calm blue vs warm orange
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white"})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.2))
x = np.arange(len(labels))
w = 0.38

ax1.bar(x - w / 2, gt_af, w, yerr=gt_afe, capsize=3, label="ground-truth geometry",
        color=GT_C, ecolor="#888")
ax1.bar(x + w / 2, cam_af, w, yerr=cam_afe, capsize=3, label="learned camera perception",
        color=CAM_C, ecolor="#888")
ax1.set_ylabel("at-fault collision rate")
ax1.set_title("Learned camera perception degrades the hard shield's safety guarantee ~10×",
              fontsize=13, weight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax1.legend(frameon=False, loc="upper left")
ax1.text(0.99, 0.95,
         "aggregate at-fault:  GT %.2f  →  camera %.2f   (%.1f×)\n"
         "progress:  GT %.2f  ≈  camera %.2f" % (
             st.mean(gt_af), st.mean(cam_af), st.mean(cam_af) / max(st.mean(gt_af), 1e-9),
             st.mean(gt_pr), st.mean(cam_pr)),
         transform=ax1.transAxes, ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", fc="#F4F0E8", ec="#CBBFA6"))
for i, s in enumerate(labels):
    if cam_af[i] > gt_af[i] + 0.15:
        ax1.annotate("camera\nmisses", (i + w / 2, cam_af[i]), (i + w / 2, cam_af[i] + 0.12),
                     ha="center", fontsize=7.5, color="#B4462F",
                     arrowprops=dict(arrowstyle="-", color="#B4462F", lw=0.8))

ax2.bar(x - w / 2, gt_pr, w, label="ground-truth geometry", color=GT_C)
ax2.bar(x + w / 2, cam_pr, w, label="learned camera perception", color=CAM_C)
ax2.set_ylabel("progress (fraction of route)")
ax2.set_title("Progress is essentially unchanged — the degradation is in safety, not mobility",
              fontsize=11, pad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.legend(frameon=False, loc="upper right", fontsize=9)

fig.text(0.5, 0.005,
         "shield over VaVAM, surround ftheta perception  |  10 NuRec scenes, n=10 rollouts each  "
         "|  obstacle field: GT actors vs camera (Depth-Anything metric + SegFormer)",
         ha="center", fontsize=8.5, color="#666")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("results/tier1_degradation.png", dpi=150)
print("wrote results/tier1_degradation.png")
