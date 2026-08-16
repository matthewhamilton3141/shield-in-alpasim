"""System diagram: how the hard shield decorates AlpaSim's VaVAM camera driver, with the pluggable
obstacle field (GT actors vs the learned camera ladder) that the Tier 1 experiment swaps. Light
theme, pure matplotlib. Writes docs/architecture.png.
"""
import matplotlib.patches as mp
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.4)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=10, weight="normal", ls="solid", tc="#222", lw=1.6, title=False):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                   fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    va = "top" if title else "center"
    ty = y + h - 0.18 if title else y + h / 2
    ax.text(x + w / 2, ty, text, ha="center", va=va, fontsize=fs, weight=weight, color=tc, zorder=3)


def arrow(x1, y1, x2, y2, text="", color="#555", rad=0.0, fs=8.5, lw=1.8, toff=(0, 0.18)):
    ax.annotate("", (x2, y2), (x1, y1), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=%s" % rad))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="center",
                fontsize=fs, color=color, style="italic")


# ---- AlpaSim (external sim) ----
box(0.3, 1.3, 2.5, 6.0,
    "AlpaSim\nclosed-loop sim\n\n• NuRec renderer\n  (5× ftheta cams)\n• physics\n• eval scorer",
    "#DCE6EF", "#6E8CA8", fs=10.5)

# ---- ShieldedDriver container (our plugin) ----
box(3.4, 0.7, 8.0, 7.2, "ShieldedDriver — our AlpaSim plugin (a decorator, not a driver)",
    "#FCF7EF", "#C9A54A", fs=11, weight="bold", ls=(0, (6, 4)), tc="#8A6D1E", title=True)

# inner policy
box(3.8, 5.6, 3.0, 1.4, "VaVAM\n(inner policy)\nfront cam → trajectory",
    "#E7E1F1", "#8A6FB0", fs=10)

# obstacle field (the pluggable seam / ladder)
box(3.8, 1.2, 3.0, 3.5, "Obstacle field  —  pluggable seam",
    "#DEEEE8", "#4C9A6A", fs=10, weight="bold", title=True)
box(4.0, 3.45, 2.6, 0.95, "GT actors (from USDZ)\nprivileged geometry",
    "#E9F2E9", "#4C9A6A", fs=8.8)
box(4.0, 1.45, 2.6, 1.75, "camera perception:\nDepth-Anything metric depth\n→ SegFormer semantic\n→ BEV occupancy → discs",
    "#FBE9DD", "#C25438", fs=8.6)

# the shield
box(7.6, 3.2, 3.4, 2.4,
    "kitti-nav hard shield\n\ncertify the tracked plan\nagainst the obstacle field;\nbrake if it can't stop\nsafely  (the guarantee)",
    "#F5E3DB", "#C25438", fs=10, weight="bold")

# ---- arrows: AlpaSim -> plugin ----
arrow(2.8, 6.4, 3.8, 6.3, "front frame", rad=0.0, toff=(0, 0.2))
arrow(2.8, 3.0, 3.8, 3.0, "5 cams + ego pose", rad=0.0, toff=(0, 0.22))

# inside: policy + field -> shield
arrow(6.8, 6.1, 7.7, 5.2, "proposed\ntrajectory", rad=-0.15, toff=(0.35, 0.25))
arrow(6.8, 3.0, 7.7, 4.0, "obstacle discs", rad=0.15, toff=(0.1, -0.3))

# shield -> out -> back to AlpaSim
arrow(11.0, 4.4, 11.9, 4.4, "", color="#C25438")
ax.annotate("", (2.8, 1.9), (11.9, 4.4), zorder=4,
            arrowprops=dict(arrowstyle="-|>", color="#C25438", lw=2.0,
                            connectionstyle="arc3,rad=0.28"))
ax.text(7.2, 0.35, "certified waypoints  →  back to AlpaSim  (scored: at-fault rate, progress)",
        ha="center", fontsize=9.5, color="#C25438", style="italic")

# ---- the experiment callout ----
ax.text(12.65, 6.7, "The experiment", ha="center", fontsize=11, weight="bold", color="#8A2E1B")
ax.add_patch(mp.FancyBboxPatch((11.75, 1.5), 2.1, 5.0, boxstyle="round,pad=0.1,rounding_size=0.1",
                               fc="#FBF1EC", ec="#C25438", lw=1.4, zorder=1))
ax.text(12.8, 4.0,
        "swap the obstacle\nfield\n\nGT  ⇄  camera\nladder\n\n(everything else\nfixed)\n\n→ how much does\nthe certificate\ndegrade?\n\nTier 1: ~4× at-fault",
        ha="center", va="center", fontsize=9, color="#7A3020")

ax.set_title("A hard safety shield filtering a learned camera policy, inside a photoreal AV sim",
             fontsize=14, weight="bold", pad=14)
fig.tight_layout()
fig.savefig("docs/architecture.png", dpi=150, bbox_inches="tight")
print("wrote docs/architecture.png")
