"""F3 --- the threat model: where a fixed placement can hide, and what caught it.

The audit's premise is that "the policy memorized the location" is not one
hypothesis but four, at four different stages, two of which are not in the model
at all. Drawn as a pipeline so the reader can see that two of the four hiding
places sit UPSTREAM of any learned weight --- in the scripted expert and in our
own checkpoint-selection procedure --- which is the part a black-box
perturbation study cannot reach.

Statuses are read from the master audit table and are deliberately not uniform:
two layers are repaired, one is bounded rather than removed, and one has a
repair that holds for the measured quantity and did not replicate for the
behavioural one.

Run: python paper/render_fig3_threatmodel.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "visuals" / "F3_threat_model.png"

OK = "#2a9d3f"        # repaired
PART = "#e08e0b"      # partial / bounded
BAD = "#c1121f"       # the shortcut itself

STAGES = [
    ("scripted\nexpert", "non-model"),
    ("corpus\n+ selection", "non-model"),
    ("trunk\n(frozen at deploy)", "model"),
    ("goal heads", "model"),
    ("shell\n(no learning)", "engineered"),
]

LAYERS = [
    (0, "L1  expert calibration constant",
     "hand-eye constant encodes\nthe pinned placement",
     "teleported episodes", "repaired", OK),
    (1, "L2  iteration-coupled selection",
     "checkpoints selected on cells\nlater rounds reused",
     "process fix + untouched band", "bounded", PART),
    (3, "L3  grasp head reads proprioception",
     "prediction tracks the eef,\nflat under uv sweeps",
     "teleported episodes", "repaired", OK),
    (3, "L4  place head memorized command$\\rightarrow$location",
     "14.5/13.0 cm off for two commands;\nswap 0.767->0.433 at n=30, p=0.031",
     "command coverage", "point repaired,\nbehaviour HALVED", PART),
]

fig, ax = plt.subplots(figsize=(12.6, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 54)
ax.axis("off")

# ---- the pipeline ---------------------------------------------------------
w, gap = 15.5, 3.6
for i, (name, kind) in enumerate(STAGES):
    x = 3 + i * (w + gap)
    fill = "#f2f2f4" if kind != "model" else "#eaf1f7"
    ax.add_patch(FancyBboxPatch((x, 44), w, 8, boxstyle="round,pad=0.4",
                                linewidth=1.5, edgecolor="#5c677d",
                                facecolor=fill, zorder=2))
    ax.text(x + w / 2, 48, name, ha="center", va="center", fontsize=8.6,
            fontweight="bold", zorder=3, linespacing=1.25)
    ax.text(x + w / 2, 44.9, kind, ha="center", va="center", fontsize=6.6,
            color="#5c677d", zorder=3, style="italic")
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((x + w, 48), (x + w + gap, 48),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.6, color="#5c677d", zorder=1))

ax.text(3, 53.4, "training-time  →  deployment", fontsize=8, color="#5c677d",
        style="italic")

# ---- the four layers hanging off their stages -----------------------------
row_y = [31.5, 21.5, 11.5, 1.5]
# Each layer names its stage in-box. An earlier version drew dashed connectors
# from the pipeline down to the boxes; they all converged on one x and crossed
# each other, which made the mapping harder to read than the label does.
for (stage, title, evidence, repair, status, col), y in zip(LAYERS, row_y):
    ax.add_patch(FancyBboxPatch((3, y), 93, 8.2, boxstyle="round,pad=0.4",
                                linewidth=1.5, edgecolor=col,
                                facecolor="#fcfcfd", zorder=2))
    ax.text(5, y + 5.7, title, fontsize=9.2, fontweight="bold", va="center",
            zorder=3)
    ax.text(5, y + 2.2, evidence, fontsize=7.4, va="center", color="#333333",
            zorder=3, linespacing=1.25)
    ax.text(40, y + 5.7, f"stage: {STAGES[stage][0].replace(chr(10), ' ')}",
            fontsize=7.8, va="center", zorder=3, color=BAD, style="italic")
    ax.text(64, y + 5.7, f"repair:  {repair}", fontsize=8.0, va="center",
            zorder=3, color="#333333")
    ax.text(64, y + 2.2, status, fontsize=8.4, va="center", zorder=3,
            color=col, fontweight="bold", linespacing=1.2)

ax.text(50, 42.0,
        "Two of the four hiding places are upstream of any learned weight — "
        "which is what a black-box perturbation study cannot reach.",
        ha="center", fontsize=9.2, style="italic", color="#333333")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
