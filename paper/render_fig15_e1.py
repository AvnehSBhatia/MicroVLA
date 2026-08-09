"""Figure 15 -- the manipulation check, drawn at the level of single trials.

Section 2 says LIBERO-Object ships states that do not sample the region its own
task files declare. Whether that MATTERS is a separate question, and this is
the experiment that answers it: run two policies on task 0 twice, once on the
fifty states LIBERO ships and once on fifty drawn from the declared region,
changing nothing else -- same weights, same seed, same trial indices, same
machine, same packages, both conditions in one batch.

The figure shows the raw outcomes rather than the summary, because the test is
paired and a pair of bar heights hides the pairing. Each column is one trial
index; a column that is green above and red below is a trial the repair cost.

  top     what changed: the target's fifty starting positions
  bottom  what happened: 30 paired trials per arm, per condition

The lookup constant is bit-identical to the shipped target position, so against
a target that actually moves it must degrade -- and it does, 25/30 to 10/30.
That is what makes the defect load-bearing rather than cosmetic.

Sources:
    results/e1_shipped_vs_repaired.json          (per-trial outcomes)
    results/resampled_init/MANIFEST.json         (the repaired draw)
    results/suite_forensics_joints.json          (the shipped positions)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F15_e1_manipulation.png"
TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
HIT, MISS = "#2e8b57", "#c0392b"

E1 = json.loads((REPO / "results/e1_shipped_vs_repaired.json").read_text())
E5 = json.loads((REPO / "results/e5_trials.json").read_text())
MAN = json.loads((REPO / "results/resampled_init/MANIFEST.json").read_text())
rec = next(t for t in MAN["tasks"] if t["task"] == TASK)

# --- what changed -----------------------------------------------------------
# Shipped: every one of the fifty states puts the target at one point, to the
# last bit of a float64. Repaired: fifty draws from the declared 5x5 cm region
# under the clearance constraint. We plot the repaired draw as recorded.
try:
    import torch
    arr = np.asarray(torch.load(REPO / "results/resampled_init" /
                                f"{TASK}.pruned_init", weights_only=False),
                     dtype=float)
    rep_xy = arr[:, rec["columns"]] * 100.0
    rep_xy -= rep_xy.mean(axis=0)
except Exception:                      # torch absent: fall back to the manifest
    rep_xy = None

fig = plt.figure(figsize=(11.6, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 2.35], hspace=0.30, wspace=0.55)

for col, (title, xy, c) in enumerate([
        ("what LIBERO ships", np.zeros((50, 2)), MISS),
        ("what its task file declares", rep_xy, HIT)]):
    ax = fig.add_subplot(gs[0, col])
    d = rec["declared_cm"]
    ax.add_patch(Rectangle((-d[0] / 2, -d[1] / 2), d[0], d[1],
                           facecolor="#f2d8a8", alpha=0.55,
                           edgecolor="#8a6d3b", lw=1.0))
    if xy is not None:
        ax.scatter(xy[:, 0], xy[:, 1], s=13, color=c, edgecolor="white",
                   lw=0.4, alpha=0.9, zorder=3)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.2, fontweight="bold", color=c, pad=4)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=7)
    ax.grid(color="#eef1f4", lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    spread = 0.0 if xy is None or not xy.any() else float(
        (xy.max(axis=0) - xy.min(axis=0)).max())
    ax.set_xlabel(f"spread {spread:.2f} cm", fontsize=8.2, color=c,
                  fontweight="bold", labelpad=2)

axn = fig.add_subplot(gs[0, 2:])
axn.axis("off")
axn.text(0.0, 0.97,
         "One variable changes: the fifty initial states.", fontsize=10.0,
         fontweight="bold", va="top")
axn.text(0.0, 0.74,
         "Same weights, same seed, same trial indices, same machine,\n"
         "same package versions, back to back in one invocation.\n\n"
         "The lookup arm's constant is bit-identical to the shipped\n"
         "target position. Against a target that actually moves it\n"
         "should degrade — so this is the manipulation check for §2,\n"
         "and a null result would have retired the criterion.",
         fontsize=8.8, va="top", color="#43505c", linespacing=1.45)

# --- what happened ----------------------------------------------------------
# Three blocks. The control is placed directly under the lookup arm because
# that adjacency IS the argument: the same thirty repaired states, red under a
# stale constant and green under a correct one.
ax = fig.add_subplot(gs[1, :])
N = 30
BLOCKS = [
    ("lookup constant, from the shipped files", 5.55,
     [("shipped", E1["cells"]["blind_shipped"]["trials"]),
      ("repaired", E1["cells"]["blind_repaired"]["trials"])]),
    ("control: true target, read once at reset", 3.05,
     [("shipped", E5["fixed_shipped"]),
      ("repaired", E5["fixed_repaired"])]),
    ("released head", 0.55,
     [("shipped", E1["cells"]["head_shipped"]["trials"]),
      ("repaired", E1["cells"]["head_repaired"]["trials"])]),
]
for label, y0, rows in BLOCKS:
    for r, (cond, t) in enumerate(rows):
        y = y0 - r * 0.92
        for j2 in range(N):
            k2 = str(j2)
            if k2 not in t:
                ax.add_patch(Rectangle((j2, y), 0.86, 0.80, facecolor="none",
                                       edgecolor="#d7dce1", lw=0.6, ls=":"))
                continue
            ax.add_patch(Rectangle((j2, y), 0.86, 0.80,
                                   facecolor=HIT if t[k2] else MISS,
                                   alpha=0.92 if t[k2] else 0.80,
                                   edgecolor="none"))
        k, n = sum(bool(v) for v in t.values()), len(t)
        ax.text(-0.6, y + 0.40, cond, fontsize=8.8, ha="right", va="center")
        ax.text(N + 0.5, y + 0.40, f"{k}/{n}", fontsize=9.2, ha="left",
                va="center", fontweight="bold",
                color=HIT if k > n / 2 else MISS)
    a, b = rows[0][1], rows[1][1]
    keys = sorted(set(a) & set(b), key=int)
    lost = sum(1 for k in keys if a[k] and not b[k])
    gained = sum(1 for k in keys if b[k] and not a[k])
    n = lost + gained
    p = min(1.0, 2 * sum(math.comb(n, i2) for i2 in
                         range(min(lost, gained) + 1)) / 2 ** n) if n else 1.0
    ax.text(-0.6, y0 + 1.12, label, fontsize=9.8, ha="right",
            fontweight="bold", va="center")
    if lost or gained:
        star = r"$\bf{p = %.4f}$" % p if p < 0.01 else f"p = {p:.2f}"
        note = f"{lost} trials lost to the repair,  {gained} gained     " + star
    else:
        note = "no trial changes: the repair costs a correct goal nothing"
    ax.text(0.0, y0 + 1.12, note, fontsize=9.0, ha="left", va="center",
            color="#1f2933" if (lost or gained) and p < 0.01 else "#5b6670")

# the contrast the paper turns on, drawn once
ax.annotate("", xy=(N + 3.6, 5.43), xytext=(N + 3.6, 2.13),
            arrowprops=dict(arrowstyle="-", color="#8a6d3b", lw=1.4))
for _y in (5.43, 2.13):
    ax.plot([N + 3.6, N + 4.0], [_y, _y], color="#8a6d3b", lw=1.4)
ax.text(N + 4.3, 3.78, "the same thirty\nrepaired states:\nred under a stale\nconstant, green\nunder a correct one",
        fontsize=8.4, va="center", color="#8a6d3b", fontweight="bold",
        linespacing=1.35)

ax.set_xlim(-9.6, N + 13.5); ax.set_ylim(-1.05, 7.05)
ax.axis("off")
ax.text(N / 2, -0.78, "trial index, matched across conditions", fontsize=8.4,
        ha="center", color="#5b6670")

fig.suptitle("The defect is exploitable: sampling the declared region costs a "
             "lookup policy half its score", fontsize=11.6, y=1.005)
fig.text(0.5, -0.045,
         "Green is a success. One variable changes between the two rows of a block — the fifty initial states — and every cell ran back to back in one invocation on one machine.",
         ha="center", fontsize=9.0, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
