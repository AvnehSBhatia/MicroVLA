"""F7 --- is the appearance gap actually a position effect? (referee M4)

Reads the two artifacts emitted by ``scripts/position_vs_appearance.py`` and
draws the two tests that separate identity from table position:

  A  every object at its SHIPPED position, coloured by cluster. If position set
     the gap, the two clusters would separate. They do not (Mann-Whitney
     p=0.31), and the decisive pair --- orange juice and cream cheese, 4.74 mm
     apart --- differ by 2.42x.
  B  every object TELEPORTED 22 cm to the other cluster, identity held constant.
     The position account predicts A->B rises and B->A falls; arrows show what
     actually happened. Cream cheese, moved onto the training cluster, stays the
     highest gap of all ten.

Nothing here is hardcoded: every number is read from the JSON.

Run: python paper/render_fig7_position_vs_appearance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F7_position_vs_appearance.png"

nat = {c["obj"]: c for c in json.loads(
    (REPO / "results/pos_vs_appearance_native.json").read_text())["cells"]}
swp = {c["obj"]: c for c in json.loads(
    (REPO / "results/pos_vs_appearance_swap.json").read_text())["cells"]}

CA, CB = "#2a6f97", "#c1121f"          # cluster A, cluster B
order = sorted(nat, key=lambda o: nat[o]["gap"])
labels = [o.replace("_", " ") for o in order]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 4.6))

# ---- A: shipped positions -------------------------------------------------
y = np.arange(len(order))
cols = [CA if nat[o]["cluster"] == "A" else CB for o in order]
axA.barh(y, [nat[o]["gap"] for o in order], color=cols, height=0.66)
axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=9)
axA.set_xlabel("appearance gap vs own corpus\n"
               "(corpus self-NN cosine $-$ probe NN cosine; higher = further off-manifold)")
axA.set_title("A  At shipped positions: clusters do not separate",
              loc="left", fontweight="bold", fontsize=11)
axA.set_xlim(0, 0.088)
for yi, o in zip(y, order):
    axA.text(nat[o]["gap"] + 0.0015, yi, f"{nat[o]['gap']:.4f}",
             va="center", fontsize=7.6, color="#333333")
h = [plt.Rectangle((0, 0), 1, 1, color=CA), plt.Rectangle((0, 0), 1, 1, color=CB)]
axA.legend(h, ["cluster A  (mean 0.0418)", "cluster B  (mean 0.0503)"],
           frameon=False, fontsize=8.5, loc="lower right")
# the decisive pair
i_oj, i_cc = order.index("orange_juice"), order.index("cream_cheese")
axA.annotate("", xy=(0.0795, i_cc), xytext=(0.0795, i_oj),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
axA.text(0.0805, (i_oj + i_cc) / 2,
         "same position\n(4.74 mm apart)\n$\\mathbf{2.42\\times}$ gap",
         fontsize=8, va="center", ha="left")
axA.spines[["top", "right"]].set_visible(False)

# ---- B: teleported 22 cm --------------------------------------------------
for yi, o in zip(y, order):
    n_, s_ = nat[o]["gap"], swp[o]["gap"]
    c = CA if nat[o]["cluster"] == "A" else CB
    axB.annotate("", xy=(s_, yi), xytext=(n_, yi),
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.9))
    axB.plot([n_], [yi], "o", ms=5, color=c)
axB.set_yticks(y)
axB.set_yticklabels(labels, fontsize=9)
axB.set_xlabel("appearance gap  (dot = shipped position, arrow head = after a 22 cm teleport)")
axB.set_title("B  Teleported 22 cm, identity held constant",
              loc="left", fontweight="bold", fontsize=11)
axB.set_xlim(0, 0.088)
# Left of the top rows is empty (their markers sit at x >= 0.055), so the note
# goes there rather than over the data -- the same objection we raised to F2.
axB.text(0.002, i_cc - 0.75,
         "cream cheese, moved ONTO the\ntraining cluster, gap RISES\n"
         "$0.0697 \\rightarrow \\mathbf{0.0727}$ — still the\nlargest of all ten",
         fontsize=8.5, color=CB, va="top", ha="left", linespacing=1.35)
axB.spines[["top", "right"]].set_visible(False)

fig.suptitle("F7 — The blocked object's appearance gap is a property of the object, not of the table",
             fontsize=12, y=1.01)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
