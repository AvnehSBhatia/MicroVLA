"""Figure 14 -- nine errors, and why checking the arithmetic caught none of them.

The paper's second contribution is a claim about verification, so it needs a
figure that makes the claim checkable at a glance rather than asserted in a
table. Three columns, one row per error:

  instrument      was the measuring device itself wrong?
  verifier        would a checker that recomputes every number have caught it?
  direction       did the error make the result look better or worse?

The pattern is the point. The first column is uniformly "correct" -- no
measurement was ever wrong. The second is almost uniformly "no" -- and in one
case we can state it as fact rather than counterfactual, because our verifier
ran, passed 76 checks, and certified a claim that was false. The third is
uniformly "flattered us", which is what makes the class dangerous: errors of
this kind do not average out.

Each row is an error documented in Table 13 with its repair.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F14_errors.png"

GOOD, BAD, WARN = "#2e8b57", "#c0392b", "#b8860b"

# (short name, what the slice was, would a recomputing verifier catch it?)
ERRORS = [
    ("a perfect null",            "which frame was read",            "no"),
    ("a withdrawn defect count",  "whether a search was a corpus",   "no"),
    ("7/7 vs a published 7/10",   "which trials had finished",       "no"),
    ("a constant beating\nour policy, p = 0.039", "which unit was analysed", "no"),
    ("constants 'identical\nto a millimetre'",    "which axes were compared", "ran"),
    ("a shipped repaired suite",  "whether it was checked at all",   "no"),
    ("a tolerance sweep\nwhere nothing moves",    "which tolerances were swept", "no"),
    ("an $\\ell_\\infty$ appendix",  "whether it was computed",         "no"),
    ("a claim already\nwithdrawn", "which parts of the document",     "no"),
]

fig, ax = plt.subplots(figsize=(10.6, 5.4))
COLS = [3.9, 5.6, 7.5, 9.2]
ax.text(0.0, len(ERRORS) + 0.55, "what we saw", fontsize=9.6, fontweight="bold")
for x, lab in zip(COLS[1:], ["was the\ninstrument wrong?",
                             "would recomputing\nthe numbers catch it?",
                             "direction"]):
    ax.text(x, len(ERRORS) + 0.80, lab, fontsize=9.0, fontweight="bold",
            ha="center", va="top")
ax.text(COLS[0], len(ERRORS) + 0.80, "the slice that\nwas chosen", fontsize=9.0,
        fontweight="bold", ha="center", va="top")

for i, (name, slice_, caught) in enumerate(ERRORS):
    y = len(ERRORS) - 1 - i
    ax.text(0.0, y, name, fontsize=8.6, va="center")
    ax.text(COLS[0], y, slice_, fontsize=8.2, va="center", ha="center",
            color="#43505c", style="italic")
    ax.text(COLS[1], y, "no", fontsize=8.8, va="center", ha="center",
            color=GOOD, fontweight="bold")
    if caught == "ran":
        ax.add_patch(Rectangle((COLS[2] - 0.82, y - 0.30), 1.64, 0.60,
                               facecolor=BAD, alpha=0.16, edgecolor=BAD, lw=1.1))
        ax.text(COLS[2], y, "it ran, and passed", fontsize=8.4, va="center",
                ha="center", color=BAD, fontweight="bold")
    else:
        ax.text(COLS[2], y, "no", fontsize=8.8, va="center", ha="center",
                color=BAD, fontweight="bold")
    ax.text(COLS[3], y, "flattered us", fontsize=8.4, va="center", ha="center",
            color=WARN)

ax.axhline(len(ERRORS) - 0.5, color="#c3c9d0", lw=1.0)
ax.set_xlim(-0.25, 10.1)
ax.set_ylim(-1.5, len(ERRORS) + 1.5)
ax.axis("off")

ax.text(0.0, -0.95,
        "Every measurement was correct. What differed each time was which slice of its output we wrote down — and every error ran the same way.",
        fontsize=9.0, color="#1f2933", fontweight="bold")
fig.text(0.5, -0.02,
         "The fifth row is the one we can state as fact rather than counterfactual: our verifier recomputes every number in the load-bearing sections from raw "
         "outcomes, it passed 76 checks, and the claim it certified was false — because it recomputed the two-dimensional distance we had chosen to compute.",
         ha="center", fontsize=8.6, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
