"""Figure 12 -- what each claim rests on, and what survives if delta is wrong.

A reviewer's first question about this paper is which conclusions depend on the
tolerance, because delta is its weakest link: our two estimates of it disagree
by a factor of 1.4, and the separation it licenses lives on a 0.32 cm window.
Rather than make a reader reconstruct that dependency from prose, we draw it.

Three tiers:

  UNCONDITIONAL   arithmetic on shipped files. No delta, no policy, no
                  simulator. These stand even if every other claim falls.
  CONDITIONAL     needs delta. Stated with the window it holds on.
  WITHDRAWN       claims this paper made and retracted, kept visible so the
                  record is legible.

The figure is honest bookkeeping, not decoration: it is the one place a
reviewer can see, at a glance, exactly how much of the paper survives losing
its weakest assumption.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F12_dependency.png"

GREEN, AMBER, GREY = "#2e8b57", "#b8860b", "#9aa5b1"

TIERS = [
    (GREEN, "Unconditional — arithmetic on shipped files", [
        "LIBERO-Object ships 0.0–1.9 cm of a declared 5×5 cm region;\n"
        "six of ten tasks are a single point   (§2, Fig. 2)",
        "Three sibling suites exercise their declared regions\n"
        "(96.7–99.8% for LIBERO-Long)   (§2, Table 2)",
        "Six of ten targets take two float64 values, 1 ULP apart   (§3)",
        "Placement radius R, exactly computed, all 40 tasks   (§3, Table 3)",
        "R ≤ δ is necessary AND sufficient for a one-constant table   (Prop. 1)",
    ]),
    (AMBER, "Conditional on δ — holds on [1.17, 1.49) cm, and not above", [
        "LIBERO-Object 10/10 admissible; elsewhere 2/30, both fixtures\n"
        "(at δ = 1.91 cm, our own second estimate, elsewhere becomes 13/30)",
        "δ ≈ 1.4 cm itself: one n = 10 cell, HARKed, confounded with trial\n"
        "index, and measured on the container rather than the grocery",
    ]),
    (GREY, "Withdrawn by this paper", [
        "“a constant beats a trained policy, p = 0.039”  →  pseudoreplication;\n"
        "task-level tests resolve nothing either way   (§6)",
        "“constants identical to 1 mm score 0/10 to 10/10”  →  computed in 2D;\n"
        "the surviving instance is 0.469 mm in 3D   (§6)",
        "r(R, success) = +0.52  →  permutation p = 0.13; sign flips on one task",
        "the renderer's 4/10 disagreement as a backend effect  →  no\n"
        "same-backend control at the contact horizon   (§9)",
    ]),
]

fig, ax = plt.subplots(figsize=(11.4, 6.6))
y = 0.0
for col, header, items in TIERS:
    ax.text(0.0, y, header, fontsize=10.6, fontweight="bold", color=col,
            va="top")
    y -= 0.62
    for it in items:
        n = it.count("\n") + 1
        h = 0.42 * n + 0.20
        ax.add_patch(FancyBboxPatch((0.06, y - h), 9.5, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    facecolor=col, alpha=0.11,
                                    edgecolor=col, lw=1.0))
        ax.text(0.28, y - 0.30, it, fontsize=8.9, va="top", color="#1f2933")
        y -= h + 0.16
    y -= 0.34

ax.set_xlim(-0.1, 10.0)
ax.set_ylim(y + 0.2, 0.55)
ax.axis("off")
fig.text(0.5, 0.005,
         "Everything in the top tier is checkable from two files the benchmark ships. Nothing in it depends on a tolerance, a controller, a policy, or a statistic — "
         "so a reader who rejects the middle tier entirely still keeps the top one.",
         ha="center", fontsize=8.8, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
