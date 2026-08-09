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
BLUE = "#2c6fbb"

TIERS = [
    (GREEN, "Unconditional — arithmetic on shipped files", [
        "LIBERO-Object ships 0.0–1.9 cm of a declared 5×5 cm region;\n"
        "six of ten tasks are a single point   (§2, Fig. 2)",
        "Three sibling suites exercise their declared regions\n"
        "(96.7–99.8% for LIBERO-Long)   (§2, Table 2)",
        "Six of ten targets take two float64 values, 1 ULP apart   (§3)",
        "Placement radius R, exactly computed, all 40 tasks   (§3, Table 3)",
        "R ≤ δ is necessary AND sufficient for a one-constant table   (Prop. 1)",
        "The decision window W(A) = [max_A R, min_A^c R): a suite-level claim\n"
        "is true iff δ ∈ W, so an estimate wider than |W| decides nothing at\n"
        "any sample size   (Prop. 2, Cor. 1)",
        "Of the 8 clean suite-level claims LIBERO can express, 7 are\n"
        "unsatisfiable by ANY tolerance; the 8th needs 0.322 cm   (§7.1)",
    ]),
    (BLUE, "Measured, and independent of δ — needs a controller and a simulator,\n"
           "but no tolerance", [
        "Sampling the declared region costs a lookup policy half its score:\n"
        "25/30 → 10/30, paired, exact McNemar p = 0.0007   (§2.1, Table 4)",
        "The repaired states are not harder: handed the true target once at\n"
        "reset, the same controller scores 30/30 on them   (§2.1, control row)",
        "So the collapse is the constant, not the states: 20 discordant\n"
        "trials on identical states, all one way, p = 2×10⁻⁶",
    ]),
    (GREY, "WITHDRAWN by §7 — the window is [1.17, 1.49) cm and the first\n"
           "direct measurement of δ on the grasped object is ≥ 2 cm", [
        "LIBERO-Object 10/10 admissible; elsewhere 2/30, both fixtures\n"
        "(at δ = 1.91 cm, our own second estimate, elsewhere becomes 13/30)",
        "δ ≈ 1.4 cm itself: one n = 10 cell, HARKed, confounded with trial\n"
        "index, and measured on the container rather than the grocery",
        "→ measured on the grocery: the score survives 2 cm and breaks by 4,\n"
        "   and at 2 cm 17 of 30 tasks elsewhere are admissible too   (§7)",
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

fig, ax = plt.subplots(figsize=(11.4, 10.4))
y = 0.0
for col, header, items in TIERS:
    ax.text(0.0, y, header, fontsize=10.6, fontweight="bold", color=col,
            va="top")
    y -= 0.62 + 0.44 * header.count("\n")
    for it in items:
        n = it.count("\n") + 1
        h = 0.64 * n + 0.20
        ax.add_patch(FancyBboxPatch((0.06, y - h), 9.5, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    facecolor=col, alpha=0.11,
                                    edgecolor=col, lw=1.0))
        ax.text(0.28, y - 0.32, it, fontsize=8.9, va="top", color="#1f2933",
                linespacing=1.45)
        y -= h + 0.20
    y -= 0.34

ax.set_xlim(-0.1, 10.0)
ax.set_ylim(y + 0.2, 0.55)
ax.axis("off")
fig.text(0.5, 0.005,
         "Everything in the top tier is checkable from two files the benchmark ships. The second tier needs a simulator but still no tolerance — so a reader who rejects δ "
         "entirely, and with it the whole third tier, keeps the defect AND the demonstration that it is exploitable.",
         ha="center", fontsize=8.8, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
