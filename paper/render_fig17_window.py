"""Figure 17 -- the decision window, and the instrument that cannot enter it.

Proposition 2 says a suite-level admissibility claim is true exactly when delta
lies in a computable interval W(A), so an estimate of delta less precise than
|W(A)| decides nothing at any sample size. That is one picture.

Top: every one of LIBERO's forty placement radii on a common axis, one row per
suite. The claim "LIBERO-Object is admissible and the rest are not" is true
precisely for tolerances in the gap between Object's largest radius and the
smallest movable radius anywhere else -- a gap of 0.322 cm, drawn to scale.

Bottom: the same axis, with what measurement can actually deliver. Both of the
paper's container-based estimates of delta, and the interval the grasp ladder
of Figure 10 resolves. The last is 2 cm wide. It cannot fit inside a 0.322 cm
window, and no number of additional episodes changes that -- sample size buys
a tighter interval only up to the noise floor of a ten-trial ladder, and the
figure is about the geometry, not the noise.

The reading is meant to be immediate: the sliver is the question, the bar is
the answer we can currently produce, and the bar is six times too wide.

Sources:
    results/admissibility.json     (the radii)
    results/decision_windows.json  (the window, recomputed there)
    results/e23_analysis.json      (the ladder's bracket)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F17_decision_window.png"

SUITES = [("libero_object", "LIBERO-Object", "#c0392b"),
          ("libero_goal", "LIBERO-Goal", "#b8860b"),
          ("libero_spatial", "LIBERO-Spatial", "#2c6fbb"),
          ("libero_10", "LIBERO-Long", "#2e8b57")]

ADM = json.loads((REPO / "results/admissibility.json").read_text())["suites"]
DW = json.loads((REPO / "results/decision_windows.json").read_text())
live = next(w for w in DW["windows"] if w["satisfiable"])
LO, HI = live["lower_cm"], live["upper_cm"]

fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.2, 5.8),
                             gridspec_kw={"height_ratios": [2.5, 1.0],
                                          "hspace": 0.30})
XMAX = 5.0

# ---------------------------------------------------------------- the radii
for row, (key, label, col) in enumerate(SUITES):
    y = len(SUITES) - 1 - row
    seen = {}
    for t in ADM[key]["tasks"]:
        r = round(t["identity_R_cm"], 4)
        fix = bool(t.get("fixture", False))
        seen[(r, fix)] = seen.get((r, fix), 0) + 1
        ax.scatter([r], [y], s=64,
                   facecolor="white" if fix else col,
                   edgecolor=col, linewidth=1.5,
                   zorder=4, alpha=0.95)
    # Radii coincide exactly -- eight of Object's ten are the same float --
    # so a reader would otherwise count three dots where there are ten.
    for (r, fix), c in seen.items():
        if c > 1:
            ax.text(r, y + 0.30, f"$\\times${c}", ha="center", fontsize=7.8,
                    color=col, fontweight="bold")
    ax.text(-0.12, y, label, ha="right", va="center", fontsize=9.6,
            fontweight="bold", color=col)
    rr = [t["identity_R_cm"] for t in ADM[key]["tasks"]]
    ax.text(XMAX + 0.08, y, f"{min(rr):.2f}–{max(rr):.2f}", ha="left",
            va="center", fontsize=8.0, color="#5b6670")

ax.add_patch(Rectangle((LO, -0.55), HI - LO, len(SUITES) + 0.1,
                       facecolor="#2e8b57", alpha=0.22, edgecolor="#2e8b57",
                       lw=1.4, zorder=1))
ax.annotate("", xy=(LO, len(SUITES) - 0.30), xytext=(HI, len(SUITES) - 0.30),
            arrowprops=dict(arrowstyle="<->", color="#2e8b57", lw=1.6))
ax.text((LO + HI) / 2, len(SUITES) - 0.08,
        f"$W$ = [{LO:.3f}, {HI:.3f}) — {HI-LO:.3f} cm",
        ha="center", fontsize=9.4, color="#2e8b57", fontweight="bold")
ax.text((LO + HI) / 2 + 1.05, len(SUITES) - 0.42,
        "the only suite-level claim LIBERO can express:\n"
        "every $\\delta$ in here makes it true, every $\\delta$ outside makes it false",
        ha="left", va="center", fontsize=8.4, color="#2e8b57")

ax.set_xlim(-1.55, XMAX + 0.75)
ax.set_ylim(-0.75, len(SUITES) + 0.30)
ax.set_yticks([])
ax.set_xticks(np.arange(0, XMAX + 0.5, 0.5))
ax.tick_params(labelsize=8)
ax.grid(axis="x", color="#eef1f4", lw=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("placement radius $R$ of the identity-bearing object, and "
              "tolerance $\\delta$ (cm)", fontsize=9.4)
ax.scatter([], [], s=64, facecolor="white", edgecolor="#5b6670", lw=1.5,
           label="fixture ($R=0$ because it is bolted down)")
ax.legend(fontsize=8.0, loc="lower right", frameon=False,
          bbox_to_anchor=(1.0, -0.02))

# ------------------------------------------------------- what we can measure
E2 = json.loads((REPO / "results/e23_analysis.json").read_text())["e2"]
b_lo, b_hi = E2["3"]["delta_interval_cm"]
bars = [("first estimate, 1.41 cm", 1.41, 1.41, "#8a6d3b", 2),
        ("second estimate, 1.91 cm", 1.91, 1.91, "#8a6d3b", 1),
        (f"this paper's grasp ladder: $\\delta \\in$ ({b_lo:g}, {b_hi:g}] cm",
         b_lo, b_hi, "#c0392b", 0)]
for lab, lo, hi, col, y in bars:
    if hi > lo:
        bx.add_patch(Rectangle((lo, y - 0.30), hi - lo, 0.60, facecolor=col,
                               alpha=0.30, edgecolor=col, lw=1.5))
    else:
        bx.plot([lo], [y], "|", color=col, ms=18, mew=2.4)
    bx.text(-0.12, y, lab, ha="right", va="center", fontsize=8.8, color=col,
            fontweight="bold" if y == 0 else "normal")
bx.add_patch(Rectangle((LO, -0.62), HI - LO, 3.0, facecolor="#2e8b57",
                       alpha=0.22, edgecolor="#2e8b57", lw=1.4, zorder=1))
bx.text(HI + 0.12, 2.0, "$W$", fontsize=9.4, color="#2e8b57",
        fontweight="bold", va="center")
bx.annotate(f"{DW['precision_shortfall']:.1f}$\\times$ too wide to enter it",
            xy=(b_hi, 0), xytext=(b_hi + 0.15, 0.62), fontsize=9.0,
            color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))
bx.set_xlim(-1.55, XMAX + 0.75)
bx.set_ylim(-0.62, 2.45)
bx.set_yticks([])
bx.set_xticks(np.arange(0, XMAX + 0.5, 0.5))
bx.tick_params(labelsize=8)
bx.grid(axis="x", color="#eef1f4", lw=0.8)
bx.set_axisbelow(True)
bx.spines[["top", "right", "left"]].set_visible(False)
bx.set_xlabel("what a measurement of $\\delta$ can currently deliver (cm)",
              fontsize=9.4, labelpad=1)

fig.suptitle("The question is a sliver; the instrument is a bar six times "
             "wider", fontsize=12.4, y=0.985)
fig.text(0.5, -0.135,
         "Top: all forty placement radii, computed from released files with no policy, simulator or training. The shaded strip is the decision window — the set of\n"
         "tolerances making “LIBERO-Object is lookup-admissible and no other suite is” true. It is the ONLY such claim of the eight LIBERO can express that any\n"
         "tolerance satisfies at all. Bottom: the same axis, with what measurement delivers. Proposition 2 says an estimate wider than the window decides nothing at\n"
         "any sample size, so this gap is not a matter of running more episodes.",
         ha="center", fontsize=8.6, color="#43505c", linespacing=1.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"  window [{LO}, {HI}) width {HI-LO:.4f} cm; "
      f"ladder ({b_lo}, {b_hi}]; shortfall {DW['precision_shortfall']}x")
