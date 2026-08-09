"""Figure 16 -- what the lookup arm's tolerance actually is, measured on the
quantity it acts on.

Appendix C confesses that both of this paper's estimates of delta were measured
on the CONTAINER during placing and then applied to the GROCERY during
grasping. Those are different physical quantities, and the whole admissibility
argument leans on the second. This measures the second directly: displace the
blind arm's grasp constant by r centimetres and see what the score does.

The displacement DIRECTION is drawn from default_rng(555000 + trial_seed), so
it is fixed by the trial index. Trial i is pushed along the same ray at every
magnitude -- the sweep is paired in direction as well as in initial state, and
the curve is a genuine dose-response rather than four independent draws.

Two readings, and the experiment was run because either was live:

  the curve falls   the arm is using the constant, and where it falls brackets
                    delta on the right quantity
  the curve is flat the arm is NOT using the constant, and every claim in this
                    paper that rests on the blind arm has to be withdrawn

Source: results/e23_analysis.json (written by scripts/analyze_e23.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F16_jitter.png"
COLS = {"0": "#c0392b", "3": "#2c6fbb"}
NAMES = {"0": "task 0  alphabet soup", "3": "task 3  bbq sauce"}

A = json.loads((REPO / "results/e23_analysis.json").read_text())["e2"]
tasks = sorted(A, key=int)

fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.9),
                         squeeze=False)
for ax, t in zip(axes[0], tasks):
    d = A[t]
    lv = d["levels_cm"]
    k = np.array([d["cells"][f"{j:g}"]["k"] for j in lv], float)
    n = np.array([d["cells"][f"{j:g}"]["n"] for j in lv], float)
    lo = np.array([d["cells"][f"{j:g}"]["wilson"][0] for j in lv])
    hi = np.array([d["cells"][f"{j:g}"]["wilson"][1] for j in lv])
    x = np.arange(len(lv))
    c = COLS.get(t, "#43505c")

    ax.fill_between(x, lo, hi, color=c, alpha=0.14, lw=0)
    ax.plot(x, k / n, "-o", color=c, lw=1.9, ms=6, mec="white", mew=1.0,
            zorder=3)
    for xi, ki, ni in zip(x, k, n):
        ax.annotate(f"{int(ki)}/{int(ni)}", (xi, ki / ni), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.6, color=c)

    # the two delta estimates this paper already carries, for scale
    for est, lab in [(1.41, r"$\delta$ = 1.41"), (1.91, r"$\delta$ = 1.91")]:
        if lv[0] <= est <= lv[-1]:
            xi = float(np.interp(est, lv, x))
            ax.axvline(xi, color="#8a6d3b", lw=1.0, ls="--", alpha=0.8)
            ax.text(xi, 1.045, lab, fontsize=7.4, color="#8a6d3b", ha="center")

    a, b = d["delta_interval_cm"]
    if a is not None and b is not None:
        ax.axvspan(float(np.interp(a, lv, x)), float(np.interp(b, lv, x)),
                   color="#2e8b57", alpha=0.10, lw=0, zorder=0)
        ax.text(0.5, 0.06,
                f"score survives {a:g} cm, breaks by {b:g} cm",
                transform=ax.transAxes, ha="center", fontsize=8.2,
                color="#2e8b57", fontweight="bold")
    elif not d["declines"]:
        ax.text(0.5, 0.06, "no decline at any displacement",
                transform=ax.transAxes, ha="center", fontsize=8.4,
                color="#c0392b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{j:g}" for j in lv])
    ax.set_xlabel("displacement of the grasp constant (cm)", fontsize=9)
    ax.set_ylim(-0.04, 1.13)
    ax.set_title(NAMES.get(t, f"task {t}"), fontsize=10.2, fontweight="bold",
                 color=c, loc="left")
    ax.grid(color="#eef1f4", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].set_ylabel("success rate", fontsize=9.4)

fig.suptitle("Displacing the lookup constant: the dose-response the "
             "admissibility argument needs", fontsize=11.4, y=1.02)
fig.text(0.5, -0.10,
         "Shaded band is the Wilson 95% interval; the green span brackets the tolerance, between the largest displacement the score survives and the smallest that "
         "breaks it (Holm-corrected paired tests against r = 0). Each trial is pushed along a ray fixed by its own index, so the four magnitudes are paired in "
         "direction as well as in initial state. Dashed lines are the two estimates of $\\delta$ this paper already carries, measured on the container rather than on the grocery.",
         ha="center", fontsize=8.4, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
for t in tasks:
    d = A[t]
    print(f"  task {t}: " + "  ".join(
        f"r={j:g}:{d['cells'][f'{j:g}']['k']}/{d['cells'][f'{j:g}']['n']}"
        for j in d["levels_cm"]) + f"   bracket {d['delta_interval_cm']}")
