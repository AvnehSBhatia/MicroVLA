"""Figure 11 -- the criterion, drawn.

Definition 1 says: R(o) is the radius of the smallest disc containing every
position the benchmark ships for object o. Proposition 1 says a single constant
serves o if and only if R <= delta. Both are one picture.

For one task per suite we draw the fifty shipped positions, the minimum
enclosing circle (whose radius IS R, and whose centre is the constant the proof
constructs), and a disc of radius delta centred on it. The criterion is then
visible rather than asserted: the shipped points fit inside the delta-disc
exactly when R <= delta.

Drawn on a common scale so the four suites can be compared by eye.

Sources:
    results/suite_forensics_joints.json  (the shipped positions)
    results/admissibility.json           (R, recomputed here as a cross-check)
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F11_radius_criterion.png"
DELTA = 1.4          # cm, the tolerance measured in sec:refute
PIN, MISS = "#c0392b", "#2c6fbb"


def mec(P):
    """Exact minimum enclosing circle: (centre, radius). Same routine as
    scripts/admissibility.py, repeated here so the figure is self-contained."""
    P = np.unique(np.asarray(P, dtype=float), axis=0)
    if len(P) == 1:
        return P[0], 0.0
    best = None
    for a, b in combinations(range(len(P)), 2):
        c = (P[a] + P[b]) / 2
        r = float(np.linalg.norm(P[a] - c))
        if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-12):
            if best is None or r < best[1]:
                best = (c, r)
    for a, b, d in combinations(range(len(P)), 3):
        A, B, C = P[a], P[b], P[d]
        den = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
        if abs(den) < 1e-15:
            continue
        ux = ((A@A)*(B[1]-C[1]) + (B@B)*(C[1]-A[1]) + (C@C)*(A[1]-B[1]))/den
        uy = ((A@A)*(C[0]-B[0]) + (B@B)*(A[0]-C[0]) + (C@C)*(B[0]-A[0]))/den
        c = np.array([ux, uy]); r = float(np.linalg.norm(A - c))
        if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-12):
            if best is None or r < best[1]:
                best = (c, r)
    return best


J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())["suites"]

# one representative task per suite: the suite's median-radius task, so the
# picture is typical rather than chosen.
PANELS = []
for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
    cands = []
    for t in J[suite]["tasks"]:
        o = next((o for o in t["objects"] if o["object"] == t["primary_target"]), None)
        if o and o.get("xy_per_state_m"):
            xy = np.asarray(o["xy_per_state_m"], dtype=float)
            cands.append((mec(xy)[1] * 100, t["task"], xy))
    cands.sort()
    PANELS.append((suite,) + cands[len(cands) // 2])

fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
for ax, (suite, R, task, xy) in zip(axes, PANELS):
    c, r = mec(xy)
    ok = R <= DELTA
    col = PIN if ok else MISS
    dx = (xy[:, 0] - c[0]) * 100
    dy = (xy[:, 1] - c[1]) * 100
    ax.add_patch(Circle((0, 0), DELTA, facecolor="#f2d8a8", alpha=0.5,
                        edgecolor="#8a6d3b", lw=1.2, ls="--", zorder=1))
    ax.add_patch(Circle((0, 0), R, facecolor="none", edgecolor=col, lw=1.8,
                        zorder=2))
    ax.scatter(dx, dy, s=26, color=col, edgecolor="white", linewidth=0.5,
               zorder=3, alpha=0.9)
    ax.plot([0], [0], "+", color="black", ms=9, mew=1.6, zorder=4)
    ax.annotate("", xy=(R * np.cos(np.deg2rad(35)), R * np.sin(np.deg2rad(35))),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax.text(R * 0.55 * np.cos(np.deg2rad(35)) - 0.15,
            R * 0.55 * np.sin(np.deg2rad(35)) + 0.22,
            f"$R$ = {R:.2f}", color=col, fontsize=9, fontweight="bold")
    ax.set_title(f"{suite.replace('libero_', 'LIBERO-').replace('10', 'Long')}",
                 loc="left", fontweight="bold", fontsize=10.4)
    ax.text(0.5, 0.02,
            r"$R \leq \delta$: one constant suffices" if ok
            else r"$R > \delta$: no constant suffices",
            transform=ax.transAxes, ha="center", fontsize=8.6,
            color=col, fontweight="bold")
    lim = 5.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.grid(color="#eef1f4", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("cm from the constant", fontsize=9)
axes[0].text(-4.6, 4.4, f"shaded disc: $\\delta$ = {DELTA} cm", fontsize=8,
             color="#8a6d3b")
for ax in axes:
    ax.set_xlabel("cm", fontsize=9)

fig.text(0.5, -0.045,
         "Each panel is one task, at its suite's median radius. Dots are the fifty shipped target positions; the solid circle is the smallest disc containing them, "
         "so its radius is $R$ and its centre is the constant Proposition 1 constructs. The criterion $R \\leq \\delta$ is then visible: the points fit the shaded disc, or they do not.",
         ha="center", fontsize=8.6, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
for suite, R, task, _ in PANELS:
    print(f"  {suite:<16} R = {R:.3f} cm   {task[:44]}")
