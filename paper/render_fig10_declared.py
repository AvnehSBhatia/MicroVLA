"""Figure 10 -- the region each LIBERO task declares, against what it ships.

This is the paper's one tolerance-free measurement, so it gets a figure that a
reader can check by eye against the numbers in the caption.

Left: LIBERO-Object task 0. The BDDL declares a 5 x 5 cm sampling box for the
alphabet soup. All fifty shipped states sit at one point in the middle of it.

Right: LIBERO-Long, same generator, an identical 5 x 5 cm declaration, drawn on
the same scale. The fifty states fill it.

The point is the contrast, and it is why this claim survived three rounds of
review that dismantled the tolerance-conditional ones: nothing here depends on
delta, a controller, a policy or a simulator.

Sources:
    .libero_src/.../bddl_files/<suite>/<task>.bddl   (the declaration)
    results/suite_forensics_joints.json              (the shipped states)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
BDDL = REPO / ".libero_src/libero/libero/bddl_files"
OUT = REPO / "paper" / "visuals" / "F10_declared_vs_shipped.png"

PIN = "#c0392b"
OK = "#2e8b57"


def declared_box(suite: str, task: str, target: str):
    txt = (BDDL / suite / f"{task}.bddl").read_text()
    init = re.search(r"\(:init(.*?)\n\s*\)", txt, re.S).group(1)
    rname = dict(re.findall(r"\(On\s+(\S+)\s+(\S+)\)", init))[target]
    for m in re.finditer(
            r"\((\w+)\s*\(:target[^)]*\)\s*\(:ranges\s*\(\s*\(([^)]*)\)", txt):
        if rname.endswith(m.group(1)):
            x0, y0, x1, y1 = (float(v) for v in m.group(2).split())
            return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
    return None


J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())["suites"]
PANELS = [("libero_object", 0, "A   LIBERO-Object, task 0", PIN),
          ("libero_10", 0, "B   LIBERO-Long, task 0 — same declared box", OK)]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
for ax, (suite, idx, title, col) in zip(axes, PANELS):
    t = J[suite]["tasks"][idx]
    tgt = t["primary_target"]
    box = declared_box(suite, t["task"], tgt)
    xy = np.asarray(next(o for o in t["objects"]
                         if o["object"] == tgt)["xy_per_state_m"], dtype=float)
    cx, cy = xy.mean(axis=0)
    # everything in cm, centred on the declared box so both panels share a scale
    bx, by, bw, bh = box
    ax.add_patch(Rectangle(((bx - cx) * 100, (by - cy) * 100), bw * 100, bh * 100,
                           facecolor="#f2d8a8", edgecolor="#8a6d3b",
                           lw=1.4, alpha=0.55, zorder=1))
    ax.scatter((xy[:, 0] - cx) * 100, (xy[:, 1] - cy) * 100, s=34, color=col,
               edgecolor="white", linewidth=0.6, zorder=3, alpha=0.9)
    spread = (xy.max(axis=0) - xy.min(axis=0)) * 100
    ax.text(0.5, 0.965,
            f"declared {bw*100:.0f} × {bh*100:.0f} cm   ·   "
            f"shipped {spread[0]:.2f} × {spread[1]:.2f} cm",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color="#43505c")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.6)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal")
    ax.set_xlabel("cm from the declared box centre", fontsize=9)
    ax.grid(color="#eef1f4", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("cm", fontsize=9)
axes[0].annotate("all 50 states\nat one point", xy=(0, 0), xytext=(1.5, 2.3),
                 fontsize=8.6, color=PIN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PIN, lw=1.2))

fig.text(0.5, -0.03,
         "The shaded square is the sampling region the task's own BDDL declares; the dots are the fifty initial states the benchmark ships. "
         "Same generator, same declared box, opposite outcomes — and no tolerance, controller or policy enters this comparison.",
         ha="center", fontsize=8.6, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
