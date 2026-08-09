"""Figure 13 -- the defect on every task it concerns, not a chosen example.

Figure 2 shows one LIBERO-Object task beside one LIBERO-Long task. A reader is
entitled to ask whether those two were picked. This figure removes the question
by drawing all twenty: every task of both suites, on identical axes, each with
the sampling box its own BDDL declares and the fifty initial states the
benchmark ships inside it.

The two rows are the whole argument of Section 2. Every LIBERO-Long panel fills
its box; every LIBERO-Object panel does not, and six of them are a single dot.
No task was selected, nothing is averaged, and the comparison needs no
tolerance, controller, policy or simulator.

Sources:
    .libero_src/.../bddl_files/<suite>/<task>.bddl
    results/suite_forensics_joints.json
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
OUT = REPO / "paper" / "visuals" / "F13_all_tasks.png"
PIN, OK = "#c0392b", "#2e8b57"


def declared_box(suite: str, task: str, target: str):
    txt = (BDDL / suite / f"{task}.bddl").read_text()
    init = re.search(r"\(:init(.*?)\n\s*\)", txt, re.S).group(1)
    rname = dict(re.findall(r"\(On\s+(\S+)\s+(\S+)\)", init)).get(target)
    if rname is None:
        return None
    for m in re.finditer(
            r"\((\w+)\s*\(:target[^)]*\)\s*\(:ranges\s*\(\s*\(([^)]*)\)", txt):
        if rname.endswith(m.group(1)):
            x0, y0, x1, y1 = (float(v) for v in m.group(2).split())
            return abs(x1 - x0) * 100, abs(y1 - y0) * 100
    return None


J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())["suites"]
ROWS = [("libero_object", PIN, "LIBERO-Object"),
        ("libero_10", OK, "LIBERO-Long")]

fig, axes = plt.subplots(2, 10, figsize=(15.0, 3.5))
for r, (suite, col, label) in enumerate(ROWS):
    for c in range(10):
        ax = axes[r, c]
        t = J[suite]["tasks"][c]
        tgt = t["primary_target"]
        o = next((o for o in t["objects"] if o["object"] == tgt), None)
        box = declared_box(suite, t["task"], tgt)
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_color("#dfe3e8")
        if box is None or o is None or not o.get("xy_per_state_m"):
            ax.text(0, 0, "—", ha="center", va="center", color="#c3c9d0")
            continue
        bw, bh = box
        ax.add_patch(Rectangle((-bw / 2, -bh / 2), bw, bh, facecolor="#f2d8a8",
                               alpha=0.55, edgecolor="#8a6d3b", lw=0.9, zorder=1))
        xy = np.asarray(o["xy_per_state_m"], dtype=float)
        cx, cy = xy.mean(axis=0)
        ax.scatter((xy[:, 0] - cx) * 100, (xy[:, 1] - cy) * 100, s=7, color=col,
                   edgecolor="none", zorder=3, alpha=0.85)
        spread = float((xy.max(axis=0) - xy.min(axis=0)).max() * 100)
        ax.text(0.5, -0.14, f"{spread:.2f} cm", transform=ax.transAxes,
                ha="center", fontsize=7.2,
                color=col, fontweight="bold" if spread < 0.01 else "normal")
        if r == 0:
            ax.set_title(f"{c}", fontsize=8, pad=3, color="#5b6670")
    axes[r, 0].set_ylabel(label, fontsize=10, fontweight="bold", color=col,
                          labelpad=8)

fig.suptitle("Every task of both suites: the region its own task file declares "
             "(shaded), and the fifty states it ships (dots)",
             fontsize=11, y=1.04)
fig.text(0.5, -0.10,
         "Identical axes throughout; the number under each panel is the shipped spread. Every LIBERO-Long task fills its declared box. No LIBERO-Object task does, "
         "and six are a single point. Nothing is selected or averaged, and no tolerance, controller, policy or simulator enters the comparison.",
         ha="center", fontsize=8.6, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
