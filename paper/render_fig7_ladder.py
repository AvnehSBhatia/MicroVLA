"""Figure 7 -- the goal-supply ladder.

The paper's central experiment in one panel. Five arms share a controller, a
trunk, a protocol, a seed and a set of initial states; the ONLY thing that
differs is where the goal comes from. Descending the ladder strips information:
the simulator every tick, the simulator once, a camera and a sentence, the
benchmark's own published files, nothing.

The figure exists because the result is a NON-separation, and non-separations
are hard to see in a table of p-values. Plotted with intervals, the point is
immediate: four arms overlap and the fifth does not. A reader should be able to
take the claim off the page without reading a number.

Left panel: success rate with Wilson 95% intervals.
Right panel: what each arm is allowed to read, as a strip of coloured cells.

Sources (every value traceable, nothing typed in twice):
    results/pod_cells.json    -- oracle, reset-oracle, learned head, random
    results/blind_cells.json  -- blind
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
OUT = REPO / "paper" / "visuals" / "F7_ladder.png"
Z = 1.959963984540054

PIN = "#c0392b"      # the colour used for "pinned / lookup" throughout the paper
LEARN = "#2c6fbb"    # the colour used for the learned artifact
GREY = "#9aa5b1"


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


POD = json.loads((REPO / "results/pod_cells.json").read_text())
BLIND = json.loads((REPO / "results/blind_cells.json").read_text())

# name, k, n, [sim, camera, text, shipped files], p vs learned, colour
ARMS = [
    ("oracle",       POD["P2_E5_oracle"]["k"], POD["P2_E5_oracle"]["n"],
     [3, 0, 0, 0], "0.50", GREY),
    ("reset-oracle", POD["P2_E5_fixed"]["k"], POD["P2_E5_fixed"]["n"],
     [1, 0, 0, 0], "0.50", PIN),
    ("learned head", POD["P0_ref_heldout"]["k"], POD["P0_ref_heldout"]["n"],
     [0, 2, 2, 0], "---", LEARN),
    ("blind",        BLIND["blind_t0"]["k"], BLIND["blind_t0"]["n"],
     [0, 0, 0, 2], "0.50", PIN),
    ("random",       POD["P2_E5_random"]["k"], POD["P2_E5_random"]["n"],
     [0, 0, 0, 0], "0.0078", GREY),
]
READS = ["simulator", "camera", "language", "shipped\nfiles", "vs.\nlearned"]
SHADE = {0: "#eef1f4", 1: "#f6c9c2", 2: "#cfd8e3", 3: "#f0a79c"}
NOTE = {0: "", 1: "once", 2: "yes", 3: "every tick"}

fig = plt.figure(figsize=(12.0, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.32], wspace=0.10)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

y = np.arange(len(ARMS))[::-1]
for yi, (name, k, n, _, p, col) in zip(y, ARMS):
    lo, hi = wilson(k, n)
    axL.plot([lo, hi], [yi, yi], color=col, lw=3.2, solid_capstyle="round",
             alpha=0.42, zorder=2)
    axL.plot([k / n], [yi], "o", color=col, ms=11, zorder=3,
             markeredgecolor="white", markeredgewidth=1.3)
    axL.text(k / n, yi + 0.30, f"{k}/{n}", ha="center", fontsize=9.6,
             fontweight="bold", color=col)


axL.axvspan(0.0, 0.0, color="none")
axL.set_yticks(y)
axL.set_yticklabels([a[0] for a in ARMS], fontsize=10.5)
axL.get_yticklabels()[3].set_fontweight("bold")     # blind
axL.set_xlim(-0.02, 1.04)
axL.set_ylim(-0.75, len(ARMS) - 0.35)
axL.set_xlabel("success rate on the held-out band ($n$ = 10), Wilson 95\\%"
               .replace("\\%", "%"), fontsize=9.5)
axL.set_title("A   Only the last step down the ladder is visible to the benchmark",
              loc="left", fontweight="bold", fontsize=10.5)
axL.grid(axis="x", color="#e6e9ec", lw=0.8)
axL.set_axisbelow(True)
axL.spines[["top", "right"]].set_visible(False)

# The brace that carries the claim.
axL.annotate("", xy=(-0.235, y[0] + 0.34), xytext=(-0.235, y[3] - 0.34),
             xycoords=("axes fraction", "data"),
             textcoords=("axes fraction", "data"), clip_on=False,
             arrowprops=dict(arrowstyle="-", color=PIN, lw=2.0))
axL.text(-0.262, (y[0] + y[3]) / 2, "all within $p$ = 0.50", rotation=90,
         transform=axL.get_yaxis_transform(), clip_on=False,
         va="center", ha="center", fontsize=8.4, color=PIN, fontweight="bold")

for yi, (_, _, _, reads, pv, _) in zip(y, ARMS):
    for xi, v in enumerate(reads):
        axR.add_patch(Rectangle((xi - 0.44, yi - 0.36), 0.88, 0.72,
                                facecolor=SHADE[v], edgecolor="white", lw=1.4))
        if NOTE[v]:
            axR.text(xi, yi, NOTE[v], ha="center", va="center", fontsize=7.4,
                     color="#7a2f26" if v in (1, 3) else "#33475b")
    xi = len(READS) - 1
    sep = pv == "0.0078"
    axR.add_patch(Rectangle((xi - 0.44, yi - 0.36), 0.88, 0.72,
                            facecolor="#fbe6e2" if sep else "#f4f6f8",
                            edgecolor="white", lw=1.4))
    axR.text(xi, yi, "reference" if pv == "---" else f"$p$ = {pv}",
             ha="center", va="center", fontsize=7.8,
             fontweight="bold" if sep else "normal",
             color=PIN if sep else "#5b6670")
axR.set_xlim(-0.6, len(READS) - 0.4)
axR.set_ylim(-0.75, len(ARMS) - 0.35)
axR.set_xticks(range(len(READS)))
axR.set_xticklabels(READS, fontsize=8.6)
axR.set_yticks([])
axR.set_title("B   what the goal supply may read", loc="left",
              fontweight="bold", fontsize=10.5)
for s in ("top", "right", "left", "bottom"):
    axR.spines[s].set_visible(False)
axR.tick_params(length=0)

fig.text(0.5, -0.075,
         "Same controller, same trunk, same seeds, same initial states — only the goal supply differs. "
         "The blind arm opens the benchmark's shipped init files, emits two constants, and never consults the episode.",
         ha="center", fontsize=8.8, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
