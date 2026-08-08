"""F6 --- the place head's command->location map, drawn on the table.

Layer 4 is currently an argument made of numbers in a paragraph. It is really a
picture: the released head, asked for three different objects, emits three place
points that are 14.5 and 13.0 cm apart, while the basket it is supposed to be
aiming at never moves more than 0.4 cm across the whole suite. A grounded place
head would emit one point per scene. This one emits one point per command.

Right panel: the same measurement after the command-coverage repair, where the
spread collapses to 0.78 cm --- the repair that demonstrably works on the
quantity we can measure directly, whatever the swap cell later says about
behaviour.

Numbers are the ones in the manuscript (App. D / S7); this script draws them and
does not recompute them, so the figure and the text cannot drift apart.

Run: python paper/render_fig6_placepoint.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "visuals" / "F6_place_point_map.png"

#: Basket (true destination) and the head's emitted place point per command,
#: in cm relative to the basket centre. Radii are the manuscript's measured
#: errors; the directions are illustrative and the caption says so.
BASKET_JITTER = 0.40          # max basket motion across all ten tasks, cm
RELEASED = {"alphabet soup\n(trained)": 0.36, "butter": 14.5, "cream cheese": 13.0}
COVERAGE_SPREAD = 0.78        # place-point spread after command coverage, cm
RELEASED_SPREAD = 14.15

ANG = {"alphabet soup\n(trained)": 20.0, "butter": 200.0, "cream cheese": 300.0}
COL = {"alphabet soup\n(trained)": "#2a9d3f", "butter": "#c1121f",
       "cream cheese": "#e08e0b"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 5.2))

for ax, title in ((axA, "A  Released head: one point per COMMAND"),
                  (axB, "B  After command coverage: one point per SCENE")):
    ax.set_aspect("equal")
    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_xlabel("cm from basket centre")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    # the basket: the thing a grounded head would aim at, and it barely moves
    ax.add_patch(plt.Circle((0, 0), 6.0, fill=False, lw=2.0, color="#333333"))
    ax.add_patch(plt.Circle((0, 0), BASKET_JITTER, color="#333333", zorder=5))
    ax.text(0, 7.0, "basket (moves $\\leq$0.40 cm\nacross all ten tasks)",
            ha="center", fontsize=7.8, color="#333333")
axA.set_ylabel("cm from basket centre")

# ---- A: released head -----------------------------------------------------
for name, r in RELEASED.items():
    th = np.deg2rad(ANG[name])
    x, y = r * np.cos(th), r * np.sin(th)
    axA.annotate("", xy=(x, y), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color=COL[name]))
    axA.plot([x], [y], "o", ms=9, color=COL[name], zorder=6)
    axA.text(x * 1.06, y * 1.06 + (1.4 if y >= 0 else -2.6), f"{name}\n{r:.2f} cm",
             ha="center", fontsize=8.0, color=COL[name], linespacing=1.25)
axA.text(-17, -16.6, f"spread across commands: {RELEASED_SPREAD:.2f} cm\n"
                     "14 cm is enough to release a grocery over open table",
         fontsize=8.4, color="#c1121f", linespacing=1.3)

# ---- B: coverage head -----------------------------------------------------
rng = np.random.default_rng(0)
for k, (name, _r) in enumerate(RELEASED.items()):
    th = np.deg2rad(ANG[name])
    r = COVERAGE_SPREAD * (0.5 + 0.5 * k / 2)
    x, y = r * np.cos(th), r * np.sin(th)
    axB.plot([x], [y], "o", ms=9, color=COL[name], zorder=6)
axB.text(-17, -16.6, f"spread across commands: {COVERAGE_SPREAD:.2f} cm\n"
                     "inside the basket, for every command",
         fontsize=8.4, color="#2a9d3f", linespacing=1.3)
h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8) for c in COL.values()]
axB.legend(h, list(COL), frameon=False, fontsize=7.8, loc="upper right")

fig.suptitle("F6 — The place head learned command$\\rightarrow$location, not basket. "
             "All ten tasks share one basket, so this was invisible until the instruction changed.",
             fontsize=10.8, y=1.00)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
