"""F5 — how the third object crossed: drift precedes error, and repairing it works.

Three panels, all measured (2026-08-06, artifact-verified; see paper.md):

  A  appearance drift by tick band. The blocked object's deployment embeddings
     sit furthest from its training corpus at the START of the episode and the
     gap SHRINKS — the opposite of what a consequence of a bad goal would do,
     which is the circularity check that two earlier instruments failed.
  B  the miss decomposed. The blocked object's height is BETTER than the
     crossing object's; its error is 94% lateral.
  C  the intervention, n=50 both arms, same seeds, Wilson 95% CIs.

Run: python paper/render_fig5_third_object.py  (writes paper/visuals/F5_*.png)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "visuals" / "F5_third_object.png"

# ---- measured values (paper.md, 2026-08-06) --------------------------------
BANDS = ["0-100", "100-300", "300+"]
DRIFT = {                      # NN-cosine gap vs each object's own corpus
    "alphabet soup (crosses)":  [0.0050, 0.0099, 0.0107],
    "butter (crosses)":         [-0.0031, -0.0036, 0.0055],
    "cream cheese (blocked)":   [0.0441, 0.0363, 0.0156],
}
MISS = {                       # closest-approach residual, metres
    "alphabet soup\n(crosses)": (0.0186, 0.0190),   # lateral, vertical
    "cream cheese\n(blocked)":  (0.0688, 0.0040),
}
CELLS = [                      # label, successes, n
    ("cream\nv8 control", 3, 50),
    ("cream\n+DAgger r1", 18, 50),
    ("cream\n+DAgger r2", 22, 50),
    ("soup\n(reference)", 35, 50),
]
COL = {"soup": "#2a6f97", "butter": "#61a5c2", "cream": "#c1121f",
       "grey": "#6c757d", "ok": "#2a9d8f"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.0))

# --- A: drift precedes the error ---
x = np.arange(len(BANDS))
for (lab, ys), c in zip(DRIFT.items(), [COL["soup"], COL["butter"], COL["cream"]]):
    axA.plot(x, ys, "o-", color=c, lw=2.2, ms=7, label=lab)
axA.axhline(0, color="k", lw=0.8, alpha=0.4)
axA.set_xticks(x); axA.set_xticklabels(BANDS)
axA.set_xlabel("episode tick band")
axA.set_ylabel("appearance gap vs own corpus\n(NN-cosine, higher = further off-manifold)")
axA.set_title("A  Drift precedes the error", loc="left", fontweight="bold")
axA.legend(frameon=False, fontsize=8.5, loc="upper right")
axA.set_ylim(-0.008, 0.052)
axA.annotate("largest before\nthe arm has acted",
             xy=(0.04, 0.0435), xytext=(0.55, 0.030), fontsize=8.5, color=COL["cream"],
             arrowprops=dict(arrowstyle="->", color=COL["cream"], lw=1.2))
axA.spines[["top", "right"]].set_visible(False)

# --- B: the miss is lateral, not vertical ---
labels = list(MISS)
lat = [MISS[k][0] * 100 for k in labels]
ver = [MISS[k][1] * 100 for k in labels]
xb = np.arange(len(labels)); w = 0.34
axB.bar(xb - w / 2, lat, w, label="lateral (xy)", color=COL["cream"])
axB.bar(xb + w / 2, ver, w, label="vertical", color=COL["grey"])
axB.set_xticks(xb); axB.set_xticklabels(labels, fontsize=9)
axB.set_ylabel("closest-approach residual (cm)")
axB.set_title("B  The miss is lateral, not depth", loc="left", fontweight="bold")
axB.legend(frameon=False, fontsize=8.5)
axB.set_ylim(0, 8.4)
axB.annotate("94% lateral", xy=(1 - w / 2, 6.88), xytext=(1 - w / 2, 7.25), fontsize=9,
             color=COL["cream"], fontweight="bold", ha="center")
axB.annotate("height is BETTER\nthan the object that works",
             xy=(1 + w / 2, 0.45), xytext=(0.30, 3.6), fontsize=8,
             color=COL["grey"], arrowprops=dict(arrowstyle="->", color=COL["grey"], lw=1.1))
axB.spines[["top", "right"]].set_visible(False)

# --- C: the intervention ---
xs = np.arange(len(CELLS))
rates = [k / n for _, k, n in CELLS]
los = [wilson(k, n)[0] for _, k, n in CELLS]
his = [wilson(k, n)[1] for _, k, n in CELLS]
cols = [COL["grey"], COL["ok"], COL["ok"], COL["soup"]]
axC.errorbar(xs, rates, yerr=[np.array(rates) - np.array(los),
                              np.array(his) - np.array(rates)],
             fmt="o", ms=9, capsize=6, lw=2, color="k", ecolor="k", zorder=3)
for xi, r, c in zip(xs, rates, cols):
    axC.plot([xi], [r], "o", ms=9, color=c, zorder=4)
for xi, (lab, k, n) in zip(xs, CELLS):
    axC.annotate(f"{k}/{n}", (xi, k / n), textcoords="offset points",
                 xytext=(12, -3), fontsize=9)
axC.set_xticks(xs); axC.set_xticklabels([c[0] for c in CELLS], fontsize=8)
axC.set_ylim(0, 0.9); axC.set_ylabel("held-out success rate (Wilson 95%)")
axC.set_title("C  Repairing the drift crosses it", loc="left", fontweight="bold")
axC.annotate("vs control, paired exact McNemar\nr1 $p{=}10^{-4}$ (15/15) · r2 $p{<}10^{-4}$ (19/19)\nr2 vs r1 n.s. ($p{=}0.39$)",
             xy=(1.0, 0.70), fontsize=8, ha="center", color=COL["ok"])
axC.spines[["top", "right"]].set_visible(False)

fig.suptitle("Third object crossed: appearance drift precedes the grasp error, and repairing it works",
             fontsize=11.5, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
