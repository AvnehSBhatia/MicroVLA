"""F4 --- the probe-only counterexample, promoted out of an annotation box.

Two heads trained the same way on the same 10-episode corpus. An on-manifold
substitution probe cannot tell them apart --- their attribution profiles agree
to within 0.73 cm on every channel. Deployed, one scores 0.000 and the other
0.700. That is the whole argument against certifying a policy from probe
evidence alone, and it previously lived inside a text box drawn on another
figure.

Left: the profiles, measured by ``scripts/attribution_profiles.py`` (which did
not exist when the original figure shipped --- regenerating it is also the fix
for that reproducibility gap). Right: what the two heads actually do.

Run: python paper/render_fig4_counterexample.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F4_probe_counterexample.png"
D = json.loads((REPO / "results/attribution_profiles.json").read_text())

PAIR = ["v2", "v2.1"]
DEPLOYED = {"v2": 0.000, "v2.1": 0.700}     # same corpus, same recipe
COL = {"v2": "#c1121f", "v2.1": "#2a6f97"}
CH = D["channels"]
LABEL = {"uv": "box centre\n(uv)", "proprio": "proprioception",
         "box_emb": "box\nembedding", "frame_emb": "frame\nembedding"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.3),
                               gridspec_kw={"width_ratios": [2.0, 1.0]})

# ---- left: the probe cannot separate them --------------------------------
x = np.arange(len(CH))
wid = 0.36
for k, h in enumerate(PAIR):
    vals = [D["heads"][h][c] for c in CH]
    axL.bar(x + (k - 0.5) * wid, vals, wid, label=f"{h}  (deployed {DEPLOYED[h]:.3f})",
            color=COL[h], edgecolor="black", linewidth=0.7)
    for xi, v in zip(x + (k - 0.5) * wid, vals):
        axL.text(xi, v + 0.15, f"{v:.2f}", ha="center", fontsize=7.4)
axL.set_xticks(x)
axL.set_xticklabels([LABEL[c] for c in CH], fontsize=8.6)
axL.set_ylabel("grasp-offset movement under\nchannel substitution (cm)")
axL.set_title("A  The probe cannot tell them apart", loc="left",
              fontweight="bold", fontsize=11)
axL.legend(frameon=False, fontsize=8.6, loc="upper left")
axL.spines[["top", "right"]].set_visible(False)
axL.set_ylim(0, max(max(D["heads"][h][c] for c in CH) for h in PAIR) * 1.32)
axL.annotate(f"largest disagreement on any channel: "
             f"{D['v2_vs_v21_max_channel_diff_cm']:.2f} cm",
             xy=(0.5, 0.86), xycoords="axes fraction", ha="center", fontsize=8.8,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#e08e0b"))

# ---- right: deployment does ----------------------------------------------
for k, h in enumerate(PAIR):
    axR.bar(k, DEPLOYED[h], 0.55, color=COL[h], edgecolor="black", linewidth=0.7)
    axR.text(k, DEPLOYED[h] + 0.03, f"{DEPLOYED[h]:.3f}", ha="center",
             fontsize=11, fontweight="bold")
axR.set_xticks([0, 1])
axR.set_xticklabels(PAIR, fontsize=10)
axR.set_ylim(0, 0.85)
axR.set_ylabel("deployed success (held-out)")
axR.set_title("B  Deployment does", loc="left", fontweight="bold", fontsize=11)
axR.spines[["top", "right"]].set_visible(False)

fig.suptitle("F4 — Matched probe profiles, opposite deployed behaviour: "
             "an on-manifold probe cannot certify off-manifold behaviour",
             fontsize=11.5, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
