"""Figure 8 -- admissibility predicts, sub-goal by sub-goal, inside one task.

The suite-level claim ("LIBERO-Object is lookup-admissible") is a statement
about ten tasks. This figure is the sharper version: a single task carries TWO
sub-goals with two different diameters, and the blind constant's outcome splits
exactly along that line. Nothing about the policy changed between the two
sub-goals -- the same constant-driven controller did both -- so the diameter is
the only thing that distinguishes them.

Panel A: the two sub-goals on one axis, against the controller's tolerance.
Panel B: what actually happened, per trial. The reach succeeded 10/10 (2 mm);
         the place succeeded 6/10; the four failures ended at the step cap
         with the object still 8 mm from the gripper.

Sources:
    results/suite_forensics_joints.json   -- the two diameters
    results/blind_logs/blind_t0_trials.txt -- the ten per-trial DONE lines
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F8_subgoal.png"

PIN = "#c0392b"
OK = "#2e8b57"
GREY = "#9aa5b1"
DELTA_CM = 2.5

# --- the two diameters, recomputed rather than typed ------------------------
J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
task0 = J["suites"]["libero_object"]["tasks"][0]
diam = {}
for o in task0["objects"]:
    xy = np.asarray(o.get("xy_per_state_m", []), dtype=float)
    if len(xy) == 0:
        continue
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).max() * 100
    diam[o["object"]] = float(d)
D_TGT = diam["alphabet_soup_1"]
D_BSK = diam["basket_1"]

# --- the ten trials ---------------------------------------------------------
lines = (REPO / "results/blind_logs/blind_t0_trials.txt").read_text().splitlines()
trials = []
for ln in lines:
    f = dict(re.findall(r"([a-z_0-9]+)=([-\w.]+)", ln))
    trials.append({"success": f["success"] == "True",
                   "min": float(f["eef_obj_dist_min"]) * 100,
                   "final": float(f["eef_obj_dist_final"]) * 100,
                   "steps": int(f["steps"])})
assert len(trials) == 10, len(trials)

fig = plt.figure(figsize=(11.6, 3.25))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- A: the two sub-goals against the tolerance ----------------------------
axA.axvspan(-0.15, DELTA_CM, color="#fbeae7", zorder=0)
axA.axvline(DELTA_CM, color="black", ls="--", lw=1.2, zorder=2)
axA.text(DELTA_CM + 0.08, 1.60, f"$\\delta$ = {DELTA_CM:g} cm", fontsize=8.6, va="top")
axA.text(DELTA_CM - 0.10, 1.60, "lookup suffices", fontsize=8.6, va="top",
         ha="right", color=PIN, fontweight="bold")

for yi, (lbl, d, col, verdict) in enumerate([
        ("basket\n(place)", D_BSK, GREY, "randomised"),
        ("alphabet soup\n(reach)", D_TGT, PIN, "one float, 50 states")]):
    axA.plot([0, d], [yi, yi], color=col, lw=7, solid_capstyle="butt", alpha=0.5)
    axA.plot([d], [yi], "|", color=col, ms=22, mew=3.0)
    if d < 0.05:
        axA.plot([d], [yi], "o", color=col, ms=8, zorder=4,
                 markeredgecolor="white", markeredgewidth=1.2)
    axA.text(d + 0.12, yi + 0.17, f"$D$ = {d:.2f} cm", fontsize=9.4,
             fontweight="bold", color=col, va="center")
    axA.text(d + 0.12, yi - 0.19, verdict, fontsize=8.0, color="#5b6670", va="center")

axA.set_yticks([0, 1])
axA.set_yticklabels(["basket\n(place)", "alphabet soup\n(reach)"], fontsize=9.4)
axA.set_ylim(-0.62, 1.72)
axA.set_xlim(-0.15, 5.3)
axA.set_xlabel("placement diameter within ONE task (cm)", fontsize=9.4)
axA.set_title("A   One task, two sub-goals, two diameters", loc="left",
              fontweight="bold", fontsize=10.5)
axA.spines[["top", "right"]].set_visible(False)

# ---- B: what the blind constant did, per trial -----------------------------
x = np.arange(10)
axB.bar(x - 0.19, [t["min"] for t in trials], width=0.36, color=PIN,
        label="closest approach to object")
axB.bar(x + 0.19, [t["final"] for t in trials], width=0.36, color="#e8a598",
        label="object$\\to$gripper at episode end")
for i, t in enumerate(trials):
    axB.text(i, 1.88, "✓" if t["success"] else "✗", ha="center",
             fontsize=12.5, color=OK if t["success"] else GREY, fontweight="bold")
axB.axhline(0.2, color="#5b6670", lw=0.9, ls=":")
axB.text(9.45, 0.28, "2 mm", fontsize=7.8, ha="right", color="#5b6670")
axB.set_xticks(x)
axB.set_xticklabels([f"{i}" for i in x], fontsize=8.4)
axB.set_ylim(0, 2.12)
axB.set_xlim(-0.7, 9.7)
axB.set_xlabel("trial (held-out band, states 10--19)", fontsize=9.4)
axB.set_ylabel("distance (cm)", fontsize=9.0)
axB.set_title("B   The reach never failed; the place did, four times",
              loc="left", fontweight="bold", fontsize=10.5)
axB.legend(fontsize=7.6, loc="upper left", frameon=False, ncol=1,
           bbox_to_anchor=(0.005, 0.80))
axB.spines[["top", "right"]].set_visible(False)

fig.text(0.5, -0.10,
         "The same constant-driven controller performed both sub-goals. It reached the pinned object in 10/10 trials and "
         "missed the randomised basket in 4 — the split the diameter predicts, computed from shipped files before anything ran.",
         ha="center", fontsize=8.8, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}  (D_tgt={D_TGT:.4f} cm, D_basket={D_BSK:.4f} cm)")
