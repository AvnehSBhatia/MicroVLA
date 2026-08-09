"""Figure 8 -- what the blind constant's failures are actually made of.

An earlier version of this figure argued that the diameter predicts, sub-goal by
sub-goal, where lookup suffices: task 0's target is pinned ($D=0$) so the reach
succeeds, its basket is not ($D=3.37$\\,cm) so the place fails. Then task 3
came back $10/10$ with a LARGER target diameter ($2.34$, the suite's largest)
and a LARGER basket diameter ($3.93$, also the suite's largest). The clean
story was wrong, and this figure reports what survives it.

What survives is stronger, because it is measured per trial rather than argued
per task:

  A  On task 0, which trials fail is predicted by how far that state's basket
     sits from the constant the blind arm was given. The four failures are the
     four largest displacements, one rank inversion; exact Mann-Whitney
     p = 0.019.
  B  On task 3, the same quantity reaches 1.91 cm -- larger than task 0's
     smallest FAILURE -- and every trial succeeds.

So the place sub-goal has a tolerance, the tolerance explains the failures
within a task, and it is not the same number across tasks. That is a direct
measurement of where assumption A1 holds and where it stops.

Sources:
    results/blind_failure_attribution.json
    results/suite_forensics_joints.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F8_subgoal.png"

FAIL = "#c0392b"
OK = "#2e8b57"
GREY = "#9aa5b1"

A = json.loads((REPO / "results/blind_failure_attribution.json").read_text())
t0, t3 = A["task0"], A["task3"]

# Rebuild task 0's per-trial series in trial order (trial i replays state 10+i).
succ_set = sorted(t0["success_disp_cm"])
fail_set = sorted(t0["failure_disp_cm"])
J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
tasks = J["suites"]["libero_object"]["tasks"]


def disp(idx: int) -> np.ndarray:
    b = next(o for o in tasks[idx]["objects"] if "basket" in o["object"])
    xy = np.asarray(b["xy_per_state_m"], dtype=float)
    return np.linalg.norm(xy - xy.mean(axis=0), axis=1) * 100


d0 = disp(0)[10:20]
is_fail0 = np.array([round(float(v), 4) in
                     [round(x, 4) for x in t0["failure_disp_cm"]] for v in d0])
d3 = np.asarray(t3["disp_cm"], dtype=float)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 3.25), sharey=True)

for ax, d, fail, title, sub in [
    (axA, d0, is_fail0,
     "A   Task 0: the failures are the largest basket errors",
     "exact Mann-Whitney $p$ = %.3f" % t0["p_two_sided"]),
    (axB, d3, np.zeros(10, bool),
     "B   Task 3: bigger errors, no failures",
     "every trial succeeded, up to %.2f cm" % t3["max_tolerated_cm"]),
]:
    x = np.arange(10)
    ax.bar(x, d, width=0.66, color=[FAIL if f else OK for f in fail],
           edgecolor="white", linewidth=0.6)
    for i, (v, f) in enumerate(zip(d, fail)):
        ax.text(i, v + 0.055, "✗" if f else "✓", ha="center", fontsize=10,
                color=FAIL if f else OK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(10 + i) for i in x], fontsize=8.2)
    ax.set_xlabel("shipped initial state", fontsize=9.4)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.4)
    ax.text(0.5, 0.955, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.4, color="#43505c")
    ax.set_ylim(0, 2.42)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eceff1", lw=0.8)
    ax.set_axisbelow(True)

axA.set_ylabel("basket distance from the\nblind constant (cm)", fontsize=9.0)

# The line that makes the two panels speak to each other.
worst_ok_A = float(max(d0[~is_fail0]))
axA.axhline(worst_ok_A, color="#5b6670", ls=":", lw=1.0)
axB.axhline(worst_ok_A, color="#5b6670", ls=":", lw=1.0)
axB.text(9.4, worst_ok_A + 0.05, "task 0's largest success", fontsize=7.4,
         ha="right", color="#5b6670")
axB.text(0.5, 1.98, "four of task 3's ten states exceed it", fontsize=7.6,
         ha="left", color=OK, fontweight="bold")

handles = [plt.Rectangle((0, 0), 1, 1, color=OK),
           plt.Rectangle((0, 0), 1, 1, color=FAIL)]
axA.legend(handles, ["episode succeeded", "episode failed"], fontsize=7.6,
           frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90))

fig.text(0.5, -0.09,
         "The blind arm is given one basket position per task, read from the shipped files. Within task 0 the error in that constant "
         "predicts which trials fail; across tasks the tolerance is not the same number — which is assumption A1 being measured rather than assumed.",
         ha="center", fontsize=8.6, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
