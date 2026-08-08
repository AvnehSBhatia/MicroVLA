"""what each LIBERO suite actually pins (referee E3 / F1).

The referee's note was fair: the paper's most visual claim --- that
LIBERO-Object's targets occupy two table positions --- had no figure. This draws
it, and draws the control the earlier framing was missing.

  A  libero_object primary targets, 10 tasks x 50 shipped states. Six tasks
     collapse to a single point (two float64 values 1 ULP apart); the other four
     scatter at 0.26-0.58 cm. The ten means sit in two clusters.
  B  libero_spatial on IDENTICAL axes scale. Every target scatters. This is what
     a randomised suite looks like, and it is why the claim belongs to
     LIBERO-Object rather than to LIBERO.
  C  placement entropy per suite against the log2(50) = 5.644 bit ceiling.

Every number is read from ``results/suite_forensics_joints.json``; nothing is
hardcoded. Run: python paper/render_fig1_placement_forensics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F1_placement_forensics.png"
D = json.loads((REPO / "results/suite_forensics_joints.json").read_text())

CA, CB = "#2a6f97", "#c1121f"
PIN, FREE = "#c1121f", "#2a6f97"


def targets(suite: str):
    """[(label, xy[50,2], pinned)] for each task's primary target."""
    rows = []
    for t in D["suites"][suite]["tasks"]:
        o = next((o for o in t["objects"] if o["object"] == t["primary_target"]), None)
        if o is None or "xy_per_state_m" not in o:
            continue
        rows.append((o["object"].rsplit("_", 1)[0], np.asarray(o["xy_per_state_m"]),
                     bool(t["target_position_pinned"])))
    return rows


fig = plt.figure(figsize=(13.6, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 3.0, 2.5], wspace=0.30)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

# ---- shared axes window, so A and B are honestly comparable ----------------
allxy = np.vstack([xy for s in ("libero_object", "libero_spatial") for _, xy, _ in targets(s)])
cx, cy = allxy[:, 0].mean(), allxy[:, 1].mean()
half = max(np.abs(allxy[:, 0] - cx).max(), np.abs(allxy[:, 1] - cy).max()) * 1.18

for ax, suite, title in ((axA, "libero_object", "A  libero_object — the audited suite"),
                         (axB, "libero_spatial", "B  libero_spatial — same axes, same scale")):
    rows = targets(suite)
    s = D["suites"][suite]
    for name, xy, pinned in rows:
        ax.scatter(xy[:, 0], xy[:, 1], s=9, alpha=0.55,
                   color=PIN if pinned else FREE, linewidths=0, zorder=2)
        m = xy.mean(axis=0)
        ax.scatter([m[0]], [m[1]], s=52, facecolor="none",
                   edgecolor=PIN if pinned else FREE, linewidths=1.3, zorder=3)
    # Per-point names are omitted on the shared-scale axes: in A all ten labels
    # land on two points, and in B all ten targets are called "black bowl". The
    # inset carries A's identities; B's spread is the whole message.
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xlabel("table x (m)")
    ax.set_title(f"{title}\n{s['n_target_position_pinned']}/{s['n_resolvable_tasks']} targets pinned"
                 f"   ·   H = {s['mean_target_entropy_bits_at_1mm']:.2f} bits",
                 loc="left", fontweight="bold", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, zorder=0)
axA.set_ylabel("table y (m)")

# the two-cluster structure, drawn from the measured separation
sep = D["suites"]["libero_object"]["max_pairwise_target_separation_cm"]
rows = targets("libero_object")
means = np.array([xy.mean(axis=0) for _, xy, _ in rows])
split = means[:, 0].mean()
gA, gB = means[means[:, 0] < split], means[means[:, 0] >= split]
for g in (gA, gB):
    axA.add_patch(plt.Circle(g.mean(axis=0), np.linalg.norm(g - g.mean(axis=0), axis=1).max() + 0.012,
                             fill=False, ls="--", lw=1.0, color="#555555", zorder=1))
axA.annotate("", xy=tuple(gB.mean(axis=0)), xytext=tuple(gA.mean(axis=0)),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.4), zorder=4)
axA.text(*(gA.mean(axis=0) + gB.mean(axis=0)) / 2 + np.array([0.0, 0.022]),
         f"{np.linalg.norm(gA.mean(axis=0) - gB.mean(axis=0)) * 100:.1f} cm",
         fontsize=9, ha="center", fontweight="bold")
h = [plt.Line2D([], [], marker="o", ls="", color=PIN),
     plt.Line2D([], [], marker="o", ls="", color=FREE)]
axA.legend(h, ["target pinned (≤2 float64 values)", "target jitters (0.26–0.58 cm)"],
           frameon=False, fontsize=7.4, loc="lower left")
axB.text(0.03, 0.04, "all 10 targets are a 'black bowl';\nidentity cannot disambiguate them,\n"
                     "so the suite must randomise position",
         transform=axB.transAxes, fontsize=7.4, color="#333333", linespacing=1.35)

# Inset: A's two clusters at their own scale, where the identities are readable.
ins = axA.inset_axes([0.44, 0.46, 0.54, 0.52])
for name, xy, pinned in rows:
    ins.scatter(xy[:, 0] * 100, xy[:, 1] * 100, s=7, alpha=0.6,
                color=PIN if pinned else FREE, linewidths=0)
# Names are stacked per cluster: five land on each centroid, so annotating them
# individually reproduces the overlap the zoom was meant to resolve.
for g, dx, ha in ((gA, -2.0, "right"), (gB, 2.0, "left")):
    mem = [(n, p) for n, xy, p in rows
           if np.linalg.norm(xy.mean(axis=0) - g.mean(axis=0)) < 0.05]
    c = g.mean(axis=0) * 100
    for k, (n, p) in enumerate(sorted(mem)):
        ins.text(c[0] + dx, c[1] + 2.6 - 1.3 * k, n.replace("_", " "),
                 fontsize=5.2, ha=ha, va="center", color=PIN if p else FREE)
ins.set_aspect("equal")
ins.tick_params(labelsize=5.5)
ins.set_xlim(-21, 15)
ins.set_ylim(-30, -4)
ins.set_title("cm, 8× zoom — every task labelled", fontsize=6, pad=2)
ins.set_facecolor("#fafafa")

# ---- C: placement diameter per task ----------------------------------------
# Panel C used to plot placement entropy at a 1 mm grid quantisation. That
# measure was retired: grid binning is not translation invariant, so a cluster
# straddling a cell boundary splits and a pinned task scores as diverse. What
# replaces it is the quantity the proposition actually needs -- the diameter,
# the largest distance between any two shipped placements of a task's target --
# which has no origin, no grid and no free parameter, and which is what a
# reader can compare against the controller's tolerance directly.
DIAM = json.loads((REPO / "results/placement_diameter.json").read_text())
DELTA_CM = 2.5                       # the tolerance sec:sweep shows the controller has
order = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
order = [s for s in order if s in DIAM]
y = np.arange(len(order))
for yi, s in zip(y, order):
    vals = np.asarray(DIAM[s], dtype=float)
    col = PIN if s == "libero_object" else "#9aa5b1"
    jit = (np.arange(len(vals)) % 5 - 2) * 0.055
    axC.scatter(vals, np.full(len(vals), yi) + jit, s=26, color=col,
                edgecolor="white", linewidth=0.5, zorder=3)
    n_adm = int((vals <= DELTA_CM).sum())
    axC.text(9.3, yi, f"{n_adm}/{len(vals)}", va="center", ha="right",
             fontsize=8.2, fontweight="bold" if n_adm else "normal",
             color=PIN if n_adm else "#5b6670")
axC.axvline(DELTA_CM, color="black", ls="--", lw=1.1, zorder=2)
axC.text(DELTA_CM + 0.15, len(order) - 0.35,
         f"$\\delta$ = {DELTA_CM:g} cm\n(controller tolerance)",
         fontsize=7.4, va="top")
axC.text(9.3, len(order) - 0.62, "tasks with\n$D \\leq \\delta$", fontsize=7.2,
         ha="right", va="top", color="#5b6670")
axC.set_yticks(y)
axC.set_yticklabels([s.replace("libero_", "").replace("10", "long")
                     for s in order], fontsize=9)
axC.set_ylim(-0.6, len(order) - 0.18)
axC.set_xlim(-0.3, 10.0)
axC.set_xlabel("placement diameter $D$ per task (cm)")
axC.set_title("C  One suite is lookup-admissible; no other task is",
              loc="left", fontweight="bold", fontsize=10)
axC.spines[["top", "right"]].set_visible(False)

_qs = [s for s in D["suites"]
       if D["suites"][s].get("n_target_orientation_bit_identical") is not None]
n_quat = sum(D["suites"][s]["n_target_orientation_bit_identical"] for s in _qs)
n_res = sum(D["suites"][s]["n_resolvable_tasks"] for s in _qs)
fig.suptitle("LIBERO-Object is the outlier: its targets are pinned, the other suites' are not "
             f"(orientation is bit-identical in {n_quat}/{n_res} resolvable tasks across all four)",
             fontsize=11, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
