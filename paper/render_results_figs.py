"""Render F3 (protocol x head success) for MANUSCRIPT_v2.md.

Every plotted value is a number that appears in the S7 table / App D run
ledger and traces to a results.json. Re-run after any cell changes:
    .venv/bin/python paper/render_results_figs.py
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


HEADS = ["memorized", "flagship v5", "v2.1", "v3"]
COLORS = ["#b0b0b0", "#1f77b4", "#8fbcd4", "#c6dbef"]
# pod-stack cells (k out of 10); memorized randomized = pod control 2026-08-05
CELLS = {
    "dev (0-9)": [7, 4, 7, 7],
    "held-out (10-19)": [3, 7, 7, 6],
    "randomized ±4 cm": [1, 4, 5, 3],
}
AUDIT = {"memorized": 6, "flagship v5": 0}  # audit-stack randomized cells

fig = plt.figure(figsize=(10.5, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[3.1, 3.1, 3.1, 1.6], wspace=0.35)

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, (proto, ks) in zip(axes, CELLS.items()):
    for j, (head, k) in enumerate(zip(HEADS, ks)):
        lo, hi = wilson(k, 10)
        p = k / 10
        lw = 2.2 if head == "flagship v5" else 0.8
        ax.bar(j, p, color=COLORS[j], edgecolor="black", linewidth=lw, width=0.72)
        ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="black",
                    elinewidth=1.1, capsize=3)
        ax.text(j, 0.02, f"{k}/10", ha="center", fontsize=8)
    ax.set_title(proto, fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(HEADS, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.0, color="#cc3333", linewidth=1.2)
    if ax is axes[0]:
        ax.set_ylabel("success rate (Wilson 95%)")
        ax.text(-0.42, 0.955, "free per-tick regression: 0/56 (red rule)",
                color="#cc3333", fontsize=7, ha="left")
    ax.spines[["top", "right"]].set_visible(False)

# audit-stack inset on the randomized panel
ins = axes[2].inset_axes([0.04, 0.71, 0.38, 0.27])
for j, head in enumerate(["memorized", "flagship v5"]):
    k = AUDIT[head]
    lo, hi = wilson(k, 10)
    p = k / 10
    ins.bar(j, p, color=COLORS[HEADS.index(head)], edgecolor="black",
            linewidth=0.8, width=0.6, hatch="//")
    ins.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="black",
                 elinewidth=0.9, capsize=2)
    ins.text(j, 0.03, f"{k}/10", ha="center", fontsize=7)
ins.set_xticks([0, 1])
ins.set_xticklabels(["mem.", "v5"], fontsize=7)
ins.set_ylim(0, 1.0)
ins.set_yticks([])
ins.set_title("audit stack (hatched)", fontsize=6.8)
axes[2].annotate("rebuilt detector stack:\nseparation inverts; never\n"
                 "compared to pod bars",
                 xy=(0.99, 0.99), xycoords="axes fraction", fontsize=6.5,
                 va="top", ha="right")

# pooled panel (variance-trained heads, pod-stack cells only)
axp = fig.add_subplot(gs[0, 3])
for i, (label, k, n) in enumerate([("fixed\n38/60", 38, 60), ("rand.\n12/30", 12, 30)]):
    lo, hi = wilson(k, n)
    p = k / n
    axp.bar(i, p, color="#1f77b4", edgecolor="black", width=0.62)
    axp.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="black",
                 elinewidth=1.1, capsize=3)
    axp.text(i, 0.02, label, ha="center", fontsize=7.5, color="white")
axp.set_title("pooled\n(variance heads)", fontsize=9)
axp.set_xticks([])
axp.set_ylim(0, 1.0)
axp.spines[["top", "right"]].set_visible(False)

fig.suptitle("F3 — Protocol × head; memorized-randomized is the pod control "
             "(identical draws as the flagship bar)", fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("paper/visuals/F3_protocol_by_head.png", dpi=180)
print("wrote paper/visuals/F3_protocol_by_head.png")
