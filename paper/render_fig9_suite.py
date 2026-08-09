"""Figure 9 -- the whole suite, lookup against learning.

The paper's other cells are one task. This is all ten, at n=10 each, for two
policies: the head we trained and released, and a policy that reads the task
index, opens the benchmark's shipped initial-state files, emits two constants
and never looks at the episode.

The result is not that lookup ties. It is that lookup WINS -- and that both
numbers are small, which is the other half of the honesty. A benchmark on which
a task-indexed constant outscores a trained vision-language policy two to one
is not measuring what its leaderboard is titled.

Sources:
    results/blind_cells.json   -- blind, task 0
    results/suite_cells.json   -- blind, tasks 1-9; head, tasks 1-9
    results/pod_cells.json     -- head, task 0 (the published reference cell)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "visuals" / "F9_suite.png"
Z = 1.959963984540054

BLIND = "#c0392b"
HEAD = "#2c6fbb"

NAMES = ["alphabet soup", "cream cheese", "salad dressing", "bbq sauce",
         "ketchup", "tomato sauce", "butter", "milk", "choc. pudding",
         "orange juice"]


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


S = json.loads((REPO / "results/suite_cells.json").read_text())
B = json.loads((REPO / "results/blind_cells.json").read_text())
P = json.loads((REPO / "results/pod_cells.json").read_text())

blind = {0: B["blind_t0"]} | {t: S[f"blind_t{t}"] for t in range(1, 10)}
head = {0: P["P0_ref_heldout"]} | {t: S[f"head_t{t}"] for t in range(1, 10)
                                   if f"head_t{t}" in S}

D = json.loads((REPO / "results/placement_diameter.json").read_text())["libero_object"]

fig, (ax, axT) = plt.subplots(1, 2, figsize=(11.8, 3.5),
                              gridspec_kw={"width_ratios": [3.0, 1.0]})

x = np.arange(10)
w = 0.38
bk = [blind[t]["k"] for t in range(10)]
hk = [head[t]["k"] if t in head else None for t in range(10)]

ax.bar(x - w / 2, bk, width=w, color=BLIND, label="blind (shipped files only)")
ax.bar(x + w / 2, [v if v is not None else 0 for v in hk], width=w,
       color=HEAD, label="released head (camera + language)")
for i in range(10):
    if bk[i]:
        ax.text(i - w / 2, bk[i] + 0.22, str(bk[i]), ha="center",
                fontsize=8.6, fontweight="bold", color=BLIND)
    if hk[i]:
        ax.text(i + w / 2, hk[i] + 0.22, str(hk[i]), ha="center",
                fontsize=8.6, fontweight="bold", color=HEAD)
    if hk[i] is None:
        ax.text(i + w / 2, 0.15, "?", ha="center", fontsize=8, color="#9aa5b1")

# The task where lookup beats learning outright is the story; mark it.
ax.annotate("constant $10/10$,\ntrained head $0/10$", xy=(3 - w / 2, 10),
            xytext=(4.15, 8.4), fontsize=8.2, color=BLIND, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLIND, lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels([f"{i}  {NAMES[i]}" for i in x], fontsize=7.2,
                   rotation=32, ha="right", rotation_mode="anchor")
ax.set_ylim(0, 12.4)
ax.set_yticks([0, 5, 10])
ax.set_ylabel("successes out of 10", fontsize=9.2)
ax.set_title("A   Every task in LIBERO-Object, $n$ = 10 each",
             loc="left", fontweight="bold", fontsize=10.6)
ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=1,
          bbox_to_anchor=(0.005, 1.0))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eef1f4", lw=0.8)
ax.set_axisbelow(True)

# ---- B: the totals --------------------------------------------------------
tot = []
BK, BN = sum(bk), 10 * 10
tot.append(("blind", BK, BN, BLIND))
if len(head) == 10:
    HK, HN = sum(head[t]["k"] for t in range(10)), 100
    tot.append(("released head", HK, HN, HEAD))
else:
    HK = sum(head[t]["k"] for t in head)
    HN = 10 * len(head)
    tot.append((f"released head\n({len(head)} tasks)", HK, HN, HEAD))

y = np.arange(len(tot))[::-1]
for yi, (lbl, k, n, col) in zip(y, tot):
    lo, hi = wilson(k, n)
    axT.plot([lo, hi], [yi, yi], color=col, lw=3.4, alpha=0.42,
             solid_capstyle="round")
    axT.plot([k / n], [yi], "o", color=col, ms=11, markeredgecolor="white",
             markeredgewidth=1.3)
    axT.text(k / n, yi + 0.22, f"{k}/{n}", ha="center", fontsize=9.6,
             fontweight="bold", color=col)
axT.set_yticks(y)
axT.set_yticklabels([t[0] for t in tot], fontsize=9)
axT.set_xlim(-0.01, 0.42)
axT.set_ylim(-0.6, len(tot) - 0.3)
axT.set_xlabel("suite success rate, Wilson 95%", fontsize=9.2)
axT.set_title("B   Suite totals", loc="left", fontweight="bold", fontsize=10.6)
axT.spines[["top", "right"]].set_visible(False)
axT.grid(axis="x", color="#eef1f4", lw=0.8)
axT.set_axisbelow(True)

fig.text(0.5, -0.10,
         "Both policies drive the identical controller; only the goal supply differs. Neither is good at this suite — that is the point of showing all ten "
         "tasks rather than the one every other cell in this paper reports.",
         ha="center", fontsize=8.6, color="#43505c")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}   blind {BK}/{BN}   head {HK}/{HN}")
