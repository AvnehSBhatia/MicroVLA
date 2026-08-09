"""Re-derive the paper's load-bearing numbers from artifacts, and RECOMPUTE the
statistics rather than reading them back.

The previous version of this script had a defect an adversarial reviewer named
exactly: it "verified" p-values by reading them out of the same JSON that wrote
them, and it never opened the raw per-trial logs. A consistency check between
two copies of a number is not a re-derivation. This version recomputes every
test from the trial outcomes, and states plainly which claims it does NOT
cover, because "all N load-bearing values" was the paper's own worst instance
of the drift it claims to police.

Usage:  python scripts/verify_paper_numbers.py
Exit 0 iff every check passes.
"""
from __future__ import annotations

import json
import math
import re
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
Z = 1.959963984540054
failures: list[str] = []
checks = 0


def check(name: str, claimed, actual, tol: float = 0.006) -> None:
    global checks
    checks += 1
    ok = (claimed == actual) if isinstance(claimed, bool) else abs(claimed - actual) <= tol
    if not ok:
        failures.append(f"{name}: paper {claimed}, artifact {actual}")
        print(f"FAIL  {name:<52} paper={claimed} actual={actual}")
    else:
        print(f"ok    {name:<52} {actual}")


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cheb(P):
    P = np.unique(np.asarray(P, float), axis=0)
    if len(P) == 1:
        return 0.0
    best = None
    for a, b in combinations(range(len(P)), 2):
        c = (P[a] + P[b]) / 2; r = float(np.linalg.norm(P[a] - c))
        if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-12):
            best = r if best is None else min(best, r)
    for a, b, d in combinations(range(len(P)), 3):
        A, B, C = P[a], P[b], P[d]
        den = 2*(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
        if abs(den) < 1e-15:
            continue
        ux = ((A@A)*(B[1]-C[1]) + (B@B)*(C[1]-A[1]) + (C@C)*(A[1]-B[1]))/den
        uy = ((A@A)*(C[0]-B[0]) + (B@B)*(A[0]-C[0]) + (C@C)*(B[0]-A[0]))/den
        c = np.array([ux, uy]); r = float(np.linalg.norm(A - c))
        if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-12):
            best = r if best is None else min(best, r)
    return float(best)


J = lambda q: json.loads((REPO / q).read_text())

print("=== §2 admissibility: radii RECOMPUTED from the shipped states ===")
FJ = J("results/suite_forensics_joints.json")
radii, fixture = {}, {}
for suite, S in FJ["suites"].items():
    rs, fx = [], []
    for t in S["tasks"]:
        o = next((o for o in t["objects"] if o["object"] == t["primary_target"]), None)
        if o and o.get("xy_per_state_m"):
            rs.append(cheb(np.asarray(o["xy_per_state_m"])) * 100); fx.append(False)
        else:
            rs.append(0.0); fx.append(True)      # fixture: never moves
    radii[suite] = np.array(rs); fixture[suite] = np.array(fx)

check("Object max identity radius (cm)", 1.17, float(radii["libero_object"].max()))
# N2: Table 2's mean and the unpinned range, so a stale sentence fails loudly.
check("Object mean identity radius (cm)", 0.30, float(radii["libero_object"].mean()))
_unp = radii["libero_object"][radii["libero_object"] > 1e-6]
check("Object unpinned range lo (cm)", 0.60, float(_unp.min()))
check("Object unpinned range hi (cm)", 1.17, float(_unp.max()))
check("Object tasks pinned to a point", 6.0,
      float((radii["libero_object"] <= 1e-6).sum()), 0.5)
for _s, _m in [("libero_spatial", 2.15), ("libero_goal", 1.48), ("libero_10", 3.10)]:
    check(f"{_s} mean identity radius (cm)", _m, float(radii[_s].mean()))
# the two tolerance estimates the appendix now reports side by side
_B = J("results/blind_failure_attribution.json")
check("task-0 tolerance estimate (cm)", 1.41, min(_B["task0"]["failure_disp_cm"]), 0.005)
check("task-3 tolerance estimate (cm)", 1.91, _B["task3"]["max_tolerated_cm"], 0.005)
check("the two estimates disagree", True,
      _B["task3"]["max_tolerated_cm"] > min(_B["task0"]["failure_disp_cm"]) + 0.3)
check("task-0 classes overlap (no clean threshold)", True,
      max(_B["task0"]["success_disp_cm"]) > min(_B["task0"]["failure_disp_cm"]))
check("Object tasks admissible at 1.4 cm", 10.0,
      float((radii["libero_object"] <= 1.4).sum()), 0.5)
others = np.concatenate([radii[s] for s in radii if s != "libero_object"])
othfx = np.concatenate([fixture[s] for s in radii if s != "libero_object"])
check("other tasks admissible at 1.4 cm", 2.0, float((others <= 1.4).sum()), 0.5)
check("other tasks total", 30.0, float(len(others)), 0.5)
check("both exceptions are fixtures", True, bool(othfx[others <= 1.4].all()))
check("smallest movable radius elsewhere (cm)", 1.49, float(others[~othfx].min()))
check("elsewhere admissible at the LARGER tolerance estimate", 13.0,
      float((others <= 1.914).sum()), 0.5)
check("measured tolerance falls inside the window", True,
      bool(float(radii["libero_object"].max()) <= 1.41 < float(others[~othfx].min())))

print("\n--- delta sensitivity (Appendix): the window, and where it ends ---")
for d, o_c, e_c in [(0.5, 6, 2), (1.0, 9, 2), (1.171, 10, 2), (1.4, 10, 2),
                    (1.492, 10, 2), (2.0, 10, 17), (2.5, 10, 19), (4.0, 10, 29)]:
    check(f"delta={d}: Object admissible", float(o_c),
          float((radii["libero_object"] <= d).sum()), 0.5)
    check(f"delta={d}: elsewhere admissible", float(e_c),
          float((others <= d).sum()), 0.5)

print("\n=== §5 suite: cells and tests RECOMPUTED from per-trial outcomes ===")
S = J("results/suite_cells.json"); B = J("results/blind_cells.json"); P = J("results/pod_cells.json")
blind = {0: B["blind_t0"]} | {t: S[f"blind_t{t}"] for t in range(1, 10)}
head = {0: P["P0_ref_heldout"]} | {t: S[f"head_t{t}"] for t in range(1, 10)}
bk = np.array([blind[t]["k"] for t in range(10)])
hk = np.array([head[t]["k"] for t in range(10)])
check("blind suite total", 16.0, float(bk.sum()), 0.5)
check("head suite total", 8.0, float(hk.sum()), 0.5)
check("head is zero on all nine untrained tasks", True, bool((hk[1:] == 0).all()))

n01 = sum(1 for t in range(10) for r in blind[t]["trials"]
          if blind[t]["trials"][r] and not head[t]["trials"].get(r))
n10 = sum(1 for t in range(10) for r in blind[t]["trials"]
          if head[t]["trials"].get(r) and not blind[t]["trials"][r])
d = n01 + n10
p_trial = min(1.0, sum(math.comb(d, i) for i in range(min(n01, n10) + 1)) / 2 ** d * 2)
check("trial-level McNemar p (reported as INVALID)", 0.039, p_trial, 0.001)
disc = {t: sum(1 for r in blind[t]["trials"]
               if blind[t]["trials"][r] != head[t]["trials"].get(r)) for t in range(10)}
check("discordant pairs confined to 2 tasks", 2.0,
      float(sum(1 for t in disc if disc[t])), 0.5)
check("discordant pairs in task 3", 10.0, float(disc[3]), 0.5)

wb, wh = int((bk > hk).sum()), int((hk > bk).sum())
dd = wb + wh
p_task = min(1.0, sum(math.comb(dd, i) for i in range(min(wb, wh) + 1)) / 2 ** dd * 2)
check("task-level sign test p", 1.0, p_task, 0.001)
check("task-level: blind wins", 1.0, float(wb), 0.5)
check("task-level: head wins", 1.0, float(wh), 0.5)
diff = (bk - hk) / 10.0
nz = np.flatnonzero(diff != 0)
obs = float(diff.mean()); cnt = tot = 0
for sg in product([1, -1], repeat=len(nz)):
    d2 = diff.copy(); d2[nz] = diff[nz] * np.array(sg)
    tot += 1; cnt += abs(float(d2.mean())) >= abs(obs) - 1e-12
check("task-level permutation p", 1.0, cnt / tot, 0.001)

print("\n=== §5 attribution: the refutation, recomputed ===")
tasks = FJ["suites"]["libero_object"]["tasks"]
C = np.array([next(o for o in t["objects"]
                   if o["object"] == t["primary_target"])["mean_xyz_m"][:2] for t in tasks])
dmm = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1) * 1000
gA, gB = [0, 4, 6, 7, 8], [1, 2, 3, 5, 9]
check("group A constants agree to (mm)", 0.95, float(dmm[np.ix_(gA, gA)].max()), 0.02)
check("group B constants agree to (mm)", 0.63, float(dmm[np.ix_(gB, gB)].max()), 0.02)
check("group A blind scores span 0..6", True,
      [int(bk[i]) for i in gA] == [6, 0, 0, 0, 0])
check("group B blind scores span 0..10", True,
      [int(bk[i]) for i in gB] == [0, 0, 10, 0, 0])
Rv = radii["libero_object"]
check("r(radius, blind successes)", 0.52,
      float(np.corrcoef(Rv, bk.astype(float))[0, 1]), 0.02)

print("\n=== §7 certification ===")
T = J("results/thread_determinism.json")
check("thread fields identical", True, bool(T["all_identical"]))
check("thread comparison horizon (steps)", 40.0,
      float(T["fields"]["4"]["steps"]), 0.5)

print(f"\n{checks - len(failures)}/{checks} checks passed")
print("\nNOT covered by this script, and stated so rather than implied: the audit\n"
      "table, the instruction-swap cells, the displacement sweep, the probe\n"
      "attribution, the positive control, the renderer cell, the parameter\n"
      "ledger, and the historical 35/50 cell whose per-trial log is not in this\n"
      "repository.")
if failures:
    print("\nDRIFT DETECTED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
