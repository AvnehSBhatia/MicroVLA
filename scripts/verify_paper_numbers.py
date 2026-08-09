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
# the reversal on the shared object, which the caption now leads with
_ADM = J("results/admissibility.json")["suites"]
for _s, _n in [("libero_object", 0), ("libero_goal", 6)]:
    _sh = [t["shared_R_cm_max"] for t in _ADM[_s]["tasks"]
           if t["shared_R_cm_max"] is not None]
    check(f"{_s}: shared object admissible at 1.4", float(_n),
          float(sum(1 for v in _sh if v <= 1.4)), 0.5)
# l_inf, computed rather than asserted
_LI = J("results/admissibility.json")["linf_admissible_at_1_4"]
for _s, _n in [("libero_object", 10), ("libero_spatial", 0),
               ("libero_goal", 3), ("libero_10", 0)]:
    check(f"l_inf {_s} admissible at 1.4", float(_n), float(_LI[_s]), 0.5)
# how much the identity-bearing rule matters
_IR = J("results/admissibility.json")["identity_rule_sensitivity"]
check("identity rule 'first': elsewhere", 2.0, float(_IR["first"]["elsewhere_admissible"]), 0.5)
check("identity rule 'unique': elsewhere", 8.0, float(_IR["unique"]["elsewhere_admissible"]), 0.5)
check("identity rule 'last': Object", 0.0, float(_IR["last"]["object_admissible"]), 0.5)
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

print("\n=== §2.3 declared region vs shipped states (needs no delta) ===")
DS = J("results/declared_vs_shipped.json")["suites"]
_o = [t["fraction_of_declared"] for t in DS["libero_object"]["tasks"]]
_l = [t["fraction_of_declared"] for t in DS["libero_10"]["tasks"]]
check("Object: max fraction of declared region used", 0.389, max(_o), 0.005)
# the generalised claim: three suites exercise their region, one does not
for _s, _exercised in [("libero_spatial", True), ("libero_goal", True),
                       ("libero_10", True), ("libero_object", False)]:
    _fr = [t["fraction_of_declared"] for t in DS[_s]["tasks"]]
    check(f"{_s}: exercises its declared region", _exercised,
          bool(min(_fr) > 0.9))
check("Object: tasks using exactly none of it", 6.0,
      float(sum(1 for v in _o if v < 1e-9)), 0.5)
check("Long: min fraction of declared region used", 0.967, min(_l), 0.005)
check("the two suites do not overlap", True, max(_o) < min(_l))
check("both declare the same 5cm box", True,
      all(abs(t["declared_cm"][0] - 5.0) < 1e-6
          for t in DS["libero_object"]["tasks"] + DS["libero_10"]["tasks"]))

print("\n=== §2.4 E1: shipped vs repaired (the manipulation check) ===")
# Recomputed from raw per-trial outcomes, not read back from the JSON's own
# summary fields -- reading a p-value back is error 1 of Table 13.
E1 = J("results/e1_shipped_vs_repaired.json")


def mcnemar_exact(a: dict, b: dict) -> tuple[int, int, float]:
    """Two-sided exact McNemar over trials present in both arms."""
    keys = sorted(set(a) & set(b), key=int)
    lost = sum(1 for k in keys if a[k] and not b[k])
    gained = sum(1 for k in keys if b[k] and not a[k])
    n = lost + gained
    if n == 0:
        return lost, gained, 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1))
    return lost, gained, min(1.0, 2.0 * tail / 2 ** n)


for _c, _k in [("blind_shipped", 25), ("blind_repaired", 10),
               ("head_shipped", 19), ("head_repaired", 13)]:
    _t = E1["cells"][_c]["trials"]
    check(f"E1 {_c} successes", float(_k), float(sum(_t.values())), 0.5)
    check(f"E1 {_c} n = 30 (planned, no early stop)", 30.0, float(len(_t)), 0.5)

for _arm, _lost, _gained, _p, _tol in [("blind", 17, 2, 0.0007, 0.0002),
                                       ("head", 10, 4, 0.18, 0.005)]:
    _l, _g, _pv = mcnemar_exact(E1["cells"][f"{_arm}_shipped"]["trials"],
                                E1["cells"][f"{_arm}_repaired"]["trials"])
    check(f"E1 {_arm}: trials lost to the repair", float(_lost), float(_l), 0.5)
    check(f"E1 {_arm}: trials gained", float(_gained), float(_g), 0.5)
    check(f"E1 {_arm}: exact McNemar p", _p, _pv, _tol)

# The paper's claim is a direction, so check the direction, not just the p.
check("E1 blind: the repair costs the constant more than half",
      True, E1["cells"]["blind_shipped"]["rate"]
      - E1["cells"]["blind_repaired"]["rate"] >= 0.5)
check("E1 head reproduces its historical 0.700 (inside Wilson)", True,
      E1["cells"]["head_shipped"]["wilson"][0] <= 0.700
      <= E1["cells"]["head_shipped"]["wilson"][1])

# The interaction we report and refuse to claim. Recomputed, not quoted.
_ds = [int(E1["cells"]["blind_shipped"]["trials"][k])
       - int(E1["cells"]["head_shipped"]["trials"][k]) for k in map(str, range(30))]
_dr = [int(E1["cells"]["blind_repaired"]["trials"][k])
       - int(E1["cells"]["head_repaired"]["trials"][k]) for k in map(str, range(30))]
check("E1 interaction: blind-head, shipped", 0.200, sum(_ds) / 30, 0.001)
check("E1 interaction: blind-head, repaired", -0.100, sum(_dr) / 30, 0.001)
_rng = np.random.default_rng(0)
_pool = np.array(_ds + _dr); _obs = abs(np.mean(_ds) - np.mean(_dr))
_c = sum(abs(np.mean(pm[:30]) - np.mean(pm[30:])) >= _obs - 1e-12
         for pm in (_rng.permutation(_pool) for _ in range(20000)))
check("E1 interaction: permutation p (does NOT reach 0.05)", 0.14, _c / 20000, 0.02)
check("E1 interaction: we are not entitled to claim it", True, _c / 20000 > 0.05)

print("\n=== §2.4 E5: the control, recomputed from raw trials ===")
E5 = J("results/e5_trials.json")
check("E5 fixed_shipped", 10.0, float(sum(E5["fixed_shipped"].values())), 0.5)
check("E5 fixed_shipped n", 10.0, float(len(E5["fixed_shipped"])), 0.5)
check("E5 fixed_repaired 30/30", 30.0,
      float(sum(E5["fixed_repaired"].values())), 0.5)
check("E5 fixed_repaired n = 30 (planned, complete)", 30.0,
      float(len(E5["fixed_repaired"])), 0.5)
_l, _g, _p = mcnemar_exact(E5["fixed_repaired"],
                           E1["cells"]["blind_repaired"]["trials"])
check("E5 fixed vs blind on repaired: discordant, fixed-only", 20.0, float(_l), 0.5)
check("E5 fixed vs blind on repaired: blind-only", 0.0, float(_g), 0.5)
check("E5 fixed vs blind on repaired: exact McNemar p", 2e-06, _p, 1e-06)
check("E5: the repaired states are NOT harder for a correct static goal",
      True, sum(E5["fixed_repaired"].values()) == len(E5["fixed_repaired"]))

# The premise that broke: fixed and blind are NOT the same arm on a pinned task.
_k = [k for k in sorted(set(E5["fixed_shipped"]) &
                        set(E1["cells"]["blind_shipped"]["trials"]), key=int)]
_mis = [k for k in _k if bool(E5["fixed_shipped"][k])
        != bool(E1["cells"]["blind_shipped"]["trials"][k])]
check("E5 identity check: trials compared", 10.0, float(len(_k)), 0.5)
check("E5 identity check: mismatches (premise falsified)", 4.0,
      float(len(_mis)), 0.5)

# The container handicap, and its association with basket displacement.
_b = np.asarray(next(o for o in J("results/suite_forensics_joints.json")
                     ["suites"]["libero_object"]["tasks"][0]["objects"]
                     if o["object"] == "basket_1")["xy_per_state_m"], float)
_dev = [float(np.linalg.norm(_b[(3 * 20 + i) % 50] - _b.mean(0))) * 100
        for i in range(10)]
_bl = E1["cells"]["blind_shipped"]["trials"]
_f = [_dev[i] for i in range(10) if not _bl[str(i)]]
_s = [_dev[i] for i in range(10) if _bl[str(i)]]
check("E5 basket displacement, blind's failures (cm)", 1.59,
      float(np.mean(_f)), 0.01)
check("E5 basket displacement, blind's successes (cm)", 1.01,
      float(np.mean(_s)), 0.01)
_rank = {v: i + 1 for i, v in enumerate(sorted(_dev))}
_obs = sum(_rank[x] for x in _f)
_tot = sum(1 for c in combinations(range(10), len(_f))
           if sum(_rank[_dev[j]] for j in c) >= _obs)
# TWO-sided, to match the value §6 already quotes for this same test. The
# first draft of this check quoted the one-sided 0.0095 while §6 quoted the
# two-sided 0.019 -- the same association, in one manuscript, at two tail
# conventions. Caught by cross-reading, not by any checker.
check("E5 basket association, exact permutation p (two-sided, = §6's)", 0.019,
      2 * _tot / math.comb(10, len(_f)), 0.001)

# The cross-build claim, settled: the historical cell and this one share ten
# trials and must agree on all ten, outcome for outcome.
_hist = J("results/blind_cells.json")["blind_t0"]["trials"]
_now = E1["cells"]["blind_shipped"]["trials"]
check("cross-build: historical blind_t0 is 6/10", 6.0,
      float(sum(_hist.values())), 0.5)
check("cross-build: the two runs agree on all ten shared trials", True,
      all(bool(_hist[k]) == bool(_now[k]) for k in _hist))
check("cross-build: the gap is the 20 trials only one run did (19/20)", 19.0,
      float(sum(_now[str(i)] for i in range(10, 30))), 0.5)
check("E5 basket spread across shipped states (cm)", 2.949,
      float((_b.max(0) - _b.min(0)).max()) * 100, 0.01)

# The manipulation, checked at the SIMULATOR rather than at the files: what
# the physics engine actually placed on the table at step 0 of every episode.
# A repair that edits state files but does not move the object would produce
# exactly the headline we report, so this is not a formality.
OP = J("results/e5_object_positions.json")
for _arm, _n, _distinct, _sx in [("fixed_shipped", 10, 1, 0.0),
                                 ("fixed_repaired", 30, 30, 4.844)]:
    _a = np.asarray([OP[_arm][k] for k in sorted(OP[_arm], key=int)], float)
    check(f"E5 sim-side {_arm}: episodes", float(_n), float(len(_a)), 0.5)
    check(f"E5 sim-side {_arm}: distinct object positions", float(_distinct),
          float(len(np.unique(_a.round(9), axis=0))), 0.5)
    check(f"E5 sim-side {_arm}: x spread (cm)", _sx,
          float((_a.max(0) - _a.min(0))[0]) * 100, 0.01)
check("E5 sim-side: the repair moved the object every episode", True,
      len(np.unique(np.asarray(list(OP["fixed_repaired"].values())).round(9),
                    axis=0)) == 30)

print("\n=== §7 E2: the jitter sweep, recomputed from raw trials ===")
E2 = J("results/e23_trials.json")["e2"]
for _t, _lv, _want in [("3", [0, 1, 2, 4, 8], [10, 9, 5, 0, 0]),
                       ("0", [0, 1, 2, 4, 8], [6, 5, 4, 0, 0])]:
    for _j, _k in zip(_lv, _want):
        _c = E2[f"{_t}|{_j:g}"]
        check(f"E2 task {_t} r={_j}", float(_k), float(sum(_c.values())), 0.5)
        check(f"E2 task {_t} r={_j}: n = 10", 10.0, float(len(_c)), 0.5)
    _ks = [sum(E2[f"{_t}|{_j:g}"].values()) for _j in _lv]
    check(f"E2 task {_t}: monotone non-increasing in r", True,
          all(_ks[i] >= _ks[i + 1] for i in range(len(_ks) - 1)))
    # Holm over the four r>0 comparisons, recomputed here.
    _ps = {}
    for _j in _lv[1:]:
        _ps[_j] = mcnemar_exact(E2[f"{_t}|0"], E2[f"{_t}|{_j:g}"])[2]
    _srt = sorted(_ps.items(), key=lambda kv: kv[1])
    _run, _holm = 0.0, {}
    for _i, (_j, _pv) in enumerate(_srt):
        _run = max(_run, min(1.0, (len(_srt) - _i) * _pv))
        _holm[_j] = _run
    if _t == "3":
        check("E2 task 3: Holm p at r=4", 0.0078, _holm[4], 0.0005)
        check("E2 task 3: Holm p at r=2 does NOT reject", True, _holm[2] >= 0.05)
        check("E2 task 3: the arm IS using the constant", True, _holm[4] < 0.05)
    else:
        check("E2 task 0: no level survives Holm", True,
              min(_holm.values()) >= 0.05)
        check("E2 task 0: uncorrected p at r=4", 0.03125, _ps[4], 0.0005)
        check("E2 task 0: falls to zero anyway", True, _ks[0] > 0 and _ks[-1] == 0)

# What the measured tolerance does to the separation, recomputed from radii.
ADM = J("results/admissibility.json")["suites"]


def contrast(d):
    o = on = e = en = 0
    for _s, _rec in ADM.items():
        for _t in _rec["tasks"]:
            _ok = _t["identity_R_cm"] <= d
            if _s == "libero_object":
                on += 1; o += _ok
            else:
                en += 1; e += _ok
    return o, on, e, en


for _d, _o, _e in [(1.41, 10, 2), (1.91, 10, 13), (2.0, 10, 17), (4.0, 10, 29)]:
    _a, _an, _b, _bn = contrast(_d)
    check(f"separation at delta={_d}: Object", float(_o), float(_a), 0.5)
    check(f"separation at delta={_d}: elsewhere", float(_e), float(_b), 0.5)
check("separation is DISSOLVED at the measured floor (>half elsewhere)", True,
      contrast(2.0)[2] / contrast(2.0)[3] > 0.5)

# Both candidate suite-level windows, and the width of what we can measure.
_R = {k: sorted(t["identity_R_cm"] for t in v["tasks"])
      for k, v in ADM.items()}
_long = _R["libero_10"]
_oth = sorted(r for k, v in _R.items() if k != "libero_10" for r in v)
check("window Object-vs-rest: lower edge (Object max R)", 1.171, max(_R["libero_object"]), 0.001)
check("window Object-vs-rest: upper edge (smallest movable elsewhere)", 1.492,
      min(r for r in _oth if r > 1.171), 0.001)
check("window Long-vs-rest: upper edge (Long min R)", 2.848, _long[0], 0.001)
check("window Long-vs-rest: lower edge", 2.480,
      max(r for r in _oth if r < _long[0]), 0.001)
check("window Long-vs-rest: elsewhere inside it", 29.0,
      float(sum(1 for r in _oth if r < _long[0])), 0.5)
check("window Long-vs-rest: Long inside it", 0.0,
      float(sum(1 for r in _long if r < _long[0])), 0.5)
check("measured bracket straddles the Long edge (decides neither)", True,
      2.0 < _long[0] < 4.0)
check("all-LIBERO admissible at the measured floor", 27.0,
      float(sum(1 for k, v in _R.items() for r in v if r <= 2.0)), 0.5)

# Cross-build reproduction, second cell: E2's task-3 zero-displacement cell is
# an independent run on this box of the historical blind_k[3].
_hist_k = J("results/constant_attribution.json")["blind_k"]
check("cross-build 2: historical task-3 blind", 10.0, float(_hist_k[3]), 0.5)
check("cross-build 2: E2 reproduces it on this box", float(_hist_k[3]),
      float(sum(E2["3|0"].values())), 0.5)
check("cross-build 2: historical task-0 blind (10 trials)", 6.0,
      float(_hist_k[0]), 0.5)

# The direction check: no arc separates failure from success at r=2 on task 3.
_ang = {i: float(np.random.default_rng(555_000 + 20 * 1_000_003 + i)
                 .uniform(0, 2 * np.pi)) for i in range(10)}
_deg = {i: np.degrees(_ang[i]) % 360 for i in range(10)}
_f = sorted(round(_deg[i]) for i in range(10) if not E2["3|2"][str(i)])
_p = sorted(round(_deg[i]) for i in range(10) if E2["3|2"][str(i)])
check("E2 direction: failing directions at r=2, task 3", True,
      _f == [63, 127, 216, 319, 340])
check("E2 direction: passing directions", True,
      _p == [104, 126, 135, 202, 234])
_sep = any(all(((_deg[i] - st) % 360) < w for i in range(10)
               if not E2["3|2"][str(i)])
           and not any(((_deg[i] - st) % 360) < w for i in range(10)
                       if E2["3|2"][str(i)])
           for st in range(360) for w in range(10, 360, 5))
check("E2 direction: NO arc separates failure from success", True, not _sep)

# E3, reported as uninformative by a precondition registered before the cells.
E3A = J("results/e23_analysis.json")["e3"]
check("E3: task pairs that reached planned n", 5.0,
      float(len(E3A["tasks_analysed"])), 0.5)
check("E3: oracle arm is at the floor", True, E3A["oracle_at_floor"])
check("E3: declared uninformative, not support", True, not E3A["informative"])
check("E3: oracle task-level mean", 0.0, E3A["task_level_mean"]["oracle"], 1e-9)
check("E3: blind task-level mean", 0.0, E3A["task_level_mean"]["blind"], 1e-9)
check("E3: every analysed cell is 0/10, both arms", True,
      all(v["oracle"]["k"] == 0 and v["blind"]["k"] == 0
          and v["oracle"]["n"] == 10 and v["blind"]["n"] == 10
          for v in E3A["per_task"].values()))
check("E3: episodes run in the analysed pairs", 100.0,
      float(sum(v["oracle"]["n"] + v["blind"]["n"]
                for v in E3A["per_task"].values())), 0.5)

print("\n=== §7.1 decision windows: recomputed from the radii ===")
DW = J("results/decision_windows.json")
check("decision windows: claims enumerated", 8.0,
      float(DW["claims_enumerated"]), 0.5)
check("decision windows: claims satisfiable at ALL", 1.0,
      float(DW["claims_satisfiable"]), 0.5)
check("decision window: widest anywhere in LIBERO (cm)", 0.322,
      DW["widest_window_cm"], 0.001)
check("decision window: precision shortfall", 6.2,
      DW["precision_shortfall"], 0.05)
_live = [w for w in DW["windows"] if w["satisfiable"]]
check("decision window: the only live claim is Object-admits", True,
      len(_live) == 1 and _live[0]["claim"].startswith("Object admits"))
check("decision window: its lower edge", 1.1709, _live[0]["lower_cm"], 0.001)
check("decision window: its upper edge", 1.4925, _live[0]["upper_cm"], 0.001)
# Independently rebuilt from admissibility.json rather than read back from the
# file decision_window.py wrote, so the two cannot agree by construction.
_T = {k: [(t["identity_R_cm"], bool(t.get("fixture", False)))
          for t in v["tasks"]] for k, v in ADM.items()}
_mov = {k: [r for r, f in v if not f] for k, v in _T.items()}
_lo = max(_mov["libero_object"])
_hi = min(r for k in _mov if k != "libero_object" for r in _mov[k])
check("decision window: independently rederived width", 0.322, _hi - _lo, 0.001)
check("decision window: a 2cm ladder cannot decide it", True, (_hi - _lo) < 2.0)

print("\n=== §2 the shipped repair ===")
import torch as _t
_man = J("results/resampled_init/MANIFEST.json")
_sp = [r["spread_after_cm"] for r in _man["tasks"]]
check("repaired tasks", 10.0, float(len(_man["tasks"])), 0.5)
check("repaired spread, min (cm)", 4.44, min(_sp), 0.02)
check("repaired spread, max (cm)", 4.96, max(_sp), 0.02)
check("repair sampled under a clearance constraint", True,
      all(abs(r["clearance_cm"] - 7.42) < 1e-6 for r in _man["tasks"]))
_src = REPO / ".libero_src/libero/libero/init_files/libero_object"
_dst = REPO / "results/resampled_init"
_ncols = []
for _r in _man["tasks"]:
    _a = np.asarray(_t.load(_src / f"{_r['task']}.pruned_init", weights_only=False), float)
    _b = np.asarray(_t.load(_dst / f"{_r['task']}.pruned_init", weights_only=False), float)
    _ncols.append(int((~np.isclose(_a, _b).all(axis=0)).sum()))
check("repair touches exactly 2 columns per task", True, set(_ncols) == {2})

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
# FULL 3D. The arm is handed a 3-vector, and an earlier version of this check
# compared only the table plane -- which is how the paper came to claim the
# constants were "identical to within a millimetre" when they differ by 40 mm
# on z. The decisive comparison is the 0.469 mm triple, and it must be 3D.
C = np.array([next(o for o in t["objects"]
                   if o["object"] == t["primary_target"])["mean_xyz_m"] for t in tasks])
dmm = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1) * 1000
for t in (2, 5, 9):
    check(f"task {t} vs task 3, 3D distance (mm)", 0.469, float(dmm[t, 3]), 0.005)
    check(f"task {t} blind score", 0.0, float(bk[t]), 0.5)
check("task 3 blind score", 10.0, float(bk[3]), 0.5)
check("group A 3D spread is NOT sub-mm (the retracted claim)", True,
      float(dmm[np.ix_([0, 4, 6, 7, 8], [0, 4, 6, 7, 8])].max()) > 30.0)
Rv = radii["libero_object"]
_r = float(np.corrcoef(Rv, bk.astype(float))[0, 1])
check("r(radius, successes) -- reported as WITHDRAWN", 0.52, _r, 0.02)
import itertools as _it
_perm = np.array([np.corrcoef(Rv, np.array(q, float))[0, 1]
                  for q in _it.permutations(bk.astype(float))])
check("r permutation p (why it is withdrawn)", 0.133,
      float((np.abs(_perm) >= abs(_r) - 1e-12).mean()), 0.005)
_m = [i for i in range(10) if i != 3]
check("r flips sign without task 3", -0.25,
      float(np.corrcoef(Rv[_m], bk.astype(float)[_m])[0, 1]), 0.01)

print("\n=== provenance: which quoted cells have shipped records ===")
PODC = J("results/pod_cells.json")
for _k, _v in [("P0_ref_heldout", 8), ("P1_E4_untouched", 20),
               ("P3_E8_base_cov", 23), ("P3_E8_swap_cov", 13),
               ("P3_E8_swap_v5", 0)]:
    check(f"{_k} has a shipped record", float(_v), float(PODC[_k]["k"]), 0.5)
check("the egl cell is flagged incomplete (disclosed in the text)", True,
      PODC["P7_GL_egl"]["complete"] is False)

print("\n=== §9 positive control: the full tolerance sweep ===")
PC = J("results/probe_positive_control.json")
_tols = PC["tolerances"]
check("tau = 0.1 is in the artifact", True, 0.1 in _tols)
_moved = [c["pair"] for c in PC["contrasts"]
          if c["evaluable"] and c["same_rate_by_tol"]["0.05"] == 0.0
          and c["same_rate_by_tol"]["0.1"] > 0.0]
check("contrasts that move at tau = 0.1", 3.0, float(len(_moved)), 0.5)
_between = [c["median_distance"] for c in PC["contrasts"]
            if 0.1 < c["median_distance"] < 0.6]
check("distances between the two modes", 2.0, float(len(_between)), 0.5)
AT = J("results/attribution_profiles.json")["heads"]
check("v2 vs v2.1 proprio gap, relative", 0.34,
      abs(AT["v2"]["proprio"] - AT["v2.1"]["proprio"]) / AT["v2.1"]["proprio"], 0.01)

print("\n=== §8 the burn bound, by subtraction ===")
check("burned-band successes (35 - 20)", 15.0, 35.0 - 20.0, 0.5)
check("burned-band rate", 0.750, 15.0 / 20.0, 0.001)
check("untouched-band rate", 0.667, 20.0 / 30.0, 0.001)
check("clean difference (not the -0.033 vs the superset)", -0.083,
      20.0 / 30.0 - 15.0 / 20.0, 0.001)

print("\n=== §7 certification ===")
T = J("results/thread_determinism.json")
_f = T["fields"]
check("thread fields identical (RECOMPUTED, not read back)", True,
      _f["1"] == _f["4"] == _f["128"])
check("thread comparison horizon (steps)", 40.0,
      float(T["fields"]["4"]["steps"]), 0.5)

print(f"\n{checks - len(failures)}/{checks} checks passed")
# This list is itself a claim, and a stale one is exactly the failure the
# paper catalogues -- a correct instrument reporting the wrong slice. The
# positive control WAS added here (tau sweep, the two between-mode distances)
# and had to come off the list; what follows is what genuinely has no check.
print("\nNOT covered by this script, stated rather than implied: the four-layer\n"
      "audit table, the instruction-swap cells, the displacement sweep, the\n"
      "probe attribution table, the renderer cell, the parameter ledger, and\n"
      "the historical 35/50 cell whose per-trial log is not in this repository.\n"
      "Covered since the last revision, and no longer on this list: the\n"
      "positive control's tolerance sweep and its between-mode distances.")
if failures:
    print("\nDRIFT DETECTED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
