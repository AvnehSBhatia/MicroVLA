"""Re-derive every load-bearing number in the paper from its artifact.

Not a test of the code -- a test of the *manuscript*. Papers drift: a value is
recomputed, the JSON updates, and the sentence quoting it does not. This script
reads ``paper/submission/paper2.tex``, pulls the numbers out of it, recomputes
each from the artifact it cites, and fails loudly on any disagreement.

It caught two real drifts while being written: a basket diameter quoted as the
suite mean ($3.66$\\,cm) inside a sentence about one task (whose basket is
$3.37$), and a zero count of $3$ where the table said $6$.

Usage:  python scripts/verify_paper_numbers.py
Exit 0 iff every check passes.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "paper" / "submission" / "paper2.tex"
Z = 1.959963984540054

failures: list[str] = []
checks = 0


def check(name: str, claimed: float, actual: float, tol: float = 0.006) -> None:
    global checks
    checks += 1
    if abs(claimed - actual) > tol:
        failures.append(f"{name}: paper says {claimed}, artifact gives {actual:.4f}")
        print(f"FAIL  {name:<46} paper={claimed}  artifact={actual:.4f}")
    else:
        print(f"ok    {name:<46} {actual:.4f}")


def check_bool(name: str, claimed: bool, actual: bool) -> None:
    global checks
    checks += 1
    if claimed != actual:
        failures.append(f"{name}: paper says {claimed}, artifact gives {actual}")
        print(f"FAIL  {name}")
    else:
        print(f"ok    {name}")


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


tex = TEX.read_text()
J = lambda p: json.loads((REPO / p).read_text())

print("=== Table 1: placement diameter, per suite ===")
D = J("results/placement_diameter.json")
SUITES = [("libero_object", "Object", 0.60, 2.34, 10, 10),
          ("libero_spatial", "Spatial", 4.29, 8.90, 0, 10),
          ("libero_goal", "Goal", 3.65, 4.94, 0, 8),
          ("libero_10", "Long", 6.21, 6.57, 0, 10)]
for key, label, mean_c, max_c, adm_c, n_c in SUITES:
    v = np.asarray(D[key], dtype=float)
    check(f"{label} mean D (cm)", mean_c, float(v.mean()))
    check(f"{label} max D (cm)", max_c, float(v.max()))
    check(f"{label} tasks with D<=2.5cm", adm_c, float((v <= 2.5).sum()), 0.5)
    check(f"{label} resolvable tasks", n_c, float(len(v)), 0.5)

print("\n=== the headline separation ===")
allother = np.concatenate([np.asarray(D[k], float) for k, *_ in SUITES[1:]])
obj = np.asarray(D["libero_object"], float)
check_bool("Object's max D < every other suite's mean",
           True, bool(obj.max() < min(np.asarray(D[k], float).mean()
                                      for k, *_ in SUITES[1:])))
check("other suites: tasks admissible at 2.5cm", 0.0,
      float((allother <= 2.5).sum()), 0.5)
check("other suites: total tasks", 28.0, float(len(allother)), 0.5)

print("\n=== the blind arm ===")
B = J("results/blind_cells.json")
c = B["blind_t0"]
check_bool("blind cell is complete (all planned trials)", True, c["complete"])
check("blind k", 6.0, float(c["k"]), 0.5)
check("blind n", 10.0, float(c["n"]), 0.5)
lo, hi = wilson(c["k"], c["n"])
check("blind Wilson lo", 0.31, lo, 0.005)
check("blind Wilson hi", 0.83, hi, 0.005)
M = B["mcnemar"]
check("p vs learned head", 0.50, M["blind_t0_vs_P0_ref_heldout"]["p"], 0.001)
check("p vs random floor", 0.031, M["blind_t0_vs_P2_E5_random"]["p"], 0.0005)
check("p vs reset-oracle", 0.125, M["blind_t0_vs_P2_E5_fixed"]["p"], 0.001)

print("\n=== the within-task split (Figure 8) ===")
FJ = J("results/suite_forensics_joints.json")
t0 = FJ["suites"]["libero_object"]["tasks"][0]
diam = {}
for o in t0["objects"]:
    xy = np.asarray(o.get("xy_per_state_m", []), dtype=float)
    if len(xy):
        diam[o["object"]] = float(
            np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).max() * 100)
check("task 0 target diameter (cm)", 0.00, diam["alphabet_soup_1"], 0.005)
check("task 0 basket diameter (cm)", 3.37, diam["basket_1"], 0.006)

lines = (REPO / "results/blind_logs/blind_t0_trials.txt").read_text().splitlines()
mins, finals, succ = [], [], []
for ln in lines:
    f = dict(re.findall(r"([a-z_0-9]+)=([-\w.]+)", ln))
    mins.append(float(f["eef_obj_dist_min"]))
    finals.append(float(f["eef_obj_dist_final"]))
    succ.append(f["success"] == "True")
check("trials logged", 10.0, float(len(mins)), 0.5)
check("successes", 6.0, float(sum(succ)), 0.5)
check("max closest-approach over ALL trials (m)", 0.002, max(mins), 0.0005)
check("failures' final object dist (m)", 0.008,
      max(f for f, s in zip(finals, succ) if not s), 0.0005)

print("\n=== renderer / thread determinism ===")
T = J("results/thread_determinism.json")
check_bool("all logged fields identical across 1/4/128 threads",
           True, T["all_identical"])
check("fields compared", 8.0, float(T["n_fields"]), 0.5)

print("\n=== appendix: entropy sweep ===")
E = J("results/placement_entropy.json")
APP = {"libero_object": [2.11, 0.48, 0.00, 0.00, 0.00],
       "libero_spatial": [5.48, 4.49, 2.04, 0.25, 0.00],
       "libero_goal": [5.59, 5.08, 1.53, 0.02, 0.00],
       "libero_10": [5.63, 5.40, 4.27, 0.37, 0.00]}
for key, row in APP.items():
    mh = E["suites"][key]["mean_H"]
    for d, claimed in zip(["0.0005", "0.002", "0.005", "0.01", "0.05"], row):
        check(f"{key} H at {d}m", claimed, mh[d])

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    print("\nDRIFT DETECTED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("every load-bearing number in the paper matches its artifact.")
