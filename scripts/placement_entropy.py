"""Placement entropy, robust to grid alignment, swept over tolerance.

The first version of this measurement quantised poses onto an axis-aligned grid
of side delta. A reviewer found the hole: grid quantisation is not translation
invariant, so a cluster straddling a cell boundary splits in two and a pinned
task can be scored as diverse. Concretely, ``pick_up_the_cream_cheese`` scored
H = 0.00 at delta = 1 cm and H = 1.00 at delta = 2 cm --- pure alignment
artifact. A definition whose value depends on where the origin happens to fall
cannot certify anything.

The fix is to cluster rather than bin. Single-linkage agglomeration at radius
delta partitions the shipped placements into groups no two of which are closer
than delta ACROSS groups, which is translation- and rotation-invariant and
depends only on the pairwise geometry. The entropy is then taken over the
cluster occupancies. Concretely: two placements land in the same group iff they
are connected by a chain of steps each shorter than delta, which is exactly
"indistinguishable at tolerance delta" made transitive.

The sweep over delta matters as much as the fix. delta is a claim about the
embodiment: 1 mm is a claim about a metrology rig, and the tolerance that
governs whether a 7-DoF arm with a parallel jaw reaches an object is closer to
1 cm. Reporting the curve rather than a point removes the choice from the
argument.

Usage:
    python scripts/placement_entropy.py --out results/placement_entropy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Tolerances spanning metrology (0.5 mm) to a generous grasp envelope (5 cm).
DELTAS_M = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def single_linkage_labels(pts: np.ndarray, delta: float) -> np.ndarray:
    """Group points connected by chains of steps shorter than ``delta``.

    Union-find over the delta-neighbourhood graph. O(n^2) in the 50 shipped
    states, which is free. Translation invariant by construction: it reads only
    pairwise distances, so no choice of origin can change the answer.
    """
    n = len(pts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    for i in range(n):
        for j in np.flatnonzero(d[i] < delta):
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def entropy_bits(pts: np.ndarray, delta: float) -> tuple[float, int]:
    labels = single_linkage_labels(np.asarray(pts, dtype=np.float64), delta)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()), int(len(counts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forensics", default="results/suite_forensics_joints.json")
    ap.add_argument("--out", default="results/placement_entropy.json")
    a = ap.parse_args()

    J = json.loads((REPO / a.forensics).read_text())
    out: dict = {"deltas_m": DELTAS_M, "method": "single-linkage at radius delta",
                 "suites": {}}
    for suite, S in J["suites"].items():
        per_task: dict = {}
        for t in S["tasks"]:
            o = next((o for o in t["objects"]
                      if o["object"] == t["primary_target"]), None)
            if o is None or "xy_per_state_m" not in o:
                continue                      # fixture goal: no placement
            xy = np.asarray(o["xy_per_state_m"], dtype=np.float64)
            per_task[t["task"]] = {
                str(d): {"H": round(entropy_bits(xy, d)[0], 4),
                         "groups": entropy_bits(xy, d)[1]}
                for d in DELTAS_M
            }
            # A physical statistic alongside the information one, because a
            # reader reasons in centimetres and it cannot have a grid artifact.
            c = xy.mean(axis=0)
            per_task[t["task"]]["max_dev_cm"] = round(
                float(np.linalg.norm(xy - c, axis=1).max() * 100), 4)
        out["suites"][suite] = {
            "n_tasks": len(per_task),
            "tasks": per_task,
            "mean_H": {str(d): round(float(np.mean(
                [per_task[k][str(d)]["H"] for k in per_task])), 4)
                for d in DELTAS_M},
            "n_zero": {str(d): int(sum(
                1 for k in per_task if per_task[k][str(d)]["H"] == 0.0))
                for d in DELTAS_M},
            "mean_max_dev_cm": round(float(np.mean(
                [per_task[k]["max_dev_cm"] for k in per_task])), 4),
        }

    hdr = "suite".ljust(16) + "".join(f"{d*100:>8.2f}cm" for d in DELTAS_M)
    print(hdr)
    for s, S in out["suites"].items():
        print(s.ljust(16) + "".join(
            f"{S['mean_H'][str(d)]:>10.2f}" for d in DELTAS_M))
    print("\nTasks with H = 0 exactly:")
    print("suite".ljust(16) + "".join(f"{d*100:>8.2f}cm" for d in DELTAS_M))
    for s, S in out["suites"].items():
        print(s.ljust(16) + "".join(
            f"{S['n_zero'][str(d)]:>7}/{S['n_tasks']:<2}" for d in DELTAS_M))
    print("\nMean max deviation from task mean (cm):")
    for s, S in out["suites"].items():
        print(f"  {s:<16} {S['mean_max_dev_cm']:.3f}")

    Path(REPO / a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
