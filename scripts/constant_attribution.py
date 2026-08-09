"""Does the constant explain the blind arm's score? No -- and this is the test.

Adversarial review asked the question the suite run had already answered and we
had not looked at: if the blind arm's successes come from the CONSTANT being
close to the truth, then tasks handed the same constant should score the same.

They are not. Four LIBERO-Object tasks receive constants agreeing to within
0.47 mm and score {0, 0, 0, 10}. Three more share an (x,y) to 0.00 mm and score
{6, 0, 0}. And the correlation between a task's placement diameter and the
blind arm's success on it is POSITIVE -- worse constants did better -- where
Proposition 1 requires it to be non-positive.

The conclusion is not that the benchmark is fine. It is that the blind arm's
per-task outcome is governed by the object, not by the goal supply, so the
suite-level comparison cannot be read as evidence about lookup. We report this
as a refutation of our own headline attribution.

The experiment that would settle the remaining question -- displace the
constant by r and see whether the score moves -- was implemented
(``--goal-anchor-jitter-cm``) and did not run before the compute window closed.

Usage: python scripts/constant_attribution.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
    tasks = J["suites"]["libero_object"]["tasks"]
    S = json.loads((REPO / "results/suite_inference.json").read_text())["per_task"]

    rec = []
    for i, t in enumerate(tasks):
        o = next(o for o in t["objects"] if o["object"] == t["primary_target"])
        xy = np.asarray(o["xy_per_state_m"], dtype=float)
        rec.append({
            "task": i, "object": t["primary_target"],
            "constant_xyz": [round(v, 6) for v in o["mean_xyz_m"]],
            "D_cm": round(float(np.linalg.norm(
                xy[:, None, :] - xy[None, :, :], axis=-1).max() * 100), 4),
            "blind": S[str(i)]["blind"], "head": S[str(i)]["head"]})

    C = np.array([r["constant_xyz"][:2] for r in rec])
    dist_mm = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1) * 1000

    # Groups of tasks whose constants agree to under 1 mm.
    groups, seen = [], set()
    for i in range(10):
        if i in seen:
            continue
        g = [j for j in range(10) if dist_mm[i, j] < 1.0]
        seen |= set(g)
        if len(g) > 1:
            groups.append({"tasks": g,
                           "max_pairwise_mm": round(float(dist_mm[np.ix_(g, g)].max()), 3),
                           "blind_scores": [rec[j]["blind"] for j in g]})

    b = np.array([r["blind"] for r in rec], dtype=float)
    D = np.array([r["D_cm"] for r in rec], dtype=float)
    r_pearson = float(np.corrcoef(D, b)[0, 1])
    exact = [r["blind"] for r in rec if r["D_cm"] < 1e-9]
    loose = [r["blind"] for r in rec if r["D_cm"] > 1.0]

    out = {"per_task": rec,
           "constant_pairwise_distance_mm": [[round(float(v), 3) for v in row]
                                             for row in dist_mm],
           "near_identical_groups": groups,
           "pearson_r_D_vs_blind_successes": round(r_pearson, 4),
           "bit_exact_constant_tasks": {"tasks": [r["task"] for r in rec
                                                  if r["D_cm"] < 1e-9],
                                        "successes": int(sum(exact)),
                                        "trials": 10 * len(exact)},
           "loosest_constant_tasks": {"tasks": [r["task"] for r in rec
                                                if r["D_cm"] > 1.0],
                                      "successes": int(sum(loose)),
                                      "trials": 10 * len(loose)},
           "verdict": ("the per-task outcome is not governed by the constant; "
                       "Proposition 1 requires r <= 0 and the measured r is "
                       f"{r_pearson:+.2f}"),
           "settling_experiment": {"flag": "--goal-anchor-jitter-cm",
                                   "status": "implemented, not run"}}
    Path(REPO / "results/constant_attribution.json").write_text(json.dumps(out, indent=2))

    print(f"{'t':<3}{'object':<20}{'constant (x,y)':<22}{'D cm':>7}{'blind':>7}{'head':>6}")
    for r in rec:
        c = r["constant_xyz"]
        print(f"{r['task']:<3}{r['object']:<20}({c[0]:+.4f}, {c[1]:+.4f})     "
              f"{r['D_cm']:>6.2f}{r['blind']:>6}/10{r['head']:>4}/10")
    print("\ngroups of tasks whose constants agree to under 1 mm:")
    for g in groups:
        print(f"   tasks {g['tasks']}  max separation {g['max_pairwise_mm']:.2f} mm"
              f"  ->  blind scores {g['blind_scores']}")
    print(f"\nbit-exact constants (D = 0): {sum(exact)}/{10*len(exact)}")
    print(f"loosest constants (D > 1 cm): {sum(loose)}/{10*len(loose)}")
    print(f"Pearson r(D, blind successes) = {r_pearson:+.3f}   "
          f"(Proposition 1 requires <= 0)")
    print("\nwrote results/constant_attribution.json")


if __name__ == "__main__":
    main()
