"""The decision window: how precisely delta must be known before a suite-level
admissibility claim is decidable at all.

This is the formal content of a complaint the paper had been making
empirically. Section 7 reports that our measurement of delta was too coarse to
decide the verdict. That reads as a statement about our experiment. It is not:
it is a statement about the benchmark, it is computable from the shipped files
alone, and it can be evaluated BEFORE any policy, simulator or training run.

    Definition.  Fix a suite A and the claim "every task in A is
    lookup-admissible and no task outside it is". By Proposition 1 that claim
    is equivalent to

        max_{t in A} R_t  <=  delta  <  min_{t not in A} R_t,

    so the set of tolerances making it true is the half-open interval

        W(A) = [ max_{t in A} R_t ,  min_{t not in A} R_t )

    which we call the DECISION WINDOW of the claim. It is empty exactly when
    the claim is unsatisfiable: no embodiment, however precise or clumsy, makes
    it true.

    Consequence.  An estimate of delta whose uncertainty exceeds |W(A)| cannot
    decide the claim, and no increase in the sample size of a BEHAVIOURAL
    experiment repairs that -- the sample size controls the uncertainty of the
    estimate, and it is the width of the window that the uncertainty must beat.
    Precision, not power, is the binding constraint.

    Why it is useful rather than merely discouraging.  W(A) needs only the
    radii, so a benchmark author can compute it the day the suite is released
    and learn what precision any admissibility claim about it will demand. Ours
    demanded 0.322 cm and we brought 2 cm.

FIXTURES. Tasks whose target is bolted to the world (a drawer, a stove) have
R = 0 for a reason that has nothing to do with a frozen randomisation, and they
sit in the complement of every suite, forcing every window empty by an
irrelevance. They are excluded from both sides here, and the count of excluded
tasks is reported rather than assumed.

Output: results/decision_windows.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "admissibility.json"
OUT = REPO / "results" / "decision_windows.json"

NAMES = {"libero_object": "Object", "libero_spatial": "Spatial",
         "libero_goal": "Goal", "libero_10": "Long"}
# The coarsest resolution our behavioural ladder achieves: the score survives
# 2 cm and is broken by 4 (sec:e2), so delta is pinned only to a 2 cm interval.
LADDER_RESOLUTION_CM = 2.0


def main() -> None:
    suites = json.loads(SRC.read_text())["suites"]
    tasks = {s: [(t["identity_R_cm"], bool(t.get("fixture", False)))
                 for t in v["tasks"]] for s, v in suites.items()}
    movable = {s: [r for r, f in t if not f] for s, t in tasks.items()}
    fixtures = {NAMES[s]: sum(f for _, f in t) for s, t in tasks.items()}

    rows = []
    for s in tasks:
        outside = [r for k in tasks if k != s for r in movable[k]]
        for direction, lo, hi in (
                ("admits, rest resist", max(movable[s]), min(outside)),
                ("resists, rest admit", max(outside), min(movable[s]))):
            rows.append({
                "suite": NAMES[s],
                "claim": f"{NAMES[s]} {direction}",
                "lower_cm": round(lo, 4),
                "upper_cm": round(hi, 4),
                "width_cm": round(max(0.0, hi - lo), 4),
                "satisfiable": hi > lo,
            })

    live = [r for r in rows if r["satisfiable"]]
    widest = max((r["width_cm"] for r in live), default=0.0)
    out = {
        "definition": ("W(A) = [max_{t in A} R_t, min_{t not in A} R_t); the "
                       "claim 'A admissible, complement not' is true exactly "
                       "when delta lies in W(A)"),
        "fixtures_excluded_per_suite": fixtures,
        "claims_enumerated": len(rows),
        "claims_satisfiable": len(live),
        "windows": rows,
        "widest_window_cm": widest,
        "ladder_resolution_cm": LADDER_RESOLUTION_CM,
        "precision_shortfall": (round(LADDER_RESOLUTION_CM / widest, 2)
                                if widest else None),
        "reading": (
            f"Of the {len(rows)} clean suite-level separations LIBERO can "
            f"express, {len(live)} is satisfiable by any tolerance whatever. "
            f"Its window is {widest:.3f} cm wide and the sharpest measurement "
            f"of delta we can make resolves {LADDER_RESOLUTION_CM:.0f} cm, a "
            f"shortfall of {LADDER_RESOLUTION_CM / widest:.1f}x."
            if widest else "no claim is satisfiable"),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}\n")
    print(f"{'claim':<32}{'window (cm)':>20}{'width':>9}")
    for r in rows:
        w = (f"[{r['lower_cm']:.3f}, {r['upper_cm']:.3f})"
             if r["satisfiable"] else "EMPTY")
        print(f"{r['claim']:<32}{w:>20}"
              + (f"{r['width_cm']:9.3f}" if r["satisfiable"] else f"{'--':>9}"))
    print("\n" + out["reading"])


if __name__ == "__main__":
    main()
