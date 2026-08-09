"""E5: the control for E1, analysed before it is run.

E1 shows the lookup arm falling 25/30 to 10/30 when the target is sampled from
the region its own task file declares. The objection that costs the paper its
headline is: the repaired positions are simply HARDER, so the drop measures
difficulty and the manipulation check proves nothing.

E5 separates those. ``--goal-anchor fixed`` differs from ``blind`` in exactly
one bit -- where the static goal comes from. ``blind`` compiles in a constant
read from the benchmark's shipped files and never looks at the episode;
``fixed`` takes one privileged look at reset and holds it, equally static
thereafter. Same controller, same single static goal, same initial states.

FIXED IN ADVANCE, AND WRITTEN DOWN BEFORE THE CELLS EXIST
--------------------------------------------------------
AMENDMENT, 2026-08-09, recorded before the primary cell was read
----------------------------------------------------------------
The IDENTITY check below fired, and it was this file's premise that was wrong,
not the harness. fixed_shipped scored 10/10 where blind_shipped scored 6/10 on
the same ten trials. The reason is in the shipped states: task 0 pins the
TARGET to a single point (0.000 cm across all fifty states) but re-places the
BASKET every episode, over 2.9 x 2.8 cm. blind must supply both goals from the
shipped files, so it carries the MEAN basket; fixed reads the true one. The two
arms therefore differ in the container, not only in the target, and the claim
that they "receive the same number" was false for half the goal.

Consequence for the primary test. fixed_repaired vs blind_repaired confounds
two changes at once and can no longer isolate the difficulty of the repaired
states. The comparison that does isolate it is fixed_shipped vs fixed_repaired:
both read the true basket, the basket's distribution is IDENTICAL in the two
conditions (the repair rewrote columns 10-11 only; the basket lives in columns
17-18 and is bit-identical between the two suites), so the sole difference is
where the target starts. That becomes the primary test.

Provenance, because an amendment made after data exists is worth distrusting:
the identity check was run at 19:11 UTC, when E5B had just been launched and
fixed_repaired did not exist; the cell completed at ~20:10 UTC; this amendment
was written and committed before any fixed_repaired outcome was read. The
ledger /workspace/jobs/e6_ledger.txt carries the timestamps. Nothing about the
old test is deleted -- it is still computed and reported below as a secondary,
with its confound named.

PRIMARY   unpaired exact test, fixed_shipped (n = 10) vs fixed_repaired
          (n = 30). Both arms read the true container; only the target's
          starting distribution differs.

            fixed_repaired holds up  ->  the repaired target positions are not
                                         intrinsically harder for a correct
                                         static goal, so blind's collapse in
                                         E1 is the lookup failing. E1 stands.
            fixed_repaired collapses ->  the repaired positions ARE harder and
                                         E1's drop stays confounded. We say so.

SECONDARY-A (retained, confounded) paired exact McNemar, fixed_repaired vs
          blind_repaired, n = 30, matched trial indices.

            rejects, fixed > blind  ->  a correct static goal SUCCEEDS on the
                                        very states the lookup fails on, so
                                        the states are not intrinsically too
                                        hard and blind's collapse is the
                                        lookup failing. E1 stands.
            anything else           ->  the arms are not separated at this n.
                                        E1's drop stays confounded and the
                                        paper says so in those words.

          Note the asymmetry, which is deliberate: only one of the two outcomes
          lets us keep the claim. A non-rejection does not prove the states are
          harder, it just fails to rule it out, and the paper must not read a
          failure to separate as support. That is error 1 of its own taxonomy.

IDENTITY  fixed_shipped against blind_shipped, trial by trial, on trials 0-9.
          Written on the premise that a pinned task hands both arms the same
          float. IT DID NOT, and the premise was the error: the target is
          pinned but the container is not. Kept exactly as it was, with its
          result, because a check that fires is worth more than one that was
          quietly rewritten to pass.

SECONDARY (descriptive, no test, no threshold) the rate of fixed_repaired
          beside blind_shipped. If a correct static goal on repaired states
          scores near what a lookup scores on frozen states, the repaired suite
          is about as tractable as the shipped one. Reported as two numbers
          with intervals and nothing else.

Input:  results/e5_trials.json
Output: results/e5_analysis.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "e5_trials.json"
E1P = REPO / "results" / "e1_shipped_vs_repaired.json"
OUT = REPO / "results" / "e5_analysis.json"
Z = 1.959963984540054


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b), key=int)
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    n = a_only + b_only
    p = 1.0
    if n:
        p = min(1.0, 2.0 * sum(math.comb(n, i)
                               for i in range(min(a_only, b_only) + 1)) / 2 ** n)
    return {"paired_n": len(keys), "a_only": a_only, "b_only": b_only,
            "p": round(p, 6)}


def rate(t: dict) -> dict:
    k, n = sum(bool(v) for v in t.values()), len(t)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson": wilson(k, n)}


def main() -> None:
    E5 = json.loads(SRC.read_text())
    E1 = json.loads(E1P.read_text())["cells"]
    out: dict = {"cells": {}, "planned_n": {"fixed_shipped": 10,
                                            "fixed_repaired": 30}}

    for name, want in out["planned_n"].items():
        t = E5.get(name, {})
        out["cells"][name] = rate(t) | {"complete": len(t) == want,
                                        "planned_n": want}
    for name in ("blind_shipped", "blind_repaired"):
        out["cells"][name] = rate(E1[name]["trials"]) | {"complete": True,
                                                         "planned_n": 30}

    # -------- IDENTITY: this check can only fail, never flatter.
    fs, bs = E5.get("fixed_shipped", {}), E1["blind_shipped"]["trials"]
    keys = sorted(set(fs) & set(bs), key=int)
    mism = [k for k in keys if bool(fs[k]) != bool(bs[k])]
    out["identity_check"] = {
        "compared_trials": len(keys),
        "mismatched_trials": mism,
        "passes": len(keys) > 0 and not mism,
        "meaning": ("fixed and blind receive the same float on a pinned task, "
                    "so any mismatch is a harness bug, not a result"),
    }

    # -------- PRIMARY
    fr = E5.get("fixed_repaired", {})
    br = E1["blind_repaired"]["trials"]
    prim = mcnemar(fr, br)
    complete = out["cells"]["fixed_repaired"]["complete"]
    fixed_higher = (out["cells"]["fixed_repaired"]["rate"] or 0) > \
        (out["cells"]["blind_repaired"]["rate"] or 0)
    rejects = prim["p"] < 0.05
    out["primary"] = {
        "test": "paired exact McNemar, fixed_repaired vs blind_repaired",
        **prim,
        "cell_complete": complete,
        "fixed_higher": fixed_higher,
        "rejects_at_05": rejects,
        "verdict": (
            "E1 STANDS: a correct static goal succeeds on the very states the "
            "lookup fails on, so the repaired states are not intrinsically too "
            "hard and blind's collapse is the lookup failing."
            if (complete and rejects and fixed_higher) else
            "NOT SEPARATED at this n. E1's drop stays confounded with the "
            "difficulty of the repaired states, and the paper says so."
            if complete else
            "CELL INCOMPLETE -- no verdict. A partial cell is not a result."),
    }

    # -------- SECONDARY, descriptive only
    out["secondary"] = {
        "fixed_repaired": out["cells"]["fixed_repaired"],
        "blind_shipped": out["cells"]["blind_shipped"],
        "note": "two rates with intervals; no test, no threshold, no verdict",
    }

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    for n, c in out["cells"].items():
        flag = "" if c["complete"] else "   << INCOMPLETE, not reportable"
        print(f"  {n:<16} {c['k']}/{c['n']} = {c['rate']}  {c['wilson']}{flag}")
    ic = out["identity_check"]
    print(f"identity: {ic['compared_trials']} trials compared, "
          f"{len(ic['mismatched_trials'])} mismatched -> "
          f"{'PASS' if ic['passes'] else 'FAIL (harness bug)'}")
    pr = out["primary"]
    print(f"primary:  fixed-only {pr['a_only']}, blind-only {pr['b_only']}, "
          f"p = {pr['p']}")
    print(f"VERDICT:  {pr['verdict']}")


if __name__ == "__main__":
    main()
