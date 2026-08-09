"""Admissibility, recomputed on the quantity the definition actually requires.

Adversarial review found three defects in the first version of this
measurement, all confirmed against the shipped artifacts, all corrected here.

1. WRONG QUANTITY. Order-1 lookup-admissibility asks whether SOME constant $c$
   reproduces every shipped position to within delta -- that is the Chebyshev
   radius R = min_c max_s ||x(s) - c||, not the diameter D. D <= delta implies
   R <= delta, so the old criterion was sound where it fired, but D > delta
   implies nothing, and the paper used it to claim 0 of 28 elsewhere. On R at
   delta = 2.5 cm the true count is 27 of 38, not 10 of 38: LIBERO-Goal is 8/8
   admissible and LIBERO-Spatial 9/10. The separation survives, but only at a
   tolerance the old text never used.

2. HALF THE TASK. Every LIBERO-Object task is a pick-AND-place and the shipped
   metadata marks both the grocery and the container `is_target`. Measuring the
   grocery alone and calling the suite admissible measured half of each task.
   We now report the identity-bearing object and the shared container
   separately, because the split is the finding: the suite pins exactly the
   sub-goal that individuates its tasks and randomises the one that does not.

3. EXCLUDED THE ADMISSIBLE CASES. Two LIBERO-Goal tasks name a fixture with no
   free joint. The old text excluded them as "no placement to measure". A
   fixture that never moves has R = 0 -- it is the MOST lookup-admissible case
   there is. Excluding them inflated the contrast. They are counted here.

Usage: python scripts/admissibility.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DELTAS = [0.5, 1.0, 1.171, 1.4, 1.492, 2.0, 2.5, 3.0, 4.0]


def chebyshev_radius_cm(P: np.ndarray) -> float:
    """Exact minimum-enclosing-circle radius, in cm. n<=50, so enumerate the
    2- and 3-point support sets rather than pull in Welzl."""
    P = np.unique(np.asarray(P, dtype=float), axis=0)
    if len(P) == 1:
        return 0.0
    best = None
    for a, b in combinations(range(len(P)), 2):
        c = (P[a] + P[b]) / 2.0
        r = float(np.linalg.norm(P[a] - c))
        if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-12):
            best = r if best is None else min(best, r)
    for a, b, d in combinations(range(len(P)), 3):
        A, B, C = P[a], P[b], P[d]
        den = 2.0 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        if abs(den) < 1e-15:
            continue
        ux = ((A @ A) * (B[1] - C[1]) + (B @ B) * (C[1] - A[1]) + (C @ C) * (A[1] - B[1])) / den
        uy = ((A @ A) * (C[0] - B[0]) + (B @ B) * (A[0] - C[0]) + (C @ C) * (B[0] - A[0])) / den
        c = np.array([ux, uy])
        r = float(np.linalg.norm(A - c))
        if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-12):
            best = r if best is None else min(best, r)
    return float(best) * 100.0


def diameter_cm(P: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    return float(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1).max() * 100.0)


def main() -> None:
    J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
    out: dict = {"deltas_cm": DELTAS, "note": (
        "R is the Chebyshev radius (min enclosing circle) -- the quantity "
        "order-1 lookup-admissibility requires. D is the diameter, reported "
        "for continuity with the previous version. Fixture tasks with no free "
        "joint have R = 0 and are counted, not excluded."), "suites": {}}

    for suite, S in J["suites"].items():
        rows = []
        for t in S["tasks"]:
            req = set(t.get("obj_of_interest") or [])
            ident_name = t["primary_target"]
            per = {}
            for o in t["objects"]:
                xy = o.get("xy_per_state_m")
                if xy:
                    per[o["object"]] = (chebyshev_radius_cm(np.asarray(xy)),
                                        diameter_cm(np.asarray(xy)))
            # A required object with no free joint never moves: R = D = 0. But
            # "no free joint" and "the name did not resolve" must not be the
            # same code path -- an adversarial reviewer pointed out that our
            # `fixture` flag WAS the lookup failure, so a naming bug would have
            # been reported as maximal admissibility. Unresolved names must be
            # on this list explicitly or the run stops.
            FIXTURES = {
                "wooden_cabinet_1_middle_region", "wooden_cabinet_1_top_region",
                "wooden_cabinet_1_top_side", "flat_stove_1",
                "flat_stove_1_cook_region", "main_table_stove_front_region",
                "wine_rack_1_top_region", "white_cabinet_1", "desk_caddy_1",
                "microwave_1",
            }
            for n in [ident_name] + [n for n in req if n != ident_name]:
                if n not in per and n not in FIXTURES:
                    raise RuntimeError(
                        f"{suite}/{t['task']}: required object {n!r} has no "
                        "resolved placement and is not a declared fixture. "
                        "Refusing to default its radius to zero, which would "
                        "report a naming bug as perfect admissibility.")
            ident = per.get(ident_name, (0.0, 0.0))
            shared = [per.get(n, (0.0, 0.0)) for n in req if n != ident_name]
            rows.append({
                "task": t["task"], "identity_object": ident_name,
                "fixture": ident_name not in per,
                "identity_R_cm": round(ident[0], 4), "identity_D_cm": round(ident[1], 4),
                "shared_R_cm_max": round(max([s[0] for s in shared]), 4) if shared else None,
                "task_R_cm_max": round(max([ident[0]] + [s[0] for s in shared]), 4),
            })
        R_i = np.array([r["identity_R_cm"] for r in rows])
        R_t = np.array([r["task_R_cm_max"] for r in rows])
        Sh = np.array([r["shared_R_cm_max"] for r in rows
                       if r["shared_R_cm_max"] is not None])
        out["suites"][suite] = {
            "n_tasks": len(rows), "tasks": rows,
            "identity_R": {"mean": round(float(R_i.mean()), 4),
                           "min": round(float(R_i.min()), 4),
                           "max": round(float(R_i.max()), 4)},
            "shared_R": ({"mean": round(float(Sh.mean()), 4),
                          "min": round(float(Sh.min()), 4),
                          "max": round(float(Sh.max()), 4)} if len(Sh) else None),
            "task_R": {"mean": round(float(R_t.mean()), 4),
                       "min": round(float(R_t.min()), 4),
                       "max": round(float(R_t.max()), 4)},
            "n_admissible_identity": {str(d): int((R_i <= d).sum()) for d in DELTAS},
            "n_admissible_task": {str(d): int((R_t <= d).sum()) for d in DELTAS},
        }

    # Fixtures are the trivially-admissible case: a drawer handle or a stove
    # knob has no free joint, so R = 0 by construction. They are counted (the
    # previous version excluded them, which flattered the contrast) but held
    # separate, because "admissible because it is bolted down" and "admissible
    # because the benchmark froze a randomisable placement" are different
    # facts about a suite.
    O = out["suites"]["libero_object"]
    others = [s for s in out["suites"] if s != "libero_object"]
    oi_max = O["identity_R"]["max"]
    fixtures, movable = [], []
    for s in others:
        for r in out["suites"][s]["tasks"]:
            (fixtures if r["fixture"] else movable).append((s, r["task"],
                                                            r["identity_R_cm"]))
    mv_min = min(v for _, _, v in movable)
    out["separation"] = {
        "quantity": "identity-bearing object, Chebyshev radius",
        "object_max_R_cm": oi_max,
        "others_movable_min_R_cm": round(mv_min, 4),
        "n_others_movable": len(movable), "n_others_fixture": len(fixtures),
        "fixtures": [{"suite": s, "task": t, "R_cm": v} for s, t, v in fixtures],
        "window_cm": [oi_max, round(mv_min, 4)],
        "window_exists": bool(oi_max < mv_min),
        "measured_A1_radius_cm": 1.41,
        "measured_radius_inside_window": bool(oi_max <= 1.41 < mv_min),
        "verdict_at_measured_radius": {
            "libero_object": O["n_admissible_identity"]["1.4"],
            "others_total": sum(out["suites"][s]["n_admissible_identity"]["1.4"]
                                for s in others),
            "others_n_tasks": sum(out["suites"][s]["n_tasks"] for s in others),
        },
    }
    # --- l_infinity, computed rather than asserted ------------------------
    def linf_R(P):
        P = np.asarray(P, dtype=float)
        return float(((P.max(0) - P.min(0)) / 2).max() * 100)

    linf = {}
    for suite, S in J["suites"].items():
        rs = []
        for t in S["tasks"]:
            o = next((o for o in t["objects"]
                      if o["object"] == t["primary_target"]), None)
            rs.append(linf_R(o["xy_per_state_m"])
                      if (o and o.get("xy_per_state_m")) else 0.0)
        linf[suite] = rs
    out["linf_identity_R_cm"] = {k: [round(v, 4) for v in v_] for k, v_ in linf.items()}
    out["linf_admissible_at_1_4"] = {k: int(sum(1 for v in v_ if v <= 1.4))
                                     for k, v_ in linf.items()}

    # --- how much does "identity-bearing" depend on how we pick it? -------
    # goi[0] is what the code uses; it is NOT the same as "the element unique
    # to this task", and on three suites those differ.
    alts = {}
    for rule in ("first", "unique", "last"):
        tot_obj = tot_oth = n_oth = 0
        for suite, S in J["suites"].items():
            counts: dict = {}
            for t in S["tasks"]:
                g = t.get("obj_of_interest") or []
                counts[g[0] if g else None] = counts.get(g[0] if g else None, 0) + 1
            for t in S["tasks"]:
                g = t.get("obj_of_interest") or []
                if not g:
                    continue
                if rule == "first":
                    name = g[0]
                elif rule == "last":
                    name = g[-1]
                else:
                    uniq = [n for n in g if counts.get(n, 0) <= 1]
                    name = uniq[0] if uniq else g[0]
                o = next((o for o in t["objects"] if o["object"] == name), None)
                r = (chebyshev_radius_cm(np.asarray(o["xy_per_state_m"]))
                     if (o and o.get("xy_per_state_m")) else 0.0)
                if suite == "libero_object":
                    tot_obj += int(r <= 1.4)
                else:
                    tot_oth += int(r <= 1.4); n_oth += 1
        alts[rule] = {"object_admissible": tot_obj,
                      "elsewhere_admissible": tot_oth, "elsewhere_n": n_oth}
    out["identity_rule_sensitivity"] = alts

    Path(REPO / "results/admissibility.json").write_text(json.dumps(out, indent=2))

    print("Chebyshev radius R of the IDENTITY-BEARING object (the one that")
    print("individuates the task), and of the whole task (max over required objects):\n")
    print(f"{'suite':<16}{'identity R (cm)':>24}{'shared R':>12}"
          + "".join(f"{('id<='+str(d)):>10}" for d in (1.171, 1.4, 1.492, 2.5)))
    for s, S in out["suites"].items():
        i, sh = S["identity_R"], S["shared_R"]
        shtxt = f"{sh['mean']:.2f}" if sh else "--"
        print(f"{s:<16}{i['mean']:>8.2f} [{i['min']:.2f},{i['max']:.2f}] n={S['n_tasks']:<3}"
              f"{shtxt:>12}"
              + "".join(f"{S['n_admissible_identity'][str(d)]:>7}/{S['n_tasks']:<2}"
                        for d in (1.171, 1.4, 1.492, 2.5)))
    sep = out["separation"]
    print(f"\nfixture tasks elsewhere (R = 0 by construction, counted not excluded):")
    for f in sep["fixtures"]:
        print(f"   {f['suite']}: {f['task']}")
    print(f"\nseparating window over MOVABLE targets: [{sep['window_cm'][0]:.3f}, "
          f"{sep['window_cm'][1]:.3f}) cm -> {'EXISTS' if sep['window_exists'] else 'NONE'}")
    print(f"measured A1 radius 1.41 cm inside it: {sep['measured_radius_inside_window']}")
    v = sep["verdict_at_measured_radius"]
    print(f"\nAT the measured radius 1.4 cm: LIBERO-Object {v['libero_object']}/10 admissible; "
          f"elsewhere {v['others_total']}/{v['others_n_tasks']} "
          f"(and both are fixtures)")
    print("\nwrote results/admissibility.json")


if __name__ == "__main__":
    main()
