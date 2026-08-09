"""E2 and E3: the analysis, written before the data.

This file was committed while the runs were still executing. That is
deliberate. Error 1 of the paper's taxonomy was reading a cell at the moment it
flattered us, and error 7 was choosing which tolerances to sweep after seeing
the sweep. The defence against both is to fix the estimator and the stopping
rule first, so the analysis cannot be a function of the outcome.

WHAT IS FIXED HERE, IN ADVANCE
------------------------------
E2 -- the jitter sweep. Displace the blind arm's grasp constant by r cm in a
direction fixed by the trial index, and read the tolerance off the curve.

  * Primary test: paired exact McNemar of each r > 0 against r = 0, on matched
    trial indices. Pairing is legitimate at the trial level and only there:
    trial i has the same initial state AND the same displacement direction at
    every magnitude, because the direction is drawn from
    default_rng(555000 + trial_seed).
  * The reported estimate of delta is an INTERVAL, not a point: the largest r
    whose paired test does not reject, and the smallest r that does. A point
    estimate from an n = 10 ladder would be false precision.
  * Multiplicity: 6 paired tests on task 0 and 4 on task 3. Holm-corrected
    within task. Reported both ways, corrected first.
  * FALSIFICATION, and it is the point of the experiment: if the score does not
    decline with r, the arm is not using the constant, and every claim in the
    paper that rests on the blind arm must be withdrawn. This is a real risk,
    not a rhetorical one -- four LIBERO-Object tasks receive constants
    identical to within 0.47 mm and score {0, 0, 0, 10}.

E3 -- the Spatial control. Reset-oracle against blind on five LIBERO-Spatial
tasks, whose placement radii the criterion says no constant can serve.

  * Trials within a task are NOT exchangeable -- that was error 4, worth
    p = 0.039 of a claim we withdrew. The unit of analysis is the TASK.
  * Primary test: exact two-sided sign test on 5 paired task-level rates.
    ITS FLOOR IS p = 0.0625. With five tasks a perfect result CANNOT reach
    0.05, and we state that here, before seeing it, so that no reader has to
    wonder whether the threshold was chosen afterwards. We will report the
    p-value and the direction and will not call 0.0625 significant.
  * Secondary: cluster bootstrap over tasks for the pooled difference, 20000
    resamples, percentile interval.
  * PRECONDITION: if the oracle arm is itself at the floor, this design cannot
    distinguish "a constant cannot serve Spatial" from "our controller cannot
    do Spatial", and we report it as uninformative rather than as support.

  * WHAT THIS CELL NOW TESTS, recorded 2026-08-09 while E3 was still running
    and before any Spatial outcome was read. E3 was designed when the paper
    claimed LIBERO-Object was distinctively admissible, so blind failing on
    Spatial would have been a confirmation. E2 has since withdrawn that claim:
    at the tolerance E2 measures (>= 2 cm) the criterion says 8 of 10 Spatial
    tasks ADMIT a constant. The arm did not change and neither did the test.
    What changed is which outcome supports what, so we fix that here:

      blind does WELL on Spatial   consistent with delta >= 2 cm. Corroborates
                                   the withdrawal: a constant is not special
                                   to LIBERO-Object.
      blind FAILS while the        evidence that delta is smaller than E2's
      oracle succeeds              floor after all, since the criterion at
                                   >= 2 cm predicted otherwise. That would
                                   argue AGAINST the withdrawal, and we would
                                   report it that way.
      both at the floor            uninformative, per the precondition above.

    Note the second row is the one that would embarrass E2, and it is on
    record before the cell was read.

Input:  results/e23_trials.json  (per-trial outcomes fetched from the runner)
Output: results/e23_analysis.json
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "e23_trials.json"
OUT = REPO / "results" / "e23_analysis.json"
Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def mcnemar(a: dict, b: dict) -> dict:
    """Two-sided exact McNemar on the trials both arms actually ran."""
    keys = sorted(set(a) & set(b), key=int)
    lost = sum(1 for k in keys if a[k] and not b[k])
    gained = sum(1 for k in keys if b[k] and not a[k])
    n = lost + gained
    p = 1.0
    if n:
        tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1))
        p = min(1.0, 2.0 * tail / 2 ** n)
    return {"paired_n": len(keys), "lost": lost, "gained": gained,
            "p": round(p, 6)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 6)
    return out


def sign_test(diffs: list[float]) -> dict:
    """Exact two-sided sign test. Ties are dropped, which is what makes the
    floor a floor: with five tasks and no ties the smallest attainable p is
    2 * 0.5**5 = 0.0625."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n_nonzero": 0, "pos": 0, "neg": 0, "p": 1.0,
                "floor": None, "note": "all tasks tied"}
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1))
    return {"n_nonzero": n, "pos": pos, "neg": neg,
            "p": round(min(1.0, 2.0 * tail / 2 ** n), 6),
            "floor": round(2.0 / 2 ** n, 6)}


def cluster_bootstrap(pairs: list[tuple[list, list]], reps: int = 20000,
                      seed: int = 11) -> dict:
    """Resample TASKS with replacement, not trials. The trial-level bootstrap
    is the same error as the trial-level test: it assumes an exchangeability
    the design does not have."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    draws = []
    for _ in range(reps):
        pick = rng.choice(idx, size=len(pairs), replace=True)
        a = np.mean([np.mean(pairs[i][0]) for i in pick])
        b = np.mean([np.mean(pairs[i][1]) for i in pick])
        draws.append(a - b)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"reps": reps, "diff_mean": round(float(np.mean(draws)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def cell(trials: dict) -> dict:
    k, n = sum(bool(v) for v in trials.values()), len(trials)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson": wilson(k, n), "trials": trials}


def main() -> None:
    raw = json.loads(SRC.read_text())
    out: dict = {"source": str(SRC.relative_to(REPO)), "e2": {}, "e3": {}}

    # ---------------------------------------------------------------- E2
    for task in sorted({t for t, _ in (k.split("|") for k in raw["e2"])}):
        levels = sorted({float(j) for t, j in
                         (k.split("|") for k in raw["e2"]) if t == task})
        cells = {j: cell(raw["e2"][f"{task}|{j:g}"]) for j in levels}
        base = cells.get(0.0)
        tests, ps = {}, {}
        if base:
            for j in levels:
                if j == 0.0:
                    continue
                tests[f"{j:g}"] = mcnemar(base["trials"], cells[j]["trials"])
                ps[f"{j:g}"] = tests[f"{j:g}"]["p"]
        corrected = holm(ps) if ps else {}
        rejected = [float(j) for j, p in corrected.items() if p < 0.05]
        not_rej = [float(j) for j, p in corrected.items() if p >= 0.05]
        out["e2"][task] = {
            "levels_cm": levels,
            "cells": {f"{j:g}": {kk: vv for kk, vv in cells[j].items()
                                 if kk != "trials"} for j in levels},
            "paired_vs_zero": tests,
            "holm": corrected,
            "delta_interval_cm": [max(not_rej) if not_rej else None,
                                  min(rejected) if rejected else None],
            "declines": bool(rejected),
            "monotone_in_r": bool(all(
                cells[levels[i]]["rate"] >= cells[levels[i + 1]]["rate"] - 1e-9
                for i in range(len(levels) - 1))),
            "trials": {f"{j:g}": cells[j]["trials"] for j in levels},
        }
        if base and not rejected:
            out["e2"][task]["FALSIFIES_BLIND_ARM"] = True
        # POST-HOC, and labelled as such. The pre-registered rule above asks
        # whether any level rejects after Holm. It cannot tell a FLAT response
        # from an UNDERPOWERED one, and on a 10-trial ladder with four
        # comparisons that distinction is the whole question. We do not touch
        # the rule -- it stands as written, with its verdict -- and report this
        # beside it so a reader can see which case fired.
        if base:
            k = [cells[j]["k"] for j in levels]
            out["e2"][task]["post_hoc"] = {
                "raw_p_min": min(ps.values()) if ps else None,
                "raw_rejects_at_05": [j for j, pv in ps.items() if pv < 0.05],
                "strictly_decreasing_endpoints": k[0] > k[-1],
                "drop_from_zero": k[0] - k[-1],
                "reading": (
                    "FLAT: the score does not move, and the arm is not using "
                    "the constant"
                    if k[0] <= k[-1] else
                    "UNDERPOWERED, NOT FLAT: the score falls monotonically to "
                    f"zero ({k[0]}/{base['n']} to {k[-1]}/{cells[levels[-1]]['n']}) "
                    "but four Holm-corrected comparisons at n = 10 cannot "
                    "resolve it"
                    if not rejected else
                    "DECLINES, and the pre-registered test rejects"),
            }

    # ---------------------------------------------------------------- E3
    # E3 is last in the run order because it is the least critical of the
    # three. If the machine went away before it started, that is what we say:
    # a cell that was never run is not a null result, and the paper reports it
    # as not run.
    if not raw.get("e3"):
        out["e3"] = {"status": "NOT RUN",
                     "why": ("last in the value-ordered queue; the box was "
                             "CPU-quota-bound at 13.6 cores and the budget "
                             "went to E1, E5 and E2. Reported as not run, "
                             "not as a null.")}
        OUT.write_text(json.dumps(out, indent=1))
        print(f"wrote {OUT.relative_to(REPO)}")
        for task, d in out["e2"].items():
            row = "  ".join(f"r={j:g}:{d['cells'][f'{j:g}']['k']}/"
                            f"{d['cells'][f'{j:g}']['n']}"
                            for j in d["levels_cm"])
            print(f"E2 task {task}: {row}")
            print(f"   delta bracketed by {d['delta_interval_cm']} cm  "
                  f"(declines={d['declines']}, monotone={d['monotone_in_r']})")
            if d.get("FALSIFIES_BLIND_ARM"):
                print("   *** NO DECLINE AT ANY r: the blind arm is not using "
                      "the constant. Claims resting on it must be withdrawn.")
        print("E3: NOT RUN")
        return
    tasks = sorted({k.split("|")[1] for k in raw["e3"]}, key=int)
    per, pairs, diffs = {}, [], []
    for t in tasks:
        o = cell(raw["e3"][f"orc|{t}"])
        b = cell(raw["e3"][f"bl|{t}"])
        per[t] = {"oracle": {k: v for k, v in o.items() if k != "trials"},
                  "blind": {k: v for k, v in b.items() if k != "trials"},
                  "paired": mcnemar(o["trials"], b["trials"]),
                  "trials": {"oracle": o["trials"], "blind": b["trials"]}}
        keys = sorted(set(o["trials"]) & set(b["trials"]), key=int)
        pairs.append(([o["trials"][k] for k in keys],
                      [b["trials"][k] for k in keys]))
        diffs.append(o["rate"] - b["rate"])
    orc_mean = float(np.mean([np.mean(a) for a, _ in pairs]))
    bl_mean = float(np.mean([np.mean(b) for _, b in pairs]))
    st = sign_test(diffs)
    out["e3"] = {
        "unit_of_analysis": "task (trials within a task are not exchangeable)",
        "per_task": per,
        "task_level_mean": {"oracle": round(orc_mean, 4),
                            "blind": round(bl_mean, 4),
                            "diff": round(orc_mean - bl_mean, 4)},
        "sign_test": st,
        "cluster_bootstrap": cluster_bootstrap(pairs),
        "oracle_at_floor": orc_mean <= 0.10,
        "informative": bool(orc_mean > 0.10),
    }
    if not out["e3"]["informative"]:
        out["e3"]["VERDICT"] = (
            "UNINFORMATIVE: the oracle arm is itself at the floor, so this "
            "design cannot separate 'a constant cannot serve Spatial' from "
            "'our controller cannot do Spatial'.")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    for task, d in out["e2"].items():
        row = "  ".join(f"r={j:g}:{d['cells'][f'{j:g}']['k']}/"
                        f"{d['cells'][f'{j:g}']['n']}" for j in d["levels_cm"])
        print(f"E2 task {task}: {row}")
        print(f"   delta bracketed by {d['delta_interval_cm']} cm  "
              f"(declines={d['declines']}, monotone={d['monotone_in_r']})")
        if d.get("FALSIFIES_BLIND_ARM"):
            print("   *** NO DECLINE AT ANY r: the blind arm is not using the "
                  "constant. Claims resting on it must be withdrawn. ***")
    e3 = out["e3"]
    print(f"E3 task-level: oracle {e3['task_level_mean']['oracle']:.3f} vs "
          f"blind {e3['task_level_mean']['blind']:.3f}, "
          f"sign test p={e3['sign_test']['p']} "
          f"(floor {e3['sign_test']['floor']}), "
          f"cluster bootstrap {e3['cluster_bootstrap']['ci95']}")
    if e3.get("VERDICT"):
        print("   " + e3["VERDICT"])


if __name__ == "__main__":
    main()
