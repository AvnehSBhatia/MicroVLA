"""Turn the pod's harvested per-trial outcomes into the paper's cells.

Reads the JSON emitted by the harvester (cell -> {k, n, trials{trial: bool}})
and reports each referee experiment with its n, Wilson interval and, where the
comparison is genuinely paired, an exact McNemar test.

Pairing discipline, because it is easy to get wrong here:

* E8 swap vs its own baseline IS paired --- same head, same seed, same init
  state per trial index, one flag differs. McNemar applies.
* E5 anchors vs the learned head ARE paired --- same seed, same states, only the
  goal supply differs. McNemar applies.
* E4 (states 20--49) vs held-out (states 10--19) is NOT paired: different init
  states. Reported as two independent proportions with a Newcombe interval on
  the difference, never as McNemar.

Usage: python scripts/analyze_pod_cells.py results/pod_cells.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a: dict, b: dict) -> tuple[int, int, int, float]:
    """Exact two-sided McNemar over trial indices present in BOTH cells."""
    keys = sorted(set(a) & set(b), key=int)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    d = n01 + n10
    if d == 0:
        return len(keys), n01, n10, 1.0
    lo = min(n01, n10)
    p = sum(math.comb(d, i) for i in range(lo + 1)) / (2 ** d) * 2
    return len(keys), n01, n10, min(1.0, p)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe hybrid-score interval for p1 - p2 (independent samples)."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text())
    cells = {k: v for k, v in d.items()}

    def show(name: str) -> None:
        c = cells.get(name)
        if not c or not c["n"]:
            print(f"  {name:<26} NOT RUN")
            return
        lo, hi = wilson(c["k"], c["n"])
        print(f"  {name:<26} {c['k']:>3}/{c['n']:<3} = {c['k']/c['n']:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]")

    print("=== P0 REFERENCE: does this pod reproduce the published cell? ===")
    show("P0_ref_heldout")
    ref = cells.get("P0_ref_heldout")
    if ref and ref["n"]:
        lo, hi = wilson(ref["k"], ref["n"])
        print(f"  published deployment-stack value for this band: 7/10 = 0.700")
        # k=8 vs k=7 out of 10 is one trial. The question is not whether the
        # counts are equal but whether the published value is compatible with
        # this cell, so test containment rather than eyeballing the difference.
        print(f"  -> published 0.700 is {'INSIDE' if lo <= 0.7 <= hi else 'OUTSIDE'} "
              f"this cell's Wilson interval [{lo:.3f}, {hi:.3f}]: "
              f"{'reproduces' if lo <= 0.7 <= hi else 'DOES NOT reproduce'}.")

    print("\n=== E4 (BLOCKING): untouched init states 20-49 ===")
    show("P1_E4_untouched")
    e4 = cells.get("P1_E4_untouched")
    if e4 and e4["n"] >= 10:
        lo, hi = newcombe(e4["k"], e4["n"], 35, 50)
        print(f"  vs published held-out 35/50 = 0.700 (INDEPENDENT states, not paired)")
        print(f"  difference {e4['k']/e4['n'] - 0.700:+.3f}, Newcombe 95% [{lo:+.3f}, {hi:+.3f}]")
        p = e4["k"] / e4["n"]
        verdict = ("WITHIN the pre-registered [0.50, 0.75]" if 0.50 <= p <= 0.75
                   else "BELOW 0.40 -- the 0.700 is substantially a selection artifact"
                   if p < 0.40 else
                   "ABOVE 0.80 -- the burn hurt rather than helped"
                   if p > 0.80 else "OUTSIDE the pre-registered band")
        print(f"  pre-registration: {verdict}")

    print("\n=== E5 (BLOCKING): shell anchors -- ceiling, floor, hand-written memorizer ===")
    for m in ("oracle", "random", "fixed"):
        show(f"P2_E5_{m}")
    base = cells.get("P0_ref_heldout")
    for m in ("oracle", "random", "fixed"):
        c = cells.get(f"P2_E5_{m}")
        if c and base and c["n"] and base["n"]:
            n, n01, n10, p = mcnemar_exact(base["trials"], c["trials"])
            print(f"  learned head vs {m:<7} (paired, n={n}): "
                  f"{n01} favour learned, {n10} favour anchor, exact p={p:.4f}")

    print("\n=== E8: instruction swap, powered up ===")
    for k in ("P3_E8_swap_v5", "P3_E8_base_cov", "P3_E8_swap_cov"):
        show(k)
    for tag, b, s in (("flagship", "P0_ref_heldout", "P3_E8_swap_v5"),
                      ("coverage", "P3_E8_base_cov", "P3_E8_swap_cov")):
        cb, cs = cells.get(b), cells.get(s)
        if cb and cs and cb["n"] and cs["n"]:
            n, n01, n10, p = mcnemar_exact(cb["trials"], cs["trials"])
            print(f"  {tag}: paired n={n}, {n01} lost by swapping, {n10} gained, "
                  f"exact McNemar p={p:.4f}")

    print("\n=== E7: randomization sweep ===")
    for r in ("02", "04", "06", "08"):
        show(f"P4_E7_v5_r{r}")


if __name__ == "__main__":
    main()
