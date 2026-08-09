"""E1: does LIBERO-Object's frozen placement matter to a policy?

Section 2 establishes that the suite ships a constant where its task file
declares a 5 x 5 cm region. That is a fact about files, and the paper has been
unable to say whether it MATTERS -- every attempt to demonstrate an exploit
failed for reasons Section 6 documents.

This experiment asks it directly and within one build: run the released head
and the blind constant on task 0, once on the fifty states LIBERO ships and
once on fifty drawn from the region its own BDDL declares. Nothing else
changes -- same weights, same seed, same trial indices, same machine, same
package versions.

  blind on repaired is the manipulation check. The constant is bit-identical
  to the shipped target; against a target that now moves it should degrade. If
  it does not, our repair is not doing what Section 2 claims and the finding is
  cosmetic.

  head on repaired is the open question, and either answer is worth having. A
  drop means the pinning was load-bearing for a trained policy. No drop bounds
  the defect's practical cost, which nobody has measured.

Reports paired McNemar within each arm (same trial indices, one variable
changed) and Wilson intervals on every cell.

Usage: python scripts/analyze_e1.py <dir-of-job-dirs> --out results/e1_shipped_vs_repaired.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
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


def mcnemar(a: dict, b: dict) -> tuple[int, int, int, float]:
    keys = sorted(set(a) & set(b), key=int)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    d = n01 + n10
    if d == 0:
        return len(keys), 0, 0, 1.0
    lo = min(n01, n10)
    return len(keys), n01, n10, min(1.0, sum(math.comb(d, i)
                                             for i in range(lo + 1)) / 2 ** d * 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--planned", type=int, default=30)
    ap.add_argument("--out", default="results/e1_shipped_vs_repaired.json")
    a = ap.parse_args()

    cells: dict[str, dict[str, bool]] = {}
    for d in sorted(Path(a.root).iterdir()):
        m = re.match(r"(head|blind)_(shipped|repaired)_r(\d+)$", d.name)
        log = d / "log.txt"
        if not m or not log.exists():
            continue
        hit = re.findall(r"mean_success ([0-9.]+)", log.read_text(errors="ignore"))
        if not hit:
            continue                      # unfinished: never counted, never dropped silently
        cells.setdefault(f"{m.group(1)}_{m.group(2)}", {})[m.group(3)] = \
            float(hit[-1]) >= 0.5

    out: dict = {"planned_n": a.planned, "cells": {}}
    for name, tr in sorted(cells.items()):
        k, n = sum(tr.values()), len(tr)
        lo, hi = wilson(k, n)
        missing = [str(i) for i in range(a.planned) if str(i) not in tr]
        out["cells"][name] = {"k": k, "n": n, "rate": k / n if n else None,
                              "wilson": [round(lo, 4), round(hi, 4)],
                              "complete": not missing, "missing": missing,
                              "trials": {t: tr[t] for t in sorted(tr, key=int)}}
        flag = "" if not missing else f"   INCOMPLETE ({len(missing)} missing)"
        print(f"{name:<18} {k:>2}/{n:<2} = {k/n if n else 0:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]{flag}")

    print()
    for arm in ("blind", "head"):
        s, r = f"{arm}_shipped", f"{arm}_repaired"
        if s in cells and r in cells:
            n, n01, n10, p = mcnemar(cells[s], cells[r])
            out.setdefault("paired", {})[arm] = {
                "n_paired": n, "shipped_only": n01, "repaired_only": n10,
                "p": round(p, 5)}
            print(f"{arm:<6} shipped vs repaired: n={n}, {n01} lost / {n10} gained, "
                  f"exact McNemar p = {p:.4f}")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
