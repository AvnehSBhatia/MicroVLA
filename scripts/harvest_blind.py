"""Harvest the blind-anchor cell from per-(task, trial) job directories.

One rule, and it is the reason this script exists rather than a shell one-liner:
a cell is reported only when every planned trial has a recorded outcome.
Successes terminate early and therefore finish first, so a partial harvest is a
CENSORED sample, not a small one -- an earlier round of this work published
7/7 for a cell that was in fact 8/10, and the difference was entirely which
trials had had time to fail. Missing trials are named, not dropped.

Usage: python scripts/harvest_blind.py <dir-of-job-dirs> --planned 10 --out results/blind_cells.json
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


def mcnemar_exact(a: dict, b: dict) -> tuple[int, int, int, float]:
    keys = sorted(set(a) & set(b), key=int)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    d = n01 + n10
    if d == 0:
        return len(keys), 0, 0, 1.0
    lo = min(n01, n10)
    p = sum(math.comb(d, i) for i in range(lo + 1)) / (2 ** d) * 2
    return len(keys), n01, n10, min(1.0, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--planned", type=int, default=10)
    ap.add_argument("--out", default="results/blind_cells.json")
    a = ap.parse_args()

    cells: dict[str, dict] = {}
    for d in sorted(Path(a.root).iterdir()):
        m = re.match(r"(?P<arm>[a-z]+)_t(?P<t>\d+)_r(?P<r>\d+)$", d.name)
        log = d / "log.txt"
        if not m or not log.exists():
            continue
        txt = log.read_text(errors="ignore")
        hit = re.findall(r"mean_success ([0-9.]+)", txt)
        if not hit:                       # killed, crashed, or still running
            continue
        cell = f"{m['arm']}_t{m['t']}"
        cells.setdefault(cell, {"trials": {}})["trials"][m["r"]] = \
            bool(float(hit[-1]) >= 0.5)

    out = {}
    for name, c in sorted(cells.items()):
        tr = c["trials"]
        k, n = sum(tr.values()), len(tr)
        lo, hi = wilson(k, n)
        missing = [str(i) for i in range(a.planned) if str(i) not in tr]
        out[name] = {"k": k, "n": n, "planned_n": a.planned,
                     "complete": not missing, "missing_trials": missing,
                     "rate": k / n if n else None, "wilson": [round(lo, 4), round(hi, 4)],
                     "trials": {t: tr[t] for t in sorted(tr, key=int)}}
        flag = "" if not missing else f"  INCOMPLETE, missing {','.join(missing)}"
        print(f"{name:<16} {k:>2}/{n:<2} = {k/n if n else 0:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]{flag}")

    # Paired against the published anchor cells where the trial indices match.
    ref = Path("results/pod_cells.json")
    if ref.exists() and "blind_t0" in out and out["blind_t0"]["complete"]:
        pod = json.loads(ref.read_text())
        for other in ("P0_ref_heldout", "P2_E5_fixed", "P2_E5_random", "P2_E5_oracle"):
            if other not in pod or not pod[other].get("complete"):
                continue
            n, n01, n10, p = mcnemar_exact(out["blind_t0"]["trials"], pod[other]["trials"])
            out.setdefault("mcnemar", {})[f"blind_t0_vs_{other}"] = {
                "n_paired": n, "n01": n01, "n10": n10, "p": round(p, 4)}
            print(f"  blind_t0 vs {other:<16} n={n} discordant {n01}/{n10}  p={p:.4f}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
