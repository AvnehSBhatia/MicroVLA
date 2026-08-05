"""Collect tonight's pod cells into the paper's paired matrix + stats.

Run AFTER the gates land (p50-rand n=50, devcells, v5 sweep, butter v6).
Reads results/logs fetched from the pod (rsync the eval_results dirs and
logs first, or run on the pod), verifies every number against its
results.json, and prints a markdown block ready for MANUSCRIPT_v2.md.

Usage:
    python scripts/collect_tonight.py --root <dir with eval_results/ + logs/>
"""
import argparse
import glob
import json
import math
import os
import re

Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cell(root, out_dir):
    js = sorted(glob.glob(os.path.join(root, out_dir, "*_results.json")))
    if not js:
        return None
    d = json.load(open(js[-1]))
    return {"k": round(d["mean_success"] * d["n_trials"]), "n": d["n_trials"],
            "rate": d["mean_success"], "file": os.path.basename(js[-1]),
            "per_task": d.get("per_task")}


def per_trial(log_path, block=1):
    """Per-trial success list from an eval log; block selects the Nth run
    (1-indexed) when a log holds several appended runs."""
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, errors="replace").read()
    runs, cur = [], None
    for line in txt.splitlines():
        if re.search(r"START .*trial 0$", line):
            cur = []
            runs.append(cur)
        m = re.search(r"DONE .*trial (\d+): success=(True|False)", line)
        if m and cur is not None:
            cur.append(m.group(2) == "True")
    if len(runs) < block:
        return None
    return runs[block - 1]


def sign_test(b, c):
    """Exact two-sided sign test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt(cl):
    if cl is None:
        return "MISSING"
    lo, hi = wilson(cl["k"], cl["n"])
    return f"{cl['k']}/{cl['n']} [{lo:.2f}, {hi:.2f}] ({cl['file']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    r = a.root

    print("## Tonight's same-stack cells (every number from its results.json)\n")
    cells = {
        "memorized rand ±4cm (control)": cell(r, "eval_results/unaided_memctl_rand"),
        "flagship rand ±4cm n=50": cell(r, "eval_results/unaided_p50_rand"),
        "flagship held-out n=50": cell(r, "eval_results/unaided_p50_heldout"),
        "memorized dev unshifted (re-measure)": cell(r, "eval_results/unaided_memdev"),
        "flagship dev unshifted (re-measure)": cell(r, "eval_results/unaided_flagdev"),
        "flagship all-tasks zero-shot": cell(r, "eval_results/v5_alltasks_zeroshot"),
        "v6 butter unaided": cell(r, "eval_results/unaided_v6_butter"),
        "v6 soup regression": cell(r, "eval_results/unaided_v6_soup"),
    }
    for k, v in cells.items():
        print(f"- {k}: {fmt(v)}")
        if v and v.get("per_task") and len(v["per_task"]) > 1:
            for t, s in v["per_task"].items():
                print(f"    - {t}: {s}")

    mem = per_trial(os.path.join(r, "logs/memctl.log"), block=1)
    p50 = per_trial(os.path.join(r, "logs/power50.log"), block=2)
    if mem and p50 and len(p50) >= len(mem):
        n = len(mem)
        b = sum(1 for i in range(n) if p50[i] and not mem[i])
        c = sum(1 for i in range(n) if mem[i] and not p50[i])
        both = sum(1 for i in range(n) if mem[i] and p50[i])
        print(f"\nPaired (identical draws, first {n}): flagship-only b={b}, "
              f"memorized-only c={c}, both={both}, exact sign p={sign_test(b, c):.3f}")
    memdev = per_trial(os.path.join(r, "logs/devcells.log"), block=1)
    if memdev and mem and len(memdev) == len(mem):
        b = sum(1 for i in range(len(mem)) if memdev[i] and not mem[i])
        c = sum(1 for i in range(len(mem)) if mem[i] and not memdev[i])
        print(f"Paired memorized dev-vs-rand (same states, same stack): "
              f"unshifted-only b={b}, shifted-only c={c}, "
              f"exact sign p={sign_test(b, c):.3f}")


if __name__ == "__main__":
    main()
