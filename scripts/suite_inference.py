"""Suite-level inference at the correct unit of analysis.

The previous version paired 100 (task, trial) cells and ran an exact McNemar,
returning p = 0.039. Adversarial review showed that is pseudoreplication and I
confirmed it: all 12 discordant pairs live in 2 of 10 tasks and 10 of them in
one, where the ten "trials" are near-identical replays (blind terminates at
steps 210-214, sd 1.08; the head runs to the 300 cap on all ten). Trials within
a task share an object, a scene, a goal supply and a near-deterministic
trajectory. They are not exchangeable, and McNemar assumes they are.

This script reports the design's actual unit -- the task -- and quantifies how
badly the trial-level test was inflated. It also runs the within-task test for
the one task where the contrast is real, which is the result that survives.

Usage: python scripts/suite_inference.py
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    keys = sorted(set(a) & set(b), key=int)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    d = n01 + n10
    if d == 0:
        return n01, n10, 1.0
    lo = min(n01, n10)
    return n01, n10, min(1.0, sum(math.comb(d, i) for i in range(lo + 1)) / 2 ** d * 2)


def main() -> None:
    S = json.loads((REPO / "results/suite_cells.json").read_text())
    B = json.loads((REPO / "results/blind_cells.json").read_text())
    P = json.loads((REPO / "results/pod_cells.json").read_text())
    blind = {0: B["blind_t0"]} | {t: S[f"blind_t{t}"] for t in range(1, 10)}
    head = {0: P["P0_ref_heldout"]} | {t: S[f"head_t{t}"] for t in range(1, 10)}

    bk = np.array([blind[t]["k"] for t in range(10)])
    hk = np.array([head[t]["k"] for t in range(10)])

    # --- what the paper used to report (kept, labelled invalid) --------------
    a = {f"{t}_{r}": v for t in range(10) for r, v in blind[t]["trials"].items()}
    b = {f"{t}_{r}": v for t in range(10) for r, v in head[t]["trials"].items()}
    n01, n10, p_trial = mcnemar(a, b)
    by_task = {t: sum(1 for r in blind[t]["trials"]
                      if blind[t]["trials"][r] != head[t]["trials"].get(r))
               for t in range(10)}

    # --- the design's actual unit: the task ---------------------------------
    wb = int((bk > hk).sum()); wh = int((hk > bk).sum()); ties = int((bk == hk).sum())
    d = wb + wh
    p_task = (1.0 if d == 0 else
              min(1.0, sum(math.comb(d, i) for i in range(min(wb, wh) + 1)) / 2 ** d * 2))

    # --- exact permutation over task-level sign flips ------------------------
    diff = (bk - hk) / 10.0
    nz = np.flatnonzero(diff != 0)
    obs = float(diff.mean())
    cnt = tot = 0
    for signs in product([1, -1], repeat=len(nz)):
        d2 = diff.copy(); d2[nz] = diff[nz] * np.array(signs)
        tot += 1
        cnt += abs(float(d2.mean())) >= abs(obs) - 1e-12
    p_perm = cnt / tot

    # --- cluster bootstrap over tasks ---------------------------------------
    rng = np.random.default_rng(0)
    boot = np.array([float((bk[i] - hk[i]).mean() / 10.0)
                     for i in (rng.integers(0, 10, (10000, 10)))])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # --- design effect -------------------------------------------------------
    def icc(k):
        p = k / 10.0
        mse_b = 10 * np.var(p, ddof=1)
        mse_w = float(np.mean(p * (1 - p) * 10 / 9)) if np.any((p > 0) & (p < 1)) else 0.0
        return max(0.0, (mse_b - mse_w) / (mse_b + 9 * mse_w)) if (mse_b + 9 * mse_w) else 1.0
    deff = 1 + 9 * max(icc(bk), icc(hk))

    n01_3, n10_3, p_t3 = mcnemar(blind[3]["trials"], head[3]["trials"])

    out = {
        "per_task": {str(t): {"blind": int(bk[t]), "head": int(hk[t])} for t in range(10)},
        "totals": {"blind_k": int(bk.sum()), "head_k": int(hk.sum()), "n": 100,
                   "blind_wilson_iid": [round(v, 4) for v in wilson(int(bk.sum()), 100)],
                   "head_wilson_iid": [round(v, 4) for v in wilson(int(hk.sum()), 100)]},
        "trial_level_INVALID": {
            "n01": n01, "n10": n10, "p": round(p_trial, 5),
            "discordant_by_task": by_task,
            "why_invalid": ("trials within a task are not exchangeable; all "
                            "discordance lives in 2 of 10 tasks and 10 of 12 "
                            "pairs in one"),
            "design_effect": round(float(deff), 2),
            "effective_n": round(100 / float(deff), 1)},
        "task_level": {"blind_wins": wb, "head_wins": wh, "ties": ties,
                       "sign_test_p": round(p_task, 4),
                       "permutation_p": round(p_perm, 4),
                       "mean_diff": round(obs, 4),
                       "cluster_bootstrap_95ci": [round(ci[0], 4), round(ci[1], 4)]},
        "task3_within": {"blind": int(bk[3]), "head": int(hk[3]),
                         "n01": n01_3, "n10": n10_3, "p": round(p_t3, 5),
                         "caveat": ("ten near-identical replays of one "
                                    "deterministic contrast; read as a single "
                                    "task-level event, not ten draws")},
    }
    Path(REPO / "results/suite_inference.json").write_text(json.dumps(out, indent=2))

    print("per task (blind / head):",
          "  ".join(f"{t}:{bk[t]}/{hk[t]}" for t in range(10)))
    print(f"\ntotals: blind {bk.sum()}/100, head {hk.sum()}/100")
    print(f"\nTRIAL-LEVEL McNemar (what we used to report): {n01}-{n10}, p = {p_trial:.4f}")
    print(f"  discordant pairs by task: {by_task}")
    print(f"  design effect {deff:.2f} -> effective n ~ {100/deff:.0f}, not 100")
    print(f"\nTASK-LEVEL (the design's unit):")
    print(f"  blind wins {wb}, head wins {wh}, ties {ties}")
    print(f"  exact sign test p = {p_task:.4f}")
    print(f"  exact sign-flip permutation p = {p_perm:.4f}")
    print(f"  cluster bootstrap 95% CI on the difference: "
          f"[{ci[0]:+.3f}, {ci[1]:+.3f}]  (contains 0: {ci[0] <= 0 <= ci[1]})")
    print(f"\nwithin task 3: {bk[3]}/10 vs {hk[3]}/10, {n01_3}-{n10_3}, p = {p_t3:.5f}")
    print("\nwrote results/suite_inference.json")


if __name__ == "__main__":
    main()
