"""Generate the LIBERO-Object evaluation states the benchmark's task files ask for.

Section 2 establishes that LIBERO-Object ships fifty initial states that do not
sample the region each task's own BDDL declares -- on six of ten tasks all
fifty place the target at one point inside a declared 5 x 5 cm box. This script
produces the repaired states: same file format, same everything else, with the
target's position drawn uniformly from the box the task already declares.

It is deliberately conservative. It changes ONLY the two columns holding the
target object's x and y, leaves every other object, every joint velocity and
the time field untouched, and refuses to write a file if it cannot locate those
columns unambiguously or if the shipped positions do not already lie inside the
declared region (which would mean the region and the state array are in
different frames and the whole operation is unsound).

What this does NOT do, and what a user must do before trusting the output: run
the states through the simulator to confirm no interpenetration. The declared
region is the benchmark's own statement of where the object may validly go, so
positions drawn from it should be admissible by construction -- but "should be"
is not a check, and we do not have a simulator here. The output is therefore a
CANDIDATE repaired suite, and the flag says so.

Usage:
    python scripts/resample_init_states.py --out results/resampled_init
    python scripts/resample_init_states.py --dry-run          # report only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / ".libero_src/libero/libero"
SUITE = "libero_object"


def load_states(path: Path) -> np.ndarray:
    import torch
    return np.asarray(torch.load(path, weights_only=False), dtype=np.float64)


def declared_box(task: str, target: str):
    """(x_lo, y_lo, x_hi, y_hi) in metres for the target's own :init region."""
    txt = (LIB / "bddl_files" / SUITE / f"{task}.bddl").read_text()
    init = re.search(r"\(:init(.*?)\n\s*\)", txt, re.S).group(1)
    rname = dict(re.findall(r"\(On\s+(\S+)\s+(\S+)\)", init)).get(target)
    if rname is None:
        return None
    for m in re.finditer(
            r"\((\w+)\s*\(:target[^)]*\)\s*\(:ranges\s*\(\s*\(([^)]*)\)", txt):
        if rname.endswith(m.group(1)):
            a, b, c, d = (float(v) for v in m.group(2).split())
            return min(a, c), min(b, d), max(a, c), max(b, d)
    return None


def target_columns(A: np.ndarray, xy: np.ndarray) -> tuple[int, int]:
    """The (x, y) columns of the target, matched against the forensics values.

    Refuses on ambiguity: if more than one column pair reproduces the shipped
    positions we cannot know which is the target's, and writing the wrong pair
    would corrupt a different object silently.
    """
    hits = []
    for c in range(A.shape[1] - 1):
        if np.allclose(A[:, c], xy[:, 0], atol=1e-12) and \
           np.allclose(A[:, c + 1], xy[:, 1], atol=1e-12):
            hits.append(c)
    if len(hits) != 1:
        raise RuntimeError(f"target columns ambiguous ({len(hits)} candidates)")
    return hits[0], hits[0] + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/resampled_init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
    rng = np.random.default_rng(a.seed)
    outdir = REPO / a.out
    if not a.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
    report, skipped = [], []

    for t in J["suites"][SUITE]["tasks"]:
        task, tgt = t["task"], t["primary_target"]
        src = LIB / "init_files" / SUITE / f"{task}.pruned_init"
        o = next((o for o in t["objects"] if o["object"] == tgt), None)
        box = declared_box(task, tgt)
        if o is None or box is None or not src.exists():
            skipped.append((task, "no region or no state file")); continue

        A = load_states(src)
        xy = np.asarray(o["xy_per_state_m"], dtype=float)
        try:
            cx, cy = target_columns(A, xy)
        except RuntimeError as e:
            skipped.append((task, str(e))); continue

        x0, y0, x1, y1 = box
        # Frame check: the shipped points must already lie in the declared box
        # (with a small margin for the sampler's own overshoot). If they do not,
        # region and state array are in different frames and resampling would
        # place the object somewhere arbitrary.
        m = 0.01
        if not (np.all(xy[:, 0] > x0 - m) and np.all(xy[:, 0] < x1 + m) and
                np.all(xy[:, 1] > y0 - m) and np.all(xy[:, 1] < y1 + m)):
            skipped.append((task, "shipped states outside declared region")); continue

        B = A.copy()
        B[:, cx] = rng.uniform(x0, x1, size=len(B))
        B[:, cy] = rng.uniform(y0, y1, size=len(B))
        before = float((xy.max(0) - xy.min(0)).max() * 100)
        after = float((B[:, [cx, cy]].max(0) - B[:, [cx, cy]].min(0)).max() * 100)
        report.append({"task": task, "target": tgt, "columns": [cx, cy],
                       "declared_cm": [round((x1 - x0) * 100, 3),
                                       round((y1 - y0) * 100, 3)],
                       "spread_before_cm": round(before, 4),
                       "spread_after_cm": round(after, 4)})
        if not a.dry_run:
            import torch
            torch.save(B, outdir / f"{task}.pruned_init")

    print(f"{'task':<44}{'declared':>10}{'before':>9}{'after':>9}")
    for r in report:
        print(f"{r['task'][:42]:<44}{r['declared_cm'][0]:>7.1f} cm"
              f"{r['spread_before_cm']:>8.2f}{r['spread_after_cm']:>9.2f}")
    for task, why in skipped:
        print(f"SKIPPED {task[:44]:<46} {why}")
    print(f"\n{len(report)} repaired, {len(skipped)} skipped")
    if not a.dry_run:
        meta = {"suite": SUITE, "seed": a.seed, "tasks": report,
                "skipped": [{"task": t, "reason": w} for t, w in skipped],
                "status": "CANDIDATE -- not validated in a simulator",
                "caveat": ("only the target's x,y columns are changed; positions "
                           "are drawn from the region the task file declares. "
                           "Physical validity has NOT been checked -- run these "
                           "through the sim before using them for evaluation.")}
        (outdir / "MANIFEST.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {len(report)} files + MANIFEST.json to {a.out}")


if __name__ == "__main__":
    main()
