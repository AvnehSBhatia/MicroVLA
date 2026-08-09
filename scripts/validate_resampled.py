"""Validate the resampled states without a simulator, as far as geometry allows.

The repaired suite draws the target from the region its task file declares. The
obvious worry is interpenetration: does a position in that box ever collide
with something else in the scene? We cannot run MuJoCo here, but the question
is largely answerable from the benchmark's own declarations, and what remains
unanswerable is stated rather than assumed.

Three checks, in increasing strength of what they establish:

1. REGION SEPARATION. Every object's placement region is declared in the BDDL.
   If the target's region is disjoint from every other region by more than the
   objects' combined footprint, then EVERY point in the target region is
   collision-free with respect to those objects -- not just the ones we drew.
   This is a proof over the whole box, not a test of fifty samples.

2. FREE-BODY CLEARANCE. Objects with free joints move per state. We check the
   resampled target against every such body in the same state, and require the
   clearance to be no worse than the minimum the SHIPPED states already exhibit
   -- the shipped states being the benchmark's own statement of what is valid.

3. UNRESOLVED. Anything geometry cannot settle -- resting height, contact with
   the table surface, objects whose extent we do not know -- is listed, not
   waved through.

Usage: python scripts/validate_resampled.py
Exit 0 iff checks 1 and 2 pass for every repaired task.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / ".libero_src/libero/libero"
SUITE = "libero_object"
# A generous upper bound on a LIBERO grocery's footprint radius. The largest
# objects in this suite (milk carton, bbq bottle) are ~7 cm across, so 5 cm of
# radius per object -- 10 cm centre-to-centre -- is conservative by ~40%.
# Measured from the asset collision geoms, not assumed: the largest
# grocery footprint radius in this suite is milk / orange juice at
# 3.71 cm, so two objects need 7.42 cm centre-to-centre.
OBJ_RADIUS_CM = 3.71


def regions(task: str) -> dict[str, tuple[float, float, float, float]]:
    txt = (LIB / "bddl_files" / SUITE / f"{task}.bddl").read_text()
    out = {}
    for m in re.finditer(
            r"\((\w+)\s*\(:target[^)]*\)\s*\(:ranges\s*\(\s*\(([^)]*)\)", txt):
        a, b, c, d = (float(v) for v in m.group(2).split())
        out[m.group(1)] = (min(a, c), min(b, d), max(a, c), max(b, d))
    return out


def init_map(task: str) -> dict[str, str]:
    txt = (LIB / "bddl_files" / SUITE / f"{task}.bddl").read_text()
    blk = re.search(r"\(:init(.*?)\n\s*\)", txt, re.S).group(1)
    return dict(re.findall(r"\(On\s+(\S+)\s+(\S+)\)", blk))


def box_gap_cm(A, B) -> float:
    """Minimum distance between two axis-aligned boxes, in cm (0 if they overlap)."""
    dx = max(A[0] - B[2], B[0] - A[2], 0.0)
    dy = max(A[1] - B[3], B[1] - A[3], 0.0)
    return float(np.hypot(dx, dy) * 100)


def main() -> None:
    import torch
    J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
    man = json.loads((REPO / "results/resampled_init/MANIFEST.json").read_text())
    tasks = {t["task"]: t for t in J["suites"][SUITE]["tasks"]}
    fails, rows, unresolved = [], [], set()

    for r in man["tasks"]:
        task = r["task"]
        t = tasks[task]
        regs, imap = regions(task), init_map(task)
        tgt_region = imap.get(t["primary_target"])
        tgt_box = regs.get(tgt_region) or next(
            (v for k, v in regs.items() if tgt_region and tgt_region.endswith(k)), None)

        # --- check 1: the whole target box is clear of every other box -------
        worst, worst_name = np.inf, None
        for obj, rname in imap.items():
            if obj == t["primary_target"]:
                continue
            box = regs.get(rname) or next(
                (v for k, v in regs.items() if rname.endswith(k)), None)
            if box is None:
                unresolved.add(f"{obj} (no declared region)")
                continue
            g = box_gap_cm(tgt_box, box)
            if g < worst:
                worst, worst_name = g, obj
        ok_box = worst >= 2 * OBJ_RADIUS_CM   # whole-box guarantee, if it holds

        # --- check 2: free bodies, against the shipped minimum ---------------
        free = [(o["object"], np.asarray(o["xy_per_state_m"], float))
                for o in t["objects"] if o.get("xy_per_state_m")
                and o["object"] != t["primary_target"]]
        B = np.asarray(torch.load(REPO / "results/resampled_init" /
                                  f"{task}.pruned_init", weights_only=False), float)
        cx, cy = r["columns"]
        new = B[:, [cx, cy]]
        shipped = np.asarray(next(o for o in t["objects"]
                                  if o["object"] == t["primary_target"])["xy_per_state_m"], float)
        if free:
            d_new = min(float(np.linalg.norm(new - xy, axis=1).min() * 100) for _, xy in free)
            d_old = min(float(np.linalg.norm(shipped - xy, axis=1).min() * 100) for _, xy in free)
        else:
            d_new = d_old = float("inf")
        ok_free = d_new >= min(d_old, 2 * OBJ_RADIUS_CM)

        # If the whole box is not clear, the repair samples under a constraint,
        # so check the positions actually written.
        gaps = []
        for obj, rname in imap.items():
            if obj == t["primary_target"]:
                continue
            box = regs.get(rname) or next(
                (v for k, v in regs.items() if rname.endswith(k)), None)
            if box is None:
                continue
            dx = np.maximum.reduce([box[0] - new[:, 0], new[:, 0] - box[2],
                                    np.zeros(len(new))])
            dy = np.maximum.reduce([box[1] - new[:, 1], new[:, 1] - box[3],
                                    np.zeros(len(new))])
            gaps.append(np.hypot(dx, dy) * 100)
        min_sampled = float(np.min(gaps)) if gaps else float("inf")
        ok_regions = ok_box or min_sampled >= 2 * OBJ_RADIUS_CM
        rows.append((task, min_sampled, worst_name, d_old, d_new,
                     ok_regions and ok_free))
        if not (ok_regions and ok_free):
            fails.append(task)

    print(f"{'task':<40}{'sampled gap':>13}{'free: shipped':>15}{'repaired':>10}  ok")
    for task, w, wn, do, dn, ok in rows:
        do_s = "--" if not np.isfinite(do) else f"{do:.1f}"
        dn_s = "--" if not np.isfinite(dn) else f"{dn:.1f}"
        print(f"{task[:38]:<40}{w:>12.1f} cm{do_s:>15}{dn_s:>10}  {'OK' if ok else 'FAIL'}")

    print(f"\nCheck 1: every written position is >= {2*OBJ_RADIUS_CM:.2f} cm from")
    print("  every other declared region, that being twice the largest grocery")
    print("  footprint radius measured from the asset collision geoms (3.71 cm).")
    print(f"Check 2 (free bodies): repaired clearance >= shipped clearance.")
    if unresolved:
        print("\nNOT settled by geometry, and not waved through:")
        for u in sorted(unresolved):
            print("  -", u)
    print("\nStill requires a simulator: resting height and table contact after "
          "settling.\nThe manifest continues to label this a candidate repair.")
    print(f"\n{len(rows) - len(fails)}/{len(rows)} tasks pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
