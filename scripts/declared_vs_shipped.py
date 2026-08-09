"""How much of the region a LIBERO task DECLARES do its shipped states use?

This is the one measurement in this paper that needs no tolerance, no
controller, no policy and no simulator -- only two files the benchmark ships.
Every other claim we make is conditional on delta; this one is not, which is
why it survived three rounds of adversarial review while the others were being
withdrawn.

Each task's BDDL declares a sampling region for its target as an axis-aligned
box, e.g. for `pick_up_the_alphabet_soup`:

    (target_object_region (:target floor)
        (:ranges ((-0.145 -0.265 -0.095 -0.215))))

That is 5 cm x 5 cm. The fifty shipped initial states place the alphabet soup
at a single point -- the box's exact centre. The generator randomises; the
released evaluation states do not.

We report, per task, the declared box side and the shipped spread, and their
ratio. A ratio near 1 means the shipped states exercise the declared
randomisation. A ratio near 0 means the benchmark ships a constant where its
own task file asks for a distribution.

Usage: python scripts/declared_vs_shipped.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BDDL = REPO / ".libero_src/libero/libero/bddl_files"
SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]


def declared_regions(path: Path) -> dict[str, tuple[float, float]]:
    """region name -> (x side, y side) in cm, from the BDDL :ranges block."""
    txt = path.read_text()
    out = {}
    for m in re.finditer(
            r"\((\w+)\s*\(:target[^)]*\)\s*\(:ranges\s*\(\s*\(([^)]*)\)", txt):
        name, nums = m.group(1), m.group(2).split()
        if len(nums) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in nums)
        out[name] = (abs(x1 - x0) * 100, abs(y1 - y0) * 100)
    return out


def init_placement(path: Path) -> dict[str, str]:
    """object -> the region its own :init line places it in.

    Matching regions by NAME is wrong and we did it first: LIBERO-Object calls
    the target's region `target_object_region`, but Spatial and Goal use
    per-object regions, so a name heuristic silently picked the plate's region
    for the bowl and reported a task using 152% of its declared area. The
    :init block states the assignment; read that instead.
    """
    txt = path.read_text()
    blk = re.search(r"\(:init(.*?)\n\s*\)", txt, re.S)
    if not blk:
        return {}
    return {m.group(1): m.group(2)
            for m in re.finditer(r"\(On\s+(\S+)\s+(\S+)\)", blk.group(1))}


def main() -> None:
    J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
    rows, out = [], {"suites": {}}
    for suite in SUITES:
        S = J["suites"][suite]
        per = []
        for t in S["tasks"]:
            f = BDDL / suite / f"{t['task']}.bddl"
            if not f.exists():
                continue
            regs = declared_regions(f)
            tgt = next((o for o in t["objects"]
                        if o["object"] == t["primary_target"]), None)
            if tgt is None or not tgt.get("xy_per_state_m"):
                continue
            xy = np.asarray(tgt["xy_per_state_m"], dtype=float)
            used = (xy.max(axis=0) - xy.min(axis=0)) * 100      # cm, per axis
            # The region this task's :init block actually places the target in.
            place = init_placement(f)
            rname = place.get(t["primary_target"])
            if rname is None:
                continue
            # :init names may carry a scene prefix ("main_table_plate_region")
            # while the :regions block declares the bare name ("plate_region").
            dec_t = regs.get(rname) or next(
                (v for k, v in regs.items() if rname.endswith(k)), None)
            if dec_t is None:
                continue
            dec = np.asarray(dec_t, dtype=float)
            ratio = float(np.max(used / np.where(dec > 0, dec, np.nan)))
            per.append({"task": t["task"], "declared_cm": [round(v, 3) for v in dec],
                        "used_cm": [round(v, 4) for v in used],
                        "fraction_of_declared": round(ratio, 4)})
            rows.append((suite, t["task"], dec, used, ratio))
        if per:
            fr = np.array([r["fraction_of_declared"] for r in per])
            out["suites"][suite] = {
                "n_tasks": len(per), "tasks": per,
                "mean_fraction_used": round(float(fr.mean()), 4),
                "max_fraction_used": round(float(fr.max()), 4),
                "n_tasks_using_under_5pct": int((fr < 0.05).sum())}

    Path(REPO / "results/declared_vs_shipped.json").write_text(json.dumps(out, indent=2))

    print(f"{'suite':<16}{'declared (cm)':>16}{'shipped spread':>17}"
          f"{'fraction used':>15}{'tasks <5%':>11}")
    for s, S in out["suites"].items():
        d = np.array([t["declared_cm"] for t in S["tasks"]]).mean(axis=0)
        u = np.array([t["used_cm"] for t in S["tasks"]]).mean(axis=0)
        print(f"{s:<16}{d[0]:7.2f} x{d[1]:6.2f}{u[0]:9.3f} x{u[1]:6.3f}"
              f"{S['mean_fraction_used']*100:13.1f}%{S['n_tasks_using_under_5pct']:>8}/{S['n_tasks']}")
    print("\nwrote results/declared_vs_shipped.json")


if __name__ == "__main__":
    main()
