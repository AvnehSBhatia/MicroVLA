"""Placement diameter, computed per required object rather than per task.

The first version of this measurement computed one diameter per task, over the
task's ``primary_target``. An adversarial review caught what that hides: every
LIBERO-Object task is a pick-AND-PLACE, the shipped metadata marks BOTH the
grocery and the basket ``is_target: true`` in ``obj_of_interest``, and the
basket is not pinned at all (3.30-3.96 cm). Reported as a single per-task
number, "LIBERO-Object is lookup-admissible on 10/10 tasks" silently meant "on
the half of each task we chose to measure".

So this script emits two families, and the paper must quote both:

  identity  the object that INDIVIDUATES the task -- the grocery, the thing
            the ten instructions differ by. A task-indexed table has to know
            this to tell the tasks apart at all.
  shared    every other object the task requires, here the container, which is
            the SAME basket in all ten tasks and therefore carries no task
            identity whatsoever.

The distinction is not a rescue, it is the finding: LIBERO-Object pins exactly
the sub-goal that carries identity and randomises the one that does not, which
is why a constant reaches the object in 10/10 trials and misses the basket.

Usage: python scripts/placement_diameter.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DELTA_CM = 2.5


def diam_cm(xy: list) -> float:
    a = np.asarray(xy, dtype=float)
    return float(np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1).max() * 100)


def main() -> None:
    J = json.loads((REPO / "results/suite_forensics_joints.json").read_text())
    out: dict = {"delta_cm": DELTA_CM, "suites": {}}
    for suite, S in J["suites"].items():
        rows = []
        for t in S["tasks"]:
            req = set(t.get("obj_of_interest") or [])
            d = {}
            for o in t["objects"]:
                if not o.get("xy_per_state_m"):
                    continue
                d[o["object"]] = diam_cm(o["xy_per_state_m"])
            ident = t["primary_target"]
            shared = [v for k, v in d.items() if k in req and k != ident]
            if ident not in d:
                continue
            rows.append({"task": t["task"], "identity_object": ident,
                         "identity_D_cm": round(d[ident], 4),
                         "shared_objects": {k: round(v, 4) for k, v in d.items()
                                            if k in req and k != ident},
                         "shared_D_cm_max": round(max(shared), 4) if shared else None,
                         "all_required_D_cm_max": round(max([d[ident]] + shared), 4)})
        I = np.array([r["identity_D_cm"] for r in rows])
        A = np.array([r["all_required_D_cm_max"] for r in rows])
        Sh = np.array([r["shared_D_cm_max"] for r in rows
                       if r["shared_D_cm_max"] is not None])
        out["suites"][suite] = {
            "n_tasks": len(rows), "tasks": rows,
            "identity": {"mean": round(float(I.mean()), 4),
                         "max": round(float(I.max()), 4),
                         "min": round(float(I.min()), 4),
                         "n_admissible": int((I <= DELTA_CM).sum())},
            "shared": ({"mean": round(float(Sh.mean()), 4),
                        "max": round(float(Sh.max()), 4),
                        "min": round(float(Sh.min()), 4),
                        "n_admissible": int((Sh <= DELTA_CM).sum())}
                       if len(Sh) else None),
            "all_required": {"mean": round(float(A.mean()), 4),
                             "max": round(float(A.max()), 4),
                             "min": round(float(A.min()), 4),
                             "n_admissible": int((A <= DELTA_CM).sum())},
        }
    Path(REPO / "results/placement_diameter_v2.json").write_text(json.dumps(out, indent=2))

    hdr = f"{'suite':<16}{'identity D (cm)':>26}{'shared D (cm)':>24}{'all-required':>22}"
    print(hdr); print("-" * len(hdr))
    for s, S in out["suites"].items():
        i, sh, a = S["identity"], S["shared"], S["all_required"]
        shtxt = (f"{sh['mean']:.2f} [{sh['min']:.2f},{sh['max']:.2f}] {sh['n_admissible']}/{S['n_tasks']}"
                 if sh else "  --")
        print(f"{s:<16}{i['mean']:>7.2f} [{i['min']:.2f},{i['max']:.2f}] {i['n_admissible']:>2}/{S['n_tasks']:<2}"
              f"{shtxt:>24}{a['mean']:>10.2f} {a['n_admissible']:>3}/{S['n_tasks']}")
    print("\nwrote results/placement_diameter_v2.json")


if __name__ == "__main__":
    main()
