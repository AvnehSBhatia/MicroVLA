"""Measure what a LIBERO suite actually pins (paper SS3 / App D).

Two modes, and the second is the one to prefer.

``--mode sim`` (original): for every task, load the shipped seeded init states
(``benchmark.get_task_init_states``), apply each with
``OffScreenRenderEnv.set_init_state`` (the exact call the eval harness uses,
``eval/libero_eval.py``), and read every scene object's start pose from the
simulator. Emits a per-task table (target mean/std/quaternion, basket std,
distractor variation) plus a SHA-256 digest of each task's init-state array.

``--mode direct``: read the shipped init files themselves --- a torch-zip
holding one float64 ``(50, 110)`` array, laid out
``[t | 9 robot qpos | 7 objects x 7 | 51 qvel]`` --- and index the target by the
BDDL's ``:obj_of_interest`` against its ``:objects`` declaration order. **No
simulator, no GPU, no detector, no rendering.** A referee reproduced our claim
this way and we should have written it first: the sim path routes a
byte-exact question through ``sim.forward()``, which injects O(1e-17) float
noise and forced the original ``pinned_1e9`` tolerance. Reading the file
answers the question exactly.

Direct mode reports, per task, what the sim path could not: the number of
DISTINCT float64 target positions, their separation in ULPs, and the number of
distinct target quaternions. That resolution matters --- it is what showed our
published word "bit-identical" to be false (each pinned task holds **two**
values 1 ULP apart, ~3e-17 m) while also showing the pinning to be *stronger*
than we claimed (**orientation is bit-identical in all ten tasks**, including
the four that jitter position).

Usage:
    python scripts/measure_placement_pinning.py --mode direct \
        --suite libero_object [--out results/placement_pinning_direct.json]

No policy, no checkpoints. ``--mode sim`` needs the LIBERO sim stack; ``--mode
direct`` needs only torch + numpy and the installed ``libero`` data files.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _target_and_basket(obj_names: list[str]) -> tuple[str | None, str | None]:
    """The libero_object family is 'pick <obj>, place in basket'."""
    basket = next((n for n in obj_names if "basket" in n), None)
    target = next((n for n in obj_names if "basket" not in n), None)
    return target, basket


#: Object qpos block is 3 position + 4 quaternion.
_OBJ_QPOS = 7
#: Prefix before the first object block: 1 time + 9 robot qpos.
_PREFIX = 10


def _ulp_gap(a: float, b: float) -> int:
    """Representable float64 steps between two values (0 == bit-identical)."""
    ia = int(np.float64(a).view(np.int64))
    ib = int(np.float64(b).view(np.int64))
    return abs(ia - ib)


def _bddl_objects(text: str) -> tuple[list[str], str]:
    """Declared object order and the target, from a BDDL file's text."""
    m = re.search(r"\(:objects\s+(.*?)\)", text, re.S)
    names: list[str] = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        names.extend(line.split("-")[0].split())
    target = re.search(r"\(:obj_of_interest\s+(.*?)\)", text, re.S)
    return names, target.group(1).strip().split()[0]


def measure_direct(suite: str) -> dict:
    """Read the shipped init files; no simulator involved.

    Answers the byte-exact question byte-exactly, which the sim path cannot:
    ``sim.forward()`` perturbs positions at 1e-17 even from identical qpos.
    """
    import torch                       # only for torch-zip load of .init files

    from libero.libero import get_libero_path

    root = Path(get_libero_path("bddl_files")).parent
    init_dir = root / "init_files" / suite
    bddl_dir = root / "bddl_files" / suite
    if not init_dir.exists():          # older layouts nest differently
        init_dir = next(root.rglob(f"init_files/{suite}"))
        bddl_dir = next(root.rglob(f"bddl_files/{suite}"))

    out: dict = {"suite": suite, "mode": "direct", "tasks": []}
    for f in sorted(init_dir.glob("*.pruned_init")):
        arr = np.asarray(torch.load(f, weights_only=False), dtype=np.float64)
        names, target = _bddl_objects((bddl_dir / f"{f.stem}.bddl").read_text())
        ti = names.index(target)
        bi = next((i for i, n in enumerate(names) if "basket" in n), None)
        s = _PREFIX + ti * _OBJ_QPOS
        tpos, tquat = arr[:, s:s + 3], arr[:, s + 3:s + 7]

        uniq = np.unique(tpos, axis=0)
        ulp = None
        if len(uniq) == 2:
            per_axis = [(ax, _ulp_gap(uniq[0][ax], uniq[1][ax])) for ax in range(3)]
            ulp = {
                "axes_differing": [{"axis": "xyz"[ax], "ulps": g}
                                   for ax, g in per_axis if g],
                "split": [int((tpos == u).all(axis=1).sum()) for u in uniq],
                "values": [uniq[0].tolist(), uniq[1].tolist()],
            }
        row = {
            "task": f.stem,
            "target": target,
            "n_states": int(len(arr)),
            "init_states_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "mean_xy_m": tpos[:, :2].mean(axis=0).round(6).tolist(),
            "std_xyz_cm": (tpos.std(axis=0) * 100).round(4).tolist(),
            "n_distinct_positions": int(len(uniq)),
            "n_distinct_quaternions": int(len(np.unique(tquat, axis=0))),
            "position_bit_identical": bool(len(uniq) == 1),
            "orientation_bit_identical": bool(len(np.unique(tquat, axis=0)) == 1),
            "ulp_structure": ulp,
        }
        if bi is not None:
            b = _PREFIX + bi * _OBJ_QPOS
            row["basket_mean_xy_m"] = arr[:, b:b + 2].mean(axis=0).round(6).tolist()
            row["basket_std_xy_cm"] = (arr[:, b:b + 2].std(axis=0) * 100).round(4).tolist()
        out["tasks"].append(row)
        print(f"{row['task'].replace('pick_up_the_', '')[:34]:<36} "
              f"xy=({row['mean_xy_m'][0]:+.4f},{row['mean_xy_m'][1]:+.4f}) "
              f"std_cm=({row['std_xyz_cm'][0]:.3f},{row['std_xyz_cm'][1]:.3f}) "
              f"#pos={row['n_distinct_positions']:<3} #quat={row['n_distinct_quaternions']}",
              flush=True)

    # Cluster structure: measured, not assumed.
    xy = np.array([t["mean_xy_m"] for t in out["tasks"]])
    split = xy[:, 0].mean()
    for t, p in zip(out["tasks"], xy):
        t["cluster"] = "A" if p[0] < split else "B"
    clusters = {}
    for c in ("A", "B"):
        pts = np.array([t["mean_xy_m"] for t in out["tasks"] if t["cluster"] == c])
        mx_e = max((float(np.linalg.norm(pts[i] - pts[j]))
                    for i, j in itertools.combinations(range(len(pts)), 2)), default=0.0)
        mx_ax = max((float(np.abs(pts[i] - pts[j]).max())
                     for i, j in itertools.combinations(range(len(pts)), 2)), default=0.0)
        clusters[c] = {
            "members": [t["task"] for t in out["tasks"] if t["cluster"] == c],
            "centroid_xy_m": pts.mean(axis=0).round(6).tolist(),
            "max_pairwise_euclidean_mm": round(mx_e * 1000, 3),
            "max_pairwise_per_axis_mm": round(mx_ax * 1000, 3),
        }
    sep = float(np.linalg.norm(np.array(clusters["A"]["centroid_xy_m"])
                               - np.array(clusters["B"]["centroid_xy_m"])))
    bxy = np.array([t["basket_mean_xy_m"] for t in out["tasks"] if "basket_mean_xy_m" in t])
    out["summary"] = {
        "n_tasks": len(out["tasks"]),
        "n_position_pinned": sum(1 for t in out["tasks"] if t["n_distinct_positions"] <= 2),
        "n_orientation_bit_identical": sum(1 for t in out["tasks"]
                                           if t["orientation_bit_identical"]),
        "clusters": clusters,
        "cluster_separation_cm": round(sep * 100, 3),
        "basket_max_pairwise_mm": round(max(
            float(np.linalg.norm(bxy[i] - bxy[j]))
            for i, j in itertools.combinations(range(len(bxy)), 2)) * 1000, 3),
    }
    s = out["summary"]
    print(f"\n{s['n_position_pinned']}/{s['n_tasks']} tasks position-pinned "
          f"(<=2 distinct float64 values); "
          f"{s['n_orientation_bit_identical']}/{s['n_tasks']} orientation "
          f"BIT-IDENTICAL across all states")
    print(f"clusters {sep * 100:.2f} cm apart; within-cluster max pairwise "
          f"A {clusters['A']['max_pairwise_euclidean_mm']:.2f} mm / "
          f"B {clusters['B']['max_pairwise_euclidean_mm']:.2f} mm")
    return out


def measure(suite: str) -> dict:
    from eval._libero_compat import prepare_libero

    prepare_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    n_tasks = bench.get_num_tasks() if hasattr(bench, "get_num_tasks") else bench.n_tasks

    out: dict = {"suite": suite, "tasks": []}
    for ti in range(n_tasks):
        task = bench.get_task(ti)
        init_states = np.asarray(bench.get_task_init_states(ti))
        digest = hashlib.sha256(init_states.tobytes()).hexdigest()
        env = OffScreenRenderEnv(
            bddl_file_name=bench.get_task_bddl_file_path(ti),
            camera_heights=64, camera_widths=64,
        )
        try:
            env.reset()
            inner = env.env if hasattr(env, "env") else env
            interest = list(getattr(inner, "obj_of_interest", []))
            # Pose of every object body across all shipped init states.
            body_names = [
                n for n in inner.sim.model.body_names
                if n.endswith("_main") and "robot" not in n and "gripper" not in n
            ]
            poses: dict[str, list] = {n: [] for n in body_names}
            for st in init_states:
                env.set_init_state(st)
                for n in body_names:
                    bid = inner.sim.model.body_name2id(n)
                    pos = np.array(inner.sim.data.body_xpos[bid])
                    quat = np.array(inner.sim.data.body_xquat[bid])
                    poses[n].append(np.concatenate([pos, quat]))
            rows = {}
            for n, arr in poses.items():
                a = np.asarray(arr)
                pos_dev = float(np.abs(a[:, :3] - a[0, :3]).max())
                quat_dev = float(np.abs(a[:, 3:] - a[0, 3:]).max())
                rows[n] = {
                    "mean_xyz": a[:, :3].mean(axis=0).round(6).tolist(),
                    "std_xyz": a[:, :3].std(axis=0).round(6).tolist(),
                    "quat_first": a[0, 3:].round(6).tolist(),
                    "quat_max_abs_dev": quat_dev,
                    "pos_max_abs_dev": pos_dev,
                    # "pinned": identical to machine precision. sim.forward()
                    # introduces O(1e-17) float noise even on byte-identical
                    # qpos, so exact `==` over xpos is the wrong test.
                    "pinned_1e9": bool(pos_dev < 1e-9 and quat_dev < 1e-9),
                    "bit_identical": bool(np.all(a == a[0])),
                }
            target, basket = _target_and_basket(
                [n for n in body_names if any(o in n for o in interest)] or body_names
            )
            out["tasks"].append({
                "task_id": ti,
                "name": getattr(task, "name", f"task_{ti}"),
                "language": task.language,
                "n_init_states": int(len(init_states)),
                "init_states_sha256": digest,
                "obj_of_interest": interest,
                "target_body": target,
                "basket_body": basket,
                "objects": rows,
            })
            print(f"[{ti}] {task.language}: target={target} "
                  f"std_xy_cm={[round(v * 100, 2) for v in rows[target]['std_xyz'][:2]] if target in rows else '?'} "
                  f"pinned={rows[target]['pinned_1e9'] if target in rows else '?'}",
                  flush=True)
        finally:
            env.close()
    return out


def as_markdown(res: dict) -> str:
    lines = [
        f"# Placement pinning measurement — {res['suite']}",
        "",
        "Generated by `scripts/measure_placement_pinning.py`. Start pose of each",
        "task's target object across all shipped init states (applied via",
        "`OffScreenRenderEnv.set_init_state`, the eval harness's own call).",
        "",
        "| task | target | n | mean x,y (m) | std x,y (cm) | max dev (m) | pinned (<1e-9 m) | quat (w,x,y,z) | basket std x,y (cm) | init-array sha256 (first 12) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in res["tasks"]:
        r = t["objects"].get(t["target_body"] or "", None)
        if r is None:
            continue
        mx, my = r["mean_xyz"][0], r["mean_xyz"][1]
        sx, sy = r["std_xyz"][0] * 100, r["std_xyz"][1] * 100
        q = ", ".join(f"{v:+.3f}" for v in r["quat_first"])
        b = t["objects"].get(t["basket_body"] or "", None)
        bstd = (f"({b['std_xyz'][0] * 100:.2f}, {b['std_xyz'][1] * 100:.2f})"
                if b else "-")
        lines.append(
            f"| {t['task_id']} | {t['target_body']} | {t['n_init_states']} "
            f"| ({mx:+.3f}, {my:+.3f}) | ({sx:.2f}, {sy:.2f}) | {r['pos_max_abs_dev']:.1e} "
            f"| {'yes' if r['pinned_1e9'] else 'no'} | ({q}) | {bstd} | {t['init_states_sha256'][:12]} |")
    return "\n".join(lines) + "\n"


def as_markdown_direct(res: dict) -> str:
    """The per-task table the paper cites for its opening claim."""
    lines = [
        f"# Placement pinning, read directly from the shipped init files — {res['suite']}",
        "",
        "Generated by `scripts/measure_placement_pinning.py --mode direct`.",
        "No simulator: each `.pruned_init` is a torch-zip holding one float64",
        "`(50, 110)` array; the target column is resolved from the BDDL's",
        "`:obj_of_interest` against its `:objects` order.",
        "",
        "`#pos` counts DISTINCT float64 target positions across the 50 states.",
        "Where that is 2, the two values differ by a single ULP (~3e-17 m) on one",
        "axis — constant to float64 rounding, but *not* bit-identical.",
        "",
        "| task | cluster | mean x,y (m) | std x,y (cm) | #pos | ULP structure | #quat | init sha256 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in res["tasks"]:
        u = t.get("ulp_structure")
        us = "—"
        if u:
            ax = ", ".join(f"{d['ulps']} ULP on {d['axis']}" for d in u["axes_differing"])
            us = f"{ax}; split {'/'.join(str(v) for v in u['split'])}"
        lines.append(
            f"| {t['task'].replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')} "
            f"| {t['cluster']} | ({t['mean_xy_m'][0]:+.4f}, {t['mean_xy_m'][1]:+.4f}) "
            f"| ({t['std_xyz_cm'][0]:.3f}, {t['std_xyz_cm'][1]:.3f}) "
            f"| {t['n_distinct_positions']} | {us} | {t['n_distinct_quaternions']} "
            f"| {t['init_states_sha256'][:12]} |")
    s = res["summary"]
    lines += [
        "",
        f"**{s['n_position_pinned']}/{s['n_tasks']}** tasks hold the target position "
        f"constant to float64 rounding. "
        f"**{s['n_orientation_bit_identical']}/{s['n_tasks']}** hold the target "
        f"orientation *bit-identical* across all 50 states — including every task "
        f"that jitters position.",
        "",
        f"Two clusters **{s['cluster_separation_cm']:.2f} cm** apart, five tasks each:",
        "",
    ]
    for c, d in s["clusters"].items():
        members = ", ".join(m.replace("pick_up_the_", "").replace(
            "_and_place_it_in_the_basket", "") for m in d["members"])
        lines.append(f"- **{c}** at ({d['centroid_xy_m'][0]:+.4f}, "
                     f"{d['centroid_xy_m'][1]:+.4f}) m — {members}. Max pairwise "
                     f"{d['max_pairwise_euclidean_mm']:.2f} mm euclidean / "
                     f"{d['max_pairwise_per_axis_mm']:.2f} mm per-axis.")
    lines += ["", f"Basket max pairwise across tasks: "
                  f"**{s['basket_max_pairwise_mm']:.2f} mm**.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--mode", choices=("sim", "direct"), default="sim",
                    help="direct reads the init files and needs no simulator; "
                         "it is the mode the paper's opening claim now cites")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "direct":
        res = measure_direct(args.suite)
        out = Path(args.out or "results/placement_pinning_direct.json")
        md_text = as_markdown_direct(res)
    else:
        res = measure(args.suite)
        out = Path(args.out or "results/placement_pinning.json")
        md_text = as_markdown(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    md = out.with_suffix(".md")
    md.write_text(md_text)
    print(f"wrote {out} and {md}")


if __name__ == "__main__":
    main()
