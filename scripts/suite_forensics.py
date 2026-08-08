"""Placement forensics across every LIBERO suite (referee E3).

The referee's blocking request: measure per-suite placement variance, bit-identity
counts and cross-task clustering, define an admissibility metric, and publish the
script plus the file hashes. ``scripts/measure_placement_pinning.py --mode direct``
already answers this exactly for ``libero_object``, but its indexing assumes the
flattened state is ``[t | 9 robot qpos | N objects x 7 | qvel]``. That holds for
libero_object (7 free-body objects, width 110 = 19 + 13*7) and for **no** suite
containing an articulated fixture: libero_spatial ships width 92 against 5
declared objects, libero_goal 79 against 4. Guessing the layout there would be
the same class of error as the cached-render null --- a number that looks like a
measurement and is not one.

So this script resolves the layout instead of assuming it. Two passes:

``columns`` (no simulator, no GPU, instant)
    A LIBERO init state is MuJoCo's flattened state, ``[time, qpos, qvel]``. In a
    shipped init file every qvel entry is zero, so the columns that vary across
    the 50 states are exactly the position degrees of freedom the suite
    randomizes. Counting distinct values per column therefore bounds what the
    suite can possibly vary, with no knowledge of which object owns which column.
    This pass runs on every suite including libero_90.

``joints`` (needs mujoco; ~2-4 s per task)
    Build each task's environment once and read ``sim.model.jnt_qposadr`` and the
    joint names, which map every column onto a named body. That gives per-OBJECT
    distinct-position counts and ULP structure for suites the fixed layout cannot
    address, and it verifies the column pass rather than replacing it.

Admissibility metric (defined here, reported per suite). For one task, let the
target's distinct start positions be quantised to 1 mm --- the scale at which a
policy could plausibly tell two placements apart --- with empirical frequencies
p_i over the 50 shipped states. Then

    H_task = -sum_i p_i log2 p_i        (bits)
    H_suite = mean over tasks           (bits per task)

H = 0 means a lookup table keyed on the task index reproduces every shipped start
pose exactly; log2(50) = 5.64 bits is the ceiling, one distinct placement per
state. The suite-level admissibility number we report is the pair
(H_suite, number of distinct 1 mm placements per task), because entropy alone
cannot distinguish "two placements, 50/50" from "two placements 22 cm apart".

Usage:
    python scripts/suite_forensics.py --pass columns --out results/suite_forensics_columns.json
    python scripts/suite_forensics.py --pass joints  --out results/suite_forensics_joints.json
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

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10", "libero_90"]
#: Quantisation for the admissibility metric. 1 mm: below this two placements are
#: not distinguishable by any policy acting through a 7-DoF arm.
QUANT_M = 1e-3


def _libero_root() -> Path:
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files")).parent


def _bddl(text: str) -> tuple[list[str], list[str]]:
    """Declared object order and the declared objects of interest."""
    m = re.search(r"\(:objects\s+(.*?)\n\s*\)", text, re.S)
    names: list[str] = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        names.extend(line.split("-")[0].split())
    goi = re.search(r"\(:obj_of_interest\s+(.*?)\)", text, re.S)
    return names, goi.group(1).strip().split() if goi else []


def _ulp_gap(a: float, b: float) -> int:
    """Representable float64 steps between two values (0 == bit-identical)."""
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def _entropy_bits(vals: np.ndarray) -> tuple[float, int]:
    """(Shannon bits, #distinct) over 1 mm-quantised rows of ``vals``."""
    q = np.round(np.asarray(vals, dtype=np.float64) / QUANT_M).astype(np.int64)
    _, counts = np.unique(q, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()), int(len(counts))


def _load(f: Path) -> np.ndarray:
    import torch

    return np.asarray(torch.load(f, weights_only=False), dtype=np.float64)


# --------------------------------------------------------------------------- #
# pass 1: columns. No simulator. Bounds what the suite can vary.
# --------------------------------------------------------------------------- #
def run_columns(suites: list[str]) -> dict:
    root = _libero_root()
    out: dict = {"pass": "columns", "quant_m": QUANT_M, "suites": {}}
    for suite in suites:
        init_dir, bddl_dir = root / "init_files" / suite, root / "bddl_files" / suite
        tasks = []
        for f in sorted(init_dir.glob("*.pruned_init")):
            arr = _load(f)
            names, goi = _bddl((bddl_dir / f"{f.stem}.bddl").read_text())
            # A column "varies" iff it takes >1 distinct float64 value.
            distinct = np.array([len(np.unique(arr[:, c])) for c in range(arr.shape[1])])
            varying = np.flatnonzero(distinct > 1)
            # ...and "moves" iff that variation exceeds the 1 mm scale.
            spans = arr.max(axis=0) - arr.min(axis=0)
            moving = np.flatnonzero(spans > QUANT_M)
            h_full, n_full = _entropy_bits(arr[:, varying]) if len(varying) else (0.0, 1)
            tasks.append({
                "task": f.stem,
                "n_states": int(arr.shape[0]),
                "width": int(arr.shape[1]),
                "n_declared_objects": len(names),
                "obj_of_interest": goi,
                "free_body_layout_consistent": bool(arr.shape[1] == 19 + 13 * len(names)),
                "n_varying_columns": int(len(varying)),
                "n_columns_moving_gt_1mm": int(len(moving)),
                "max_span_m": float(spans.max()),
                "distinct_full_states": int(len(np.unique(arr, axis=0))),
                "distinct_states_at_1mm": n_full,
                "entropy_bits_at_1mm": round(h_full, 4),
                "init_states_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            })
        out["suites"][suite] = {
            "n_tasks": len(tasks),
            "tasks": tasks,
            "mean_varying_columns": round(float(np.mean([t["n_varying_columns"] for t in tasks])), 3),
            "mean_columns_moving_gt_1mm": round(
                float(np.mean([t["n_columns_moving_gt_1mm"] for t in tasks])), 3),
            "mean_entropy_bits_at_1mm": round(
                float(np.mean([t["entropy_bits_at_1mm"] for t in tasks])), 4),
            "mean_distinct_states_at_1mm": round(
                float(np.mean([t["distinct_states_at_1mm"] for t in tasks])), 3),
            "n_tasks_fully_frozen_at_1mm": sum(
                1 for t in tasks if t["n_columns_moving_gt_1mm"] == 0),
            "entropy_ceiling_bits": round(float(np.log2(tasks[0]["n_states"])), 4),
        }
        s = out["suites"][suite]
        print(f"{suite:<16} tasks={s['n_tasks']:<3} "
              f"varying_cols={s['mean_varying_columns']:<7} "
              f">1mm_cols={s['mean_columns_moving_gt_1mm']:<6} "
              f"H@1mm={s['mean_entropy_bits_at_1mm']:.3f}/{s['entropy_ceiling_bits']:.2f} bits  "
              f"distinct@1mm={s['mean_distinct_states_at_1mm']:<6} "
              f"frozen_tasks={s['n_tasks_fully_frozen_at_1mm']}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# pass 2: joints. Resolves columns -> named bodies via the model itself.
# --------------------------------------------------------------------------- #
def _joint_map(env) -> dict[str, tuple[int, int]]:
    """{joint_name: (qpos_start, qpos_len)} from the compiled model."""
    sim = env.sim
    m = sim.model
    names = [m.joint_id2name(i) for i in range(m.njnt)]
    adr = [int(m.jnt_qposadr[i]) for i in range(m.njnt)]
    nq = int(m.nq)
    out = {}
    for i, n in enumerate(names):
        end = adr[i + 1] if i + 1 < len(adr) else nq
        out[n] = (adr[i], end - adr[i])
    return out


def run_joints(suites: list[str]) -> dict:
    from eval._libero_compat import prepare_libero

    prepare_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    root = _libero_root()
    bench_dict = benchmark.get_benchmark_dict()
    out: dict = {"pass": "joints", "quant_m": QUANT_M, "suites": {}}
    for suite in suites:
        bench = bench_dict[suite]()
        n_tasks = bench.get_num_tasks() if hasattr(bench, "get_num_tasks") else bench.n_tasks
        bddl_dir = root / "bddl_files" / suite
        tasks = []
        for ti in range(n_tasks):
            task = bench.get_task(ti)
            bddl = bddl_dir / task.problem_folder / task.bddl_file \
                if (bddl_dir / task.problem_folder).exists() else bddl_dir / task.bddl_file
            names, goi = _bddl(Path(bddl).read_text())
            env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                     camera_heights=64, camera_widths=64)
            try:
                jm = _joint_map(env.env if hasattr(env, "env") else env)
            finally:
                env.close()
            arr = np.asarray(bench.get_task_init_states(ti), dtype=np.float64)
            # flattened state is [time, qpos, qvel]; qpos starts at column 1.
            per_obj = []
            for obj in names:
                cols = [(v[0] + 1, v[1]) for k, v in jm.items() if k.startswith(obj + "_")]
                if not cols:
                    continue
                s, ln = min(cols)
                block = arr[:, s:s + ln]
                pos = block[:, :3] if ln >= 3 else block
                quat = block[:, 3:7] if ln >= 7 else None
                uq = np.unique(pos, axis=0)
                h, nd = _entropy_bits(pos)
                rec = {
                    "object": obj,
                    "qpos_len": int(ln),
                    "is_target": obj in goi,
                    "mean_xyz_m": pos.mean(axis=0).round(6).tolist(),
                    "std_xyz_cm": (pos.std(axis=0) * 100).round(4).tolist(),
                    "n_distinct_positions": int(len(uq)),
                    "n_distinct_quaternions": int(len(np.unique(quat, axis=0))) if quat is not None else None,
                    "entropy_bits_at_1mm": round(h, 4),
                    "n_distinct_at_1mm": nd,
                }
                if len(uq) == 2 and pos.shape[1] == 3:
                    rec["ulps_per_axis"] = [_ulp_gap(uq[0][a], uq[1][a]) for a in range(3)]
                if obj in goi:
                    # Every shipped xy for the objects of interest, so figures
                    # can be drawn from the artifact rather than from a re-run.
                    rec["xy_per_state_m"] = pos[:, :2].round(8).tolist()
                per_obj.append(rec)
            # ``:obj_of_interest`` lists the MANIPULATED object first and the
            # destination second ("pick up the alphabet soup and place it in the
            # basket" -> [alphabet_soup_1, basket_1]). Summarising over both at
            # once averages a frozen target with a jittering receptacle and
            # reports every libero_object task as unpinned, which is false. The
            # primary target is goi[0]; the receptacle is reported beside it,
            # never pooled into it.
            prim = next((o for o in per_obj if o["object"] == goi[0]), None) if goi else None
            dest = [o for o in per_obj if o["is_target"] and o is not prim]
            tasks.append({
                "task": Path(task.bddl_file).stem,
                "obj_of_interest": goi,
                "primary_target": goi[0] if goi else None,
                "objects": per_obj,
                "target_entropy_bits_at_1mm": prim["entropy_bits_at_1mm"] if prim else None,
                "target_distinct_at_1mm": prim["n_distinct_at_1mm"] if prim else None,
                "target_position_pinned": bool(prim and prim["n_distinct_positions"] <= 2),
                "target_orientation_bit_identical": bool(
                    prim and prim["n_distinct_quaternions"] == 1),
                "destination_entropy_bits_at_1mm": [o["entropy_bits_at_1mm"] for o in dest],
            })
            t = tasks[-1]
            print(f"  {suite:<15}{t['task'][:44]:<46} H={t['target_entropy_bits_at_1mm']} "
                  f"distinct@1mm={t['target_distinct_at_1mm']} "
                  f"pinned={t['target_position_pinned']} "
                  f"quat_identical={t['target_orientation_bit_identical']}", flush=True)
        # A task is "resolvable" iff its primary target owns a free joint. Tasks
        # whose goal is a fixture ("turn on the stove") have no placement to
        # measure; they are excluded from the denominator and counted, never
        # silently scored as unpinned.
        resolved = [t for t in tasks if t["target_entropy_bits_at_1mm"] is not None]
        # Cross-task clustering of primary-target placements, measured not assumed.
        pts = np.array([[o["mean_xyz_m"][0], o["mean_xyz_m"][1]]
                        for t in tasks for o in t["objects"]
                        if o["object"] == t["primary_target"]])
        pair_max = max((float(np.linalg.norm(pts[i] - pts[j]))
                        for i, j in itertools.combinations(range(len(pts)), 2)), default=0.0)
        out["suites"][suite] = {
            "n_tasks": len(tasks),
            "n_resolvable_tasks": len(resolved),
            "tasks": tasks,
            "n_target_position_pinned": sum(1 for t in resolved if t["target_position_pinned"]),
            "n_target_orientation_bit_identical": sum(
                1 for t in resolved if t["target_orientation_bit_identical"]),
            "mean_target_entropy_bits_at_1mm": round(
                float(np.mean([t["target_entropy_bits_at_1mm"] for t in resolved])), 4)
            if resolved else None,
            "mean_target_distinct_at_1mm": round(
                float(np.mean([t["target_distinct_at_1mm"] for t in resolved])), 3)
            if resolved else None,
            "entropy_ceiling_bits": round(float(np.log2(50)), 4),
            "max_pairwise_target_separation_cm": round(pair_max * 100, 3),
        }
        s = out["suites"][suite]
        print(f"{suite:<16} pinned={s['n_target_position_pinned']}/{s['n_resolvable_tasks']} "
              f"quat_identical={s['n_target_orientation_bit_identical']}/{s['n_resolvable_tasks']} "
              f"H={s['mean_target_entropy_bits_at_1mm']}/{s['entropy_ceiling_bits']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=["columns", "joints"], default="columns")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suites = a.suites or (SUITES if a.which == "columns" else SUITES[:4])
    res = run_columns(suites) if a.which == "columns" else run_joints(suites)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
