"""Is the appearance-drift gap actually a POSITION effect? (referee M4)

The addendum's surviving mechanism is appearance-side off-manifold drift: the
blocked object's deployment embeddings sit 3.1x further from its own corpus
than the crossing object's. A referee observed that our design confounds this
with table position. LIBERO-Object's ten targets occupy exactly two clusters
22 cm apart (``scripts/measure_placement_pinning.py``), and the three objects we
attempted are soup (cluster A, crosses), butter (cluster A, crosses) and cream
cheese (**cluster B, blocked**). "Blocked" and "22 cm from the training
position" are therefore perfectly confounded, and the gap may be caused by
*where the object sits* rather than by *what it looks like*.

Note first what the confound CANNOT be. Within an object, corpus position ==
deployment position: cream's corpus is cream at cluster B and cream's
deployment is cream at cluster B, so absolute position is matched by
construction and cannot create a gap directly. Two mediated routes remain:

  (a) the head emits cluster-A-ish goals, the arm flies to the wrong place, and
      the viewpoints that follow are unlike the corpus. This is a CONSEQUENCE
      of the error and is what the tick-band split already excludes -- the gap
      is largest in ticks 0-100, before the arm has acted.
  (b) at early ticks a cluster-B object is further from the home pose, so its
      crops are smaller and its embeddings noisier. This would ALSO be largest
      early and shrink on approach, so the tick-band check does not separate
      it. This is the live alternative and this script tests it.

Two measurements, both at the HOME POSE so that arm pose is identical by
construction across every cell (the pre-effect slice -- the same discipline
that rescued the tick-band analysis):

  A  ``--mode native``  -- gap vs own corpus for all ten objects at their
     shipped positions. All five cluster-B objects sit at the SAME xy within
     0.6 mm. If position sets the gap, the five must agree; spread inside a
     cluster that rivals the spread between clusters refutes (b) without
     moving anything.

  B  ``--mode swap``    -- the within-object manipulation the referee asked
     for: teleport each object to the OTHER cluster's centroid and re-measure.
     Identity is held exactly constant, position changes by 22 cm. This is a
     direct causal test rather than an observational contrast.

The readout is deliberately the embedding gap, not task success: a 22 cm
teleport is far outside the shell's +-6 cm probe envelope, so every teleported
cell would collapse for reasons unrelated to the mechanism.

Usage:
    python scripts/position_vs_appearance.py --mode native --states 50 \
        --corpus data/libero_object_wrist --out results/pos_vs_appearance.json
    python scripts/position_vs_appearance.py --mode swap --states 50 ...

Requires the LIBERO sim stack and the real detector. Records the stack versions
it ran under: a rebuild of this stack is known to invert head rankings, so a
number from here is only comparable to another number from here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Cluster centroids, measured from the shipped init files (not assumed).
#: Filled at runtime by :func:`cluster_centroids`.
CLUSTER_A = "A"
CLUSTER_B = "B"


def _stack() -> dict:
    """Versions that produced these numbers, read not transcribed."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "torch", "ultralytics", "mujoco", "robosuite"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "absent"
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, timeout=10).stdout.strip()
    except Exception:
        out["git_commit"] = "unknown"
    out["argv"] = sys.argv
    return out


def task_object(task_name: str) -> str:
    """'pick_up_the_alphabet_soup_and_place_it_in_the_basket' -> 'alphabet_soup'."""
    s = task_name.replace("pick_up_the_", "")
    return s.replace("_and_place_it_in_the_basket", "")


def source_body(inner, obj: str) -> str | None:
    """The sim body for this task's target object."""
    bodies = list(getattr(inner, "obj_body_id", {}) or {})
    for b in bodies:
        if b.lower().startswith(obj.lower()):
            return b
    return None


def body_xy(inner, body: str) -> tuple[float, float, float]:
    sim = inner.sim
    bid = inner.obj_body_id[body]
    p = sim.data.body_xpos[bid]
    return float(p[0]), float(p[1]), float(p[2])


def set_body_xy(inner, body: str, x: float, y: float) -> bool:
    """Absolutely place a free-joint body at (x, y), keeping z and rotation.

    Absolute, unlike ``eval.libero_eval.randomize_source_xy`` which applies a
    random delta -- this test needs a DIRECTED move to a named position.
    """
    sim = inner.sim
    model, data = sim.model, sim.data
    jname = f"{body}_joint0"
    try:
        jid = model.joint_name2id(jname)
    except Exception:
        try:
            jid = list(model.joint_names).index(jname)
        except Exception:
            return False
    if hasattr(model, "jnt_type") and int(model.jnt_type[jid]) != 0:
        return False                      # free joints only
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr] = x
    data.qpos[adr + 1] = y
    try:
        sim.forward()
    except Exception:
        return False
    return True


def corpus_embs(corpus_dir: str, task_name: str) -> np.ndarray:
    """Corpus source-box embeddings for one task, confidence-gated as baked."""
    out = []
    for f in sorted(glob.glob(os.path.join(corpus_dir, f"{task_name}_demo__demo_*.npz"))):
        with np.load(f) as z:
            if "source_box_embs" not in z:
                continue
            e = np.asarray(z["source_box_embs"], dtype=np.float64)
            w = np.asarray(z["box_weights"], dtype=np.float64) if "box_weights" in z else None
            if w is not None and w.ndim > 1:
                w = w[:, 0]
            keep = (w > 0) if w is not None else np.ones(len(e), bool)
            if keep.any():
                out.append(e[keep])
    return np.concatenate(out) if out else np.zeros((0, 512))


def unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-9)


def self_nn(corp: np.ndarray) -> float:
    """Mean cosine of each corpus vector to its nearest OTHER corpus vector."""
    if len(corp) < 2:
        return float("nan")
    u = unit(corp)
    sim = u @ u.T
    np.fill_diagonal(sim, -np.inf)
    return float(sim.max(axis=1).mean())


def nn_to(probe: np.ndarray, corp: np.ndarray) -> float:
    if len(probe) == 0 or len(corp) == 0:
        return float("nan")
    return float((unit(probe) @ unit(corp).T).max(axis=1).mean())


def cluster_centroids(bench, n_tasks: int) -> dict:
    """Measure the two clusters from the shipped init states themselves."""
    pos = {}
    for ti in range(n_tasks):
        init = np.asarray(bench.get_task_init_states(ti), dtype=np.float64)
        # target slot is resolved per-task below via the sim; here we only need
        # a coarse split, which the sim-read positions give us in collect().
        pos[ti] = init
    return pos


def collect(args) -> dict:
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "osmesa")
    from eval._libero_compat import prepare_libero

    prepare_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    from microvla.config import DEFAULT_CONFIG as cfg
    from microvla.perception.command_parser import strip_article
    from microvla.perception.prompts import role_chains
    from microvla.perception.text_encoder import ClipTaskEncoder
    from microvla.perception.yolo_world import YoloWorldPerception
    from microvla.utils.camera import upright

    perception = YoloWorldPerception(
        device=args.device, det_conf=cfg.det_conf,
        role_disjoint_iou=cfg.role_disjoint_iou,
        grid_size=0)
    task_encoder = ClipTaskEncoder(perception)

    bench = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = bench.get_num_tasks() if hasattr(bench, "get_num_tasks") else bench.n_tasks

    # ---- pass 1: read every task's native target xy, so clusters are MEASURED
    native = {}
    for ti in range(n_tasks):
        task = bench.get_task(ti)
        name = Path(bench.get_task_bddl_file_path(ti)).stem
        obj = task_object(name)
        env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(ti),
                                 camera_heights=args.res, camera_widths=args.res)
        try:
            env.reset()
            env.set_init_state(np.asarray(bench.get_task_init_states(ti))[0])
            inner = env.env if hasattr(env, "env") else env
            body = source_body(inner, obj)
            native[ti] = dict(name=name, obj=obj, body=body,
                              xy=body_xy(inner, body)[:2] if body else None,
                              instruction=task.language)
        finally:
            env.close()

    xs = np.array([native[t]["xy"][0] for t in native])
    split = xs.mean()
    for ti in native:
        native[ti]["cluster"] = CLUSTER_A if native[ti]["xy"][0] < split else CLUSTER_B
    cent = {}
    for c in (CLUSTER_A, CLUSTER_B):
        pts = np.array([native[t]["xy"] for t in native if native[t]["cluster"] == c])
        cent[c] = pts.mean(axis=0).tolist()
    sep = float(np.linalg.norm(np.array(cent[CLUSTER_A]) - np.array(cent[CLUSTER_B])))

    # ---- pass 2: probe each task at the HOME POSE
    cells = []
    for ti in range(n_tasks):
        meta = native[ti]
        if args.objects and meta["obj"] not in args.objects:
            continue
        name, obj, body = meta["name"], meta["obj"], meta["body"]
        parsed = task_encoder.encode(meta["instruction"]).parsed
        src, tgt = strip_article(parsed.source), strip_article(parsed.target)
        role_src, role_tgt = role_chains(src, tgt)
        perception.set_role_prompts(role_src, role_tgt)

        other = CLUSTER_B if meta["cluster"] == CLUSTER_A else CLUSTER_A
        dest = cent[other]

        env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(ti),
                                 camera_heights=args.res, camera_widths=args.res)
        init = np.asarray(bench.get_task_init_states(ti))
        embs, confs, placed = [], [], []
        n_frame_unchanged = 0
        try:
            for i in range(min(args.states, len(init))):
                env.reset()
                obs = env.set_init_state(init[i])
                inner = env.env if hasattr(env, "env") else env
                if args.mode == "swap":
                    if body is None or not set_body_xy(inner, body, dest[0], dest[1]):
                        continue
                    before = np.asarray(obs[args.camera]).copy()
                    # force_update=True is LOAD-BEARING: robosuite's
                    # _get_observations() defaults to force_update=False and
                    # returns the CACHED observation, so without it every
                    # teleported cell is rendered from the pre-teleport frame.
                    # The first run of this script did exactly that and
                    # produced swap gaps identical to native to four decimals
                    # -- a broken instrument that reads as a clean null.
                    obs = inner._get_observations(force_update=True)
                    if np.array_equal(before, np.asarray(obs[args.camera])):
                        n_frame_unchanged += 1
                placed.append(body_xy(inner, body)[:2] if body else None)
                frame = upright(obs[args.camera], args.camera)
                # perceive() takes BGR (ultralytics convention); frames are RGB
                p = perception.perceive(np.ascontiguousarray(frame[..., ::-1]))
                embs.append(p.source.emb.numpy().astype(np.float64))
                confs.append(float(p.source.confidence))
        finally:
            env.close()

        # An instrument that silently no-ops is worse than one that fails.
        if args.mode == "swap" and n_frame_unchanged:
            raise RuntimeError(
                f"{obj}: the rendered frame was IDENTICAL after teleporting the "
                f"object 22 cm on {n_frame_unchanged}/{len(placed)} states. The "
                f"render did not follow the sim. Refusing to emit numbers that "
                f"would read as 'position has no effect'.")

        corp = corpus_embs(args.corpus, name)
        probe = np.asarray(embs, dtype=np.float64)
        base = self_nn(corp)
        dep = nn_to(probe, corp)
        cells.append(dict(
            task=name, obj=obj, cluster=meta["cluster"], mode=args.mode,
            native_xy=[round(v, 5) for v in meta["xy"]],
            probe_xy_mean=[round(float(v), 5) for v in np.array(
                [p for p in placed if p is not None]).mean(axis=0)] if placed else None,
            n_probe=len(probe), n_corpus=int(len(corp)),
            corpus_self_nn=round(base, 6),
            probe_nn_to_corpus=round(dep, 6),
            gap=round(base - dep, 6),
            mean_conf=round(float(np.mean(confs)) if confs else float("nan"), 4),
            det_rate=round(float(np.mean([c > 0 for c in confs])) if confs else 0.0, 3),
        ))
        print(f"[{args.mode}] {obj:<18} cluster {meta['cluster']}  "
              f"gap {cells[-1]['gap']:+.4f}  conf {cells[-1]['mean_conf']:.3f}  "
              f"det {cells[-1]['det_rate']:.2f}  n={len(probe)}")

    return dict(suite=args.suite, mode=args.mode, states=args.states,
                camera=args.camera, corpus=args.corpus,
                cluster_centroids=cent, cluster_separation_m=round(sep, 5),
                cells=cells, stack=_stack())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--mode", choices=("native", "swap"), default="native",
                   help="native: shipped positions. swap: teleport each object "
                        "to the OTHER cluster's centroid (identity held fixed)")
    p.add_argument("--states", type=int, default=50)
    p.add_argument("--corpus", default="data/libero_object_wrist")
    p.add_argument("--camera", default="robot0_eye_in_hand_image")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--objects", nargs="*", default=None,
                   help="restrict to these object slugs (default: all ten)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = collect(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
