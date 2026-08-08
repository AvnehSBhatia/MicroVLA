"""Positive control for the identity-blindness probe (referee E9 / M7).

The probe has only ever been shown FIRING: on this stack two objects' prompt
chains select the same box at rates 0.87--1.00, which we read as "this stage
carries no object identity". A referee's objection is exact and correct: an
instrument that has never been shown *not* firing is not yet an instrument. The
same-box verdict could be a property of ``prompt_agreement`` rather than of the
stack, and nothing published so far distinguishes those.

So this runs the identical function, through the identical deployed perception
object, on pairs chosen so a working grounder MUST separate them, and on the
pair the paper reports, on the same frames in one process:

It runs the commanded object's prompt chain against every other object's chain
on the same frames in one process, and reports each contrast separately.

If SOME contrast returns "different boxes", the probe registers difference when
difference exists, and a same-box result elsewhere is evidence about the stack.
If NO contrast separates, the probe cannot distinguish and every same-box number
in the paper must be withdrawn as uninformative. Both outcomes are reportable;
only one is comfortable, which is the point of running it.

One correction worth recording, because the first version of this script made
exactly the error it exists to catch. The basket was designated the positive
control -- and it can never be one: the deployed ``source_max_area`` filter
rejects basket-sized boxes by construction, so that pair compares zero frames.
The script nonetheless read ``same_rate=0.0`` off an EMPTY comparison and
printed "INSTRUMENT DISCRIMINATES". A verdict now requires ``n_compared >=
MIN_N``, and unevaluable contrasts are labelled, never counted.

Also emits the threshold sweep the referee asked for: same-rate as a function of
``tol``, so readers can see whether any verdict rides on the 0.02 default.

Usage:
    python scripts/probe_positive_control.py --task-id 0 --frames 24 \
        --out results/probe_positive_control.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Deployed source chains, in the same shape the policy uses (first prompt that
#: detects anything wins), keyed by the libero_object noun.
CHAINS = {
    "alphabet_soup": ["alphabet soup", "soup can", "can"],
    "cream_cheese": ["cream cheese", "cheese box", "box"],
    "butter": ["butter", "butter box", "box"],
    "milk": ["milk", "milk carton", "carton"],
    "tomato_sauce": ["tomato sauce", "sauce can", "can"],
    "ketchup": ["ketchup", "ketchup bottle", "bottle"],
    "orange_juice": ["orange juice", "juice carton", "carton"],
    "bbq_sauce": ["bbq sauce", "sauce bottle", "bottle"],
    "salad_dressing": ["salad dressing", "dressing bottle", "bottle"],
    "chocolate_pudding": ["chocolate pudding", "pudding box", "box"],
    "basket": ["basket", "bin", "container"],
}


def collect_frames(task_id: int, n: int, camera: str, size: int) -> list[np.ndarray]:
    """Wrist frames from a real reset, stepping a zero action between grabs.

    Real deployed-viewpoint frames matter here: a probe run on corpus frames
    tells you about the corpus. force_update=True is load-bearing --- robosuite
    caches observations and would hand back the same frame every time, which
    would make every contrast look identical for a reason that has nothing to
    do with grounding.
    """
    from eval._libero_compat import prepare_libero

    prepare_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()["libero_object"]()
    task = bench.get_task(task_id)
    bddl = Path(bench.get_task_bddl_file_path(task_id))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=size, camera_widths=size)
    frames: list[np.ndarray] = []
    try:
        env.seed(20)
        obs = env.reset()
        obs = env.set_init_state(np.asarray(bench.get_task_init_states(task_id))[0])
        zero = np.zeros(7, dtype=np.float64)
        zero[-1] = -1.0
        seen = set()
        for i in range(n):
            from microvla.utils.camera import upright
            f = np.ascontiguousarray(upright(obs[camera], camera))
            h = hash(f.tobytes())
            if h in seen:
                raise RuntimeError(
                    "identical frame returned twice: the renderer is handing "
                    "back a cached observation, so every contrast below would "
                    "be measured on one frame. Refusing to emit.")
            seen.add(h)
            frames.append(f)
            obs, _, _, _ = env.step(zero)
    finally:
        env.close()
    return frames, task.language


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--camera", default="robot0_eye_in_hand_image")
    ap.add_argument("--render-size", type=int, default=256)
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--det-conf", type=float, default=0.02)
    ap.add_argument("--out", default="results/probe_positive_control.json")
    a = ap.parse_args()

    from eval.probes import prompt_agreement
    from microvla.perception.yolo_world import YoloWorldPerception

    frames, lang = collect_frames(a.task_id, a.frames, a.camera, a.render_size)
    # Same detector construction the eval harness uses, including the two role
    # filters -- a probe run on a differently-configured detector would measure
    # a stack the policy never deploys.
    per = YoloWorldPerception(weights=a.weights, device=a.device,
                              det_conf=a.det_conf, role_disjoint_iou=0.1,
                              source_max_area=0.12)

    # The commanded object for this task, and everything else in the scene.
    tgt = {0: "alphabet_soup", 1: "cream_cheese", 6: "butter"}.get(a.task_id, "alphabet_soup")
    others = [k for k in CHAINS if k not in (tgt,)]

    TOLS = [0.005, 0.01, 0.02, 0.05, 0.10]
    rows = []
    for other in others:
        r = prompt_agreement(per, CHAINS[tgt], CHAINS[other], frames)
        sweep = {}
        for t in TOLS:
            rt = prompt_agreement(per, CHAINS[tgt], CHAINS[other], frames, tol=t)
            sweep[str(t)] = round(rt.same_rate, 4)
        rows.append({
            "pair": f"{tgt} vs {other}",
            "kind": ("container (filtered out by source_max_area; "
                     "expected not evaluable)" if other == "basket"
                     else "distractor"),
            "n_compared": r.n_compared,
            "same_rate_at_0.02": round(r.same_rate, 4),
            "median_distance": round(r.median_distance, 6),
            "detect_rate_a": round(r.detect_rate_a, 4),
            "detect_rate_b": round(r.detect_rate_b, 4),
            "same_rate_by_tol": sweep,
        })
        print(f"{rows[-1]['pair']:<34} same@0.02={r.same_rate:.3f} "
              f"median_d={r.median_distance:.5f} "
              f"det={r.detect_rate_a:.2f}/{r.detect_rate_b:.2f}", flush=True)

    out = {
        "task_id": a.task_id, "instruction": lang, "n_frames": len(frames),
        "camera": a.camera, "det_conf": a.det_conf, "target": tgt,
        "tolerances": TOLS, "contrasts": rows,
    }
    # A contrast with no comparable frames is NOT evidence of anything. The
    # first version of this script scored the basket contrast "different boxes"
    # off n_compared=0 -- the deployed source_max_area filter rejects
    # basket-sized boxes by construction, so that pair can never compare a
    # single frame -- and printed a confident verdict from an empty set. Any
    # contrast below MIN_N is reported as not evaluable and excluded from the
    # verdict rather than counted as a separation.
    MIN_N = 3
    evaluable = [r for r in rows if r["n_compared"] >= MIN_N]
    for r in rows:
        r["evaluable"] = r["n_compared"] >= MIN_N
    separating = [r for r in evaluable if r["same_rate_at_0.02"] <= 0.2]
    colliding = [r for r in evaluable if r["same_rate_at_0.02"] >= 0.5]
    if not evaluable:
        out["verdict"] = ("NOT EVALUABLE: no contrast had >= %d frames in which "
                          "both chains detected. This says nothing about the "
                          "probe or the stack." % MIN_N)
    elif separating:
        out["verdict"] = (
            "INSTRUMENT DISCRIMINATES: %d of %d evaluable contrasts return "
            "DIFFERENT boxes (%s), so a same-box result is evidence about the "
            "stack and not an artifact of the probe. %d contrasts collide (%s)."
            % (len(separating), len(evaluable),
               ", ".join(r["pair"].split(" vs ")[1] for r in separating),
               len(colliding),
               ", ".join(r["pair"].split(" vs ")[1] for r in colliding)))
    else:
        out["verdict"] = (
            "INSTRUMENT DOES NOT DISCRIMINATE on this scene: no evaluable "
            "contrast separated. Every same-box number from this probe must be "
            "treated as uninformative until a separating contrast is found.")
    out["min_n_for_verdict"] = MIN_N
    out["n_evaluable"] = len(evaluable)
    print("\n" + out["verdict"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
