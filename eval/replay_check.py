"""Replay a demo's own actions through the eval env. The control experiment.

Closed-loop success has been 0.000 across every run this project has produced —
150 trials on v8 alone — while open-loop metrics looked healthy (``corr`` 0.54,
``std_ratio`` 0.44-0.56). That combination has two very different explanations
and they need separating before another policy is trained:

1. the policy is not good enough, or
2. the eval path cannot succeed with ANY actions.

(2) is cheap to rule out and nobody has. If the demonstration's OWN actions,
replayed from the demonstration's OWN initial state, do not solve the task, then
no policy could, and every success number this project has ever produced was
measuring a broken pipeline rather than a broken model.

What a failure here would implicate, in rough order of likelihood: the
controller configuration (LIBERO demos are recorded under OSC_POSE; a different
controller reinterprets the same 7 numbers), the init-state indexing (replaying
demo *i* from init state *j* starts the arm somewhere the actions do not fit),
and the action convention (dimension order, gripper sign, absolute vs delta).

Usage::

    python -m eval.replay_check --hdf5 <task>.hdf5 --suite libero_object \\
        --task-name pick_up_the_alphabet_soup_and_place_it_in_the_basket \\
        --n-demos 10
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from microvla.utils.signals import ignore_sigterm


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, help="one LIBERO task .hdf5 of demos")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-name", default=None,
                   help="benchmark task name; defaults to the hdf5 stem minus '_demo'")
    p.add_argument("--n-demos", type=int, default=10)
    p.add_argument("--camera", default="robot0_eye_in_hand_image")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--use-demo-init", action="store_true", default=True,
                   help="reset to the demo's own recorded initial state (default). "
                        "The whole point: a demo replayed from a DIFFERENT start "
                        "is not a control, it is a new experiment.")
    p.add_argument("--use-bench-init", dest="use_demo_init", action="store_false",
                   help="use the benchmark's init state i instead — this is what "
                        "eval/libero_eval.py does, so a pass here and a fail "
                        "there localizes the bug to init-state selection.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ignore_sigterm()

    import os
    import platform
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if platform.system() == "Darwin" else "egl"

    from eval._libero_compat import prepare_libero
    prepare_libero()

    import h5py
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    task_name = args.task_name
    if task_name is None:
        stem = os.path.basename(args.hdf5).replace(".hdf5", "")
        task_name = stem[:-5] if stem.endswith("_demo") else stem

    bench = benchmark.get_benchmark_dict()[args.suite]()
    names = [bench.get_task(i).name for i in range(bench.n_tasks)]
    if task_name not in names:
        raise SystemExit(f"task {task_name!r} not in {args.suite}; have {names}")
    idx = names.index(task_name)
    task = bench.get_task(idx)
    init_states = bench.get_task_init_states(idx)

    from libero.libero import get_libero_path
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                             camera_widths=args.size)

    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[: args.n_demos]
        print(f"task {task_name}\n{len(demos)} demos | init_states {init_states.shape}\n")
        print(f"{'demo':>6} {'steps':>6} {'success':>8}  final |action| mean")
        ok = 0
        for n, d in enumerate(demos):
            actions = np.asarray(f[f"data/{d}/actions"])
            env.reset()
            if args.use_demo_init and "states" in f[f"data/{d}"]:
                # The demo's OWN first recorded sim state.
                env.set_init_state(np.asarray(f[f"data/{d}/states"])[0])
            else:
                env.set_init_state(init_states[n % len(init_states)])
            done = False
            for a in actions:
                _obs, _r, done, _info = env.step(np.asarray(a, dtype=np.float64))
                if done:
                    break
            success = bool(env.check_success())
            ok += success
            print(f"{n:>6} {len(actions):>6} {str(success):>8}  "
                  f"{np.abs(actions).mean():.4f}")

    rate = ok / max(len(demos), 1)
    print(f"\nreplay success {ok}/{len(demos)} = {rate:.3f}")
    print()
    if rate >= 0.9:
        print("PASS — the env, controller, init states and action convention are")
        print("sound. A policy CAN succeed here, so 0.000 is the policy, not the")
        print("pipeline. Next suspect is the emitted-action path: normalization")
        print("round-trip, the waypoint actuator, and the trust brake.")
    else:
        print("FAIL — the demonstration's own actions do not solve the task.")
        print("No policy could. Every closed-loop number this project has")
        print("produced was measuring a broken pipeline. Check, in order: the")
        print("controller config (demos are OSC_POSE), init-state selection, and")
        print("the action convention (dim order, gripper sign, delta vs absolute).")
    sys.exit(0 if rate >= 0.9 else 1)


if __name__ == "__main__":
    main()
