"""Push a demo's actions through the POLICY's actuation path, stage by stage.

``eval/replay_check.py`` established that the environment is sound: a demo's own
actions, stepped directly, solve the task 5/5. So the 0.000 closed-loop success
lives between the network's output and ``env.step`` — in the normalizer, the
waypoint actuator, or the trust brake.

This isolates those. Each stage is applied to GROUND-TRUTH actions and replayed:
whatever stage first drops success below 1.0 is the one that cannot carry a
correct action to the robot. A policy cannot do better than ground truth pushed
through the same pipe, so any stage that fails here is an upper bound on the
whole system.

Stages, in the order the deployed path applies them:

``raw``
    demo actions, stepped directly. The control; must be ~1.0 or the harness
    itself is broken and nothing below means anything.
``norm``
    ``normalizer.inverse(normalizer(a))``. A quantile normalizer CLIPS, so any
    action outside the fitted q01/q99 comes back shortened — and clipping is
    exactly what a demo's fast reaching motions will hit.
``brake``
    the corrector's trust scaling, at a fixed representative trust. Measured in
    deployment at ~50% of steps attenuated (trust mean 0.521 against
    ``cfg.brake_trust`` 0.5).
``norm+brake``
    both, i.e. the full path minus the network.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys

import numpy as np

from microvla.utils.signals import ignore_sigterm


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-name", required=True)
    p.add_argument("--norm-stats", required=True,
                   help="the norm_stats.json paired with the corpus under test")
    p.add_argument("--n-demos", type=int, default=5)
    p.add_argument("--trust", type=float, default=0.521,
                   help="representative trust for the brake stage; the deployed "
                        "mean was 0.521 against cfg.brake_trust 0.5")
    p.add_argument("--stages", default="raw,norm,brake,norm+brake")
    p.add_argument("--scale-sweep", default=None,
                   help="comma-separated magnitude multipliers applied to the POSE "
                        "columns of ground-truth actions (gripper exempt). Maps how "
                        "much magnitude error the task tolerates at all, which is the "
                        "bar any policy has to clear: measured 1.0 -> 5/5 but "
                        "0.8 -> 0/4, so a policy at std_ratio 0.5 cannot succeed no "
                        "matter how correct its direction is.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ignore_sigterm()
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if platform.system() == "Darwin" else "osmesa"

    from eval._libero_compat import prepare_libero
    prepare_libero()

    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from microvla.config import DEFAULT_CONFIG as cfg
    from preprocess.common import ActionNormalizer

    norm = ActionNormalizer.load(args.norm_stats)

    bench = benchmark.get_benchmark_dict()[args.suite]()
    names = [bench.get_task(i).name for i in range(bench.n_tasks)]
    idx = names.index(args.task_name)
    task = bench.get_task(idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    # ---- stage 0: does the normalizer even round-trip? ----------------------
    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[: args.n_demos]
        sample = np.concatenate([np.asarray(f[f"data/{d}/actions"]) for d in demos])
    rt = norm.inverse(norm(sample))
    err = np.abs(rt - sample)
    clipped = float((np.abs(norm(sample)) >= 0.999).mean()) * 100
    print("normalizer round-trip on demo actions")
    print(f"  max |a - inverse(norm(a))| : {err.max():.4f}")
    print(f"  mean                        : {err.mean():.4f}")
    print(f"  per-dim max                 : {np.round(err.max(0), 4).tolist()}")
    print(f"  actions CLIPPED at +/-1     : {clipped:.1f}% of all dims")
    if clipped > 5:
        print("  ^ a quantile normalizer clips outside q01/q99, and a clipped")
        print("    action is SHORTER than the demo's. That is magnitude loss")
        print("    applied before the policy ever runs.")
    print()

    def apply(a: np.ndarray, stage: str) -> np.ndarray:
        out = a
        if "norm" in stage:
            out = norm.inverse(norm(out))
        if "brake" in stage:
            # Delta-mode brake: plan = min(1, tau/brake_trust) * raw, gripper exempt.
            scale = min(1.0, args.trust / cfg.brake_trust) if cfg.brake_trust > 0 else 1.0
            out = out.copy()
            out[..., :-1] = out[..., :-1] * scale
        return out

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)

    if args.scale_sweep:
        scales = [float(s) for s in args.scale_sweep.split(",")]
        print(f"{'scale':>7} {'success':>10}   (pose columns only; gripper untouched)")
        with h5py.File(args.hdf5, "r") as f:
            for k in scales:
                ok = 0
                for d in demos:
                    a = np.asarray(f[f"data/{d}/actions"], dtype=np.float64).copy()
                    a[:, :-1] *= k
                    env.reset()
                    env.set_init_state(np.asarray(f[f"data/{d}/states"])[0])
                    for step in a:
                        _o, _r, done, _i = env.step(step)
                        if done:
                            break
                    ok += bool(env.check_success())
                bar = "#" * int(round(ok / len(demos) * 20))
                print(f"{k:7.2f} {ok}/{len(demos)} = {ok/len(demos):4.2f}  {bar}")
        print()
        print("Read this as the accuracy bar. Any policy whose emitted magnitude")
        print("falls outside the passing band cannot solve the task regardless of")
        print("how well it predicts DIRECTION — and std_ratio measures exactly")
        print("that magnitude ratio.")
        return

    print(f"{'stage':12s} {'success':>9} {'|a| vs raw':>12}")
    results = {}
    with h5py.File(args.hdf5, "r") as f:
        for stage in args.stages.split(","):
            ok, mags = 0, []
            for d in demos:
                actions = np.asarray(f[f"data/{d}/actions"], dtype=np.float64)
                mod = apply(actions, stage)
                mags.append(np.abs(mod[:, :-1]).mean() / max(np.abs(actions[:, :-1]).mean(), 1e-9))
                env.reset()
                env.set_init_state(np.asarray(f[f"data/{d}/states"])[0])
                for a in mod:
                    _o, _r, done, _i = env.step(a)
                    if done:
                        break
                ok += bool(env.check_success())
            rate = ok / len(demos)
            results[stage] = rate
            print(f"{stage:12s} {ok}/{len(demos)} = {rate:4.2f} {np.mean(mags):11.3f}x")

    print()
    base = results.get("raw", 1.0)
    culprits = [s for s, r in results.items() if s != "raw" and r < base - 1e-9]
    if not culprits:
        print("No stage degrades ground truth. The actuation path carries a correct")
        print("action faithfully, so the loss is in the NETWORK's output, not in")
        print("what happens to it afterwards.")
    else:
        print(f"DEGRADED BY: {', '.join(culprits)}")
        print("A policy cannot beat ground truth through the same pipe, so this")
        print("bounds the whole system regardless of how good the network gets.")


if __name__ == "__main__":
    main()
