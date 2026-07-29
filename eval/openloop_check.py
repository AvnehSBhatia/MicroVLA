"""Drive the DEPLOYMENT stack with a demonstration's own frames (teacher forcing).

Separates the two remaining explanations for a policy that emits a constant
gripper in closed loop while its stage-B validation reports ``grip_acc 0.94``:

1. **A defect** — the checkpoint, the load, or the deployment assembly is wrong,
   in which case the policy is broken on ANY input, including a demonstration's.
2. **Covariate shift** — the policy is fine on states the demonstrations visit,
   but its own early errors take it somewhere no training state resembles, and
   everything after that is off-distribution. The classic behaviour-cloning
   failure, and NOT a bug.

``eval/planner_probe.py`` established that the planner's deployment inputs are
in-distribution at the group level and that the per-dimension proprio
differences (EEF ``z`` 0.048 vs corpus 0.205, gripper always open) are
CONSEQUENCES of the policy's own descent rather than causes — the reset state
matches the corpus. That rules out the input path but not the two explanations
above, because both produce identical closed-loop telemetry.

This distinguishes them. The policy is stepped through a demo's recorded frames
and proprio in order, so it never leaves the demonstration's state
distribution — exactly the condition stage B validated under, but through the
real ``MicroVLAPolicy`` / ``JEPALoop`` code path rather than the trainer's. If
the gripper closes at roughly the demo's timing, the deployment stack is sound
and the closed-loop failure is compounding error. If it stays pinned, the defect
is in the stack and stage-B's 0.94 is not reproducible through it.

Usage::

    MUJOCO_GL=osmesa python -m eval.openloop_check \\
        --checkpoint checkpoints/full_stageB_v8_act.pt \\
        --corpus data/libero_object_v8 \\
        --hdf5 /root/libero_raw/libero_object/<task>_demo.hdf5 --n-demos 3
"""
from __future__ import annotations

import argparse
import os
import platform

import numpy as np

from microvla.utils.signals import ignore_sigterm


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--norm-stats", default=None)
    p.add_argument("--waypoint-stats", default=None)
    p.add_argument("--camera", default="eye_in_hand_rgb",
                   help="hdf5 obs key; must be the view the corpus was baked from")
    p.add_argument("--n-demos", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--task", default=None, help="language string; defaults to the file stem")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ignore_sigterm()
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if platform.system() == "Darwin" else "osmesa"

    import h5py

    from eval.policy import MicroVLAPolicy
    from microvla.utils.proprio import build_proprio

    task = args.task
    if task is None:
        stem = os.path.basename(args.hdf5).replace("_demo.hdf5", "").replace(".hdf5", "")
        task = stem.replace("_", " ")

    norm = args.norm_stats or os.path.join(args.corpus, "norm_stats.json")
    wp = args.waypoint_stats or os.path.join(args.corpus, "waypoint_stats.json")
    policy = MicroVLAPolicy(
        checkpoint=args.checkpoint, norm_stats=norm,
        waypoint_stats=wp if os.path.exists(wp) else None,
        device=args.device, heads_device=args.device,
    )

    print(f"task: {task!r}\n")
    print(f"{'demo':>5} {'steps':>6} {'grip agree':>11} {'demo close%':>12} "
          f"{'ours close%':>12} {'pose corr':>10}")

    agrees, ours_all, demo_all = [], [], []
    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[: args.n_demos]
        for d in demos:
            grp = f["data"][d]
            frames = np.asarray(grp["obs"][args.camera])
            actions = np.asarray(grp["actions"], dtype=np.float64)
            obs = grp["obs"]
            n = min(len(frames), len(actions), args.max_steps)

            policy.reset(task)
            emitted = []
            for t in range(n):
                # The demo's OWN state at t: teacher forcing, so the policy never
                # leaves the demonstrated distribution.
                pro = build_proprio(
                    np.asarray(obs["ee_pos"][t]),
                    np.asarray(obs["ee_ori"][t]) if "ee_ori" in obs else None,
                    np.asarray(obs["gripper_states"][t]) if "gripper_states" in obs else None,
                )
                emitted.append(policy.act(np.asarray(frames[t]), proprio=pro))
            E = np.stack(emitted)
            A = actions[:n]

            agree = float(((E[:, -1] > 0) == (A[:, -1] > 0)).mean())
            ours = float((E[:, -1] > 0).mean())
            demo = float((A[:, -1] > 0).mean())
            cc = [np.corrcoef(E[:, i], A[:, i])[0, 1] for i in range(3)
                  if E[:, i].std() > 1e-9 and A[:, i].std() > 1e-9]
            corr = float(np.mean(cc)) if cc else float("nan")
            agrees.append(agree); ours_all.append(ours); demo_all.append(demo)
            print(f"{d:>5} {n:>6} {agree:11.3f} {demo*100:11.1f}% {ours*100:11.1f}% "
                  f"{corr:10.3f}")

    print()
    print(f"gripper agreement with the demo : {np.mean(agrees):.3f}")
    print(f"demo closes on                  : {np.mean(demo_all)*100:.1f}% of steps")
    print(f"we close on                     : {np.mean(ours_all)*100:.1f}% of steps")
    print()
    if np.mean(ours_all) < 0.01:
        print("The gripper NEVER closes even on the demonstration's own frames, so")
        print("the policy is not merely off-distribution in closed loop — the")
        print("deployment stack does not reproduce stage B's grip_acc 0.94 on the")
        print("data stage B measured it on. The defect is in the stack or the")
        print("checkpoint, not in compounding error.")
    elif np.mean(agrees) > 0.8:
        print("The gripper tracks the demo when the states are the demo's, so the")
        print("stack and the checkpoint are sound. The closed-loop failure is")
        print("COMPOUNDING ERROR: early mistakes move the arm somewhere no")
        print("training state resembles. That is a policy/data problem (more")
        print("coverage, DAgger, or action chunking), not a bug to find.")
    else:
        print("Partial tracking: the head fires but not at the demo's timing.")


if __name__ == "__main__":
    main()
