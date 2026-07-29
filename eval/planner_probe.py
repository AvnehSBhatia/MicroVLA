"""Compare the planner's ACTUAL inputs at deployment against the corpus.

Four defects in this stack have been the same shape: two sides of a train/deploy
pair disagreeing about what a value means, each side individually correct and
individually tested (paper.md 4t). Three of them were found by guessing which
input might differ and checking that one — grounding prompts, proprio
orientation, frame rotation — and after fixing the two that were real, the
policy's behaviour did not move: the emitted gripper stayed pinned at exactly
-1.0 with std 0.000000 across 9000 ticks, against a checkpoint whose stage-B
validation grip_acc is 0.94 (a 0.477 always-open baseline).

Guessing one input at a time does not scale and does not terminate. This
instruments the planner instead: a forward hook captures every tensor the
planner is actually handed at deployment, and the same statistics are computed
over the baked corpus the planner was trained on. Whichever memory group is far
out of distribution is the defect — no hypothesis required.

Read the output as: for each planner input group, deployment mean/std next to
corpus mean/std. A group whose deployment std collapses to ~0, or whose mean sits
many corpus-sds away, is the one carrying the policy off-distribution.

Usage::

    MUJOCO_GL=osmesa python -m eval.planner_probe \\
        --checkpoint checkpoints/full_stageB_v8_act.pt \\
        --corpus data/libero_object_v8 --steps 60
"""
from __future__ import annotations

import argparse
import glob
import os
import platform

import numpy as np

from microvla.utils.signals import ignore_sigterm


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True, help="the baked dir this ckpt trained on")
    p.add_argument("--norm-stats", default=None, help="defaults to <corpus>/norm_stats.json")
    p.add_argument("--waypoint-stats", default=None)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--episodes", type=int, default=40, help="corpus episodes to profile")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def _stats(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def main(argv=None) -> None:
    args = parse_args(argv)
    ignore_sigterm()
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if platform.system() == "Darwin" else "osmesa"

    import torch

    from eval._libero_compat import prepare_libero
    prepare_libero()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from eval.policy import MicroVLAPolicy
    from microvla.utils.proprio import proprio_from_obs

    norm = args.norm_stats or os.path.join(args.corpus, "norm_stats.json")
    wp = args.waypoint_stats or os.path.join(args.corpus, "waypoint_stats.json")
    policy = MicroVLAPolicy(
        checkpoint=args.checkpoint, norm_stats=norm,
        waypoint_stats=wp if os.path.exists(wp) else None,
        device=args.device, heads_device=args.device,
    )

    # ---- capture what the planner is actually handed, at deployment ---------
    seen: dict[str, list[np.ndarray]] = {}

    def hook(_mod, _a, kwargs):
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                seen.setdefault(k, []).append(v.detach().float().cpu().numpy())
        return None

    planner = policy.loop.planner
    planner.register_forward_pre_hook(hook, with_kwargs=True)

    bench = benchmark.get_benchmark_dict()[args.suite]()
    task = bench.get_task(args.task_index)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    obs = env.reset()
    init = bench.get_task_init_states(args.task_index)
    obs = env.set_init_state(init[0])
    policy.reset(task.language)

    plans = []
    for _ in range(args.steps):
        frame = np.asarray(obs["robot0_eye_in_hand_image"])
        a = policy.act(frame, proprio=proprio_from_obs(obs))
        plans.append(np.asarray(a, dtype=np.float64))
        obs, _r, done, _i = env.step(np.asarray(a, dtype=np.float64))
        if done:
            break

    # ---- the same groups, over the corpus the planner trained on -----------
    files = sorted(glob.glob(os.path.join(args.corpus, "*.npz")))[: args.episodes]
    corpus: dict[str, np.ndarray] = {}
    if files:
        for key, dest in (("frame_embs", "current_emb"), ("proprio", "proprio")):
            try:
                corpus[dest] = np.concatenate([np.load(f)[key] for f in files])
            except KeyError:
                pass

    print(f"\nplanner inputs: deployment ({args.steps} ticks) vs corpus "
          f"({len(files)} episodes)\n")
    print(f"{'group':14s} {'source':>6}  {'mean':>9} {'std':>9} {'min':>9} {'max':>9}")
    for k in sorted(seen):
        d = np.concatenate([v.reshape(-1) for v in seen[k]])
        m, s, lo, hi = _stats(d)
        print(f"{k:14s} {'DEPLOY':>6}  {m:9.4f} {s:9.4f} {lo:9.4f} {hi:9.4f}")
        if k in corpus:
            m2, s2, lo2, hi2 = _stats(corpus[k])
            print(f"{'':14s} {'corpus':>6}  {m2:9.4f} {s2:9.4f} {lo2:9.4f} {hi2:9.4f}")
            if s2 > 1e-8:
                z = abs(m - m2) / s2
                flag = "   <== OFF-DISTRIBUTION" if z > 2 or s < 0.1 * s2 else ""
                print(f"{'':14s} {'delta':>6}  mean is {z:.1f} corpus-sd away, "
                      f"std ratio {s / s2:.3f}{flag}")

    P = np.stack(plans)
    print(f"\nemitted action over {len(P)} ticks")
    print(f"  gripper unique : {np.unique(np.round(P[:, -1], 3)).tolist()[:8]}")
    print(f"  gripper std    : {P[:, -1].std():.6f}")
    print(f"  pose std/dim   : {np.round(P[:, :3].std(0), 4).tolist()}")


if __name__ == "__main__":
    main()
