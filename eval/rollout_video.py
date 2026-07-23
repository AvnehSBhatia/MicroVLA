"""Run ONE episode with the trained policy and save a visual + action trace.

The success number tells you *whether* it works; this tells you *why*. It runs
a single LIBERO episode, saves a montage of agentview frames across the episode
(so you can see what the arm actually does), and prints the per-dimension
action statistics (so you can see if it ever commands the gripper, whether the
deltas are sane, etc.).

    PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO MUJOCO_GL=osmesa \
    python -m eval.rollout_video --suite libero_object --task 0 \
        --checkpoint checkpoints/full_stageB.pt --norm-stats data/libero/norm_stats.json \
        --device cuda:0 --max-steps 200

Reads:  the arm should reach TOWARD the object, close the gripper on it, lift,
        and move to the basket. If it drifts away / freezes / jitters in place,
        that localizes the failure (orientation, precision, or grounding).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoints/full_stageB.pt")
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--montage-frames", type=int, default=12)
    ap.add_argument("--out", default="eval_results/rollout.png")
    args = ap.parse_args(argv)

    from eval._libero_compat import prepare_libero
    prepare_libero()
    from libero.libero import benchmark          # registry only — no mujoco/GL
    from eval.policy import MicroVLAPolicy        # torch + ultralytics FIRST

    bench = benchmark.get_benchmark_dict()[args.suite]()
    task = bench.get_task(args.task)
    bddl = bench.get_task_bddl_file_path(args.task)
    inits = np.asarray(bench.get_task_init_states(args.task))
    print(f"task: {task.language!r}")

    # CRITICAL ORDER (matches libero_eval, which runs without segfaulting):
    # build the torch/YOLO policy, and only THEN import + construct the
    # robosuite/mujoco env. Importing mujoco's GL stack before torch is
    # initialized segfaults. Render at 128 (the size env_smoke/eval used).
    policy = MicroVLAPolicy(checkpoint=args.checkpoint, norm_stats=args.norm_stats,
                            device=args.device)
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    try:
        obs = env.reset()
        if len(inits) > 0:
            obs = env.set_init_state(inits[0])
        policy.reset(task.language)

        agent_frames, wrist_frames, actions, trusts = [], [], [], []
        for step in range(args.max_steps):
            wrist = np.asarray(obs["robot0_eye_in_hand_image"])
            action = np.asarray(policy.act(wrist), dtype=np.float32)
            agent_frames.append(np.asarray(obs["agentview_image"]))
            wrist_frames.append(wrist)
            actions.append(action)
            trusts.append(policy.loop.corrector.trust if hasattr(policy.loop, "corrector") else 1.0)
            obs, _r, done, info = env.step(action)
            if done or (hasattr(env, "check_success") and env.check_success()):
                print(f"episode ended at step {step}, success={getattr(env, 'check_success', lambda: '?')()}")
                break

        A = np.stack(actions)
        names = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "grip"]
        print(f"\nran {len(actions)} steps | final trust {trusts[-1]:.3f}")
        print("per-dim action stats (denormalized, env units):")
        for i, nm in enumerate(names):
            print(f"  {nm:8s} mean {A[:,i].mean():+.3f}  std {A[:,i].std():.3f}  "
                  f"min {A[:,i].min():+.3f}  max {A[:,i].max():+.3f}")
        grip = A[:, 6]
        print(f"gripper: opened(<0) {np.mean(grip<0)*100:.0f}% of steps, closed(>0) {np.mean(grip>0)*100:.0f}%")

        # montage of agentview frames sampled across the episode
        import cv2
        n = min(args.montage_frames, len(agent_frames))
        idxs = np.linspace(0, len(agent_frames) - 1, n).astype(int)
        tiles = [agent_frames[i] for i in idxs]
        cols = min(6, n); rows = (n + cols - 1) // cols
        h, w = tiles[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = t
        Path(args.out).parent.mkdir(exist_ok=True)
        cv2.imwrite(args.out, grid[..., ::-1])  # RGB->BGR
        print(f"\nsaved montage -> {args.out} (agentview, left->right = start->end)")
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
