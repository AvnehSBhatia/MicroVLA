"""Stage-1 sim validation: bring up ONE real LIBERO env, no MicroVLA policy.

Run this on the GPU box FIRST, before wiring the trained policy in. It proves
the LIBERO physics stack is installed and rendering, and — critically — dumps
the wrist-camera frame the policy will consume so we can confirm its
orientation/format match what the model trained on (LIBERO eye_in_hand HDF5,
used un-rotated). If this passes, the closed-loop eval is just plumbing.

    MUJOCO_GL=egl python -m eval.env_smoke --suite libero_object --task 0

Checks, in order:
  1. libero.libero.benchmark enumerates the suite (no physics needed).
  2. OffScreenRenderEnv builds for the task's BDDL scene (needs assets).
  3. reset() + set_init_state() + 5 random steps run without error.
  4. robot0_eye_in_hand_image comes out [H,W,3] uint8; saved to
     eval_results/env_smoke_wrist.png for eyeball comparison to training frames.
  5. Reports how success is queried (info['success'] vs env.check_success()).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--camera", default="robot0_eye_in_hand_image")
    ap.add_argument("--size", type=int, default=128, help="render size (training used 128)")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args(argv)

    if "MUJOCO_GL" not in os.environ:
        # Linux GPU box: egl is the headless renderer (osmesa is the CPU fallback).
        os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/libero_home")
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}")

    # 1. registry
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    if args.suite not in bd:
        raise SystemExit(f"unknown suite {args.suite!r}; have: {sorted(bd)}")
    bench = bd[args.suite]()
    n = bench.get_num_tasks() if hasattr(bench, "get_num_tasks") else bench.n_tasks
    task = bench.get_task(args.task)
    bddl = bench.get_task_bddl_file_path(args.task)
    inits = np.asarray(bench.get_task_init_states(args.task))
    print(f"suite {args.suite}: {n} tasks | task {args.task}: {getattr(task,'name','?')!r}")
    print(f"  language: {task.language!r}")
    print(f"  bddl: {bddl}")
    print(f"  init_states: {inits.shape}")

    # 2. env
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=args.size, camera_widths=args.size)
    try:
        # 3. reset + init + steps
        obs = env.reset()
        if len(inits) > 0:
            obs = env.set_init_state(inits[0])
        print(f"  obs keys: {sorted(obs.keys())[:12]}{' ...' if len(obs)>12 else ''}")
        assert args.camera in obs, f"camera {args.camera!r} not in obs keys!"
        act_dim = env.action_space.shape[0] if hasattr(env, "action_space") else "?"
        print(f"  action dim: {act_dim} (expect 7)")

        done = False
        for i in range(args.steps):
            a = np.random.uniform(-1, 1, size=7).astype(np.float32)
            obs, reward, done, info = env.step(a)
        # 4. wrist frame
        frame = np.asarray(obs[args.camera])
        print(f"  {args.camera}: shape {frame.shape} dtype {frame.dtype} "
              f"min/max {frame.min()}/{frame.max()}")
        out = Path("eval_results"); out.mkdir(exist_ok=True)
        try:
            import cv2
            cv2.imwrite(str(out / "env_smoke_wrist.png"), frame[..., ::-1])  # RGB->BGR for cv2
            print(f"  saved {out/'env_smoke_wrist.png'} — EYEBALL vs a training frame "
                  "(same object/wrist view, upright?). If upside-down, the eval "
                  "must flip to match training.")
        except Exception as e:
            print(f"  (could not save png: {e})")

        # 5. success API
        has_check = hasattr(env, "check_success")
        info_success = isinstance(info, dict) and "success" in info
        print(f"  success API: env.check_success()={has_check}, info['success']={info_success}")
        print("\nSTAGE 1 PASS — sim renders, env steps, wrist frame captured.")
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
