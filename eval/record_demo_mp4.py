"""Record a side-by-side MP4 of a LIBERO demo replay (ground-truth actions).

Used to compare how a *successful* demonstrator clears neighbour cans/cartons
vs the policy's blocked last-centimetre descent (paper.md §5q).
"""
from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import numpy as np

from microvla.utils.camera import AGENTVIEW, ENV_KEY, WRIST, upright
from microvla.utils.signals import ignore_sigterm


def _label(img, text, cv2):
    scale = max(0.35, img.shape[0] / 400.0)
    org = (4, int(14 * scale) + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def _side_by_side(agent_rgb, wrist_rgb, cv2):
    agent = np.ascontiguousarray(agent_rgb).astype(np.uint8)
    wrist = np.ascontiguousarray(wrist_rgb).astype(np.uint8)
    if agent.shape[0] != wrist.shape[0]:
        h = min(agent.shape[0], wrist.shape[0])
        agent, wrist = agent[:h], wrist[:h]
    _label(agent, "3rd person (DEMO)", cv2)
    _label(wrist, "wrist (DEMO)", cv2)
    sep = np.zeros((agent.shape[0], 2, 3), np.uint8)
    return np.concatenate([agent, sep, wrist], axis=1)


def _write_mp4(path: str, frames, fps: int) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    for f in frames:
        writer.write(f[..., ::-1])
    writer.release()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-name", default=None)
    ap.add_argument("--demo-id", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-dir", default="eval_results/demo_videos")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = use full demo length")
    args = ap.parse_args(argv)
    ignore_sigterm()

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if platform.system() == "Darwin" else "egl"

    from eval._libero_compat import prepare_libero
    prepare_libero()

    import cv2
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
        # language vs name — fall back to language match
        langs = [bench.get_task(i).language for i in range(bench.n_tasks)]
        if task_name in langs:
            tid = langs.index(task_name)
        else:
            raise SystemExit(f"task {task_name!r} not in suite; have {names[:5]}…")
    else:
        tid = names.index(task_name)
    task = bench.get_task(tid)
    bddl = bench.get_task_bddl_file_path(tid)

    with h5py.File(args.hdf5, "r") as f:
        key = f"data/demo_{args.demo_id}"
        if key not in f:
            raise SystemExit(f"{key} missing; demos={[k for k in f['data'].keys()][:8]}")
        actions = np.asarray(f[f"{key}/actions"], dtype=np.float32)
        states = np.asarray(f[f"{key}/states"])

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=args.res, camera_widths=args.res
    )
    obs = env.reset()
    obs = env.set_init_state(states[0])

    n = len(actions) if args.max_steps <= 0 else min(len(actions), args.max_steps)
    frames, success = [], False
    for i in range(n):
        agent = upright(obs[ENV_KEY[AGENTVIEW]], ENV_KEY[AGENTVIEW])
        wrist = upright(obs[ENV_KEY[WRIST]], ENV_KEY[WRIST])
        frames.append(_side_by_side(agent, wrist, cv2))
        obs, _r, done, _info = env.step(actions[i])
        if hasattr(env, "check_success") and env.check_success():
            success = True
        if done or success:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if success else "fail"
    slug = "".join(c if c.isalnum() else "_" for c in task.language)[:40]
    out = out_dir / f"demo_{args.demo_id}_{tag}_{slug}.mp4"
    _write_mp4(str(out), frames, args.fps)
    print(f"wrote {out}  steps={len(frames)} success={success}")
    env.close()


if __name__ == "__main__":
    main()
