"""Record full-episode MP4s of the trained policy driving the robot.

Unlike ``rollout_video`` (one PNG montage), this renders EVERY env step to a
frame and muxes them into a real 30 fps MP4 you can scrub. It picks N random
tasks (and random init states) from a LIBERO suite so you get a fresh sample
each run.

    PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO MUJOCO_GL=osmesa \
    python -m eval.record_mp4 --suite libero_spatial --n-videos 2 \
        --checkpoint checkpoints/full_stageB.pt \
        --norm-stats data/libero/norm_stats.json \
        --device cuda:0 --max-steps 300

Writes one ``.mp4`` per episode into ``--out-dir`` (default eval_results/videos).
Each frame is BOTH cameras side by side: the third-person agentview (LEFT, what
is really happening) and the wrist ``eye_in_hand`` camera (RIGHT, exactly what
the policy sees). Agentview is rotated 180° so it renders right-side up
(LIBERO's agentview is mounted upside down); the wrist cam is un-rotated.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def _label(img, text, cv2):
    """Draws a small caption in the top-left of an RGB uint8 frame (in place)."""
    scale = max(0.35, img.shape[0] / 400.0)
    org = (4, int(14 * scale) + 2)
    # black outline then yellow text, so it's legible on any background.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def _side_by_side(agent_rgb, wrist_rgb, cv2):
    """Third-person (left) + wrist/policy view (right), captioned, concatenated."""
    agent = np.ascontiguousarray(agent_rgb).astype(np.uint8)
    wrist = np.ascontiguousarray(wrist_rgb).astype(np.uint8)
    # Match heights if the two cameras ever differ in size (they don't by default).
    if agent.shape[0] != wrist.shape[0]:
        h = min(agent.shape[0], wrist.shape[0])
        agent, wrist = agent[:h], wrist[:h]
    _label(agent, "3rd person", cv2)
    _label(wrist, "wrist (policy sees this)", cv2)
    sep = np.zeros((agent.shape[0], 2, 3), np.uint8)  # thin divider
    return np.concatenate([agent, sep, wrist], axis=1)


def _write_mp4(path: str, frames, fps: int) -> None:
    """Mux RGB uint8 frames -> MP4 via OpenCV's built-in mp4v codec (no ffmpeg dep)."""
    import cv2

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path} (codec/dir issue)")
    for f in frames:
        writer.write(f[..., ::-1])  # RGB -> BGR
    writer.release()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--n-videos", type=int, default=2)
    ap.add_argument("--checkpoint", default="checkpoints/full_stageB.pt")
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--res", type=int, default=128,
                    help="render height/width (px). osmesa is CPU software "
                         "rendering, so cost scales ~res^2; 128 matches the "
                         "eval. Bump to 256/384 only if you can wait.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-dir", default="eval_results/videos")
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a genuinely random pick each run")
    ap.add_argument("--zero-center-actions", action="store_true",
                    help="denormalize actions zero-centered (x=0 -> no motion), so a "
                         "collapsed policy stays still instead of drifting into a wall.")
    args = ap.parse_args(argv)

    from eval._libero_compat import prepare_libero
    prepare_libero()
    from libero.libero import benchmark            # registry only — no mujoco/GL
    from eval.policy import MicroVLAPolicy          # torch + ultralytics FIRST

    bench = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = bench.n_tasks
    rng = random.Random(args.seed)
    task_ids = rng.sample(range(n_tasks), k=min(args.n_videos, n_tasks))
    # if more videos than tasks, allow repeats (different init states)
    while len(task_ids) < args.n_videos:
        task_ids.append(rng.randrange(n_tasks))
    print(f"suite {args.suite!r}: {n_tasks} tasks, recording task ids {task_ids}")

    # CRITICAL ORDER (matches libero_eval / rollout_video): build the torch/YOLO
    # policy BEFORE importing the robosuite/mujoco env, or mujoco's GL stack
    # segfaults. Build the policy ONCE and reuse it across episodes.
    policy = MicroVLAPolicy(checkpoint=args.checkpoint, norm_stats=args.norm_stats,
                            device=args.device, zero_center_actions=args.zero_center_actions)
    from libero.libero.envs import OffScreenRenderEnv

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vi, tid in enumerate(task_ids):
        task = bench.get_task(tid)
        bddl = bench.get_task_bddl_file_path(tid)
        inits = np.asarray(bench.get_task_init_states(tid))
        print(f"\n[{vi+1}/{len(task_ids)}] task {tid}: {task.language!r}")

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=args.res, camera_widths=args.res
        )
        import cv2  # lazy: box-only (present wherever the real env renders)
        try:
            obs = env.reset()
            if len(inits) > 0:
                obs = env.set_init_state(inits[rng.randrange(len(inits))])
            policy.reset(task.language)

            frames, success = [], False
            for step in range(args.max_steps):
                wrist = np.asarray(obs["robot0_eye_in_hand_image"])  # policy's view
                action = np.asarray(policy.act(wrist), dtype=np.float32)
                # agentview is mounted upside down -> rotate 180 for viewing.
                agent = np.rot90(np.asarray(obs["agentview_image"]), 2)
                # Side by side: 3rd person (left) + wrist/policy view (right).
                frames.append(_side_by_side(agent, wrist, cv2))
                obs, _r, done, _info = env.step(action)
                if hasattr(env, "check_success") and env.check_success():
                    success = True
                if done or success:
                    break

            tag = "success" if success else "fail"
            slug = "".join(c if c.isalnum() else "_" for c in task.language)[:40]
            out = out_dir / f"{args.suite}_t{tid}_{tag}_{slug}.mp4"
            _write_mp4(str(out), frames, args.fps)
            print(f"    {len(frames)} steps, success={success} -> {out}")
        finally:
            if hasattr(env, "close"):
                env.close()

    print(f"\ndone. {args.n_videos} video(s) in {out_dir}/")


if __name__ == "__main__":
    main()
