"""Teacher-rollout recorder + converter (UNAIDED_PLAN Phase B).

The calibrated PhasedIBVS stack (handeye_v4/v5 configs; paper/paper.md
§5r–§5t) is the first controller that completes LIBERO picks on this
project's eval distribution. This module turns its SUCCESSFUL rollouts into
training shards in the exact schema `train/dataset.py` consumes, so stage B
can behavior-clone the teacher and convert the assisted success into an
unaided policy number (Phase C).

Two subcommands, mirroring the shard pipeline's record→convert→delete rule:

  record   Run the teacher on a real LIBERO task; keep only successful
           episodes; write RAW npz (uint8 frames + raw actions + proprio)
           under --raw-dir. Init-state indices start at --init-offset
           (default 20) so the teacher NEVER rolls the eval trials (0..19):
           training data and the eval protocol stay disjoint.

  convert  Stream the raw npz through preprocess.common.run_conversion —
           the SAME two-pass driver the demo bake uses (normalizer fit,
           perception features, spatial grid, manifest, provenance) — then
           optionally delete the raw dir (--purge-raw).

Teacher flags are forwarded verbatim from the winning eval configs; the
recorder imposes no policy of its own. Success only, by design: the point
is to imitate what worked.

DAgger mode (`record --dagger-student-flags "..."`): the STUDENT policy
(mostly) drives — teacher acts on a `--dagger-beta` fraction of ticks as
recovery mixing — while the teacher labels every visited state; saved
`actions` are always the teacher's labels (`executed_actions` records what
actually ran). Failures are kept: off-distribution states with teacher
labels are exactly the covariate-shift medicine BC lacks.

Pod usage (cream, teacher = v5 config):
  MUJOCO_GL=osmesa python -m preprocess.teacher_rollouts record \
      --suite libero_object --task-id 1 --n-success 30 --raw-dir data/teacher_raw \
      -- <every eval/libero_eval flag of the winning config>
  python -m preprocess.teacher_rollouts convert \
      --raw-dir data/teacher_raw --out data/teacher_cream --spatial-grid 4 \
      --camera eye_in_hand_rgb --purge-raw
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LIBERO_HZ = 20.0


# --------------------------------------------------------------------- record
def record(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description="record successful teacher rollouts")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--n-success", type=int, default=30,
                   help="stop after this many SUCCESSFUL episodes")
    p.add_argument("--max-attempts", type=int, default=120)
    p.add_argument("--init-offset", type=int, default=20,
                   help="first init-state index (eval uses 0..n_trials-1; "
                        "keep them disjoint)")
    p.add_argument("--raw-dir", default="data/teacher_raw")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--dagger-student-flags", default=None,
                   help="DAgger mode: libero_eval flags (one quoted string) "
                        "for the STUDENT policy. The student (mostly) drives, "
                        "the teacher labels every visited state; saved "
                        "`actions` are ALWAYS the teacher's labels.")
    p.add_argument("--dagger-beta", type=float, default=0.3,
                   help="probability of executing the TEACHER's action on a "
                        "given tick (recovery mixing); 0 = pure student roll")
    p.add_argument("--keep-failures", action="store_true",
                   help="save failed episodes too (default ON in DAgger mode: "
                        "off-distribution states with teacher labels are the "
                        "point; success-only for plain teacher recording)")
    args, teacher_flags = p.parse_known_args(argv)
    dagger = args.dagger_student_flags is not None
    keep_failures = args.keep_failures or dagger

    # The teacher is exactly the eval policy: reuse libero_eval's own arg
    # parser + factory so a winning config transfers flag-for-flag.
    from eval import libero_eval as LE
    tf = list(teacher_flags)
    if tf and tf[0] == "--":
        tf = tf[1:]
    targs = LE.parse_args(tf + ["--suite", args.suite])
    factory = LE._make_policy_factory(targs)
    policy = factory()

    student = None
    if dagger:
        import shlex
        sargs = LE.parse_args(shlex.split(args.dagger_student_flags)
                              + ["--suite", args.suite])
        student = LE._make_policy_factory(sargs)()

    tasks = LE._real_tasks(args.suite)
    task = tasks[args.task_id]
    out = Path(args.raw_dir)
    out.mkdir(parents=True, exist_ok=True)

    from eval._libero_compat import prepare_libero
    prepare_libero()
    from libero.libero.envs import OffScreenRenderEnv
    from microvla.utils.camera import upright
    from microvla.utils.proprio import proprio_from_obs

    camera = getattr(targs, "camera", "robot0_eye_in_hand_image")
    cam_base = camera[:-6] if camera.endswith("_image") else camera
    size = max(64, int(getattr(targs, "render_size", 256)))
    try:
        env = OffScreenRenderEnv(bddl_file_name=task.bddl_file,
                                 camera_heights=size, camera_widths=size,
                                 camera_names=[cam_base])
    except TypeError:
        env = OffScreenRenderEnv(bddl_file_name=task.bddl_file,
                                 camera_heights=size, camera_widths=size)

    n_ok = 0
    try:
        for attempt in range(args.max_attempts):
            if n_ok >= args.n_success:
                break
            idx = args.init_offset + attempt
            if hasattr(env, "seed"):
                env.seed(idx)
            obs = env.reset()
            if task.init_states is not None and len(task.init_states) > 0:
                obs = env.set_init_state(
                    task.init_states[idx % len(task.init_states)])
            policy.reset(task.instruction)
            if student is not None:
                student.reset(task.instruction)
            mix_rng = np.random.default_rng(10_000 + idx)

            frames, actions, proprios, executed = [], [], [], []
            success = False
            for _step in range(args.max_steps):
                frame = upright(obs[camera], camera)
                pro = proprio_from_obs(obs)
                # Teacher runs on EVERY tick so its phase machine tracks the
                # trajectory; its output is the label regardless of who drives.
                label = policy.act(frame, proprio=pro)
                label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
                if student is not None:
                    s_act = student.act(frame, proprio=pro)
                    s_act = np.nan_to_num(s_act, nan=0.0, posinf=0.0,
                                          neginf=0.0)
                    action = (label if mix_rng.random() < args.dagger_beta
                              else s_act)
                else:
                    action = label
                frames.append(np.asarray(frame, dtype=np.uint8))
                actions.append(np.asarray(label, dtype=np.float32))
                executed.append(np.asarray(action, dtype=np.float32))
                proprios.append(np.asarray(pro, dtype=np.float32))
                obs, _r, done, info = env.step(action)
                success = (bool(info.get("success", False))
                           if isinstance(info, dict) else False)
                if not success and hasattr(env, "check_success"):
                    success = bool(env.check_success())
                if done or success:
                    break
            logger.info("attempt %d (init %d): success=%s steps=%d",
                        attempt, idx, success, len(actions))
            if not success and not keep_failures:
                continue
            n_ok += 1
            np.savez_compressed(
                out / f"teacher_{args.suite}_t{args.task_id}_i{idx:04d}.npz",
                frames=np.stack(frames),
                actions=np.stack(actions),
                proprio=np.stack(proprios),
                executed_actions=np.stack(executed),
                success=np.array(success),
                dagger_beta=np.array(args.dagger_beta if dagger else 1.0),
                instruction=np.array(task.instruction),
                init_index=np.array(idx),
                camera=np.array(camera),
            )
            logger.info("saved episode %d/%d (success=%s)",
                        n_ok, args.n_success, success)
    finally:
        if hasattr(env, "close"):
            env.close()
    print(f"recorded {n_ok} successful episodes -> {out}")


# -------------------------------------------------------------------- convert
def _iter_raw(raw_dir: Path):
    from preprocess.common import SourceEpisode

    for f in sorted(raw_dir.glob("teacher_*.npz")):
        d = np.load(f, allow_pickle=False)
        frames = d["frames"]
        proprio = d["proprio"]
        yield SourceEpisode(
            frames=list(frames),
            actions=np.asarray(d["actions"], dtype=np.float32),
            instruction=str(d["instruction"]),
            source_hz=LIBERO_HZ,
            episode_id=f.stem,
            proprio_raw=np.asarray(proprio, dtype=np.float32),
            eef_pos_raw=np.asarray(proprio[:, :3], dtype=np.float32),
        )


def convert(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description="convert raw teacher rollouts to shards")
    p.add_argument("--raw-dir", default="data/teacher_raw")
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="eye_in_hand_rgb",
                   help="provenance record: which camera the frames came from")
    p.add_argument("--spatial-grid", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--role-disjoint-iou", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--purge-raw", action="store_true",
                   help="delete the raw dir after a successful conversion "
                        "(the shard pipeline's download->convert->delete rule)")
    args = p.parse_args(argv)

    from microvla.config import DEFAULT_CONFIG
    from preprocess.common import run_conversion

    cfg = DEFAULT_CONFIG
    import dataclasses
    if args.role_disjoint_iou is not None:
        cfg = dataclasses.replace(cfg, role_disjoint_iou=float(args.role_disjoint_iou))
    if args.det_conf is not None:
        cfg = dataclasses.replace(cfg, det_conf=float(args.det_conf))

    raw = Path(args.raw_dir)
    out = run_conversion(
        lambda: _iter_raw(raw),
        args.out,
        cfg=cfg,
        mock=args.dry_run,
        device=args.device,
        limit=args.limit,
        grid_size=args.spatial_grid,
        store_frames=True,
        provenance={"camera": args.camera, "deflip": True,
                    "source_hz": LIBERO_HZ, "teacher": "PhasedIBVS-handeye",
                    "eval_camera": {"eye_in_hand_rgb": "robot0_eye_in_hand_image",
                                    "agentview_rgb": "agentview_image"}[args.camera]},
    )
    if args.purge_raw:
        import shutil
        shutil.rmtree(raw)
        logger.info("purged raw dir %s", raw)
    print(f"converted -> {out}")


def main() -> None:
    from microvla.utils.signals import ignore_sigterm

    ignore_sigterm()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "convert"):
        print(__doc__)
        raise SystemExit("usage: teacher_rollouts.py {record,convert} ...")
    (record if sys.argv[1] == "record" else convert)(sys.argv[2:])


if __name__ == "__main__":
    main()
