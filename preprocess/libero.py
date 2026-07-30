"""LIBERO -> MicroVLA episode converter (fine-tune + eval set).

LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO) ships task suites
(libero_spatial / libero_object / libero_goal / libero_90 / libero_10) as HDF5
files, one file per task, ~50 human demos per file:

    data/
      demo_0/
        obs/agentview_rgb      [T, H, W, 3] uint8 RGB
        obs/eye_in_hand_rgb    [T, H, W, 3] uint8 RGB
        actions                [T, 7] float  (Δxyz, Δrpy, gripper), ~20 Hz
      demo_1/ ...

Known quirk handled here: robosuite renders through an OpenGL framebuffer whose
origin is bottom-left, so BOTH camera streams arrive row-reversed. Putting them
upright is a row flip and nothing else — see ``microvla/utils/camera.py``, which
holds that convention for the bake and the deployment path alike. Historically
this file rotated agentview a full 180° (row flip plus a spurious left-right
mirror) and left the wrist stream untouched; ``--no-deflip`` reproduces the
"untouched" half for a deliberate ablation, nothing reproduces the mirror.

Instructions come from ``data.attrs['problem_info']`` (JSON with
``language_instruction``) when present, else are reconstructed from the
filename (``..._SCENE1_put_the_bowl_on_the_plate_demo.hdf5`` ->
"put the bowl on the plate").

Usage (nothing is downloaded by this script — point it at your local copy):

    python -m preprocess.libero /path/to/libero_object ./data/libero_object \\
        [--camera agentview_rgb] [--limit N] [--dry-run] [--device cpu]
        [--teacher tinyvla --teacher-checkpoint ... --teacher-repo ... --teacher-cache ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from microvla.config import DEFAULT_CONFIG
from microvla.utils.camera import upright
from preprocess.common import SourceEpisode, run_conversion
from preprocess.teacher import build_teacher
from microvla.utils.signals import ignore_sigterm

logger = logging.getLogger(__name__)

#: LIBERO demos are recorded at the robosuite control rate.
LIBERO_HZ = 20.0

_SCENE_PREFIX_RE = re.compile(r"^[A-Z0-9_]+?_SCENE\d+_")
_DEMO_SUFFIX_RE = re.compile(r"_demo$")


def instruction_from_filename(stem: str) -> str:
    """Recovers the language instruction from a LIBERO hdf5 filename.

    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo" ->
    "put the black bowl on the plate".

    Args:
        stem: Filename without extension.

    Returns:
        Best-effort instruction string.
    """
    s = _SCENE_PREFIX_RE.sub("", stem)
    s = _DEMO_SUFFIX_RE.sub("", s)
    return s.replace("_", " ").strip().lower()


def _instruction_for_file(f, stem: str) -> str:
    """Instruction from problem_info attrs, falling back to the filename."""
    try:
        info = json.loads(f["data"].attrs["problem_info"])
        lang = str(info.get("language_instruction", "")).strip().strip('"')
        if lang:
            return lang.lower()
    except (KeyError, ValueError, TypeError):
        pass
    return instruction_from_filename(stem)


def _demo_sort_key(name: str) -> int:
    """demo_10 must sort after demo_2 (numeric, not lexicographic)."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def iter_libero_episodes(
    root: str | Path,
    camera: str = "agentview_rgb",
    detect_camera: str | None = None,
    deflip: bool = True,
) -> Iterator[SourceEpisode]:
    """Streams every demo of every LIBERO hdf5 under ``root``.

    Args:
        root: Directory containing (possibly nested) LIBERO ``*.hdf5`` files.
        camera: Observation key to use as the video stream.
        deflip: Put frames upright via :func:`microvla.utils.camera.upright`.
            BOTH LIBERO streams need it — they share one OpenGL framebuffer
            whose origin is bottom-left — and the deployment path applies the
            same function to the live observation, so this must stay on unless
            you are deliberately baking a mis-oriented control corpus.

            This replaces a ``rotate_180`` flag that applied ``[:, ::-1, ::-1]``
            to agentview: the correct row flip PLUS a spurious left-right
            mirror, which cost source detection duty 0.850 → 0.613 and mirrored
            every baked box center with respect to the action frame. See
            ``microvla/utils/camera.py``.

    Yields:
        One :class:`SourceEpisode` per demo (frames RGB, actions ``[T, 7]``).

    Raises:
        FileNotFoundError: If no hdf5 files exist under ``root``.
    """
    import h5py  # lazy heavy dep (``pip install microvla[data]``)

    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no .hdf5 files under {root}")

    for h5path in files:
        with h5py.File(h5path, "r") as f:
            instruction = _instruction_for_file(f, h5path.stem)
            for demo in sorted(f["data"].keys(), key=_demo_sort_key):
                grp = f["data"][demo]
                frames = np.asarray(grp["obs"][camera])  # [T, H, W, 3] RGB
                det_frames = None
                if detect_camera and detect_camera != camera:
                    det_frames = np.asarray(grp["obs"][detect_camera])
                    if deflip:
                        det_frames = upright(det_frames, detect_camera)
                if deflip:
                    frames = upright(frames, camera)
                actions = np.asarray(grp["actions"], dtype=np.float32)  # [T, 7]
                T = min(len(frames), len(actions))

                # v7: raw robot state per native step (proprio + absolute EEF).
                from microvla.utils.proprio import build_proprio

                obs = grp["obs"]

                def _first(keys):
                    for k in keys:
                        if k in obs:
                            return np.asarray(obs[k])
                    return None

                pos = _first(("ee_pos", "robot0_eef_pos", "ee_states"))
                proprio_raw = eef_pos_raw = None
                if pos is not None:
                    pos = pos[:T, :3]
                    ori = _first(("ee_ori", "robot0_eef_quat"))
                    grip = _first(("gripper_states", "robot0_gripper_qpos"))
                    proprio_raw = np.stack([
                        build_proprio(pos[i],
                                      ori[i] if ori is not None else None,
                                      grip[i] if grip is not None else None)
                        for i in range(min(T, len(pos)))
                    ])
                    eef_pos_raw = pos.astype(np.float32)

                yield SourceEpisode(
                    frames=list(frames[:T]),
                    actions=actions[:T],
                    instruction=instruction,
                    source_hz=LIBERO_HZ,
                    episode_id=f"{h5path.stem}__{demo}",
                    proprio_raw=proprio_raw,
                    eef_pos_raw=eef_pos_raw,
                    detect_frames=None if det_frames is None else list(det_frames[:T]),
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="directory with LIBERO *.hdf5 files (already downloaded)")
    parser.add_argument("out", help="output directory for MicroVLA .npz episodes")
    # REQUIRED, no default. A silent default cost a full bake + two stage-B
    # trainings + a closed-loop eval: the corpus was built from the
    # third-person `agentview_rgb` (rotated 180°) while eval/libero_eval.py
    # reads the WRIST camera `robot0_eye_in_hand_image`, so every visual
    # feature the policy learned was from a viewpoint it never sees at
    # deployment. The npz key is called `wrist_frames` regardless of this
    # flag, which is what made the mismatch invisible.
    parser.add_argument("--detect-camera", default=None,
                        choices=["eye_in_hand_rgb", "agentview_rgb"],
                        help="run OBJECT DETECTION on this view instead of --camera. "
                             "The wrist view supplies 0.68 proposals/frame with 47%% "
                             "of frames empty (paper.md 4r); the third-person view of "
                             "the same scenes yields 3.40. --camera still drives the "
                             "frame embedding the world model predicts, so the "
                             "train/eval ego-view coupling 4f requires is unchanged — "
                             "only the detector's pixels move. agentview is "
                             "de-rotated automatically.")
    parser.add_argument("--camera", required=True,
                        choices=["eye_in_hand_rgb", "agentview_rgb"],
                        help="LIBERO obs key to bake. Use eye_in_hand_rgb to match "
                             "eval/libero_eval.py's robot0_eye_in_hand_image. "
                             "agentview_rgb is a THIRD-PERSON view and needs "
                             "--rotate-180; the wrist view must NOT be rotated.")
    parser.add_argument("--no-deflip", "--no-rotate-180", dest="deflip",
                        action="store_false",
                        help="bake frames as robosuite rendered them, i.e. UPSIDE "
                             "DOWN. Both LIBERO streams come out of a bottom-left-"
                             "origin framebuffer, so the default (on) is correct for "
                             "both cameras and you normally never pass this. Kept "
                             "only so the shipped-corpus orientation stays "
                             "reproducible as an ablation.")
    parser.add_argument("--deflip", "--rotate-180", dest="deflip",
                        action="store_true",
                        help="row-flip frames upright (the default).")
    parser.set_defaults(deflip=True)
    parser.add_argument("--det-conf", type=float, default=None,
                        help="OVERRIDE cfg.det_conf (%(default)s = use the config "
                             "value, currently the one eval/policy.py also reads). "
                             "The bake used to take the detector class default "
                             "0.10 while eval passed 0.02 -- defect 26. Whatever "
                             "you set is recorded in manifest.json provenance, and "
                             "the robot must be run with the same number.")
    parser.add_argument("--frame-hz", type=float, default=None,
                        help="OVERRIDE cfg.real_frame_hz for this bake, i.e. the "
                             "sampling rate. The default 2 Hz keeps 1 frame in 10 "
                             "of a 20 Hz LIBERO demo, so a 150-step demo becomes 15 "
                             "supervised decision points and the corpus holds ~7.7k "
                             "for 500 episodes -- an order of magnitude below what "
                             "LIBERO BC baselines train on. That rate was chosen for "
                             "the PERCEPTION budget and then inherited, unexamined, "
                             "by the ACTION supervision. Pass 20 to supervise every "
                             "control step (paper.md 4w).")
    parser.add_argument("--spatial-grid", type=int, default=0,
                        help="bake a GxG coarse spatial map per frame (0 = off). "
                             "GAP throws away WHERE, and on a wrist camera WHERE is "
                             "the servo error; the two role boxes were the only "
                             "spatial channel and yield 0.68 proposals/frame, so "
                             "roughly half of frames carried none. 4 is a good "
                             "default (16 tokens x vis_dim per frame).")
    parser.add_argument("--limit", type=int, default=None, help="max episodes")
    parser.add_argument("--dry-run", action="store_true", help="mock perception (no weights)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-frames", action="store_true",
                        help="skip baking wrist_frames (v7 default bakes them; "
                             "they make the TQSA/perception trainable)")
    parser.add_argument("--teacher", choices=["mock", "tinyvla"], default=None,
                        help="relabel actions with a distillation teacher")
    parser.add_argument("--teacher-checkpoint", default=None,
                        help="TRAINED TinyVLA output dir (after scripts/process_ckpts.sh) — "
                             "NOT the HF Llava-Pythia base VLM")
    parser.add_argument("--teacher-base", default=None,
                        help="base VLM dir/HF id the VLA was trained from (e.g. a local "
                             "clone of lesjie/Llava-Pythia-400M); required for LoRA ckpts")
    parser.add_argument("--teacher-stats", default=None,
                        help="dataset_stats.pkl used to denormalize teacher actions "
                             "(defaults to <checkpoint>/dataset_stats.pkl)")
    parser.add_argument("--teacher-repo", default=None)
    parser.add_argument("--teacher-cache", default=None)
    args = parser.parse_args(argv)
    ignore_sigterm()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.detect_camera and args.detect_camera != args.camera:
        # The flag's documented contract is "--camera still drives the frame
        # embedding; only the detector's pixels move". EpisodeBuilder does not
        # honour it: preprocess/common.py makes exactly ONE perceive() call per
        # frame, on the DETECT view, and frame_embs/spatial_grid/box embs/obj_*
        # all come from that call. So a two-view bake silently trains the world
        # model on a viewpoint the robot never sees -- defect 25 again, from the
        # tool built to avoid it.
        #
        # It is also unusable downstream even if fixed here: eval/libero_eval.py
        # renders a single camera (`camera_names=[cam_base]`), so a two-view
        # corpus's boxes have no deployment counterpart. And manifest.json's
        # `eval_camera` is derived from --camera, the view that never reached
        # the encoder, so the provenance guard would certify the swap CLEAN.
        raise SystemExit(
            "--detect-camera is disabled: preprocess/common.py runs the frozen "
            "encoder ONLY on the detect view, so --camera would not drive the "
            "frame embedding as documented, and eval/libero_eval.py renders one "
            "camera anyway. Bake a single view (--camera agentview_rgb grounds "
            "the source role on 85% of libero_object frames against the wrist's "
            "22%; see paper.md 5n).")

    logger.info("baking camera=%s deflip=%s", args.camera, args.deflip)
    if not args.deflip:
        logger.warning(
            "--no-deflip: frames stay as robosuite rendered them (upside down). "
            "eval/libero_eval.py puts the live observation upright, so this "
            "bakes a train/deploy orientation mismatch on purpose.")
    logger.info(
        "eval/libero_eval.py must be run with --camera %s to match this bake; "
        "the corpus records the choice in manifest.json.",
        {"eye_in_hand_rgb": "robot0_eye_in_hand_image",
         "agentview_rgb": "agentview_image"}[args.camera])
    teacher = build_teacher(args.teacher, args.teacher_checkpoint, args.teacher_repo,
                            args.teacher_cache, device=args.device,
                            model_base=args.teacher_base, stats_path=args.teacher_stats)
    cfg = DEFAULT_CONFIG
    if args.det_conf is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, det_conf=float(args.det_conf))
        logger.info("detector threshold %.3f (cfg default %.3f)",
                    args.det_conf, DEFAULT_CONFIG.det_conf)
    if args.frame_hz is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, real_frame_hz=float(args.frame_hz))
        logger.info("sampling at %.1f Hz (stride %d on a 20 Hz demo) -- %.0fx the "
                    "default supervision density", args.frame_hz,
                    max(1, round(20.0 / args.frame_hz)), args.frame_hz / 2.0)
    run_conversion(
        lambda: iter_libero_episodes(args.root, camera=args.camera,
                                     detect_camera=args.detect_camera,
                                     deflip=args.deflip),
        args.out,
        cfg=cfg,
        provenance={"camera": args.camera, "detect_camera": args.detect_camera,
                    "deflip": bool(args.deflip), "source_hz": LIBERO_HZ,
                    "eval_camera": {"eye_in_hand_rgb": "robot0_eye_in_hand_image",
                                    "agentview_rgb": "agentview_image"}[args.camera]},
        grid_size=args.spatial_grid,
        mock=args.dry_run,
        device=args.device,
        limit=args.limit,
        teacher=teacher,
        store_frames=not args.no_frames,  # v7: frames make perception trainable
    )


if __name__ == "__main__":
    main()
