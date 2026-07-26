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

Known quirk handled here: robosuite renders the agentview camera upside down —
frames are rotated 180° by default (``--no-rotate-180`` to disable), matching
what OpenVLA/Octo do for LIBERO.

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

from preprocess.common import SourceEpisode, run_conversion
from preprocess.teacher import build_teacher

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
    rotate_180: bool = True,
) -> Iterator[SourceEpisode]:
    """Streams every demo of every LIBERO hdf5 under ``root``.

    Args:
        root: Directory containing (possibly nested) LIBERO ``*.hdf5`` files.
        camera: Observation key to use as the video stream.
        rotate_180: Rotate frames 180° (robosuite renders agentview flipped).

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
                if rotate_180:
                    frames = frames[:, ::-1, ::-1]
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
    parser.add_argument("--camera", required=True,
                        choices=["eye_in_hand_rgb", "agentview_rgb"],
                        help="LIBERO obs key to bake. Use eye_in_hand_rgb to match "
                             "eval/libero_eval.py's robot0_eye_in_hand_image. "
                             "agentview_rgb is a THIRD-PERSON view and needs "
                             "--rotate-180; the wrist view must NOT be rotated.")
    parser.add_argument("--no-rotate-180", dest="rotate", action="store_false",
                        help="robosuite renders AGENTVIEW upside down; the wrist view is "
                             "already upright. Defaults per --camera, so you normally "
                             "never pass this.")
    parser.add_argument("--rotate-180", dest="rotate", action="store_true",
                        help="force the 180° flip (implied by --camera agentview_rgb).")
    parser.set_defaults(rotate=None)
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Rotation follows the camera unless overridden: robosuite renders agentview
    # flipped, the wrist view upright. Getting this wrong is as damaging as the
    # camera itself and just as silent.
    if args.rotate is None:
        args.rotate = args.camera == "agentview_rgb"
    logger.info("baking camera=%s rotate_180=%s", args.camera, args.rotate)
    if args.camera != "eye_in_hand_rgb":
        logger.warning(
            "camera=%s is NOT the view eval/libero_eval.py reads "
            "(robot0_eye_in_hand_image). A policy trained on this corpus will "
            "see a viewpoint it never encounters at deployment.", args.camera)
    teacher = build_teacher(args.teacher, args.teacher_checkpoint, args.teacher_repo,
                            args.teacher_cache, device=args.device,
                            model_base=args.teacher_base, stats_path=args.teacher_stats)
    run_conversion(
        lambda: iter_libero_episodes(args.root, camera=args.camera, rotate_180=args.rotate),
        args.out,
        mock=args.dry_run,
        device=args.device,
        limit=args.limit,
        teacher=teacher,
        store_frames=not args.no_frames,  # v7: frames make perception trainable
    )


if __name__ == "__main__":
    main()
