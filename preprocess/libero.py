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
    files = sorted(root.rglob("*.hdf5"))
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
                yield SourceEpisode(
                    frames=list(frames[:T]),
                    actions=actions[:T],
                    instruction=instruction,
                    source_hz=LIBERO_HZ,
                    episode_id=f"{h5path.stem}__{demo}",
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="directory with LIBERO *.hdf5 files (already downloaded)")
    parser.add_argument("out", help="output directory for MicroVLA .npz episodes")
    parser.add_argument("--camera", default="agentview_rgb")
    parser.add_argument("--no-rotate-180", dest="rotate", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="max episodes")
    parser.add_argument("--dry-run", action="store_true", help="mock perception (no weights)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teacher", choices=["mock", "tinyvla"], default=None,
                        help="relabel actions with a distillation teacher")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-repo", default=None)
    parser.add_argument("--teacher-cache", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    teacher = build_teacher(args.teacher, args.teacher_checkpoint, args.teacher_repo,
                            args.teacher_cache, device=args.device)
    run_conversion(
        lambda: iter_libero_episodes(args.root, camera=args.camera, rotate_180=args.rotate),
        args.out,
        mock=args.dry_run,
        device=args.device,
        limit=args.limit,
        teacher=teacher,
    )


if __name__ == "__main__":
    main()
