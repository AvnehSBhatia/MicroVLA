"""BridgeData V2 -> MicroVLA episode converter (main pretraining set).

BridgeData V2 (https://rail-berkeley.github.io/bridgedata/) is ~60k WidowX-250
teleop trajectories at 5 Hz. This converter targets the RAW release layout:

    bridgedata_raw/<domain>/<task>/<date>/raw/traj_group*/traj*/
        images0/im_0.jpg im_1.jpg ...     (480x640 RGB, camera 0)
        policy_out.pkl                    (per-step dicts with 'actions' [7])
        lang.txt                          (language annotation; may be absent)

Actions are (Δxyz, Δrpy, gripper) at 5 Hz — the same 7-DoF convention as
LIBERO and ``cfg.num_servos=7``. Trajectories without a language annotation
are skipped by default (MicroVLA is language-conditioned end to end); pass
``--fallback-task-lang`` to fall back to the task directory name
("put_carrot_in_pot" -> "put carrot in pot") instead of skipping.

Usage (nothing is downloaded by this script — point it at your local copy):

    python -m preprocess.bridge /path/to/bridgedata_raw ./data/bridge \\
        [--camera images0] [--limit N] [--dry-run] [--device cpu]
        [--teacher tinyvla --teacher-checkpoint ... --teacher-repo ... --teacher-cache ...]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from preprocess.common import SourceEpisode, run_conversion
from preprocess.teacher import build_teacher

logger = logging.getLogger(__name__)

#: BridgeData V2 control / camera rate.
BRIDGE_HZ = 5.0

_IM_INDEX_RE = re.compile(r"im_(\d+)\.jpg$")


def task_name_to_instruction(task_dir_name: str) -> str:
    """"put_carrot_in_pot_cardboardfence" -> "put carrot in pot cardboardfence".

    Crude fallback when ``lang.txt`` is missing; prefer real annotations.

    Args:
        task_dir_name: Name of the task-level directory.

    Returns:
        Space-separated, lowercased instruction string.
    """
    return task_dir_name.replace("_", " ").strip().lower()


def read_lang(traj_dir: Path) -> str | None:
    """Reads the first non-empty, non-comment line of ``lang.txt`` if present."""
    lang_file = traj_dir / "lang.txt"
    if not lang_file.exists():
        return None
    for line in lang_file.read_text(errors="ignore").splitlines():
        line = line.strip()
        # Bridge annotation files may carry confidence lines starting with
        # 'confidence:'; the instruction is the first plain line.
        if line and not line.lower().startswith("confidence"):
            return line.lower()
    return None


def read_actions(traj_dir: Path) -> np.ndarray | None:
    """Loads the ``[T, 7]`` action array from ``policy_out.pkl``.

    The pickle is a list of per-step dicts (key ``'actions'``) in most of the
    release; some shards store a plain array. Returns None when unreadable.
    """
    pkl = traj_dir / "policy_out.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError):
        logger.warning("unreadable policy_out.pkl in %s; skipping", traj_dir)
        return None
    if isinstance(data, list):
        rows = [np.asarray(step["actions"], dtype=np.float32) for step in data if "actions" in step]
        return np.stack(rows, axis=0) if rows else None
    return np.asarray(data, dtype=np.float32)


def _sorted_images(img_dir: Path) -> list[Path]:
    """im_10.jpg after im_2.jpg (numeric sort on the frame index)."""
    def key(p: Path) -> int:
        m = _IM_INDEX_RE.search(p.name)
        return int(m.group(1)) if m else 0

    return sorted(img_dir.glob("im_*.jpg"), key=key)


def iter_bridge_episodes(
    root: str | Path,
    camera: str = "images0",
    fallback_task_lang: bool = False,
) -> Iterator[SourceEpisode]:
    """Streams every raw-format trajectory under ``root``.

    Args:
        root: ``bridgedata_raw``-style directory tree (already downloaded).
        camera: Image subdirectory to read (``images0`` = primary camera).
        fallback_task_lang: Use the task directory name as the instruction
            when ``lang.txt`` is absent instead of skipping the trajectory.

    Yields:
        One :class:`SourceEpisode` per usable trajectory.

    Raises:
        FileNotFoundError: If no trajectories exist under ``root``.
    """
    import cv2  # lazy heavy dep (``pip install microvla[perception]``)

    root = Path(root)
    traj_dirs = sorted(p for p in root.rglob("traj*") if (p / camera).is_dir())
    if not traj_dirs:
        raise FileNotFoundError(f"no traj*/{camera} directories under {root}")

    skipped_lang = 0
    for traj_dir in traj_dirs:
        instruction = read_lang(traj_dir)
        if instruction is None:
            if not fallback_task_lang:
                skipped_lang += 1
                continue
            # .../<task>/<date>/raw/traj_group*/traj* -> <task>
            try:
                task_name = traj_dir.parents[3].name
            except IndexError:
                task_name = traj_dir.parent.name
            instruction = task_name_to_instruction(task_name)

        actions = read_actions(traj_dir)
        if actions is None or actions.ndim != 2 or actions.shape[1] != 7:
            continue

        image_paths = _sorted_images(traj_dir / camera)
        if not image_paths:
            continue
        frames = []
        for p in image_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                break
            frames.append(np.ascontiguousarray(bgr[..., ::-1]))  # -> RGB
        T = min(len(frames), len(actions))
        if T < 2:
            continue

        rel_id = "__".join(traj_dir.relative_to(root).parts)
        yield SourceEpisode(
            frames=frames[:T],
            actions=actions[:T],
            instruction=instruction,
            source_hz=BRIDGE_HZ,
            episode_id=re.sub(r"[^A-Za-z0-9_.-]", "_", rel_id),
        )

    if skipped_lang:
        logger.info("skipped %d trajectories without lang.txt "
                    "(pass --fallback-task-lang to keep them)", skipped_lang)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="bridgedata_raw directory (already downloaded)")
    parser.add_argument("out", help="output directory for MicroVLA .npz episodes")
    parser.add_argument("--camera", default="images0")
    parser.add_argument("--fallback-task-lang", action="store_true")
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
        lambda: iter_bridge_episodes(args.root, camera=args.camera,
                                     fallback_task_lang=args.fallback_task_lang),
        args.out,
        mock=args.dry_run,
        device=args.device,
        limit=args.limit,
        teacher=teacher,
    )


if __name__ == "__main__":
    main()
