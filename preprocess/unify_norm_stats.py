"""Put separately-baked dataset dirs on ONE shared action normalization.

Why this exists (disk budget). The LIBERO converter globs its input root
recursively, so baking three suites under one ``norm_stats.json`` means having
all three raw HDF5 suites on disk at once — roughly 10-12 GB, which is the
entire project budget (CLAUDE.md: 10 GB total, ever, including transient
download state). The disk-safe route is download -> bake -> delete, ONE suite
at a time. That leaves each suite with its own normalizer, fitted to its own
action quantiles.

For training that is already fine (the trainer reads baked, already-normalized
``pwm_targets`` and never loads ``norm_stats.json`` — that is how bridge and
libero have always been mixed). It is NOT fine at EVAL: the policy emits one
normalized action and ``ActionNormalizer.inverse`` maps it back through ONE
stats file, so if the suites disagree on scale, every dim carries a systematic
gain error against whichever suite you happened to pass.

This pass fixes that with pure arithmetic — no re-bake, no YOLO, embeddings
untouched. Given symmetric per-dir scales ``s_i`` (``q_high``, with
``q_low = -s_i``), it takes the widest per dim::

    S = max_i s_i         and rewrites   pwm_new = pwm_old * (s_i / S)

so every dir now means the same thing by the same numbers, and one shared
``norm_stats.json`` denormalizes all of them. Taking the MAX (never the mean)
guarantees no episode's actions are clipped by the change.

    python -m preprocess.unify_norm_stats \\
        --data-dir data/libero_object_v7 --data-dir data/libero_spatial_v7 \\
        --data-dir data/libero_goal_v7

Idempotent: a second run finds every ``s_i`` already equal to ``S`` and does
nothing. Per-file atomic (tmp + ``os.replace``), so peak disk overhead is one
episode. Inputs MUST already be symmetric (0 <=> zero motion) — run
``preprocess/renorm_symmetric.py`` first if a dir predates that, since mixing
an offset normalizer into a shared scale would silently bake in the
drift-into-wall bug this project already paid for once.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from preprocess.common import ActionNormalizer


def _symmetric_scale(stats_path: Path) -> np.ndarray:
    """Returns a dir's per-dim scale ``s``, refusing anything not symmetric."""
    stats = ActionNormalizer.load(stats_path)
    s = np.asarray(stats.q_high, dtype=np.float64)
    if not np.allclose(stats.q_low, -s):
        raise ValueError(
            f"{stats_path} is NOT symmetric (q_low != -q_high). Unifying an "
            f"offset normalizer would re-introduce the neutral-action drift "
            f"bug. Run `python preprocess/renorm_symmetric.py --data-dir "
            f"{stats_path.parent}` first."
        )
    return np.where(np.abs(s) > 1e-8, s, 1.0)


def unify(data_dirs: list[str | Path]) -> dict:
    """Rescales every dir's ``pwm_targets`` onto one shared symmetric scale.

    Args:
        data_dirs: Baked episode directories, each with a symmetric
            ``norm_stats.json``.

    Returns:
        ``{"scale": [num_servos], "dirs": {dir: n_rewritten}}``.

    Raises:
        FileNotFoundError: If a directory has no ``norm_stats.json`` or no
            episodes.
        ValueError: If a directory's normalizer is not symmetric, or the dirs
            disagree on the action dimension.
    """
    dirs = [Path(d) for d in data_dirs]
    scales = {}
    for d in dirs:
        stats_path = d / "norm_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"{stats_path} not found — is this a baked dataset dir?")
        scales[d] = _symmetric_scale(stats_path)
    widths = {len(s) for s in scales.values()}
    if len(widths) != 1:
        raise ValueError(f"action dims disagree across dirs: "
                         f"{ {str(d): len(s) for d, s in scales.items()} }")

    shared = np.max(np.stack(list(scales.values())), axis=0)
    print(f"shared symmetric scale S = {np.round(shared, 4).tolist()}")

    rewritten: dict[str, int] = {}
    for d in dirs:
        ratio = scales[d] / shared           # <= 1 by construction: never clips
        files = []
        for f in sorted(d.glob("*.npz")):
            if f.name.endswith(".tmp.npz"):
                f.unlink()                    # leftover from a crashed run
                continue
            files.append(f)
        if not files:
            raise FileNotFoundError(f"no .npz episodes in {d}")

        changed = 0
        if not np.allclose(ratio, 1.0):
            for f in files:
                with np.load(f) as z:
                    data = {k: z[k] for k in z.files}
                new = (data["pwm_targets"].astype(np.float64) * ratio).astype(np.float32)
                if np.allclose(new, data["pwm_targets"], atol=1e-7):
                    continue
                # np.savez appends ".npz" to names lacking it — keep the tmp
                # name ending in .npz so os.replace moves the file we wrote.
                tmp = f.with_name(f.stem + ".tmp.npz")
                np.savez_compressed(tmp, **data | {"pwm_targets": new})
                os.replace(tmp, f)
                changed += 1
        ActionNormalizer(q_low=-shared, q_high=shared).save(d / "norm_stats.json")
        rewritten[str(d)] = changed
        print(f"[{d}] scale {np.round(scales[d], 4).tolist()} -> shared; "
              f"rewrote {changed}/{len(files)} episodes")

    return {"scale": shared.tolist(), "dirs": rewritten}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", action="append", required=True,
                    help="baked episode dir (repeatable; needs >= 2 to be useful)")
    args = ap.parse_args(argv)
    unify(args.data_dir)
    print("done — every dir now shares one norm_stats.json; pair ANY of them "
          "with the checkpoint trained on the set.")


if __name__ == "__main__":
    main()
