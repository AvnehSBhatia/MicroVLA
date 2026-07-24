"""Re-normalize baked pwm_targets to a SYMMETRIC (zero-centered) action space.

Why this exists (v5 action-interface redesign): the original quantile min-max
normalization maps raw ``q01 -> -1`` and ``q99 -> +1``, so a NEUTRAL normalized
action (0) denormalizes to the range midpoint ``(q01+q99)/2`` — which for the
LIBERO/Bridge delta actions is a nonzero constant push (+dx, +dy, +yaw). A
policy that hedges toward neutral therefore commands a persistent drift and
sails into a wall (observed; root-caused via eval/replay_probe).

This pass rewrites every episode's ``pwm_targets`` (pure arithmetic — no YOLO
re-bake, embeddings untouched) so that **0 <=> zero motion**:

    raw   = inverse_old(pwm)                     # back to env units
    new   = clip(raw / s, -1, 1),  s = max(|q01|, |q99|) per dim

and writes a symmetric ``norm_stats.json`` (``q_low = -s, q_high = +s``), so
the UNCHANGED ``ActionNormalizer.inverse`` maps 0 -> 0 by construction — no
special eval flags needed. The old stats are backed up alongside as
``norm_stats.asymmetric.json`` (first run only). Idempotent: a second run is a
no-op (modulo the original clipping).

    python preprocess/renorm_symmetric.py --data-dir data/libero --data-dir data/bridge

Checkpoints trained on the OLD normalization pair with the OLD stats — retrain
after renorming (stage A is ~10 min on the box; the action-token input
distribution shifts too, so a full A+B retrain is the honest path).

Disk budget: rewrites are per-file atomic (tmp + os.replace), compressed npz,
peak overhead = one episode file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.common import ActionNormalizer


def renorm_dir(data_dir: Path) -> None:
    """Symmetrically re-normalizes every episode in one dataset directory."""
    stats_path = data_dir / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found — is this a baked dataset dir?")
    old = ActionNormalizer.load(stats_path)

    s = np.maximum(np.abs(old.q_low), np.abs(old.q_high))
    s = np.where(s > 1e-8, s, 1.0)
    already_symmetric = np.allclose(old.q_low, -s) and np.allclose(old.q_high, s)
    print(f"[{data_dir}] per-dim symmetric scale s = {np.round(s, 4).tolist()}")
    if already_symmetric:
        print(f"[{data_dir}] stats already symmetric — verifying episodes anyway.")

    # Skip stale .tmp.npz leftovers from a crashed prior run (and clean them).
    files = []
    for f in sorted(data_dir.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            f.unlink()
            continue
        files.append(f)
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {data_dir}")

    changed = 0
    for f in files:
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        pwm = data["pwm_targets"].astype(np.float64)        # [T, steps, servos]
        raw = old.inverse(pwm).astype(np.float64)           # env units
        new = np.clip(raw / s, -1.0, 1.0).astype(np.float32)
        if np.allclose(new, data["pwm_targets"], atol=1e-6):
            continue
        data["pwm_targets"] = new
        # NOTE: np.savez appends ".npz" to names that lack it — keep the tmp
        # name ending in .npz so the write lands where we os.replace from.
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, **data)
        os.replace(tmp, f)                                   # atomic per file
        changed += 1

    backup = data_dir / "norm_stats.asymmetric.json"
    if not backup.exists() and not already_symmetric:
        old.save(backup)
    ActionNormalizer(q_low=-s, q_high=s).save(stats_path)
    print(f"[{data_dir}] rewrote {changed}/{len(files)} episodes; "
          f"norm_stats.json is now symmetric (0 <=> zero motion); "
          f"old stats saved to {backup.name}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", action="append", required=True,
                    help="baked dataset dir containing *.npz + norm_stats.json "
                         "(repeatable)")
    args = ap.parse_args(argv)
    for d in args.data_dir:
        renorm_dir(Path(d))
    print("done. RETRAIN before evaluating — old checkpoints pair with the old stats.")


if __name__ == "__main__":
    main()
