"""Fit the EEF response gain that the waypoint controller inverts.

The v7.2 waypoint head predicts WHERE the end-effector should be; actuating
that prediction needs the one number the head cannot know — how far the arm
actually travels per unit of commanded action. For LIBERO's OSC controller
that is very close to a per-axis constant, so a least-squares fit over baked
episodes is enough::

    Δeef_j  ≈  gain_j · action_j          (per axis j, per control step)

Reads baked ``.npz`` episodes plus their ``norm_stats.json`` (actions are
stored NORMALIZED; the gain must be in raw action units, so they are inverted
first), and writes ``waypoint_stats.json`` next to them::

    python -m preprocess.fit_waypoint_gain data/libero_v7_full

Pair that file with the checkpoint exactly like ``norm_stats.json`` — a gain
fitted on one action normalization is meaningless under another.

Episodes without proprioception are zero-filled by the converter, so pairs are
kept only where the proprio validity flag is set; the printed pair count and
per-axis R² are the honest read on whether the fit means anything (LIBERO's
translation axes should land well above 0.5; a near-zero R² means that axis
does not respond linearly and the waypoint controller should not be trusted
on it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from preprocess.common import ActionNormalizer


def fit_gain(
    data_dirs: list[str | Path],
    norm_stats: str | Path | None = None,
    max_episodes: int | None = None,
) -> dict:
    """Least-squares per-axis gain over every valid ``(action, Δeef)`` pair.

    Args:
        data_dirs: Directories of baked ``.npz`` episodes.
        norm_stats: ``norm_stats.json`` path; defaults to the one inside the
            FIRST data dir.
        max_episodes: Optional cap (per directory) for a quick look.

    Returns:
        ``{"gain": [3], "r2": [3], "n": int}`` — the payload of
        ``waypoint_stats.json``.

    Raises:
        FileNotFoundError: If no episodes or no ``norm_stats.json`` are found.
        ValueError: If no episode carries valid proprioception.
    """
    dirs = [Path(d) for d in data_dirs]
    stats_path = Path(norm_stats) if norm_stats else dirs[0] / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"no action normalizer at {stats_path}; pass --norm-stats explicitly"
        )
    normalizer = ActionNormalizer.load(stats_path)

    acts: list[np.ndarray] = []
    disps: list[np.ndarray] = []
    n_eps = n_skipped = 0
    for d in dirs:
        files = sorted(d.glob("*.npz"))
        if max_episodes is not None:
            files = files[:max_episodes]
        if not files:
            raise FileNotFoundError(f"no .npz episodes in {d}")
        for f in files:
            with np.load(f) as ep:
                if "eef_pos_chunk" not in ep or "proprio" not in ep:
                    n_skipped += 1
                    continue
                chunk = np.asarray(ep["eef_pos_chunk"], dtype=np.float64)
                valid = np.asarray(ep["proprio"], dtype=np.float64)[:, -1] > 0.5
                pwm = np.asarray(ep["pwm_targets"], dtype=np.float64)
            if chunk.shape[1] < 2 or not valid.any():
                n_skipped += 1
                continue
            n_eps += 1
            # One pair per (timestep, intra-chunk step): the action commanded at
            # chunk row k moves the arm from chunk[k] to chunk[k+1].
            k = chunk.shape[1] - 1
            # Denormalize the FULL action vector — the normalizer's per-dim
            # stats are 7-wide — then take the translation dims.
            raw = normalizer.inverse(pwm[valid, :k, :])[..., :3]  # [n, k, 3]
            step = np.diff(chunk[valid], axis=1)[:, :k, :]        # [n, k, 3]
            acts.append(raw.reshape(-1, 3))
            disps.append(step.reshape(-1, 3))

    if not acts:
        raise ValueError(
            f"no episode in {[str(d) for d in dirs]} carries valid proprio + "
            f"eef_pos_chunk ({n_skipped} skipped) — re-bake with the v7 "
            f"converter or run preprocess/patch_proprio.py first."
        )
    A = np.concatenate(acts, axis=0)
    D = np.concatenate(disps, axis=0)
    # Through-the-origin least squares per axis: a zero command must mean no
    # motion (the same symmetry the action normalizer enforces), so no intercept.
    denom = (A * A).sum(axis=0)
    gain = np.where(denom > 1e-12, (A * D).sum(axis=0) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    resid = D - A * gain
    ss_tot = ((D - D.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - (resid**2).sum(axis=0) / np.where(ss_tot > 1e-12, ss_tot, 1.0), 0.0)
    return {"gain": gain.tolist(), "r2": r2.tolist(), "n": int(A.shape[0]),
            "episodes": n_eps, "skipped": n_skipped}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data_dir", nargs="+", help="baked episode directory(ies)")
    ap.add_argument("--norm-stats", default=None,
                    help="norm_stats.json (default: the first data dir's)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="cap episodes per directory (quick look)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <first data dir>/waypoint_stats.json)")
    args = ap.parse_args(argv)

    result = fit_gain(args.data_dir, args.norm_stats, args.episodes)
    out = Path(args.out) if args.out else Path(args.data_dir[0]) / "waypoint_stats.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"pairs: {result['n']:,} from {result['episodes']} episodes "
          f"({result['skipped']} skipped: no valid proprio)")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: gain {result['gain'][i]:+.5f} m/unit-action/step   "
              f"R2 {result['r2'][i]:.3f}")
    weak = [a for i, a in enumerate("xyz") if result["r2"][i] < 0.5]
    if weak:
        print(f"WARNING: axes {weak} fit poorly (R2 < 0.5) — the arm does not "
              f"respond linearly there; waypoint actuation will be unreliable "
              f"on them. Check the controller type before trusting this.")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
