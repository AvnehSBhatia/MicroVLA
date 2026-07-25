"""Patch baked LIBERO episodes with proprioception — NO YOLO re-bake.

The v6 planner conditions on arm state (see microvla/utils/proprio.py for the
diagnosis). The baked ``.npz`` episodes carry embeddings but no proprio, and
the raw hdf5s were deleted under the 10 GB budget. This pass re-acquires ONLY
the proprio arrays and merges them into the EXISTING npz files by episode
identity (npz filename == ``{hdf5_stem}__{demo}`` — the converter's
``episode_id``), leaving every embedding byte untouched.

Adds two keys per episode (sampled at the SAME ``subsample_indices`` the
converter used, so frame t and proprio t are the same instant):

    proprio        [T, 10]           (see utils/proprio.py layout; valid=1)
    eef_pos_chunk  [T, plan_steps, 3] absolute EEF xyz at the pwm chunk steps
                                      (future absolute-waypoint action head)

Two acquisition modes:

    # A. hdf5s already on disk (repeatable):
    python preprocess/patch_proprio.py --data-dir data/libero --hdf5 /path/to/suite_dir

    # B. streaming download -> patch -> delete, budget-guarded (reuses the
    #    shard pipeline's list-file format: one URL per line, # comments):
    python preprocess/patch_proprio.py --data-dir data/libero \
        --shards preprocess/shards_libero.txt --workdir /tmp/proprio_scratch

Episodes whose npz is missing (or whose sampled length mismatches) are
reported and skipped — never corrupted. Atomic per-file rewrites. Idempotent:
already-patched files are skipped unless --force.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microvla.config import DEFAULT_CONFIG
from microvla.utils.proprio import PROPRIO_DIM, build_proprio
from preprocess.common import chunk_actions, subsample_indices

#: LIBERO native control rate (matches preprocess/libero.py::LIBERO_HZ).
_LIBERO_HZ = 20.0

#: Candidate hdf5 obs keys (LIBERO spelling first, then raw robosuite).
_POS_KEYS = ("ee_pos", "robot0_eef_pos", "ee_states")
_ORI_KEYS = ("ee_ori", "robot0_eef_quat")
_GRIP_KEYS = ("gripper_states", "robot0_gripper_qpos")


def _obs_first(obs, keys):
    for k in keys:
        if k in obs:
            return np.asarray(obs[k])
    return None


def patch_from_hdf5(hdf5_root: Path, data_dir: Path, force: bool = False) -> dict:
    """Merges proprio from every demo under ``hdf5_root`` into ``data_dir`` npzs."""
    import h5py  # lazy heavy dep

    cfg = DEFAULT_CONFIG
    files = [hdf5_root] if hdf5_root.is_file() else sorted(hdf5_root.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no .hdf5 files under {hdf5_root}")

    stats = {"patched": 0, "skipped_done": 0, "no_npz": 0, "mismatch": 0, "no_proprio": 0}
    for h5path in files:
        with h5py.File(h5path, "r") as f:
            for demo in f["data"].keys():
                npz_path = data_dir / f"{h5path.stem}__{demo}.npz"
                if not npz_path.exists():
                    stats["no_npz"] += 1
                    continue
                with np.load(npz_path) as z:
                    existing = {k: z[k] for k in z.files}
                if "proprio" in existing and not force:
                    stats["skipped_done"] += 1
                    continue

                grp = f["data"][demo]
                obs = grp["obs"]
                pos = _obs_first(obs, _POS_KEYS)
                if pos is None:
                    stats["no_proprio"] += 1
                    continue
                pos = pos[:, :3]
                ori = _obs_first(obs, _ORI_KEYS)
                grip = _obs_first(obs, _GRIP_KEYS)
                n_actions = np.asarray(grp["actions"]).shape[0]
                # The converter truncated to min(frames, actions); pos length
                # tracks frames. Reproduce the same T before sampling.
                T_raw = min(pos.shape[0], n_actions)
                indices = subsample_indices(T_raw, _LIBERO_HZ, cfg.real_frame_hz)

                T_npz = existing["frame_embs"].shape[0]
                if len(indices) != T_npz:
                    stats["mismatch"] += 1
                    print(f"  ! {npz_path.name}: sampled T {len(indices)} != baked T {T_npz} — skipped")
                    continue

                proprio = np.stack([
                    build_proprio(
                        pos[i],
                        ori[i] if ori is not None else None,
                        grip[i] if grip is not None else None,
                    )
                    for i in indices
                ]).astype(np.float32)                                  # [T, 10]
                eef_pos_chunk = chunk_actions(
                    pos[:T_raw].astype(np.float32), indices, cfg.plan_steps
                )                                                       # [T, steps, 3]

                existing["proprio"] = proprio
                existing["eef_pos_chunk"] = eef_pos_chunk
                tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
                np.savez_compressed(tmp, **existing)
                os.replace(tmp, npz_path)
                stats["patched"] += 1
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True, help="dir of baked *.npz to patch")
    ap.add_argument("--hdf5", action="append", default=[],
                    help="hdf5 file or dir (repeatable) — mode A")
    ap.add_argument("--shards", default=None,
                    help="shard list file (URLs) — mode B: download->patch->delete")
    ap.add_argument("--workdir", default="./_proprio_scratch",
                    help="mode B scratch dir (deleted per shard)")
    ap.add_argument("--budget-gb", type=float, default=10.0,
                    help="mode B total disk budget (matches the shard pipeline)")
    ap.add_argument("--force", action="store_true", help="re-patch already-patched files")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    totals: dict[str, int] = {}

    def _acc(s: dict) -> None:
        for k, v in s.items():
            totals[k] = totals.get(k, 0) + v

    for h in args.hdf5:
        _acc(patch_from_hdf5(Path(h), data_dir, force=args.force))

    if args.shards:
        import shutil

        from preprocess.shard_pipeline import BudgetGuard, _download, _extract

        work = Path(args.workdir)
        work.mkdir(parents=True, exist_ok=True)
        guard = BudgetGuard(args.budget_gb, tracked=[work, data_dir])
        lines = [ln.strip() for ln in Path(args.shards).read_text().splitlines()]
        shards = [ln for ln in lines if ln and not ln.startswith("#")]
        for url in shards:
            name = url.rsplit("/", 1)[-1] or "shard"
            dest = work / name
            print(f"[shard] {url}")
            guard.ensure(4.0, f"download {name}")  # conservative headroom
            src = Path(url)
            if src.exists():
                dest = src
            else:
                _download(url, dest)
            extracted = _extract(dest, work / (name + ".x")) if dest.suffix not in (".hdf5", ".h5") else dest
            _acc(patch_from_hdf5(Path(extracted), data_dir, force=args.force))
            # delete transient state before the next shard (budget rule)
            if dest != src and dest.exists():
                dest.unlink()
            xdir = work / (name + ".x")
            if xdir.exists():
                shutil.rmtree(xdir)

    print(f"\ndone: {totals}")
    if totals.get("patched"):
        print("proprio merged. Retrain STAGE B ONLY (--load-stage-a) — the world "
              "model does not consume proprio and stays valid.")


if __name__ == "__main__":
    main()
