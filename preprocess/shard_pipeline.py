"""Budget-guarded shard pipeline: download -> convert -> delete, under a disk cap.

Built for a hard constraint: **total disk usage must never exceed a fixed
budget (default 10 GB), including transient download/extraction state.** Full
BridgeData V2 raw is ~1-2 TB — it must never exist on disk at once. Instead,
this pipeline streams SHARDS (per-domain archives or directories):

    for each shard:
        1. BudgetGuard: refuse the shard unless (current usage + download +
           extraction headroom) fits the budget
        2. download (curl) into the scratch workdir
        3. extract, delete the archive immediately
        4. convert with the frozen perception stack -> tiny .npz episodes
           (UNNORMALIZED action chunks + a running action-stats sample)
        5. delete the raw shard
    finalize:
        fit the GLOBAL quantile ActionNormalizer on the accumulated samples,
        rewrite every episode's pwm_targets in place, write norm_stats.json
        and a merged manifest.json

Why normalization is deferred: converting shard-by-shard with per-shard stats
would give every shard a different action scaling — silently corrupting
training. Raw chunks are tiny, so the finalize rewrite costs seconds.

Shards are listed in a plain text file (one URL or local path per line, ``#``
comments allowed). For BridgeData V2 raw, list the per-domain archive URLs
from https://rail-berkeley.github.io/bridgedata/ — pick shards individually
smaller than roughly half your remaining budget (download + extraction
coexist briefly). For LIBERO, the per-suite zips (a few GB each) fit easily.

Usage:

    python -m preprocess.shard_pipeline shards.txt data/bridge --dataset bridge \\
        --budget-gb 10 [--workdir .shard_tmp] [--device mps] [--dry-run] \\
        [--limit-per-shard N] [--teacher ... etc.]

Episodes are written with ``np.savez_compressed`` (see train/dataset.py), and
the budget guard counts the output directory, the workdir, and any teacher
cache — the sum stays under the cap or the pipeline stops with a clear error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from preprocess.common import ActionNormalizer, EpisodeBuilder, SourceEpisode, chunk_actions
from preprocess.teacher import build_teacher

logger = logging.getLogger(__name__)

_GB = 1024**3
#: Max raw action rows kept for quantile fitting (uniform reservoir; ~60 MB).
_STATS_RESERVOIR = 1_000_000


def dir_size_gb(path: str | Path) -> float:
    """Recursive on-disk size of a directory in GB (0 if absent)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / _GB


class BudgetGuard:
    """Enforces the hard disk cap across every directory this pipeline touches.

    Args:
        budget_gb: The cap. NOTHING tracked may push the sum past it.
        tracked: Directories whose combined size counts against the budget
            (output episodes, scratch workdir, teacher cache).
    """

    def __init__(self, budget_gb: float, tracked: list[Path]) -> None:
        self.budget_gb = budget_gb
        self.tracked = [Path(t) for t in tracked]

    def used_gb(self) -> float:
        return sum(dir_size_gb(t) for t in self.tracked)

    def ensure(self, extra_gb: float, what: str) -> None:
        """Raises before an operation that would exceed the budget.

        Args:
            extra_gb: Estimated additional disk the operation needs at peak.
            what: Human-readable description for the error message.

        Raises:
            RuntimeError: If ``used + extra`` would exceed the budget.
        """
        used = self.used_gb()
        if used + extra_gb > self.budget_gb:
            raise RuntimeError(
                f"disk budget: {what} needs ~{extra_gb:.2f} GB but only "
                f"{self.budget_gb - used:.2f} GB of the {self.budget_gb:.0f} GB "
                f"budget remain (used {used:.2f} GB). Free space (delete "
                "converted shards you no longer need, shrink the teacher "
                "cache) or choose a smaller shard."
            )


class _IdentityNormalizer:
    """Pass-through stand-in used during shard conversion (no clipping)."""

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        return np.asarray(actions, dtype=np.float32)


class _StatsReservoir:
    """Uniform reservoir sample of raw action rows across every shard."""

    def __init__(self, capacity: int = _STATS_RESERVOIR, seed: int = 0) -> None:
        self.capacity = capacity
        self.rows: list[np.ndarray] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, actions: np.ndarray) -> None:
        for row in np.asarray(actions, dtype=np.float64):
            self.seen += 1
            if len(self.rows) < self.capacity:
                self.rows.append(row)
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.capacity:
                    self.rows[j] = row

    def fit(self) -> ActionNormalizer:
        return ActionNormalizer.fit([np.stack(self.rows, axis=0)])


def _download(url: str, dest: Path) -> Path:
    """Downloads ``url`` with curl (resumable, fail-on-error)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(dest), url],
        check=True,
    )
    return dest


def _extract(archive: Path, dest: Path) -> Path:
    """Extracts zip/tar archives; plain files and directories pass through.

    Plain (non-archive) files — LIBERO ``.hdf5`` task files, Bridge RLDS
    ``.tfrecord`` shards — are themselves the shard root: readers accept a
    file path as ``root``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        return archive
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    else:
        return archive  # plain-file shard (hdf5 / tfrecord); deleted with workdir
    archive.unlink()  # free the archive's disk before conversion
    return dest


def _episode_iter_for(dataset: str, root: Path, **kwargs) -> Iterator[SourceEpisode]:
    if dataset == "bridge":
        from preprocess.bridge import iter_bridge_episodes

        return iter_bridge_episodes(root, **kwargs)
    if dataset == "bridge_rlds":
        from preprocess.bridge_rlds import iter_bridge_rlds_episodes

        return iter_bridge_rlds_episodes(root, **kwargs)
    if dataset == "libero":
        from preprocess.libero import iter_libero_episodes

        return iter_libero_episodes(root, **kwargs)
    raise ValueError(
        f"unknown dataset {dataset!r} (expected 'bridge', 'bridge_rlds', or 'libero')"
    )


def convert_shard(
    episodes: Iterator[SourceEpisode],
    out_dir: Path,
    builder: EpisodeBuilder,
    reservoir: _StatsReservoir,
    manifest: list[dict],
    teacher=None,
    limit: int | None = None,
) -> int:
    """Converts one shard's episodes with UNNORMALIZED action chunks.

    Args:
        episodes: The shard's episode iterator.
        out_dir: Episode output directory.
        builder: Shared perception builder (loaded once for all shards).
        reservoir: Global action-stats accumulator.
        manifest: Global manifest list (appended in place).
        teacher: Optional distillation teacher (relabels actions).
        limit: Optional per-shard episode cap.

    Returns:
        Number of episodes written.
    """
    from train.dataset import save_episode

    identity = _IdentityNormalizer()
    n = 0
    for ep in episodes:
        if limit is not None and n >= limit:
            break
        if teacher is not None:
            ep = dataclasses.replace(ep, actions=teacher.relabel(ep))
        reservoir.add(ep.actions)
        arrays = builder.build(ep, identity)  # pwm_targets = RAW chunks for now
        path = out_dir / f"{ep.episode_id}.npz"
        save_episode(path, arrays)
        manifest.append({"file": path.name, "id": ep.episode_id,
                         "T": int(arrays["frame_embs"].shape[0]),
                         "instruction": ep.instruction})
        n += 1
    return n


def finalize(out_dir: Path, reservoir: _StatsReservoir, manifest: list[dict],
             label_source: str) -> None:
    """Fits global stats and rewrites every episode's pwm_targets in place."""
    normalizer = reservoir.fit()
    normalizer.save(out_dir / "norm_stats.json")
    for entry in manifest:
        path = out_dir / entry["file"]
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        raw = arrays["pwm_targets"]
        arrays["pwm_targets"] = normalizer(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape)
        np.savez_compressed(path, **arrays)
    (out_dir / "manifest.json").write_text(
        json.dumps({"label_source": label_source, "episodes": manifest}, indent=2)
    )
    logger.info("finalized %d episodes with global norm stats -> %s", len(manifest), out_dir)


def run_shards(
    shards: list[str],
    out_dir: str | Path,
    dataset: str,
    budget_gb: float = 10.0,
    workdir: str | Path = ".shard_tmp",
    cfg: MicroVLAConfig = DEFAULT_CONFIG,
    mock: bool = False,
    device: str = "cpu",
    teacher=None,
    teacher_cache_dir: str | Path | None = None,
    limit_per_shard: int | None = None,
    reader_kwargs: dict | None = None,
    downloader: Callable[[str, Path], Path] = _download,
) -> Path:
    """Runs the full budget-guarded shard pipeline.

    Args:
        shards: URLs or local paths (archives or directories), processed in
            order. Local paths are converted in place (never deleted).
        out_dir: Episode output directory (counts against the budget).
        dataset: ``"bridge"`` or ``"libero"``.
        budget_gb: The hard disk cap for out_dir + workdir + teacher cache.
        workdir: Scratch space for downloads/extraction (deleted per shard).
        cfg: Canonical config.
        mock: Mock perception (dry-runs/tests).
        device: Torch device for the real detector (``"mps"`` on the MacBook).
        teacher: Optional distillation teacher.
        teacher_cache_dir: Teacher cache dir, tracked against the budget.
        limit_per_shard: Optional per-shard episode cap.
        reader_kwargs: Extra kwargs for the dataset's episode iterator.
        downloader: Injectable download function (tests use a copy stub).

    Returns:
        The output directory path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(workdir)
    tracked = [out, work] + ([Path(teacher_cache_dir)] if teacher_cache_dir else [])
    guard = BudgetGuard(budget_gb, tracked)

    builder = EpisodeBuilder(cfg, mock=mock, device=device)
    reservoir = _StatsReservoir()
    manifest: list[dict] = []

    for i, shard in enumerate(shards):
        local = Path(shard)
        is_local = local.exists()
        logger.info("shard %d/%d: %s", i + 1, len(shards), shard)

        if is_local:
            shard_root = local if local.is_dir() else _extract_guarded(local, work, guard, delete_archive=False)
        else:
            # Remote: reserve headroom for download + extraction coexisting.
            # Without a size hint we require 40% of the budget free — pick
            # shards well under that.
            guard.ensure(0.4 * budget_gb, f"downloading shard {shard}")
            archive = downloader(shard, work / Path(shard.split("?")[0]).name)
            shard_root = _extract_guarded(archive, work / f"shard_{i}", guard, delete_archive=True)

        n = convert_shard(
            _episode_iter_for(dataset, shard_root, **(reader_kwargs or {})),
            out, builder, reservoir, manifest,
            teacher=teacher, limit=limit_per_shard,
        )
        logger.info("  shard done: %d episodes (disk used %.2f / %.0f GB)",
                    n, guard.used_gb(), budget_gb)

        if not is_local and work.exists():
            shutil.rmtree(work)  # raw shard gone before the next download
        guard.ensure(0.0, "post-shard check")

    if not manifest:
        raise RuntimeError("no episodes converted — check shard contents/layout")
    finalize(out, reservoir, manifest,
             label_source=type(teacher).__name__ if teacher else "dataset")
    return out


def _extract_guarded(archive: Path, dest: Path, guard: BudgetGuard,
                     delete_archive: bool) -> Path:
    """Extraction with a budget check (extracted ~= archive size or larger)."""
    size_gb = archive.stat().st_size / _GB if archive.is_file() else 0.0
    guard.ensure(1.5 * size_gb, f"extracting {archive.name}")
    if archive.is_file() and not delete_archive:
        # Local archive the user owns: extract a COPY path, keep the original
        # (it lives outside our tracked dirs and is not ours to delete).
        dest.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
            return dest
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                t.extractall(dest, filter="data")
            return dest
        return archive  # plain-file shard (hdf5 / tfrecord), used in place
    return _extract(archive, dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shard_list", help="text file: one shard URL/path per line (# comments)")
    parser.add_argument("out", help="output directory for .npz episodes")
    parser.add_argument("--dataset", choices=["bridge", "bridge_rlds", "libero"], required=True)
    parser.add_argument("--budget-gb", type=float, default=10.0)
    parser.add_argument("--workdir", default=".shard_tmp")
    parser.add_argument("--device", default="cpu", help="'mps' on Apple silicon")
    parser.add_argument("--dry-run", action="store_true", help="mock perception")
    parser.add_argument("--limit-per-shard", type=int, default=None)
    parser.add_argument("--teacher", choices=["mock", "tinyvla"], default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-repo", default=None)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument(
        "--rlds-meta", default=None,
        help="bridge_rlds only: directory holding features.json/dataset_info.json "
        "(default <out>/_rlds_meta)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    shards = [
        line.strip() for line in Path(args.shard_list).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    teacher = build_teacher(args.teacher, args.teacher_checkpoint, args.teacher_repo,
                            args.teacher_cache, device=args.device)
    reader_kwargs = {}
    if args.dataset == "bridge_rlds":
        reader_kwargs["features_dir"] = args.rlds_meta or str(Path(args.out) / "_rlds_meta")
    run_shards(
        shards, args.out, args.dataset,
        budget_gb=args.budget_gb, workdir=args.workdir,
        mock=args.dry_run, device=args.device,
        teacher=teacher, teacher_cache_dir=args.teacher_cache,
        limit_per_shard=args.limit_per_shard,
        reader_kwargs=reader_kwargs,
    )


if __name__ == "__main__":
    main()
