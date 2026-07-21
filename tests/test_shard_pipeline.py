"""Tests for the budget-guarded shard pipeline (mocks only, no network)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from microvla import DEFAULT_CONFIG
from preprocess.shard_pipeline import (
    BudgetGuard,
    _StatsReservoir,
    dir_size_gb,
    run_shards,
)
from train.dataset import EPISODE_KEYS, EpisodeDataset

CFG = DEFAULT_CONFIG


class TestBudgetGuard:
    def test_dir_size_and_ensure(self, tmp_path):
        d = tmp_path / "tracked"
        d.mkdir()
        (d / "blob.bin").write_bytes(b"x" * 1024)
        guard = BudgetGuard(budget_gb=1e-5, tracked=[d])  # ~10.7 KB budget
        assert 0 < guard.used_gb() < 1e-5
        guard.ensure(0.0, "fits")
        with pytest.raises(RuntimeError, match="disk budget"):
            guard.ensure(1e-5, "too big")

    def test_missing_dir_counts_zero(self, tmp_path):
        assert dir_size_gb(tmp_path / "nope") == 0.0


class TestStatsReservoir:
    def test_reservoir_caps_and_fits(self):
        res = _StatsReservoir(capacity=100, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            res.add(rng.normal(size=(50, 7)))
        assert len(res.rows) == 100
        assert res.seen == 500
        norm = res.fit()
        z = norm(rng.normal(size=(20, 7)))
        assert z.shape == (20, 7) and np.abs(z).max() <= 1.0


class TestRunShards:
    def _patch_iter(self, monkeypatch, episodes_per_shard: int = 2, T: int = 30):
        """Routes _episode_iter_for to synthetic episodes seeded per shard root."""
        from preprocess.common import SourceEpisode
        import preprocess.shard_pipeline as sp

        def fake_iter(dataset, root, **kwargs):
            seed = abs(hash(str(root))) % 1000
            for j in range(episodes_per_shard):
                rng = np.random.default_rng(seed + j)
                yield SourceEpisode(
                    frames=[rng.integers(0, 256, (32, 48, 3), dtype=np.uint8) for _ in range(T)],
                    actions=rng.uniform(-0.6, 0.6, (T, CFG.num_servos)).astype(np.float32),
                    instruction="put the fork on the plate",
                    source_hz=20.0,
                    episode_id=f"{root.name}_ep{j}",
                )

        monkeypatch.setattr(sp, "_episode_iter_for", fake_iter)

    def test_local_shards_end_to_end(self, tmp_path, monkeypatch):
        self._patch_iter(monkeypatch)
        shards = []
        for name in ("shard_a", "shard_b"):
            d = tmp_path / name
            d.mkdir()
            (d / "marker").write_text("raw")
            shards.append(str(d))

        out = run_shards(shards, tmp_path / "out", dataset="bridge",
                         budget_gb=1.0, workdir=tmp_path / "wk", mock=True)

        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest["episodes"]) == 4
        assert (out / "norm_stats.json").exists()
        # Finalize must have normalized ALL shards with ONE set of stats.
        ds = EpisodeDataset(out)
        for i in range(len(ds)):
            item = ds[i]
            assert set(item) == set(EPISODE_KEYS)
            assert float(item["pwm_targets"].abs().max()) <= 1.0

    def test_budget_violation_stops_pipeline(self, tmp_path, monkeypatch):
        self._patch_iter(monkeypatch)
        shard = tmp_path / "shard"
        shard.mkdir()
        big = tmp_path / "out"
        big.mkdir()
        (big / "existing.bin").write_bytes(b"x" * 4096)

        with pytest.raises(RuntimeError, match="disk budget"):
            run_shards(["http://example.invalid/shard.zip"], big, dataset="bridge",
                       budget_gb=1e-6, workdir=tmp_path / "wk", mock=True,
                       downloader=lambda url, dest: (_ for _ in ()).throw(
                           AssertionError("guard must fire before download")))

    def test_per_shard_limit(self, tmp_path, monkeypatch):
        self._patch_iter(monkeypatch, episodes_per_shard=5)
        d = tmp_path / "only"
        d.mkdir()
        out = run_shards([str(d)], tmp_path / "out", dataset="bridge",
                         budget_gb=1.0, workdir=tmp_path / "wk", mock=True,
                         limit_per_shard=2)
        assert len(json.loads((out / "manifest.json").read_text())["episodes"]) == 2

    def test_empty_conversion_raises(self, tmp_path, monkeypatch):
        import preprocess.shard_pipeline as sp
        monkeypatch.setattr(sp, "_episode_iter_for", lambda ds, root, **kw: iter(()))
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(RuntimeError, match="no episodes"):
            run_shards([str(d)], tmp_path / "out", dataset="bridge",
                       budget_gb=1.0, workdir=tmp_path / "wk", mock=True)
