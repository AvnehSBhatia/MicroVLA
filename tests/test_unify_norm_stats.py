"""Shared-normalization pass over separately-baked suites (disk-safe bake path)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microvla.config import DEFAULT_CONFIG as CFG
from preprocess.common import ActionNormalizer
from preprocess.unify_norm_stats import unify


def _bake(d: Path, scale: np.ndarray, raw: np.ndarray) -> None:
    """Writes one 'baked' dir: symmetric stats + episodes normalized by them."""
    d.mkdir(parents=True, exist_ok=True)
    ActionNormalizer(q_low=-scale, q_high=scale).save(d / "norm_stats.json")
    np.savez(d / "ep0.npz",
             pwm_targets=np.clip(raw / scale, -1.0, 1.0).astype(np.float32))


class TestUnify:
    def test_raw_actions_are_preserved_across_the_rescale(self, tmp_path):
        """THE property: denormalizing through the shared stats must return the
        same env-unit actions each dir encoded under its own."""
        rng = np.random.default_rng(0)
        a = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 1.0])
        b = np.array([1.0, 0.25, 0.5, 0.4, 0.1, 0.2, 1.0])
        raw_a = rng.uniform(-0.4, 0.4, size=(4, CFG.plan_steps, CFG.num_servos))
        raw_b = rng.uniform(-0.2, 0.2, size=(4, CFG.plan_steps, CFG.num_servos))
        _bake(tmp_path / "a", a, raw_a)
        _bake(tmp_path / "b", b, raw_b)

        # What must be preserved is the raw action each file ENCODES — read it
        # through each dir's own normalizer first, so the comparison is immune
        # to any clipping that happened at bake time.
        before = {}
        for d in (tmp_path / "a", tmp_path / "b"):
            with np.load(d / "ep0.npz") as z:
                before[d] = ActionNormalizer.load(d / "norm_stats.json").inverse(
                    z["pwm_targets"])

        out = unify([tmp_path / "a", tmp_path / "b"])
        assert np.allclose(out["scale"], np.maximum(a, b))

        shared = ActionNormalizer.load(tmp_path / "a" / "norm_stats.json")
        assert json.loads((tmp_path / "b" / "norm_stats.json").read_text()) == \
            json.loads((tmp_path / "a" / "norm_stats.json").read_text())
        for d in (tmp_path / "a", tmp_path / "b"):
            with np.load(d / "ep0.npz") as z:
                np.testing.assert_allclose(shared.inverse(z["pwm_targets"]),
                                           before[d], atol=1e-6)

    def test_widest_scale_wins_so_nothing_clips(self, tmp_path):
        narrow, wide = np.full(CFG.num_servos, 0.1), np.full(CFG.num_servos, 0.9)
        raw = np.full((2, CFG.plan_steps, CFG.num_servos), 0.1)   # saturates 'narrow'
        _bake(tmp_path / "n", narrow, raw)
        _bake(tmp_path / "w", wide, raw)
        unify([tmp_path / "n", tmp_path / "w"])
        with np.load(tmp_path / "n" / "ep0.npz") as z:
            assert np.abs(z["pwm_targets"]).max() <= 1.0

    def test_idempotent(self, tmp_path):
        a = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 1.0])
        b = np.array([1.0, 0.25, 0.5, 0.4, 0.1, 0.2, 1.0])
        raw = np.random.default_rng(1).uniform(-0.3, 0.3,
                                               size=(3, CFG.plan_steps, CFG.num_servos))
        _bake(tmp_path / "a", a, raw)
        _bake(tmp_path / "b", b, raw)
        unify([tmp_path / "a", tmp_path / "b"])
        with np.load(tmp_path / "a" / "ep0.npz") as z:
            once = z["pwm_targets"].copy()
        second = unify([tmp_path / "a", tmp_path / "b"])
        assert second["dirs"][str(tmp_path / "a")] == 0
        with np.load(tmp_path / "a" / "ep0.npz") as z:
            np.testing.assert_array_equal(z["pwm_targets"], once)

    def test_refuses_asymmetric_stats(self, tmp_path):
        """An offset normalizer is the drift-into-wall bug; refuse, don't average."""
        d = tmp_path / "old"
        d.mkdir()
        ActionNormalizer(q_low=np.full(CFG.num_servos, -0.2),
                         q_high=np.full(CFG.num_servos, 0.9)).save(d / "norm_stats.json")
        np.savez(d / "ep0.npz",
                 pwm_targets=np.zeros((1, CFG.plan_steps, CFG.num_servos), dtype=np.float32))
        with pytest.raises(ValueError, match="NOT symmetric"):
            unify([d])

    def test_missing_stats_and_empty_dirs_are_errors(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="norm_stats.json"):
            unify([tmp_path / "empty"])
        scale = np.ones(CFG.num_servos)
        (tmp_path / "nodata").mkdir()
        ActionNormalizer(q_low=-scale, q_high=scale).save(
            tmp_path / "nodata" / "norm_stats.json")
        with pytest.raises(FileNotFoundError, match="no .npz episodes"):
            unify([tmp_path / "nodata"])
