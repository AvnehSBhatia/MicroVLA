"""Tests for preprocess/renorm_symmetric.py (v5 symmetric action space).

The invariant that matters: re-normalization must PRESERVE the raw env-unit
actions (inverse_new(new_pwm) == inverse_old(old_pwm)) while making the
mapping symmetric (normalized 0 <=> raw 0). CPU-only, tmp-dir, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocess.common import ActionNormalizer
from preprocess.renorm_symmetric import renorm_dir


@pytest.fixture()
def asymmetric_dir(tmp_path):
    """A fake baked dataset dir: 2 episodes normalized with asymmetric stats."""
    rng = np.random.default_rng(0)
    q_low = np.array([-0.78, -0.81, -0.94, -0.12, -0.17, -0.17, -1.0])
    q_high = np.array([0.94, 0.89, 0.94, 0.13, 0.21, 0.34, 1.0])
    old = ActionNormalizer(q_low, q_high)
    raws = []
    for i in range(2):
        raw = rng.uniform(q_low, q_high, size=(6, 5, 7))  # in-range raw actions
        raws.append(raw)
        np.savez_compressed(
            tmp_path / f"ep{i}.npz",
            pwm_targets=old(raw),
            frame_embs=rng.standard_normal((6, 8)).astype(np.float32),  # untouched key
        )
    old.save(tmp_path / "norm_stats.json")
    return tmp_path, old, raws


class TestRenormSymmetric:
    def test_raw_actions_preserved_and_zero_centered(self, asymmetric_dir):
        d, old, raws = asymmetric_dir
        renorm_dir(d)

        new = ActionNormalizer.load(d / "norm_stats.json")
        # Stats are symmetric -> normalized 0 denormalizes to exactly 0.
        assert np.allclose(new.q_low, -new.q_high)
        assert np.allclose(new.inverse(np.zeros(7)), 0.0, atol=1e-9)

        for i, raw in enumerate(raws):
            with np.load(d / f"ep{i}.npz") as z:
                pwm_new = z["pwm_targets"]
                assert "frame_embs" in z.files  # other keys survive the rewrite
            # The raw env-unit actions are unchanged by the re-normalization.
            assert np.allclose(new.inverse(pwm_new), raw, atol=1e-4)

        # Old stats are backed up for provenance.
        assert (d / "norm_stats.asymmetric.json").exists()

    def test_idempotent(self, asymmetric_dir):
        d, _old, _raws = asymmetric_dir
        renorm_dir(d)
        with np.load(d / "ep0.npz") as z:
            first = z["pwm_targets"].copy()
        renorm_dir(d)  # second run must be a no-op
        with np.load(d / "ep0.npz") as z:
            second = z["pwm_targets"].copy()
        assert np.allclose(first, second, atol=1e-6)
