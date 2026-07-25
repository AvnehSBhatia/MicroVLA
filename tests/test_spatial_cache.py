"""Precomputed frozen-backbone maps must be a pure speedup, never a change.

CPU-only, mock-only, no cv2, no network: the real backbone is stood in for by a
deterministic fake with the same interface (``feature_maps(list_of_bgr) ->
[B, vis_dim, Hf, Wf]``).
"""

from __future__ import annotations

import numpy as np
import torch

import train.train_batched as TB
from microvla.config import DEFAULT_CONFIG as CFG
from microvla.perception.spatial_adapter import TextQueriedSpatialAdapter

_HF = 5  # stand-in map side (the real backbone yields 20x20 at min_side=512)


class FakeBackbone:
    """Deterministic frozen map extractor: content-dependent, no weights."""

    def __init__(self) -> None:
        self.calls = 0

    def feature_maps(self, frames_bgr: list) -> torch.Tensor:
        self.calls += 1
        out = []
        for f in frames_bgr:
            g = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32).mean(-1)
            out.append(torch.nn.functional.adaptive_avg_pool2d(
                g[None].repeat(CFG.vis_dim, 1, 1)[None], (_HF, _HF))[0])
        return torch.stack(out)


def _bucket(n: int = 5, t: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "wrist_frames": torch.as_tensor(
            rng.integers(0, 255, (n, t, 32, 32, 3), dtype=np.uint8)),
        "text_tokens": torch.randn(n, CFG.n_text_tokens, CFG.text_dim),
    }


class TestSpatialCache:
    def test_cached_matches_the_live_backbone_path(self):
        """The whole justification: same maps, computed once instead of per epoch."""
        torch.manual_seed(0)
        tqsa = TextQueriedSpatialAdapter(CFG).eval()
        b = _bucket()
        T = b["wrist_frames"].shape[1]

        live = [TB._batch_spatial(b, t, tqsa, FakeBackbone(), "cpu") for t in range(T)]
        TB.precompute_spatial_maps({(T, True): b}, FakeBackbone(), 2, "test")
        cached = [TB._batch_spatial(b, t, tqsa, None, "cpu") for t in range(T)]

        for t in range(T):
            assert set(live[t]) == set(cached[t])
            for k in live[t]:
                torch.testing.assert_close(cached[t][k], live[t][k], rtol=2e-3, atol=2e-3)

    def test_backbone_is_never_touched_again(self):
        """Post-cache the adapter must not re-enter the backbone (that IS the cost)."""
        tqsa = TextQueriedSpatialAdapter(CFG).eval()
        b = _bucket()
        T = b["wrist_frames"].shape[1]
        TB.precompute_spatial_maps({(T, True): b}, FakeBackbone(), 2, "test")

        spy = FakeBackbone()
        for t in range(T):
            assert TB._batch_spatial(b, t, tqsa, spy, "cpu") is not None
        assert spy.calls == 0

    def test_cache_shape_and_dtype(self):
        b = _bucket(n=3, t=2)
        n = TB.precompute_spatial_maps({(2, True): b}, FakeBackbone(), 2, "test")
        maps = b["spatial_maps"]
        assert maps.shape == (3, 2, CFG.vis_dim, _HF, _HF)
        assert maps.dtype == torch.float16 and maps.device.type == "cpu"
        assert n == maps.numel() * 2

    def test_frameless_buckets_are_skipped(self):
        """Bridge episodes carry no frames; they must stay planner-only."""
        frameless = {"text_tokens": torch.randn(2, CFG.n_text_tokens, CFG.text_dim)}
        assert TB.precompute_spatial_maps({(4, False): frameless},
                                          FakeBackbone(), 2, "test") == 0
        assert "spatial_maps" not in frameless
        tqsa = TextQueriedSpatialAdapter(CFG).eval()
        assert TB._batch_spatial(frameless, 0, tqsa, None, "cpu") is None

    def test_no_tqsa_means_no_spatial_even_with_a_cache(self):
        b = _bucket(n=2, t=2)
        TB.precompute_spatial_maps({(2, True): b}, FakeBackbone(), 2, "test")
        assert TB._batch_spatial(b, 0, None, None, "cpu") is None

    def test_batch_size_does_not_change_the_cache(self):
        """Chunking is an implementation detail, not part of the training signal."""
        a, b = _bucket(n=6, t=3, seed=1), _bucket(n=6, t=3, seed=1)
        TB.precompute_spatial_maps({(3, True): a}, FakeBackbone(), 1, "test")
        TB.precompute_spatial_maps({(3, True): b}, FakeBackbone(), 4, "test")
        torch.testing.assert_close(a["spatial_maps"], b["spatial_maps"])
