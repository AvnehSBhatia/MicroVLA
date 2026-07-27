"""Tests for the text-space region path (``perception/text_region.py``, v8).

The claim under test is that perception stops making an irreversible decision
for the rest of the stack. v7 grounded two boxes by a hard argmax over two
text-conditioned classes and pooled a *backbone* (SPPF) map inside them, so the
only causal path from language to vision was "which box got picked" — and a
missed detection deleted the visual evidence outright. v8 keeps K class-agnostic
proposals whose embeddings come from the contrastive head (text space), scores
every one of them softly against every task phrase, and exposes that as a graded
weight in ``[0, 1]``.

So these tests pin down exactly the properties downstream modules rely on:
static ``cfg.max_objects`` shapes with weight-0 padding, weights that are a
product of objectness and text match, canonical (standardized) embeddings,
ranking by objectness alone, an all-pad observation instead of an exception when
the detector sees nothing, and determinism of the mock. CPU-only, mock-only: a
fake ultralytics-shaped model exercises the real extractor's hook plumbing, so
even the model-coupled code is covered without ``ultralytics`` installed.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from microvla.config import DEFAULT_CONFIG
from microvla.perception.text_region import (
    MockTextRegionExtractor,
    RegionObs,
    TextRegionExtractor,
    assemble_regions,
)

#: Small dims keep the tests fast AND prove nothing hardcodes 512/8.
CFG = replace(DEFAULT_CONFIG, vis_dim=16, max_objects=4)


def phrases(cfg=CFG, n=3, seed=7) -> torch.Tensor:
    """``[n, vis_dim]`` L2-normalized pseudo text features (CLIP-space stand-in)."""
    gen = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, cfg.vis_dim, generator=gen), dim=-1)


def frame(tag: int = 0, h: int = 16, w: int = 24) -> np.ndarray:
    """Deterministic uint8 BGR frame; ``tag`` changes its bytes."""
    values = (np.arange(h * w * 3, dtype=np.int64) * 7 + tag * 13) % 251
    return values.astype(np.uint8).reshape(h, w, 3)


def mock(cfg=CFG, **kwargs) -> MockTextRegionExtractor:
    """A mock extractor with task phrases already set."""
    extractor = MockTextRegionExtractor(cfg, **kwargs)
    extractor.set_text_features(phrases(cfg))
    return extractor


def maps_and_boxes(cfg=CFG, n_boxes: int = 3):
    """Synthetic cv4 maps (3 FPN levels) plus ``n_boxes`` distinct boxes."""
    gen = torch.Generator().manual_seed(3)
    levels = [
        torch.randn(1, cfg.vis_dim, side, side, generator=gen) for side in (8, 4, 2)
    ]
    boxes = torch.tensor(
        [[10.0 * i, 5.0 * i, 10.0 * i + 30.0, 5.0 * i + 40.0] for i in range(n_boxes)]
    )
    return levels, boxes


class TestShapesAndPadding:
    """Static ``cfg.max_objects`` rows, whatever the detector found."""

    def test_shapes_are_config_driven(self):
        obs = mock(n_proposals=2).extract(frame())
        assert obs.emb.shape == (CFG.max_objects, CFG.vis_dim)
        assert obs.center.shape == (CFG.max_objects, 2)
        assert obs.weight.shape == (CFG.max_objects,)
        assert obs.match.shape == (CFG.max_objects, 3)

    def test_default_config_shapes(self):
        # The dims the loop will actually see: 8 proposals x 512-d.
        obs = mock(DEFAULT_CONFIG).extract(frame())
        assert obs.emb.shape == (DEFAULT_CONFIG.max_objects, DEFAULT_CONFIG.vis_dim)

    def test_fewer_proposals_are_padded(self):
        obs = mock(n_proposals=2).extract(frame())
        assert obs.n_valid == 2
        assert torch.equal(obs.emb[2:], torch.zeros(CFG.max_objects - 2, CFG.vis_dim))
        assert torch.equal(obs.match[2:], torch.zeros(CFG.max_objects - 2, 3))
        # Pad centers reuse the (0.5, 0.5) missed-detection convention; weight 0
        # is what keeps them distinguishable from a real object at frame center.
        assert torch.equal(obs.center[2:], torch.full((CFG.max_objects - 2, 2), 0.5))

    def test_more_proposals_than_k_are_truncated(self):
        obs = mock(n_proposals=CFG.max_objects + 5).extract(frame())
        assert obs.n_valid == CFG.max_objects
        assert bool((obs.weight > 0).all())

    def test_shape_is_the_same_whatever_the_count(self):
        shapes = {
            tuple(mock(n_proposals=n).extract(frame(n)).emb.shape)
            for n in (0, 1, CFG.max_objects, CFG.max_objects + 3)
        }
        assert shapes == {(CFG.max_objects, CFG.vis_dim)}


class TestWeightSemantics:
    """``weight`` is graded evidence in [0, 1]; pads are exactly 0.0."""

    @pytest.mark.parametrize("tag", [0, 1, 2, 3, 4])
    def test_weight_in_unit_interval(self, tag):
        weight = mock().extract(frame(tag)).weight
        assert float(weight.min()) >= 0.0
        assert float(weight.max()) <= 1.0

    def test_pad_weight_is_exactly_zero(self):
        weight = mock(n_proposals=1).extract(frame()).weight
        assert float(weight[0]) > 0.0
        assert torch.equal(weight[1:], torch.zeros(CFG.max_objects - 1))

    def test_weight_is_objectness_times_match_peak(self):
        # The evidence rule downstream relies on: objectness says something is
        # there, the match peak says the language cares about it.
        levels, boxes = maps_and_boxes()
        obj = torch.tensor([0.25, 0.75, 0.5])
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, obj, frame_hw=(64, 64), text_feats=phrases()
        )
        peak = obs.match[: obs.n_valid].max(dim=-1).values
        expected = torch.tensor([0.75, 0.5, 0.25]) * peak  # sorted by objectness
        assert torch.allclose(obs.weight[: obs.n_valid], expected, atol=1e-6)

    def test_objectness_is_clamped_before_it_multiplies_the_match(self):
        # weight must stay in [0, 1] because every consumer treats it as the
        # same graded evidence scalar fusion's box_weight was. Clamping the
        # objectness FIRST (rather than only clamping the product) is what makes
        # an out-of-range confidence saturate instead of rescaling the match.
        levels, boxes = maps_and_boxes(n_boxes=2)
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, torch.tensor([3.0, -1.0]), frame_hw=(64, 64),
            text_feats=phrases(),
        )
        peak = obs.match.max(dim=-1).values
        assert float(obs.weight[0]) == pytest.approx(float(peak[0]), abs=1e-6)
        assert float(obs.weight[1]) == 0.0

    def test_weak_objectness_is_kept_not_dropped(self):
        # A near-zero-confidence proposal must survive with a small weight: the
        # v7 failure was that discarded evidence is unrecoverable.
        levels, boxes = maps_and_boxes(n_boxes=1)
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, torch.tensor([1e-3]), frame_hw=(64, 64), text_feats=phrases()
        )
        assert obs.n_valid == 1
        assert 0.0 < float(obs.weight[0]) < 0.01


class TestDeterminism:
    """The mock must be a pure function of the frame bytes."""

    def test_same_frame_same_observation(self):
        a, b = mock().extract(frame(5)), mock().extract(frame(5))
        assert torch.equal(a.emb, b.emb)
        assert torch.equal(a.center, b.center)
        assert torch.equal(a.weight, b.weight)
        assert torch.equal(a.match, b.match)

    def test_repeated_calls_on_one_instance_agree(self):
        extractor = mock()
        assert torch.equal(extractor.extract(frame(2)).weight, extractor.extract(frame(2)).weight)

    def test_different_frames_differ(self):
        a, b = mock().extract(frame(1)), mock().extract(frame(2))
        assert not torch.equal(a.emb, b.emb)

    def test_global_rng_is_untouched(self):
        torch.manual_seed(1234)
        before = torch.rand(4)
        torch.manual_seed(1234)
        mock().extract(frame(9))
        assert torch.equal(before, torch.rand(4))


class TestCanonicalSpace:
    """Embeddings leave perception standardized (CLAUDE.md's canonical space)."""

    def test_mock_valid_rows_are_standardized(self):
        obs = mock(n_proposals=3).extract(frame())
        valid = obs.emb[: obs.n_valid]
        assert torch.allclose(valid.mean(dim=-1), torch.zeros(3), atol=1e-5)
        assert torch.allclose(valid.std(dim=-1, unbiased=False), torch.ones(3), atol=1e-3)

    def test_pooled_rows_are_standardized(self):
        levels, boxes = maps_and_boxes()
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, torch.tensor([0.9, 0.5, 0.2]), frame_hw=(64, 64),
            text_feats=phrases(),
        )
        valid = obs.emb[: obs.n_valid]
        assert torch.allclose(valid.mean(dim=-1), torch.zeros(3), atol=1e-5)
        assert torch.allclose(valid.std(dim=-1, unbiased=False), torch.ones(3), atol=1e-3)

    def test_pad_rows_are_exact_zeros_not_standardized_noise(self):
        obs = mock(n_proposals=1).extract(frame())
        assert float(obs.emb[1:].abs().max()) == 0.0


class TestTopKOrdering:
    """Ranking is class-agnostic: by objectness over ALL proposals."""

    def test_rows_are_sorted_by_objectness(self):
        levels, boxes = maps_and_boxes(n_boxes=CFG.max_objects + 2)
        obj = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, obj, frame_hw=(64, 64), text_feats=phrases()
        )
        centers = torch.stack(
            [(boxes[:, 0] + boxes[:, 2]) * 0.5 / 64, (boxes[:, 1] + boxes[:, 3]) * 0.5 / 64],
            dim=-1,
        )
        expected = centers[torch.argsort(obj, descending=True)[: CFG.max_objects]]
        assert torch.allclose(obs.center, expected, atol=1e-6)

    def test_low_objectness_proposals_are_the_ones_dropped(self):
        levels, boxes = maps_and_boxes(n_boxes=CFG.max_objects + 1)
        obj = torch.linspace(0.1, 0.9, CFG.max_objects + 1)
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, obj, frame_hw=(64, 64), text_feats=phrases()
        )
        # The weakest box (index 0, the top-left one) must not appear.
        dropped = torch.tensor([(boxes[0, 0] + boxes[0, 2]) * 0.5 / 64,
                                (boxes[0, 1] + boxes[0, 3]) * 0.5 / 64])
        assert not bool((obs.center - dropped).abs().sum(dim=-1).lt(1e-6).any())

    def test_ties_keep_detector_order(self):
        levels, boxes = maps_and_boxes(n_boxes=3)
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, torch.full((3,), 0.5), frame_hw=(64, 64), text_feats=phrases()
        )
        centers = torch.stack(
            [(boxes[:, 0] + boxes[:, 2]) * 0.5 / 64, (boxes[:, 1] + boxes[:, 3]) * 0.5 / 64],
            dim=-1,
        )
        assert torch.allclose(obs.center[:3], centers, atol=1e-6)


class TestBlindFrame:
    """Zero proposals is a weight-0 observation, never an exception."""

    def test_mock_with_no_proposals(self):
        obs = mock(n_proposals=0).extract(frame())
        assert isinstance(obs, RegionObs)
        assert obs.n_valid == 0
        assert float(obs.weight.abs().sum()) == 0.0
        assert float(obs.emb.abs().sum()) == 0.0
        assert torch.equal(obs.center, torch.full((CFG.max_objects, 2), 0.5))

    def test_regions_from_with_no_boxes(self):
        levels, _ = maps_and_boxes()
        obs = TextRegionExtractor(CFG).regions_from(
            levels, torch.zeros(0, 4), torch.zeros(0), frame_hw=(64, 64),
            text_feats=phrases(),
        )
        assert obs.n_valid == 0
        assert obs.emb.shape == (CFG.max_objects, CFG.vis_dim)
        assert obs.match.shape == (CFG.max_objects, 3)

    def test_assemble_with_no_proposals(self):
        obs = assemble_regions(
            emb=torch.zeros(0, CFG.vis_dim),
            objectness=torch.zeros(0),
            center=torch.zeros(0, 2),
            text_feats=phrases(),
            vis_dim=CFG.vis_dim,
            max_objects=CFG.max_objects,
        )
        assert obs.n_valid == 0


class TestSoftTextMatching:
    """The replacement for the hard argmax: every proposal scores every phrase."""

    def test_match_rows_are_a_distribution_over_phrases(self):
        obs = mock(n_proposals=3).extract(frame())
        rows = obs.match[: obs.n_valid]
        assert torch.allclose(rows.sum(dim=-1), torch.ones(3), atol=1e-5)
        assert float(rows.min()) > 0.0  # soft: no phrase is ever exactly ruled out

    def test_pad_match_is_outside_the_simplex(self):
        # All-zero (not uniform) so a pad can never read as a genuine match.
        obs = mock(n_proposals=1).extract(frame())
        assert float(obs.match[1:].abs().sum()) == 0.0

    def test_match_follows_where_the_phrase_lives_in_the_map(self):
        # A map whose left half IS phrase 0 and right half IS phrase 1: the box
        # on the left must match phrase 0 and the one on the right phrase 1.
        # This is the whole point of cv4 — similarity is a dot product in text
        # space — and it also pins the box -> map coordinate mapping.
        text = phrases()
        region_map = torch.zeros(1, CFG.vis_dim, 8, 8)
        region_map[..., :4] = text[0].view(1, -1, 1, 1)
        region_map[..., 4:] = text[1].view(1, -1, 1, 1)
        boxes = torch.tensor([[2.0, 20.0, 22.0, 44.0], [42.0, 20.0, 62.0, 44.0]])
        obs = TextRegionExtractor(CFG).regions_from(
            region_map, boxes, torch.tensor([0.9, 0.8]), frame_hw=(64, 64), text_feats=text
        )
        assert int(obs.match[0].argmax()) == 0
        assert int(obs.match[1].argmax()) == 1

    def test_temperature_controls_sharpness(self):
        text = phrases()
        region_map = torch.zeros(1, CFG.vis_dim, 8, 8) + text[0].view(1, -1, 1, 1)
        boxes = torch.tensor([[8.0, 8.0, 56.0, 56.0]])
        sharp = TextRegionExtractor(CFG, match_temperature=0.02).regions_from(
            region_map, boxes, torch.ones(1), frame_hw=(64, 64), text_feats=text
        )
        soft = TextRegionExtractor(CFG, match_temperature=1.0).regions_from(
            region_map, boxes, torch.ones(1), frame_hw=(64, 64), text_feats=text
        )
        assert float(sharp.match[0, 0]) > float(soft.match[0, 0])
        # Sharper matching means a higher weight for the same objectness.
        assert float(sharp.weight[0]) > float(soft.weight[0])

    def test_missing_text_features_raise(self):
        # Silently dropping language would recreate the defect this module fixes.
        with pytest.raises(RuntimeError, match="text features"):
            MockTextRegionExtractor(CFG).extract(frame())

    def test_text_features_of_the_wrong_width_raise(self):
        extractor = MockTextRegionExtractor(CFG)
        extractor.set_text_features(torch.randn(3, CFG.vis_dim + 1))
        with pytest.raises(ValueError, match="text features"):
            extractor.extract(frame())

    def test_txt_feats_batch_layout_is_accepted(self):
        # ultralytics hands over [1, P, dim]; TaskEncoding.tokens() gives [P, dim].
        extractor = MockTextRegionExtractor(CFG, n_proposals=2)
        extractor.set_text_features(phrases().unsqueeze(0))
        assert extractor.extract(frame()).match.shape == (CFG.max_objects, 3)


class TestBoxGeometry:
    """Centers are frame-normalized; pooling undoes ultralytics' letterbox."""

    def test_centers_are_normalized_and_clamped(self):
        levels, _ = maps_and_boxes()
        boxes = torch.tensor([[-20.0, -20.0, 10.0, 10.0], [0.0, 0.0, 64.0, 32.0]])
        obs = TextRegionExtractor(CFG).regions_from(
            levels, boxes, torch.tensor([0.5, 0.9]), frame_hw=(32, 64), text_feats=phrases()
        )
        # Sorted by objectness: the full-frame box first.
        assert torch.allclose(obs.center[0], torch.tensor([0.5, 0.5]), atol=1e-6)
        assert bool(((obs.center >= 0.0) & (obs.center <= 1.0)).all())

    def test_letterbox_mapping_matches_ultralytics_geometry(self):
        # frame 100x200 (h, w) into a 640x640 input: r = 3.2, pad_h = 160, so a
        # full-frame box maps to y in [0.25, 0.75] and x in [0, 1].
        extractor = TextRegionExtractor(CFG)
        unit = extractor._boxes_to_unit(
            torch.tensor([[0.0, 0.0, 200.0, 100.0]]), (100, 200), (640, 640)
        )
        assert torch.allclose(unit[0], torch.tensor([0.0, 0.25, 1.0, 0.75]), atol=1e-6)

    def test_pooling_averages_the_box_extent_not_its_center(self):
        # A map whose middle third is phrase 0 and whose border is phrase 1: a
        # full-frame box must read as phrase 1, because a region embedding is
        # the average over the box's extent. Sampling only the box center — the
        # cheap shortcut — would report phrase 0 from one pixel.
        text = phrases()
        region_map = torch.zeros(1, CFG.vis_dim, 9, 9) + text[1].view(1, -1, 1, 1)
        region_map[:, :, 3:6, 3:6] = text[0].view(1, -1, 1, 1)
        obs = TextRegionExtractor(CFG).regions_from(
            region_map, torch.tensor([[0.0, 0.0, 64.0, 64.0]]), torch.ones(1),
            frame_hw=(64, 64), text_feats=text,
        )
        assert int(obs.match[0].argmax()) == 1

    def test_all_fpn_levels_contribute(self):
        # One level says phrase 0, the other says phrase 1. Averaging lands
        # exactly between them (both text vectors are unit-norm, so the midpoint
        # is equidistant); using only the first level would report phrase 0.
        # The average is well-defined precisely because every level is scored
        # against ONE shared text-feature matrix.
        text = phrases()
        box = torch.tensor([[0.0, 0.0, 64.0, 64.0]])
        extractor = TextRegionExtractor(CFG)
        level0 = torch.zeros(1, CFG.vis_dim, 8, 8) + text[0].view(1, -1, 1, 1)
        level1 = torch.zeros(1, CFG.vis_dim, 4, 4) + text[1].view(1, -1, 1, 1)

        both = extractor.regions_from(
            [level0, level1], box, torch.ones(1), frame_hw=(64, 64), text_feats=text
        )
        first_only = extractor.regions_from(
            [level0], box, torch.ones(1), frame_hw=(64, 64), text_feats=text
        )
        assert int(first_only.match[0].argmax()) == 0
        assert float(both.match[0, 0]) == pytest.approx(float(both.match[0, 1]), abs=1e-5)

    def test_pooling_reads_the_letterboxed_region(self):
        # Only the TOP QUARTER of the map is phrase 0. Letterboxing a 100x200
        # frame into 640x640 squeezes it into the middle band (y -> 0.25..0.75),
        # so a box near the top of the FRAME is nowhere near the top of the MAP.
        # Skipping the correction would read phrase 0; the correct mapping reads
        # phrase 1, which is what makes this test worth having.
        text = phrases()
        region_map = torch.zeros(1, CFG.vis_dim, 8, 8)
        region_map[:, :, :2] = text[0].view(1, -1, 1, 1)
        region_map[:, :, 2:] = text[1].view(1, -1, 1, 1)
        box = torch.tensor([[20.0, 5.0, 180.0, 20.0]])
        extractor = TextRegionExtractor(CFG)

        letterboxed = extractor.regions_from(
            region_map, box, torch.ones(1), frame_hw=(100, 200),
            input_hw=(640, 640), text_feats=text,
        )
        naive = extractor.regions_from(
            region_map, box, torch.ones(1), frame_hw=(100, 200), text_feats=text
        )
        assert int(letterboxed.match[0].argmax()) == 1
        assert int(naive.match[0].argmax()) == 0


class TestFrozenPath:
    """No parameters, no heavy imports: this is a frozen extraction path."""

    def test_import_does_not_pull_ultralytics(self):
        # Re-imported from scratch rather than just inspecting sys.modules, so
        # the assertion cannot be decided by whatever ran earlier in the suite:
        # what is checked is that THIS module adds no heavy dependency.
        heavies = ("ultralytics", "torchvision", "cv2")
        before = {name for name in heavies if name in sys.modules}
        saved = sys.modules.pop("microvla.perception.text_region")
        try:
            importlib.import_module("microvla.perception.text_region")
            after = {name for name in heavies if name in sys.modules}
        finally:
            sys.modules["microvla.perception.text_region"] = saved
        assert after == before, f"{after - before} must be imported lazily"

    def test_extractors_own_no_parameters(self):
        for extractor in (TextRegionExtractor(CFG), MockTextRegionExtractor(CFG)):
            assert not isinstance(extractor, nn.Module)
            assert not any(
                isinstance(value, (nn.Parameter, nn.Module))
                for value in vars(extractor).values()
            )

    def test_outputs_carry_no_grad(self):
        obs = mock(n_proposals=2).extract(frame())
        assert not obs.emb.requires_grad
        assert obs.emb.dtype == torch.float32


# --- Fake ultralytics-shaped model: exercises the REAL hook plumbing ---------
# Named and shaped like the pieces attach() matches on (a "WorldDetect" module
# whose cv4 is a ModuleList of contrastive heads taking (embedding, text)), so
# the model-coupled half of TextRegionExtractor is covered without ultralytics.


class _FakeContrastiveHead(nn.Module):
    """Consumes a ``[1, D, H, W]`` embedding and emits per-phrase logits."""

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        sim = torch.einsum(
            "bchw,kc->bkhw", F.normalize(x, dim=1), F.normalize(text, dim=-1)
        )
        return sim * self.logit_scale.exp()


class _FakeBNContrastiveHead(nn.Module):
    """The world**v2** variant: scores ``BatchNorm2d(x)``, never L2-normalized.

    Shaped exactly like ultralytics' ``BNContrastiveHead`` — a ``norm`` child, a
    ``logit_scale`` that multiplies an UNNORMALIZED dot product, and a bias — so
    the tests can pin that ``attach`` taps the post-norm vector and refuses to
    read that ``logit_scale`` as a cosine temperature.
    """

    def __init__(self, dim: int, logit_scale: float = 0.5801) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        sim = torch.einsum("bchw,kc->bkhw", self.norm(x), F.normalize(text, dim=-1))
        return sim * self.logit_scale.exp() + self.bias


class WorldDetect(nn.Module):
    """Stand-in for ultralytics' head; ``attach`` finds it by this class name."""

    def __init__(self, dim: int, levels: int = 2, bn: bool = False) -> None:
        super().__init__()
        self.cv2 = nn.ModuleList(nn.Identity() for _ in range(levels))
        self.cv3 = nn.ModuleList(nn.Identity() for _ in range(levels))
        make = (lambda: _FakeBNContrastiveHead(dim)) if bn else _FakeContrastiveHead
        self.cv4 = nn.ModuleList(make() for _ in range(levels))

    def forward(self, feats: list[torch.Tensor], text: torch.Tensor) -> list[torch.Tensor]:
        return [head(f, text) for head, f in zip(self.cv4, feats)]


class _FakeDetectionModel(nn.Module):
    """Turns an input image tensor into two levels of per-anchor embeddings.

    ``feats`` pins the per-anchor embedding to a constant vector instead of
    drawing it, which is what lets a test say exactly which phrase the tapped
    tensor should match.
    """

    def __init__(self, dim: int, bn: bool = False, feats: torch.Tensor = None) -> None:
        super().__init__()
        self.head = WorldDetect(dim, bn=bn)
        self.dim = dim
        self.feats = feats

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """ultralytics' ``BaseModel.predict``, present purely to be a trap.

        The inner detection model has a ``predict`` too, with a signature that
        shares no keyword with the wrapper's. Reproducing it here is what makes
        ``attach`` pick the RIGHT of the two objects a testable property rather
        than a coincidence.
        """
        raise AssertionError("attach() bound the inner detection model, not the wrapper")

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        gen = torch.Generator().manual_seed(int(image.shape[-1]))
        if self.feats is None:
            feats = [torch.randn(1, self.dim, side, side, generator=gen) for side in (8, 4)]
        else:
            feats = [
                self.feats.view(1, -1, 1, 1).expand(1, self.dim, side, side)
                for side in (8, 4)
            ]
        return self.head(feats, torch.randn(3, self.dim, generator=gen))


class _FakeBoxes:
    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor) -> None:
        self.xyxy = xyxy
        self.conf = conf

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class _FakeWrapper(nn.Module):
    """ultralytics ``YOLOWorld`` stand-in: ``.model`` + ``.predict``."""

    def __init__(self, dim: int, boxes, bn: bool = False, feats: torch.Tensor = None) -> None:
        super().__init__()
        self.model = _FakeDetectionModel(dim, bn=bn, feats=feats)
        self._boxes = boxes
        self.calls: list[dict] = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        with torch.inference_mode():  # ultralytics runs predict this way
            self.model(torch.zeros(1, 3, 64, 64))
        return [_FakeResult(self._boxes)]


class _FakePerception:
    """``YoloWorldPerception`` stand-in: what the loop will hand to attach()."""

    def __init__(self, wrapper: _FakeWrapper, device: str = "cpu") -> None:
        self.model = wrapper
        self.device = device


def fake_perception(boxes=None, bn: bool = False, feats=None) -> _FakePerception:
    if boxes is None:
        boxes = _FakeBoxes(
            torch.tensor([[4.0, 4.0, 20.0, 20.0], [30.0, 10.0, 60.0, 40.0]]),
            torch.tensor([0.3, 0.8]),
        )
    return _FakePerception(_FakeWrapper(CFG.vis_dim, boxes, bn=bn, feats=feats))


class TestRealExtractorPlumbing:
    """attach() + extract() against the fake ultralytics-shaped model."""

    def test_extract_before_attach_raises(self):
        with pytest.raises(RuntimeError, match="attach"):
            TextRegionExtractor(CFG).extract(frame())

    def test_attach_requires_a_predictable_model(self):
        with pytest.raises(TypeError, match="predict"):
            TextRegionExtractor(CFG).attach(object())

    def test_missing_head_raises(self):
        class _Headless(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Identity()

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Headless()

            def predict(self, *a, **k):  # pragma: no cover - never reached
                return []

        with pytest.raises(RuntimeError, match="WorldDetect"):
            TextRegionExtractor(CFG).attach(_Wrapper())

    def test_missing_cv4_raises(self):
        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                head = WorldDetect(CFG.vis_dim)
                del head.cv4
                self.model = nn.Sequential(head)

            def predict(self, *a, **k):  # pragma: no cover - never reached
                return []

        with pytest.raises(RuntimeError, match="cv4"):
            TextRegionExtractor(CFG).attach(_Wrapper())

    def test_attach_adopts_the_heads_learned_temperature(self):
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(fake_perception())
        assert extractor.head_kind == "cosine"
        assert extractor.match_temperature == pytest.approx(0.05, rel=1e-4)

    def test_explicit_temperature_survives_attach(self):
        extractor = TextRegionExtractor(CFG, match_temperature=0.3, min_side=1)
        extractor.attach(fake_perception())
        assert extractor.match_temperature == pytest.approx(0.3)

    def test_extract_end_to_end(self):
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(fake_perception())
        extractor.set_text_features(phrases())
        obs = extractor.extract(frame(h=64, w=64))

        assert obs.emb.shape == (CFG.max_objects, CFG.vis_dim)
        assert obs.n_valid == 2
        # Ranked by objectness: the 0.8 box (center 45/64, 25/64) comes first.
        assert torch.allclose(obs.center[0], torch.tensor([45.0 / 64, 25.0 / 64]), atol=1e-6)
        assert float(obs.weight[0]) > float(obs.weight[1])
        assert torch.equal(obs.weight[2:], torch.zeros(CFG.max_objects - 2))
        # The hooked map is cloned out of inference_mode; an inference tensor
        # would explode the moment a trainable consumer used it.
        assert not obs.emb.is_inference()

    def test_extract_requests_class_agnostic_nms(self):
        perception = fake_perception()
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(perception)
        extractor.set_text_features(phrases())
        extractor.extract(frame(h=64, w=64))
        kwargs = perception.model.calls[-1]
        assert kwargs["agnostic_nms"] is True
        assert kwargs["half"] is False
        assert kwargs["conf"] == extractor.det_conf

    def test_extract_with_no_detections_is_all_pad(self):
        empty = _FakeBoxes(torch.zeros(0, 4), torch.zeros(0))
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(fake_perception(empty))
        extractor.set_text_features(phrases())
        obs = extractor.extract(frame(h=64, w=64))
        assert obs.n_valid == 0
        assert obs.emb.shape == (CFG.max_objects, CFG.vis_dim)

    def test_attaching_the_bare_wrapper_works(self):
        # The docstring promises attach() accepts an ultralytics YOLOWorld
        # directly, not only a YoloWorldPerception. That object's `.model` is a
        # WorldModel, which inherits BaseModel.predict — so an attach() that
        # reaches for `.model` first binds the wrong object and only fails at
        # extract() time.
        wrapper = fake_perception().model
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(wrapper)
        extractor.set_text_features(phrases())
        assert extractor.model is wrapper
        assert extractor.extract(frame(h=64, w=64)).n_valid == 2

    def test_reattaching_does_not_pile_up_hooks(self):
        # The detector is shared and frozen — the v7 box path taps the same
        # object — so a second attach() must replace this extractor's hooks,
        # not add a second set that keeps firing forever.
        perception = fake_perception()
        level = perception.model.model.head.cv4[0]
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(perception)
        after_first = len(level._forward_pre_hooks) + len(level._forward_hooks)
        extractor.attach(perception)
        assert len(level._forward_pre_hooks) + len(level._forward_hooks) == after_first
        extractor.set_text_features(phrases())
        assert extractor.extract(frame(h=64, w=64)).n_valid == 2

    def test_bn_variant_is_detected(self):
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(fake_perception(bn=True))
        assert extractor.head_kind == "bn"

    def test_bn_logit_scale_is_not_read_as_a_cosine_temperature(self):
        # BNContrastiveHead's logit_scale multiplies an UNNORMALIZED dot
        # product. On yolov8s-worldv2.pt it is 0.5801, and 1/exp(0.5801) = 0.560
        # used as a cosine temperature flattens a three-phrase softmax to a
        # measured peak of ~0.35 against a 0.333 floor — the language term in
        # `weight` would become a constant.
        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(fake_perception(bn=True))
        assert extractor.match_temperature == pytest.approx(0.07)

    def test_bn_variant_taps_the_post_norm_vector(self):
        # The head scores norm(x), not x, and the BN's affine is not a no-op:
        # measured on yolov8s-worldv2.pt the pre-BN and post-BN cosine argmax
        # over three phrases agree on only 7% / 94% / 0% of anchors per FPN
        # level. Here the BN is set to map phrase 0 exactly onto phrase 1, so a
        # tap on cv4's INPUT reports phrase 0 and the correct tap reports 1.
        text = phrases()
        perception = fake_perception(bn=True, feats=text[0])
        for head in perception.model.model.head.cv4:
            with torch.no_grad():
                head.norm.running_mean.zero_()
                head.norm.running_var.fill_(1.0 - head.norm.eps)
                head.norm.weight.fill_(1.0)
                head.norm.bias.copy_(text[1] - text[0])
        perception.model.eval()  # a loaded detector is always in eval mode

        extractor = TextRegionExtractor(CFG, min_side=1)
        extractor.attach(perception)
        extractor.set_text_features(text)
        obs = extractor.extract(frame(h=64, w=64))
        assert int(obs.match[0].argmax()) == 1

    def test_wrong_head_width_fails_loudly(self):
        # A head that no longer exposes cfg.vis_dim-wide embeddings must raise
        # rather than quietly feed logits into the canonical embedding space.
        extractor = TextRegionExtractor(replace(CFG, vis_dim=CFG.vis_dim + 1), min_side=1)
        extractor.attach(fake_perception())
        extractor.set_text_features(phrases(replace(CFG, vis_dim=CFG.vis_dim + 1)))
        with pytest.raises(RuntimeError, match="cv4"):
            extractor.extract(frame(h=64, w=64))
