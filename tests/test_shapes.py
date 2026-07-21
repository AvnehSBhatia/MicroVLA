"""Shape and range tests for every MicroVLA v2 module (CPU-only, mocks only)."""

from __future__ import annotations

import numpy as np
import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.perception.text_encoder import MockTaskEncoder
from microvla.perception.video_stream import VideoStreamSampler
from microvla.perception.yolo_world import MockYoloWorldPerception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.mock_trm import MockTRM

CFG = DEFAULT_CONFIG


def _frame(seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _fusion_inputs(batch: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    text_tokens = torch.randn(batch, CFG.n_text_tokens, CFG.text_dim, generator=g)
    frame_emb = torch.randn(batch, CFG.vis_dim, generator=g)
    source_box_emb = torch.randn(batch, CFG.vis_dim, generator=g)
    target_box_emb = torch.randn(batch, CFG.vis_dim, generator=g)
    source_center = torch.rand(batch, 2, generator=g)
    target_center = torch.rand(batch, 2, generator=g)
    return text_tokens, frame_emb, source_box_emb, target_box_emb, source_center, target_center


class TestSlotResonanceFusion:
    def test_output_shape(self):
        fusion = SlotResonanceFusion(CFG)
        fused = fusion(*_fusion_inputs(batch=2))
        assert fused.shape == (2, CFG.fused_rows, CFG.fused_cols)

    def test_batch_handling(self):
        fusion = SlotResonanceFusion(CFG)
        for batch in (1, 4):
            fused = fusion(*_fusion_inputs(batch=batch))
            assert fused.shape == (batch, CFG.fused_rows, CFG.fused_cols)
            assert torch.isfinite(fused).all()

    def test_dream_mode_with_zeroed_boxes(self):
        fusion = SlotResonanceFusion(CFG)
        batch = 2
        text_tokens = torch.randn(batch, CFG.n_text_tokens, CFG.text_dim)
        frame_emb = torch.randn(batch, CFG.vis_dim)  # corrected TRM latent in real use
        zeros_box = torch.zeros(batch, CFG.vis_dim)
        zeros_center = torch.zeros(batch, 2)

        fused = fusion(
            text_tokens,
            frame_emb,
            zeros_box,
            zeros_box,
            zeros_center,
            zeros_center,
            dream=True,
        )
        assert fused.shape == (batch, CFG.fused_rows, CFG.fused_cols)
        assert torch.isfinite(fused).all()

    def test_dream_flag_changes_output(self):
        fusion = SlotResonanceFusion(CFG)
        fusion.eval()  # avoid train-time Bernoulli dropout muddying the comparison
        inputs = _fusion_inputs(batch=2, seed=7)
        grounded = fusion(*inputs, dream=False)
        dreamed = fusion(*inputs, dream=True)
        assert not torch.allclose(grounded, dreamed)


class TestAnchoredDriftEncoder:
    def test_output_shape_and_finiteness(self):
        drift = AnchoredDriftEncoder(CFG)
        drift.reset()
        frame_emb = torch.randn(2, CFG.vis_dim)
        for _ in range(3):
            out = drift(frame_emb)
            assert out.shape == (2, CFG.state_dim)
            assert torch.isfinite(out).all()

    def test_first_call_after_reset_is_zero(self):
        drift = AnchoredDriftEncoder(CFG)
        drift.reset()
        out = drift(torch.randn(3, CFG.vis_dim))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_reset_restores_initial_state(self):
        drift = AnchoredDriftEncoder(CFG)
        g = torch.Generator().manual_seed(1)
        frames = [torch.randn(1, CFG.vis_dim, generator=g) for _ in range(3)]

        drift.reset()
        first_run = [drift(f).detach().clone() for f in frames]
        drift.reset()
        second_run = [drift(f).detach().clone() for f in frames]
        for a, b in zip(first_run, second_run):
            assert torch.allclose(a, b), "reset() must clear anchor + hidden state"

    def test_batch_size_change_is_handled(self):
        drift = AnchoredDriftEncoder(CFG)
        drift.reset()
        out1 = drift(torch.randn(1, CFG.vis_dim))
        assert out1.shape == (1, CFG.state_dim)
        # A different batch size must not crash (encoder re-resets internally).
        out2 = drift(torch.randn(4, CFG.vis_dim))
        assert out2.shape == (4, CFG.state_dim)


class TestMockTRM:
    def test_output_shape(self):
        trm = MockTRM(CFG)
        fused = torch.randn(2, CFG.fused_rows, CFG.fused_cols)
        state_delta = torch.randn(2, CFG.state_dim)
        next_emb = trm(fused, state_delta)
        assert next_emb.shape == (2, CFG.vis_dim)
        assert torch.isfinite(next_emb).all()

    def test_batch_handling(self):
        trm = MockTRM(CFG)
        for batch in (1, 4):
            next_emb = trm(
                torch.randn(batch, CFG.fused_rows, CFG.fused_cols),
                torch.randn(batch, CFG.state_dim),
            )
            assert next_emb.shape == (batch, CFG.vis_dim)


class TestChronoQueryPlanner:
    def test_output_shape_and_range(self):
        planner = ChronoQueryPlanner(CFG)
        next_emb = torch.randn(2, CFG.vis_dim)
        plan = planner(next_emb)
        assert plan.shape == (2, CFG.plan_steps, CFG.num_servos)
        assert plan.min() >= -1.0
        assert plan.max() <= 1.0

    def test_batch_handling(self):
        planner = ChronoQueryPlanner(CFG)
        for batch in (1, 4):
            plan = planner(torch.randn(batch, CFG.vis_dim))
            assert plan.shape == (batch, CFG.plan_steps, CFG.num_servos)


class TestMockTaskEncoder:
    def test_shapes_and_norms(self):
        enc = MockTaskEncoder(CFG.text_dim)
        out = enc.encode("pick up the red block")
        assert out.command_emb.shape == (CFG.text_dim,)
        assert out.source_emb.shape == (CFG.text_dim,)
        assert out.target_emb.shape == (CFG.text_dim,)
        for emb in (out.command_emb, out.source_emb, out.target_emb):
            assert emb.dtype == torch.float32
            assert abs(emb.norm().item() - 1.0) < 1e-4

    def test_tokens_stacks_in_order(self):
        enc = MockTaskEncoder(CFG.text_dim)
        out = enc.encode("push the box left")
        tokens = out.tokens()
        assert tokens.shape == (3, CFG.text_dim)
        assert torch.allclose(tokens[0], out.command_emb)
        assert torch.allclose(tokens[1], out.source_emb)
        assert torch.allclose(tokens[2], out.target_emb)

    def test_deterministic_per_text(self):
        enc = MockTaskEncoder(CFG.text_dim)
        a = enc.encode("stack the cups").tokens()
        b = enc.encode("stack the cups").tokens()
        c = enc.encode("open the drawer").tokens()
        assert torch.equal(a, b)
        assert not torch.allclose(a, c)

    def test_parsed_command_attached(self):
        enc = MockTaskEncoder(CFG.text_dim)
        out = enc.encode("move can to ball")
        assert out.parsed.source == "can"
        assert out.parsed.target == "ball"


class TestMockYoloWorldPerception:
    def test_perception_shapes(self):
        perception = MockYoloWorldPerception()
        perception.set_classes(["can", "ball"])
        p = perception.perceive(_frame(0))
        assert p.frame_emb.shape == (CFG.vis_dim,)
        for box in (p.source, p.target):
            assert box.emb.shape == (CFG.vis_dim,)
            assert box.center.shape == (2,)
            assert box.xyxy.shape == (4,)
            assert 0.0 <= box.center[0].item() <= 1.0
            assert 0.0 <= box.center[1].item() <= 1.0
            assert isinstance(box.confidence, float)

    def test_source_and_target_are_distinct_boxes(self):
        perception = MockYoloWorldPerception()
        perception.set_classes(["can", "ball"])
        p = perception.perceive(_frame(1))
        assert not torch.allclose(p.source.emb, p.target.emb)
        assert not torch.allclose(p.source.center, p.target.center)

    def test_deterministic_per_frame(self):
        perception = MockYoloWorldPerception()
        perception.set_classes(["can", "ball"])
        p1 = perception.perceive(_frame(3))
        p2 = perception.perceive(_frame(3))
        assert torch.allclose(p1.frame_emb, p2.frame_emb)
        assert torch.allclose(p1.source.emb, p2.source.emb)
        assert torch.allclose(p1.target.center, p2.target.center)


class TestVideoStreamSampler:
    def test_default_target_fps_matches_config(self):
        frames = [_frame(i) for i in range(3)]
        sampler = VideoStreamSampler(frames)
        assert sampler.target_fps == DEFAULT_CONFIG.real_frame_hz

    def test_timestamped_iterable_downsampling(self):
        # 10 frames at 10 fps (t = 0.0 .. 0.9) sampled at 2 fps ->
        # emits at t=0.0 and t=0.5 only.
        frames = [(_frame(i), i * 0.1) for i in range(10)]
        sampler = VideoStreamSampler(frames, target_fps=2.0)
        emitted = list(sampler)
        assert len(emitted) == 2
        for frame, t in emitted:
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8
            assert isinstance(t, float)
        assert emitted[0][1] == 0.0
        assert abs(emitted[1][1] - 0.5) < 1e-9

    def test_plain_frame_iterable_synthesizes_timestamps(self):
        frames = [_frame(i) for i in range(6)]
        sampler = VideoStreamSampler(frames, target_fps=2.0)
        emitted = list(sampler)
        assert len(emitted) >= 1  # at least the first frame is emitted
        frame, t = emitted[0]
        assert isinstance(frame, np.ndarray)
        assert isinstance(t, float)
