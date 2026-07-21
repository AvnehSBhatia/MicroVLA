"""Mock end-to-end pipeline tests (CPU-only, no downloads, no cv2, v2 types)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microvla import MicroVLAPipeline, StepResult
from microvla.config import DEFAULT_CONFIG

CFG = DEFAULT_CONFIG


def _frame(seed: int, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _timestamped_frames(n: int, dt: float | None = None) -> list[tuple[np.ndarray, float]]:
    """Frames spaced exactly one sample period apart so every frame emits."""
    dt = dt if dt is not None else 1.0 / CFG.fps
    return [(_frame(i), i * dt) for i in range(n)]


class TestBuildMock:
    def test_builds_and_runs_end_to_end(self):
        pipe = MicroVLAPipeline.build_mock()
        results = pipe.run(_timestamped_frames(6), "pick up the red block")
        assert len(results) == 6
        for r in results:
            assert isinstance(r, StepResult)
            assert r.fused.shape == (CFG.fused_rows, CFG.fused_cols)
            assert r.state_delta.shape == (CFG.state_dim,)
            assert r.next_emb.shape == (CFG.vis_dim,)
            assert r.plan.shape == (CFG.plan_steps, CFG.num_servos)
            assert r.plan.min() >= -1.0
            assert r.plan.max() <= 1.0
            # StepResult tensors are unbatched and gradient-free.
            assert not r.plan.requires_grad
            # v2 dual-box perception.
            assert r.perception.frame_emb.shape == (CFG.vis_dim,)
            assert r.perception.source.emb.shape == (CFG.vis_dim,)
            assert r.perception.target.emb.shape == (CFG.vis_dim,)
            assert r.perception.source.center.shape == (2,)
            assert r.perception.target.center.shape == (2,)

    def test_max_steps_caps_output(self):
        pipe = MicroVLAPipeline.build_mock()
        results = pipe.run(_timestamped_frames(8), "stack the cups", max_steps=3)
        assert len(results) == 3

    def test_step_before_set_task_raises(self):
        pipe = MicroVLAPipeline.build_mock()
        with pytest.raises(RuntimeError):
            pipe.step(_frame(0))


class TestDriftReset:
    def test_set_task_resets_drift_state(self):
        pipe = MicroVLAPipeline.build_mock()
        frames = [_frame(i) for i in range(4)]

        pipe.set_task("push the button")
        first_run = [pipe.step(f).state_delta.clone() for f in frames]

        # Re-setting the task resets the drift anchor + hidden state, so the
        # same frames must reproduce the same state-delta sequence.
        pipe.set_task("push the button")
        second_run = [pipe.step(f).state_delta.clone() for f in frames]

        for a, b in zip(first_run, second_run):
            assert torch.allclose(a, b, atol=1e-6)

    def test_drift_evolves_across_steps(self):
        pipe = MicroVLAPipeline.build_mock()
        pipe.set_task("push the button")
        d0 = pipe.step(_frame(0)).state_delta
        d1 = pipe.step(_frame(1)).state_delta
        d2 = pipe.step(_frame(2)).state_delta
        # Distinct frames after the anchor should produce distinct drift codes.
        assert not torch.allclose(d1, d2)
        assert d0.shape == d1.shape == d2.shape == (CFG.state_dim,)


class TestMockDeterminism:
    def test_same_pipeline_same_inputs_same_outputs(self):
        pipe = MicroVLAPipeline.build_mock()
        frames = _timestamped_frames(4)

        run1 = pipe.run(frames, "pick up the red block")
        run2 = pipe.run(frames, "pick up the red block")

        for a, b in zip(run1, run2):
            assert torch.allclose(a.fused, b.fused, atol=1e-6)
            assert torch.allclose(a.state_delta, b.state_delta, atol=1e-6)
            assert torch.allclose(a.next_emb, b.next_emb, atol=1e-6)
            assert torch.allclose(a.plan, b.plan, atol=1e-6)

    def test_mock_perception_is_deterministic_across_pipelines(self):
        # Frozen-mock perception (unlike the randomly initialized heads) must
        # be reproducible across independently built pipelines.
        pipe_a = MicroVLAPipeline.build_mock()
        pipe_b = MicroVLAPipeline.build_mock()
        pipe_a.set_task("grab the mug")
        pipe_b.set_task("grab the mug")
        frame = _frame(7)
        pa = pipe_a.step(frame).perception
        pb = pipe_b.step(frame).perception
        assert torch.allclose(pa.frame_emb, pb.frame_emb)
        assert torch.allclose(pa.source.emb, pb.source.emb)
        assert torch.allclose(pa.target.emb, pb.target.emb)
        assert torch.allclose(pa.source.center, pb.source.center)
        assert torch.allclose(pa.target.center, pb.target.center)

    def test_different_text_changes_fused_output(self):
        pipe = MicroVLAPipeline.build_mock()
        frame = _frame(11)
        pipe.set_task("pick up the red block")
        fused_red = pipe.step(frame).fused
        pipe.set_task("open the drawer")
        fused_drawer = pipe.step(frame).fused
        assert not torch.allclose(fused_red, fused_drawer)
