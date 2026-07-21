"""JEPA loop tests: real/dream tick cadence, corrector semantics, shapes.

CPU-only, mocks only, no network, no cv2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microvla import DEFAULT_CONFIG, InnovationCorrector, JEPALoop, TickResult

CFG = DEFAULT_CONFIG


def _frame(seed: int, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _frames(n: int) -> list[np.ndarray]:
    return [_frame(i) for i in range(n)]


class TestBuildMock:
    def test_build_mock_returns_working_loop(self):
        loop = JEPALoop.build_mock()
        assert isinstance(loop, JEPALoop)


class TestRealDreamCadence:
    def test_61_ticks_at_30fps_gives_5_real_56_dream(self):
        loop = JEPALoop.build_mock()
        results = loop.run(_frames(61), "pick up the red block")
        assert len(results) == 61

        real_ticks = int(round(CFG.tick_hz / CFG.real_frame_hz))  # 15
        expected_real_idx = set(range(0, 61, real_ticks))  # {0, 15, 30, 45, 60}
        assert expected_real_idx == {0, 15, 30, 45, 60}

        real_count = 0
        dream_count = 0
        for i, r in enumerate(results):
            assert isinstance(r, TickResult)
            if i in expected_real_idx:
                assert r.is_real is True
                assert r.perception is not None
                real_count += 1
            else:
                assert r.is_real is False
                assert r.perception is None
                dream_count += 1
        assert real_count == 5
        assert dream_count == 56


class TestTickResultShapes:
    def test_all_tick_result_shapes_and_plan_bounds(self):
        loop = JEPALoop.build_mock()
        results = loop.run(_frames(31), "grab the mug")
        assert len(results) == 31
        for r in results:
            assert r.latent.shape == (CFG.vis_dim,)
            assert r.fused.shape == (CFG.fused_rows, CFG.fused_cols)
            assert r.state_delta.shape == (CFG.state_dim,)
            assert r.next_emb.shape == (CFG.vis_dim,)
            assert r.plan.shape == (CFG.plan_steps, CFG.num_servos)
            assert r.plan.min().item() >= -1.0
            assert r.plan.max().item() <= 1.0
            assert not r.plan.requires_grad
            assert isinstance(r.trust, float)
            assert torch.isfinite(r.latent).all()
            assert torch.isfinite(r.fused).all()
            assert torch.isfinite(r.state_delta).all()
            assert torch.isfinite(r.next_emb).all()


class TestManualTick:
    def test_dream_tick_before_any_real_frame_raises(self):
        loop = JEPALoop.build_mock()
        loop.set_task("go to the kitchen")
        with pytest.raises(RuntimeError):
            loop.tick(None)

    def test_real_tick_then_dream_tick_succeeds(self):
        loop = JEPALoop.build_mock()
        loop.set_task("go to the kitchen")
        real = loop.tick(_frame(0))
        assert real.is_real is True
        assert real.perception is not None
        dream = loop.tick(None)
        assert dream.is_real is False
        assert dream.perception is None

    def test_set_task_resets_state_for_a_fresh_episode(self):
        loop = JEPALoop.build_mock()
        loop.set_task("go to the kitchen")
        loop.tick(_frame(0))
        loop.tick(None)
        # Re-set the task: a dream tick must again require a prior real frame.
        loop.set_task("go to the kitchen")
        with pytest.raises(RuntimeError):
            loop.tick(None)


class TestInnovationCorrector:
    def test_reset_defaults(self):
        corrector = InnovationCorrector(CFG)
        corrector.reset()
        assert corrector.trust == pytest.approx(1.0)

    def test_correct_without_measurement_is_identity(self):
        corrector = InnovationCorrector(CFG)
        corrector.reset()
        pred = torch.randn(CFG.vis_dim)
        out = corrector.correct(pred)
        assert torch.allclose(out, pred)

    def test_correction_decays_over_successive_dream_steps(self):
        corrector = InnovationCorrector(CFG)
        corrector.reset()
        pred = torch.zeros(CFG.vis_dim)
        real = torch.ones(CFG.vis_dim)
        corrector.on_measurement(pred, real)

        deltas = [(corrector.correct(pred) - pred).norm().item() for _ in range(4)]
        for earlier, later in zip(deltas, deltas[1:]):
            assert later < earlier

    def test_on_measurement_resets_the_decay_counter(self):
        # k (not c) is what on_measurement resets to 0: after several dream
        # steps have decayed the applied correction down, a fresh
        # measurement snaps the next correction back up to the undecayed
        # (k=0) magnitude, even though c itself keeps EMA-accumulating
        # (never reset to zero by on_measurement).
        corrector = InnovationCorrector(CFG)
        corrector.reset()
        pred = torch.zeros(CFG.vis_dim)
        real = torch.ones(CFG.vis_dim)
        corrector.on_measurement(pred, real)

        decayed = None
        for _ in range(3):
            decayed = (corrector.correct(pred) - pred).norm().item()

        corrector.on_measurement(pred, real)  # resets k -> 0
        reset_delta = (corrector.correct(pred) - pred).norm().item()
        assert reset_delta > decayed

    def test_trust_drops_for_orthogonal_prediction(self):
        pred = torch.zeros(CFG.vis_dim)
        pred[0] = 1.0

        aligned = InnovationCorrector(CFG)
        aligned.reset()
        real_aligned = torch.zeros(CFG.vis_dim)
        real_aligned[0] = 1.0
        aligned.on_measurement(pred, real_aligned)

        orthogonal = InnovationCorrector(CFG)
        orthogonal.reset()
        real_orthogonal = torch.zeros(CFG.vis_dim)
        real_orthogonal[1] = 1.0
        orthogonal.on_measurement(pred, real_orthogonal)

        assert orthogonal.trust < aligned.trust
