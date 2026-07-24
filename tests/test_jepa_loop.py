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


class TestV3Behaviors:
    """Fixes from the architecture review: self-calibrating trust, plan
    hold-blending, action feedback, and held (not zeroed) dream evidence."""

    def test_trust_is_self_calibrating(self):
        """A typical-sized error keeps tau moderate; a spike tanks it."""
        corr = InnovationCorrector(CFG)
        base = torch.randn(CFG.vis_dim)
        # Establish a baseline of similar-sized innovations.
        for _ in range(5):
            corr.on_measurement(base, base + 0.1 * torch.randn(CFG.vis_dim))
        tau_baseline = corr.trust
        # A 20x error spike must produce much lower trust than baseline.
        corr.on_measurement(base, base + 2.0 * torch.randn(CFG.vis_dim))
        assert corr.trust < tau_baseline * 0.5
        # Near-zero error must push trust toward 1.
        corr.on_measurement(base, base + 1e-4 * torch.randn(CFG.vis_dim))
        assert corr.trust > 0.9

    def test_low_trust_delta_mode_brakes_toward_zero_motion(self):
        # Default action_space="delta": zero IS "no motion", so zero trust must
        # BRAKE the pose toward a stop — holding the previous plan would be
        # momentum (a held delta is a continued motion) and perpetuate drift.
        assert CFG.action_space == "delta"
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        loop.tick(_frame(0))
        loop.corrector.tau = 0.0
        dream = loop.tick(None)
        assert torch.allclose(dream.plan[:, :-1], torch.zeros_like(dream.plan[:, :-1]),
                              atol=1e-6)
        # Gripper stays a hard +/-1 decision (never a blended fraction).
        assert torch.all((dream.plan[:, -1] == 1.0) | (dream.plan[:, -1] == -1.0))

    def test_low_trust_absolute_mode_holds_previous_plan_not_zero(self):
        # action_space="absolute" (the Pi's PWM rig): zero commands servo
        # mid-range, so zero trust must HOLD the previously emitted plan.
        import dataclasses

        cfg = dataclasses.replace(CFG, action_space="absolute")
        loop = JEPALoop.build_mock(cfg)
        loop.set_task("move can to ball")
        first = loop.tick(_frame(0))
        loop.corrector.tau = 0.0
        dream = loop.tick(None)
        assert torch.allclose(dream.plan[:, :-1], first.plan[:, :-1], atol=1e-6)
        assert torch.all((dream.plan[:, -1] == 1.0) | (dream.plan[:, -1] == -1.0))
        assert dream.plan.abs().sum() > 0 or first.plan.abs().sum() == 0

    def test_plan_row0_feeds_back_as_last_action(self):
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        result = loop.tick(_frame(0))
        assert loop._last_action is not None
        assert torch.allclose(loop._last_action, result.plan[0])

    def test_dream_ticks_hold_last_real_boxes_with_decaying_weight(self):
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        real = loop.tick(_frame(0))
        assert real.perception is not None
        held_conf = real.perception.source.confidence
        loop.tick(None)
        loop.tick(None)
        # After 2 dream ticks the internal staleness counter must be 2 and
        # the held percept must still be the real tick's.
        assert loop._dream_k == 2
        assert loop._last_percept is real.perception
        expected_w = held_conf * CFG.staleness_decay**2
        assert 0.0 < expected_w < held_conf

    def test_dream_latent_is_standardized(self):
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        loop.tick(_frame(0))
        dream = loop.tick(None)
        assert abs(float(dream.latent.mean())) < 1e-3
        assert abs(float(dream.latent.std(unbiased=False)) - 1.0) < 1e-2

    def test_drift_code_held_constant_across_dream_ticks(self):
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        loop.tick(_frame(0))
        d1 = loop.tick(None).state_delta
        d2 = loop.tick(None).state_delta
        assert torch.equal(d1, d2), "drift must not step on dream ticks"

    def test_trm_context_window_fills_and_caps(self):
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        loop.tick(_frame(0))
        assert len(loop._latent_ctx) == 1
        for i in range(CFG.context_window + 3):
            loop.tick(None)
        assert len(loop._latent_ctx) == CFG.context_window
        loop.set_task("grab the mug")
        assert len(loop._latent_ctx) == 0, "set_task must clear the context window"


class TestRealTickMissHold:
    """v5: a real tick whose detection MISSES holds the last-known box.

    The wrist camera loses the object exactly at approach/grasp; before v5 a
    real-tick miss reset geometry to the (0.5, 0.5)/weight-0 fallback at the
    moment it mattered most. Now the last-known box is held per role at
    ``cfg.miss_decay ** age`` weight until the detector reacquires.
    """

    class _BlinkingPerception:
        """Delegates to the mock, but every perceive() after the first 'misses'."""

        def __init__(self, cfg):
            from microvla.perception.yolo_world import MockYoloWorldPerception

            self._mock = MockYoloWorldPerception(vis_dim=cfg.vis_dim)
            self.calls = 0

        def set_classes(self, classes):
            self._mock.set_classes(classes)

        def set_role_prompts(self, source, target=None):
            self._mock.set_role_prompts(source, target)

        def perceive(self, frame):
            from microvla.perception.yolo_world import BoxObs, Perception

            p = self._mock.perceive(frame)
            self.calls += 1
            if self.calls == 1:
                return p
            fallback = BoxObs(
                emb=p.frame_emb.clone(),
                center=torch.tensor([0.5, 0.5], dtype=torch.float32),
                xyxy=torch.zeros(4, dtype=torch.float32),
                confidence=0.0,
            )
            return Perception(frame_emb=p.frame_emb, source=fallback, target=fallback)

    def test_miss_holds_last_known_box_with_decayed_weight(self):
        loop = JEPALoop.build_mock(CFG)
        loop.perception = self._BlinkingPerception(CFG)
        loop.set_task("move can to ball")

        first = loop.tick(_frame(0))
        held_center = first.perception.source.center.clone()
        held_conf = first.perception.source.confidence
        assert held_conf > 0.0

        second = loop.tick(_frame(1))  # real tick, detector misses
        assert torch.allclose(second.perception.source.center, held_center)
        assert second.perception.source.confidence == pytest.approx(
            held_conf * CFG.miss_decay
        )

        third = loop.tick(_frame(2))  # still missing -> weight keeps decaying
        assert third.perception.source.confidence == pytest.approx(
            held_conf * CFG.miss_decay**2
        )

    def test_cold_miss_without_history_keeps_fallback(self):
        loop = JEPALoop.build_mock(CFG)
        blinker = self._BlinkingPerception(CFG)
        blinker.calls = 1  # every perceive() from now on misses, incl. the first
        loop.perception = blinker
        loop.set_task("move can to ball")
        first = loop.tick(_frame(0))
        # No last-known box exists: the fallback stands, weight 0.
        assert first.perception.source.confidence == 0.0
