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

    class _FixedTrust:
        """Corrector stub: constant trust, identity correction (isolates the
        brake formula from trust's legitimate effect on the dream latent)."""

        def __init__(self, tau: float) -> None:
            self.trust = tau

        def correct(self, pred):
            return pred

        def on_measurement(self, *args) -> None:
            pass

        def reset(self) -> None:
            pass

    def test_healthy_trust_delta_mode_does_not_attenuate(self):
        # v5.1 progressive brake: trust ABOVE cfg.brake_trust -> scale exactly
        # 1 (braking is for divergence, not a standing tax on every action);
        # trust at brake_trust/2 -> pose exactly halved. Identity-correcting
        # fixed-trust stubs make the raw plan identical across the copies, so
        # the ONLY difference is the brake scale.
        import copy

        assert 0.0 < CFG.brake_trust < 0.9
        loop = JEPALoop.build_mock(CFG)
        loop.set_task("move can to ball")
        loop.tick(_frame(0))

        plans = {}
        for tau in (0.9, 1.0, CFG.brake_trust / 2.0):
            twin = copy.deepcopy(loop)
            twin.corrector = self._FixedTrust(tau)
            plans[tau] = twin.tick(None).plan
        # Above threshold: full magnitude, identical plans at 0.9 and 1.0.
        assert torch.allclose(plans[0.9], plans[1.0], atol=1e-6)
        # At half the threshold: pose dims exactly halved vs full magnitude.
        half = plans[CFG.brake_trust / 2.0]
        assert torch.allclose(half[:, :-1], 0.5 * plans[1.0][:, :-1], atol=1e-6)

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
    """v5 miss-hold, now OPT-IN (``cfg.miss_hold``) rather than the default.

    The wrist camera loses the object exactly at approach/grasp, so holding the
    last-known box at ``cfg.miss_decay ** age`` is an appealing deployment
    behaviour. It is off by default because the CORPUS does not do it: a miss is
    baked as weight 0 at the (0.5, 0.5) fallback, so holding makes deployment
    feed the policy evidence it was never trained to read. See paper.md 4v and
    ``TestMissHoldMatchesTraining`` below.

    Turning it back on is a corpus decision, not a loop decision: the bake would
    have to hold too.
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
        import dataclasses

        loop = JEPALoop.build_mock(dataclasses.replace(CFG, miss_hold=True))
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


class TestMissHoldMatchesTraining:
    """A missed detection on a REAL tick must pass zero evidence, as baked.

    The loop used to hold a role's last-known box at ``miss_decay ** age`` when
    the detector missed on a real tick. The bake does not: ``preprocess`` writes
    weight 0 at the (0.5, 0.5) fallback for a miss and
    ``train_batched._boxes`` feeds exactly that, so the policy learned that
    weight 0 means "no evidence" while deployment handed it a confident stale
    box on the same ticks.

    Measured by A/B-ing one corpus episode through both paths with perception
    held identical (``eval/train_vs_deploy.py``): on ticks the corpus zeroed, the
    loop emitted weights 0.156 / 0.109 / 0.077 and held the previous centers,
    which moved ``fused``, ``geometry``, ``relational`` and ``wm_msg`` off
    distribution while ``current_emb``, ``proprio`` and ``state_delta`` matched
    exactly. See paper.md 4v.
    """

    def _percept(self, cfg, conf):
        import torch

        from microvla.perception.yolo_world import BoxObs, Perception

        box = lambda c, ctr: BoxObs(emb=torch.zeros(cfg.vis_dim),
                                    center=torch.tensor(ctr), xyxy=torch.zeros(4),
                                    confidence=c)
        ctr = (0.8, 0.9) if conf > 0 else (0.5, 0.5)
        return Perception(frame_emb=torch.zeros(cfg.vis_dim),
                          source=box(conf, ctr), target=box(conf, ctr))

    def _run(self, cfg):
        import numpy as np

        from microvla.jepa.loop import JEPALoop

        loop = JEPALoop.build_mock(cfg)
        seq = [0.9, 0.0, 0.0]        # a hit, then two misses
        outer = self

        class P:
            def __init__(self):
                self.i = 0

            def set_role_prompts(self, source, target=None):
                pass

            def set_classes(self, names):
                pass

            def perceive(self, _f):
                c = seq[min(self.i, len(seq) - 1)]
                self.i += 1
                return outer._percept(cfg, c)

        loop.set_task("move can to ball")
        loop.perception = P()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        return [loop.tick(frame) for _ in range(len(seq))]

    def test_miss_passes_zero_weight_by_default(self):
        from microvla.config import DEFAULT_CONFIG

        res = self._run(DEFAULT_CONFIG)
        for r in res[1:]:
            assert r.perception.source.confidence == 0.0, (
                "a missed detection carried non-zero weight; the corpus bakes "
                "0 for a miss, so this trains one thing and deploys another."
            )

    def test_miss_hold_still_available_when_explicitly_enabled(self):
        import dataclasses

        from microvla.config import DEFAULT_CONFIG

        cfg = dataclasses.replace(DEFAULT_CONFIG, miss_hold=True)
        res = self._run(cfg)
        assert res[1].perception.source.confidence > 0.0
        assert res[2].perception.source.confidence < res[1].perception.source.confidence


class TestProposalsSurviveTheLoop:
    """The class-agnostic scene must reach the relational head at deployment.

    ``Perception.proposals`` is an OPTIONAL field with an empty default, and the
    loop rebuilds ``Perception`` twice per real tick — once in ``_percept_to``
    for the device move, once for the miss-hold. Both rebuilds omitted it, so
    the deployed RelationalHead read all-zero object evidence on 100% of ticks
    while the trainer fed it the baked scene (52.7% of baked frames carry at
    least one proposal).

    An optional field with an empty default is silent when dropped: no error, no
    shape change, just evidence quietly replaced by zeros.
    """

    def test_percept_to_carries_proposals(self):
        import torch

        from microvla.jepa.loop import _percept_to
        from microvla.perception.yolo_world import BoxObs, Perception

        b = lambda c: BoxObs(emb=torch.zeros(CFG.vis_dim), center=torch.zeros(2),
                             xyxy=torch.zeros(4), confidence=c)
        p = Perception(frame_emb=torch.zeros(CFG.vis_dim), source=b(0.9),
                       target=b(0.8), proposals=(b(0.7), b(0.6)))
        out = _percept_to(p, torch.device("cpu"))
        assert len(out.proposals) == 2, "proposals dropped on the device move"
        assert [x.confidence for x in out.proposals] == [0.7, 0.6]

    def test_a_real_tick_hands_the_relational_head_nonzero_evidence(self):
        """End-to-end: the mock detector returns proposals; they must arrive."""
        import dataclasses

        import numpy as np
        import torch

        from microvla.jepa.loop import JEPALoop
        from microvla.relational import RelationalHead

        cfg = dataclasses.replace(
            CFG, planner_inputs=tuple(n for n in CFG.planner_inputs
                                      if n not in ("fused", "geometry", "pred_box_emb",
                                                   "spatial", "wm_msg", "wm_latent"))
            + ("relational",))
        loop = JEPALoop.build_mock(cfg)
        loop.relational = RelationalHead(cfg).eval()
        seen = {}
        real_fwd = loop.relational.forward
        loop.relational.forward = lambda ne, obj, ctr, w, tt, **kw: (
            seen.update(w=w.detach().clone()), real_fwd(ne, obj, ctr, w, tt, **kw))[1]

        loop.set_task("move can to ball")
        loop.tick(_frame(0))
        assert "w" in seen, "the relational head was never called on a real tick"
        assert float(seen["w"].abs().sum()) > 0.0, (
            "relational head received all-zero object weights on a real tick "
            "where the detector reported proposals"
        )
