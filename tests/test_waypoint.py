"""v7.2 waypoint-absolute actuation: targets, loss masking, control law, gain fit.

CPU-only, mock-only, no network — same contract as the rest of the suite.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from microvla.config import DEFAULT_CONFIG
from microvla.perception.text_encoder import MockTaskEncoder
from microvla.perception.yolo_world import MockYoloWorldPerception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.utils.waypoint import WaypointActuator, WaypointGain, waypoint_targets
from train.losses import waypoint_loss

CFG = DEFAULT_CONFIG
WP_CFG = dataclasses.replace(CFG, waypoint_action=True)


class TestWaypointTargets:
    def test_displacement_convention(self):
        """Target for plan row k is (eef[k+1] - eef[0]) / waypoint_range."""
        # A straight 1 cm/step ramp along +x, one batch, one timestep.
        chunk = torch.zeros(1, 1, CFG.plan_steps, 3)
        chunk[..., 0] = torch.arange(CFG.plan_steps, dtype=torch.float32) * 0.01
        target, row_mask = waypoint_targets(chunk, CFG.plan_steps, waypoint_range=0.05)
        # rows 0..3 supervised by steps 1..4; row 4 has no source row.
        assert row_mask.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
        expected = torch.tensor([0.01, 0.02, 0.03, 0.04]) / 0.05
        assert torch.allclose(target[0, 0, :4, 0], expected)
        assert torch.allclose(target[0, 0, :4, 1:], torch.zeros(4, 2))
        assert torch.allclose(target[0, 0, 4], torch.zeros(3))  # masked row

    def test_full_length_chunk_supervises_every_row(self):
        """A chunk with plan_steps+1 rows needs no mask (forward compatible)."""
        chunk = torch.zeros(2, 3, CFG.plan_steps + 1, 3)
        chunk[..., 1] = torch.arange(CFG.plan_steps + 1, dtype=torch.float32) * 0.02
        target, row_mask = waypoint_targets(chunk, CFG.plan_steps, waypoint_range=0.2)
        assert row_mask.tolist() == [1.0] * CFG.plan_steps
        assert torch.allclose(target[0, 0, :, 1],
                              torch.arange(1, CFG.plan_steps + 1) * 0.02 / 0.2)

    def test_clamped_to_head_range(self):
        """A move larger than waypoint_range cannot ask for |target| > 1."""
        chunk = torch.zeros(1, 1, CFG.plan_steps, 3)
        chunk[..., 2] = torch.arange(CFG.plan_steps, dtype=torch.float32) * 1.0  # metres
        target, _ = waypoint_targets(chunk, CFG.plan_steps, waypoint_range=0.05)
        assert target.max() <= 1.0 and target.min() >= -1.0


class TestWaypointLoss:
    def test_invalid_samples_contribute_nothing(self):
        """Zero-filled (no-proprio) episodes must not teach 'the arm never moves'."""
        pred = torch.full((4, 2, CFG.plan_steps, 3), 0.5)
        target = torch.zeros_like(pred)
        row_mask = torch.ones(CFG.plan_steps)
        valid = torch.zeros(4, 2)
        assert float(waypoint_loss(pred, target, row_mask, valid)) == 0.0
        # ...and with one valid sample the loss is exactly that sample's MSE.
        valid[0, 0] = 1.0
        assert float(waypoint_loss(pred, target, row_mask, valid)) == pytest.approx(0.25)

    def test_masked_rows_excluded(self):
        pred = torch.zeros(1, 1, CFG.plan_steps, 3)
        target = torch.zeros_like(pred)
        pred[0, 0, -1] = 10.0                      # only the masked row is wrong
        row_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        assert float(waypoint_loss(pred, target, row_mask)) == 0.0

    def test_stays_differentiable_when_nothing_is_supervised(self):
        pred = torch.zeros(2, 1, CFG.plan_steps, 3, requires_grad=True)
        loss = waypoint_loss(pred, torch.zeros_like(pred), torch.ones(CFG.plan_steps),
                             torch.zeros(2, 1))
        loss.backward()  # must not raise: an all-masked batch is a valid batch
        assert pred.grad is not None and float(pred.grad.abs().sum()) == 0.0


class TestPlannerWaypointHead:
    def test_head_only_exists_when_enabled(self):
        assert ChronoQueryPlanner(CFG).wp_disp_head is None
        assert ChronoQueryPlanner(WP_CFG).wp_disp_head is not None

    def test_shape_and_range(self):
        planner = ChronoQueryPlanner(WP_CFG)
        plan, grip, wp = planner(torch.randn(3, CFG.vis_dim), return_wp=True)
        assert plan.shape == (3, CFG.plan_steps, CFG.num_servos)
        assert grip.shape == (3, CFG.plan_steps)
        assert wp.shape == (3, CFG.plan_steps, CFG.waypoint_dim)
        assert wp.min() >= -1.0 and wp.max() <= 1.0

    def test_returns_none_without_the_head(self):
        _, _, wp = ChronoQueryPlanner(CFG)(torch.randn(2, CFG.vis_dim), return_wp=True)
        assert wp is None

    def test_head_is_cheap(self):
        base = sum(p.numel() for p in ChronoQueryPlanner(CFG).parameters())
        with_wp = sum(p.numel() for p in ChronoQueryPlanner(WP_CFG).parameters())
        assert with_wp - base == (CFG.d_plan + 1) * CFG.waypoint_dim


class TestWaypointActuator:
    def _act(self, **kw):
        kw.setdefault("waypoint_range", 0.1)
        kw.setdefault("horizon", 1)
        return WaypointActuator(np.array([0.05, 0.05, 0.05]), **kw)

    def test_inverts_the_gain(self):
        """command = displacement / gain: a 0.05 m move at gain 0.05 is 1.0."""
        act = self._act()
        wp = torch.zeros(CFG.plan_steps, 3)
        wp[:, 0] = 0.5                       # 0.5 * 0.1 m range = 0.05 m
        cmd = act.command(wp.numpy(), np.zeros(3), is_real=True)
        assert cmd[0] == pytest.approx(1.0, abs=1e-6)
        assert cmd[1] == pytest.approx(0.0, abs=1e-6)

    def test_horizon_selects_the_row(self):
        act = self._act(horizon=CFG.plan_steps, clip=10.0)
        wp = torch.zeros(CFG.plan_steps, 3)
        wp[:, 0] = torch.linspace(0.1, 0.5, CFG.plan_steps)
        cmd = act.command(wp.numpy(), np.zeros(3), is_real=True)
        assert cmd[0] == pytest.approx(0.5 * 0.1 / 0.05, abs=1e-5)

    def test_clip(self):
        act = self._act(clip=0.3)
        wp = torch.ones(CFG.plan_steps, 3)
        cmd = act.command(wp.numpy(), np.zeros(3), is_real=True)
        assert np.all(np.abs(cmd) <= 0.3 + 1e-6)

    def test_zero_gain_axis_does_not_explode(self):
        act = WaypointActuator(np.array([0.05, 0.0, 0.05]), waypoint_range=0.1,
                               horizon=1, clip=10.0)
        cmd = act.command(np.full((CFG.plan_steps, 3), 0.5), np.zeros(3), is_real=True)
        assert np.all(np.isfinite(cmd))

    def test_lagging_arm_keeps_being_pushed(self):
        """THE design claim: while the arm has not arrived, the command holds.

        A delta head that under-predicts makes the arm crawl forever. Anchoring
        the target in absolute coordinates and re-measuring makes the command a
        function of the REMAINING error, so a timid prediction only delays
        arrival — it does not shrink the command as the arm falls behind.
        """
        act = self._act(horizon=1, clip=10.0)
        wp = np.zeros((CFG.plan_steps, 3))
        wp[:, 0] = 0.5                                  # target: +0.05 m in x
        eef = np.zeros(3)
        first = act.command(wp, eef, is_real=True)      # anchors at 0.05 m ahead
        # The arm responds at a TENTH of the commanded speed (simulated lag).
        commands = [first]
        for _ in range(3):
            eef = eef + np.array([0.05 * 0.1 * commands[-1][0], 0.0, 0.0])
            commands.append(act.command(wp, eef, is_real=False))
        # Every follow-up command still points the same way and stays within
        # 60% of the first — no silent decay toward zero.
        assert all(c[0] > 0.6 * first[0] for c in commands[1:])
        # ...and it DOES shrink once the arm actually arrives.
        arrived = act.command(wp, np.array([0.05, 0.0, 0.0]), is_real=False)
        assert abs(arrived[0]) < 1e-6

    def test_real_tick_re_anchors(self):
        act = self._act(horizon=1, clip=10.0)
        wp = np.zeros((CFG.plan_steps, 3))
        wp[:, 0] = 0.5
        act.command(wp, np.zeros(3), is_real=True)
        moved = np.array([0.05, 0.0, 0.0])              # arm arrived
        assert abs(act.command(wp, moved, is_real=False)[0]) < 1e-6   # held target
        assert act.command(wp, moved, is_real=True)[0] > 0.5          # re-anchored


class TestWaypointGainIO:
    def test_round_trip(self, tmp_path: Path):
        g = WaypointGain(np.array([0.01, 0.02, 0.03]), np.array([0.9, 0.8, 0.7]), n=42)
        p = tmp_path / "waypoint_stats.json"
        g.save(p)
        back = WaypointGain.load(p)
        assert np.allclose(back.gain, g.gain) and back.n == 42

    def test_fit_recovers_a_known_gain(self, tmp_path: Path):
        """Synthesize episodes whose EEF responds with a known per-axis gain."""
        from preprocess.common import ActionNormalizer
        from preprocess.fit_waypoint_gain import fit_gain

        rng = np.random.default_rng(0)
        true_gain = np.array([0.02, 0.05, 0.01])
        span = np.array([1.0] * CFG.num_servos)
        ActionNormalizer(-span, span).save(tmp_path / "norm_stats.json")

        for e in range(3):
            T = 12
            raw = rng.uniform(-1.0, 1.0, size=(T, CFG.plan_steps, CFG.num_servos))
            eef = np.zeros((T, CFG.plan_steps, 3))
            for t in range(T):
                for k in range(1, CFG.plan_steps):
                    eef[t, k] = eef[t, k - 1] + true_gain * raw[t, k - 1, :3]
            proprio = np.zeros((T, 10), dtype=np.float32)
            proprio[:, -1] = 1.0
            np.savez(tmp_path / f"ep{e}.npz", pwm_targets=raw.astype(np.float32),
                     eef_pos_chunk=eef.astype(np.float32), proprio=proprio)

        out = fit_gain([tmp_path])
        assert np.allclose(out["gain"], true_gain, atol=1e-6)
        assert min(out["r2"]) > 0.999
        assert out["episodes"] == 3

    def test_fit_refuses_data_without_proprio(self, tmp_path: Path):
        from preprocess.common import ActionNormalizer
        from preprocess.fit_waypoint_gain import fit_gain

        span = np.ones(CFG.num_servos)
        ActionNormalizer(-span, span).save(tmp_path / "norm_stats.json")
        np.savez(tmp_path / "ep0.npz",
                 pwm_targets=np.zeros((4, CFG.plan_steps, CFG.num_servos), dtype=np.float32),
                 eef_pos_chunk=np.zeros((4, CFG.plan_steps, 3), dtype=np.float32),
                 proprio=np.zeros((4, 10), dtype=np.float32))   # valid flag = 0
        with pytest.raises(ValueError, match="valid proprio"):
            fit_gain([tmp_path])


class TestPolicyWaypointPath:
    """End-to-end through MicroVLAPolicy with mock perception (no sim, no net)."""

    def _policy(self, tmp_path: Path, cfg):
        from eval.policy import MicroVLAPolicy

        WaypointGain(np.full(3, 0.05), np.ones(3), n=1).save(tmp_path / "wp.json")
        return MicroVLAPolicy(
            checkpoint=None,
            norm_stats=str(Path(__file__).resolve().parents[1]
                           / "eval" / "identity_norm_stats.json"),
            cfg=cfg,
            perception=MockYoloWorldPerception(),
            task_encoder=MockTaskEncoder(),
            waypoint_stats=str(tmp_path / "wp.json"),
        )

    def test_translation_comes_from_the_actuator(self, tmp_path: Path):
        policy = self._policy(tmp_path, WP_CFG)
        assert policy.actuator is not None
        policy.reset("pick up the red block")
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        proprio = np.zeros(10, dtype=np.float32)
        proprio[-1] = 1.0
        action = policy.act(frame, proprio=proprio)
        rec = policy.telemetry[-1]
        assert rec["waypoint_cmd"] is not None
        assert np.allclose(action[:3], rec["waypoint_cmd"], atol=1e-5)

    def test_falls_back_to_the_plan_without_proprio(self, tmp_path: Path):
        policy = self._policy(tmp_path, WP_CFG)
        policy.reset("pick up the red block")
        action = policy.act(np.zeros((128, 128, 3), dtype=np.uint8), proprio=None)
        assert policy.telemetry[-1]["waypoint_cmd"] is None
        assert action.shape == (CFG.num_servos,)

    def test_disabled_without_a_trained_head(self, tmp_path: Path):
        """A gain file against a non-waypoint checkpoint warns, does not crash."""
        policy = self._policy(tmp_path, CFG)
        assert policy.actuator is None
        policy.reset("pick up the red block")
        proprio = np.zeros(10, dtype=np.float32)
        proprio[-1] = 1.0
        assert policy.act(np.zeros((128, 128, 3), dtype=np.uint8),
                          proprio=proprio).shape == (CFG.num_servos,)
