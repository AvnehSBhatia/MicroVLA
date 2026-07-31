"""Centering + depth IBVS-shaped stage-B aux losses."""
from __future__ import annotations

import torch

from microvla.utils.phase import grasp_place_masks
from train.losses import centering_loss, depth_loss


def _toy_batch(B=2, T=8, steps=5, servos=7):
    pwm = torch.zeros(B, T, steps, servos)
    # Episode 0: close at t=3, open at t=6. Episode 1: never closes.
    pwm[0, 3:6, 0, -1] = 1.0
    pwm[0, 6:, 0, -1] = -1.0
    src = torch.zeros(B, T, 2)
    tgt = torch.zeros(B, T, 2)
    src[:] = torch.tensor([0.5, 0.55])   # already centred
    tgt[:] = torch.tensor([0.5, 0.55])
    # Off-center at grasp window for episode 0 → centering residual fires.
    src[0, 2:5] = torch.tensor([0.7, 0.55])
    bw = torch.ones(B, T, 2)
    return pwm, src, tgt, bw


def test_grasp_place_masks_fire_around_transitions():
    pwm, *_ = _toy_batch()
    g, p = grasp_place_masks(pwm, half_window=1)
    assert g[0, 3] == 1.0 and g[1].sum() == 0.0
    assert p[0, 6] == 1.0


def test_centering_loss_zero_when_already_on_uv():
    pwm, src, tgt, bw = _toy_batch()
    src[:] = torch.tensor([0.5, 0.55])
    g, p = grasp_place_masks(pwm, half_window=1)
    plan = torch.zeros(2 * 8, 5, 7, requires_grad=True)
    target = torch.zeros_like(plan)
    loss = centering_loss(
        plan, target, src.reshape(-1, 2), tgt.reshape(-1, 2),
        g.reshape(-1), p.reshape(-1), box_weights=bw.reshape(-1, 2),
        gain=0.5)
    assert float(loss) == 0.0
    loss.backward()  # stays in the graph even when zero


def test_centering_loss_pulls_xy_toward_canceling_error():
    pwm, src, tgt, bw = _toy_batch()
    g, p = grasp_place_masks(pwm, half_window=1)
    plan = torch.zeros(2 * 8, 5, 7, requires_grad=True)
    target = torch.zeros_like(plan)
    loss = centering_loss(
        plan, target, src.reshape(-1, 2), tgt.reshape(-1, 2),
        g.reshape(-1), p.reshape(-1), box_weights=bw.reshape(-1, 2),
        grasp_uv=(0.5, 0.55), sign=(1.0, -1.0), gain=0.5)
    assert float(loss) > 0.0
    loss.backward()
    # Wanted residual on grasp steps: gain * 1.0 * (0.7-0.5) = 0.1 in x.
    # Grad on pred should push plan x toward +0.1.
    assert plan.grad is not None
    assert plan.grad[:, 0, 0].abs().sum() > 0


def test_depth_loss_silent_when_off_center():
    pwm, src, tgt, bw = _toy_batch()
    # Far from uv → descend gate closed for both roles.
    src[:] = torch.tensor([0.9, 0.9])
    tgt[:] = torch.tensor([0.9, 0.9])
    g, p = grasp_place_masks(pwm, half_window=1)
    plan = torch.zeros(2 * 8, 5, 7, requires_grad=True)
    target = torch.zeros_like(plan)
    loss = depth_loss(
        plan, target, src.reshape(-1, 2), tgt.reshape(-1, 2),
        g.reshape(-1), p.reshape(-1), box_weights=bw.reshape(-1, 2),
        descend=-0.3, descend_tol=0.2)
    assert float(loss) == 0.0


def test_depth_loss_engages_when_centred():
    pwm, src, tgt, bw = _toy_batch()
    src[:] = torch.tensor([0.5, 0.55])
    tgt[:] = torch.tensor([0.5, 0.55])
    g, p = grasp_place_masks(pwm, half_window=1)
    plan = torch.zeros(2 * 8, 5, 7, requires_grad=True)
    target = torch.zeros_like(plan)
    loss = depth_loss(
        plan, target, src.reshape(-1, 2), tgt.reshape(-1, 2),
        g.reshape(-1), p.reshape(-1), box_weights=bw.reshape(-1, 2),
        descend=-0.3, descend_tol=0.2)
    assert float(loss) > 0.0
    loss.backward()
    assert plan.grad[:, 0, 2].abs().sum() > 0
