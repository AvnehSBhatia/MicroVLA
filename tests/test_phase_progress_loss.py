"""phase_progress_loss unit tests (CPU, synthetic episodes).

Builds a toy pick-and-place: EEF marches toward a grasp point, gripper
closes, carries to a drop point, opens. The loss must prefer predictions
whose row-0 pose POINTS the right way and whose gripper logits time the
close/open — while being indifferent to exact magnitudes above the floor.
"""
import torch

from train.losses import phase_progress_loss


def _episode(T=20, t_close=10, t_open=16):
    grasp = torch.tensor([0.3, 0.1, 0.02])
    drop = torch.tensor([0.0, 0.35, 0.15])
    eef = torch.zeros(T, 3)
    for t in range(T):
        if t <= t_close:
            eef[t] = grasp * (t / t_close)
        elif t <= t_open:
            eef[t] = grasp + (drop - grasp) * ((t - t_close) / (t_open - t_close))
        else:
            eef[t] = drop
    target = torch.zeros(T, 5, 7)
    target[t_close:t_open, :, -1] = 1.0          # gripper closed span
    return eef, target, grasp, drop


def _aligned_preds(eef, grasp, drop, target, mag=0.5):
    T = eef.shape[0]
    preds = torch.zeros(T, 5, 7)
    closed = target[:, 0, -1] > 0
    t_close = int(closed.float().argmax())
    for t in range(T):
        goal = grasp if t < t_close else drop
        v = goal - eef[t]
        n = float(v.norm())
        if n > 1e-6:
            preds[t, 0, :3] = v / n * mag
    return preds


def test_aligned_beats_misaligned_direction():
    eef, target, grasp, drop = _episode()
    good = _aligned_preds(eef, grasp, drop, target)
    bad = good.clone()
    bad[:, 0, :3] = -bad[:, 0, :3]               # points away
    glog = torch.where(target[:, :, -1] > 0, 5.0, -5.0)  # perfect grip timing
    lg = phase_progress_loss(good[None], glog[None], target[None], eef[None])
    lb = phase_progress_loss(bad[None], glog[None], target[None], eef[None])
    assert float(lg) < float(lb)


def test_magnitude_open_above_floor():
    eef, target, grasp, drop = _episode()
    glog = torch.where(target[:, :, -1] > 0, 5.0, -5.0)
    slow = _aligned_preds(eef, grasp, drop, target, mag=0.2)
    fast = _aligned_preds(eef, grasp, drop, target, mag=0.9)
    ls = phase_progress_loss(slow[None], glog[None], target[None], eef[None])
    lf = phase_progress_loss(fast[None], glog[None], target[None], eef[None])
    # Above the floor, magnitude must NOT be penalized: near-equal losses.
    assert abs(float(ls) - float(lf)) < 1e-5


def test_frozen_policy_pays_magnitude_floor():
    eef, target, grasp, drop = _episode()
    glog = torch.where(target[:, :, -1] > 0, 5.0, -5.0)
    frozen = torch.zeros_like(_aligned_preds(eef, grasp, drop, target))
    moving = _aligned_preds(eef, grasp, drop, target, mag=0.5)
    lf = phase_progress_loss(frozen[None], glog[None], target[None], eef[None])
    lm = phase_progress_loss(moving[None], glog[None], target[None], eef[None])
    assert float(lm) < float(lf)


def test_grip_timing_dominates_window():
    eef, target, grasp, drop = _episode()
    preds = _aligned_preds(eef, grasp, drop, target)
    good = torch.where(target[:, :, -1] > 0, 5.0, -5.0)
    never = torch.full_like(good, -5.0)          # never closes
    lg = phase_progress_loss(preds[None], good[None], target[None], eef[None])
    ln = phase_progress_loss(preds[None], never[None], target[None], eef[None])
    assert float(lg) < float(ln)


def test_no_close_episode_is_finite_zero():
    eef, target, _, _ = _episode()
    target = torch.zeros_like(target)            # gripper never closes
    preds = torch.randn(1, eef.shape[0], 5, 7)
    glog = torch.randn(1, eef.shape[0], 5)
    loss = phase_progress_loss(preds, glog, target[None], eef[None])
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_gradient_flows():
    eef, target, grasp, drop = _episode()
    preds = _aligned_preds(eef, grasp, drop, target)[None].requires_grad_(True)
    glog = torch.zeros(1, eef.shape[0], 5, requires_grad=True)
    loss = phase_progress_loss(preds, glog, target[None], eef[None])
    loss.backward()
    assert preds.grad is not None and glog.grad is not None
    assert preds.grad.abs().sum() > 0 and glog.grad.abs().sum() > 0
