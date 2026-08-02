"""pose_magnitude_loss / gain_magnitude_loss — anti mean-collapse auxiliaries."""
from __future__ import annotations

import torch

from train.losses import gain_magnitude_loss, pose_magnitude_loss


def test_pose_magnitude_zero_when_matched():
    tgt = torch.randn(16, 5, 7)
    pred = tgt.clone()
    assert float(pose_magnitude_loss(pred, tgt)) < 1e-6


def test_pose_magnitude_penalizes_shrink():
    tgt = torch.randn(32, 5, 7) * 0.5
    pred = tgt.clone()
    pred[:, 0, :3] *= 0.2  # 5× undershoot on executed row
    assert float(pose_magnitude_loss(pred, tgt)) > 0.01


def test_gain_magnitude_moves_gains():
    gains = torch.nn.Parameter(torch.ones(4, 3) * 0.01)
    tgt = torch.zeros(4, 5, 7)
    tgt[:, 0, :3] = 0.5  # normalized
    scale = torch.ones(7) * 0.6
    loss = gain_magnitude_loss(gains, tgt, scale)
    loss.backward()
    assert gains.grad is not None
    assert float(gains.grad.abs().sum()) > 0
