"""Neighbour-clearance residual (cream-cheese can-block, paper.md §5q)."""
from __future__ import annotations

import numpy as np

from microvla.utils.ibvs import clearance_aim_shift, clearance_residual


def test_clearance_repels_from_left_neighbor():
    # Source at centre, can to the left → residual should push +x (sign[0]=+1).
    out = clearance_residual(
        (0.50, 0.55),
        [(0.30, 0.55)],
        gain=0.2,
        radius=0.35,
        sign=(1.0, -1.0, 0.0),
        lift=0.1,
    )
    assert out is not None
    assert out[0] > 0.0  # away from neighbour
    assert out[2] > 0.0  # lift while close


def test_clearance_silent_when_far():
    out = clearance_residual(
        (0.50, 0.55),
        [(0.95, 0.55)],
        gain=0.2,
        radius=0.35,
    )
    assert out is None


def test_clearance_aim_shifts_toward_neighbor():
    # Can on the left → aim shifts left so grasp sits on the clear (right) edge.
    tu, tv = clearance_aim_shift(
        (0.50, 0.55),
        [(0.30, 0.55)],
        (0.50, 0.60),
        bias=0.10,
        radius=0.35,
    )
    assert tu < 0.50
    assert abs(tv - 0.60) < 0.05


def test_clearance_gain_zero_disables():
    assert clearance_residual((0.5, 0.5), [(0.3, 0.5)], gain=0.0) is None
