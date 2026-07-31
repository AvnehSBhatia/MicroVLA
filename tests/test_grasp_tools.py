"""Unit tests for callable grasp tools (reach-to-center accuracy path)."""
from __future__ import annotations

import numpy as np

from microvla.tools import GraspToolController, ToolObs, image_error, reach_center


def _obs(src=(0.7, 0.55), conf=0.9, z=0.2):
    proprio = np.zeros(10, dtype=np.float64)
    proprio[2] = z
    proprio[7:9] = 1.0  # jaws open
    return ToolObs(src, conf, (0.5, 0.55), 0.9, proprio, action_dim=7)


def test_reach_center_cancels_image_error():
    act = reach_center(_obs(), gain=1.0, sign=(1.0, -1.0), grasp_uv=(0.5, 0.55))
    # eu = +0.2 → +dx; ev = 0
    assert act[0] == np.float32(0.2) or abs(float(act[0]) - 0.2) < 1e-5
    assert abs(float(act[1])) < 1e-6
    assert act[-1] < 0  # open while reaching


def test_image_error_none_when_low_conf():
    assert image_error(_obs(conf=0.01), conf_floor=0.05) is None
    assert image_error(_obs(conf=0.9), conf_floor=0.05) == np.float64(0.2) or abs(
        image_error(_obs(conf=0.9), conf_floor=0.05) - 0.2) < 1e-6


class _Box:
    def __init__(self, uv, conf):
        self.center = uv
        self.confidence = conf


def test_controller_stays_in_reach_until_centred():
    ctl = GraspToolController(gain=1.0, center_tol=0.04, descend_tol=0.12,
                              conf_floor=0.05, descend_rate=-0.35,
                              settle_persist=3, i_gain=0.0)
    proprio = np.zeros(10); proprio[2] = 0.20; proprio[7:9] = 1.0
    # Far off-center: stay in reach_src
    a = ctl.step(_Box((0.8, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert ctl.phase == "reach_src"
    assert a[0] > 0
    # Near enough → settle
    ctl.step(_Box((0.55, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert ctl.phase == "settle"
    # Persist inside tol → descend
    for _ in range(5):
        ctl.step(_Box((0.51, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert ctl.phase == "descend"
    # Low z + centred → grasp
    proprio[2] = 0.05
    ctl.step(_Box((0.51, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert ctl.phase == "grasp"


def test_integral_pushes_past_steady_offset():
    ctl = GraspToolController(gain=0.5, i_gain=0.5, i_clamp=0.3, settle_persist=99)
    proprio = np.zeros(10); proprio[2] = 0.2; proprio[7:9] = 1.0
    # Same small offset for many ticks — integral should grow the command.
    a0 = ctl.step(_Box((0.55, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    for _ in range(20):
        a1 = ctl.step(_Box((0.55, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert abs(float(a1[0])) > abs(float(a0[0]))
