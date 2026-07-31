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
    ctl = GraspToolController(gain=1.0, center_tol=0.05, descend_tol=0.08,
                              conf_floor=0.05, descend_rate=-0.4)
    proprio = np.zeros(10); proprio[2] = 0.20; proprio[7:9] = 1.0
    # Far off-center: must NOT enter grasp
    a = ctl.step(_Box((0.8, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert ctl.phase == "reach_src"
    assert a[0] > 0  # pushing toward center
    # Near center + low z → grasp
    proprio[2] = 0.04
    ctl.step(_Box((0.51, 0.55), 0.9), _Box((0.5, 0.55), 0.9), proprio)
    assert ctl.phase == "grasp"


def test_call_dispatches_named_tool():
    ctl = GraspToolController()
    obs = _obs()
    a = ctl.call("descend", obs)
    assert a[2] < 0
    assert ctl.last_tool == "descend"
