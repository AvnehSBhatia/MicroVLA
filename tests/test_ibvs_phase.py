"""Pins the PhasedIBVS state machine contract (eval/ibvs_phase.py).

CPU-only, no torch models: the machine is pure numpy over role detections and
the canonical proprio vector.
"""
import numpy as np
import pytest

from eval.ibvs_phase import GRASP_Z, HELD_JAW_MIN, LIFT_Z, PhasedIBVS


class _Det:
    def __init__(self, center=(0.5, 0.55), confidence=1.0):
        self.center = center
        self.confidence = confidence


def _proprio(z=0.3, jaws=1.0):
    p = np.zeros(10, dtype=np.float32)
    p[2] = z
    p[7:9] = jaws
    p[9] = 1.0
    return p


def _machine():
    return PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                      target_uv=(0.5, 0.55), conf_floor=0.005)


def test_full_pick_and_place_progression():
    m = _machine()
    src, tgt = _Det(), _Det()

    # Centered + low: transitions to grasp, gripper starts closing.
    a = m.step(src, tgt, _proprio(z=GRASP_Z - 0.01), action_dim=7)
    assert m.phase == "grasp"

    # 12 close ticks with jaws stalled on an object -> lift.
    for _ in range(12):
        a = m.step(src, tgt, _proprio(z=0.03, jaws=0.6))
        assert a[6] == 1.0
    assert m.phase == "lift"

    # Rises until clear, gripper stays closed.
    a = m.step(src, tgt, _proprio(z=0.1, jaws=0.6))
    assert a[2] > 0 and a[6] == 1.0
    m.step(src, tgt, _proprio(z=LIFT_Z + 0.01, jaws=0.6))
    assert m.phase == "servo_tgt"

    # Centered on the target -> release -> done, gripper opens.
    a = m.step(src, tgt, _proprio(z=LIFT_Z + 0.01, jaws=0.6))
    assert m.phase == "release" and a[6] == 1.0
    for _ in range(9):
        a = m.step(src, tgt, _proprio(z=LIFT_Z, jaws=1.0))
        assert a[6] == -1.0
    assert m.phase == "done"


def test_closed_on_air_retries_servo():
    m = _machine()
    m.phase = "grasp"
    for _ in range(11):
        m.step(_Det(), _Det(), _proprio(z=0.03, jaws=0.0))
    a = m.step(_Det(), _Det(), _proprio(z=0.03, jaws=0.0))
    assert m.phase == "servo_src"
    assert a[6] == -1.0 and a[2] > 0  # reopen and rise to retry


def test_lost_source_rises_to_reacquire():
    m = _machine()
    a = m.step(_Det(confidence=0.0), _Det(), _proprio(z=0.3))
    assert m.phase == "servo_src"
    assert a[2] > 0 and a[6] == -1.0
    assert a[0] == 0.0 and a[1] == 0.0  # no servo without a fix


def test_dropped_object_restarts():
    m = _machine()
    m.phase = "servo_tgt"
    m.step(_Det(), _Det(), _proprio(z=LIFT_Z, jaws=0.0))
    assert m.phase == "servo_src"


def test_servo_direction_matches_sign_convention():
    m = _machine()
    # Object right-and-below the grasp point -> positive u error, positive v
    # error; with sign (1, 1, .) both action components are positive.
    a = m.step(_Det(center=(0.7, 0.8)), _Det(), _proprio(z=0.3))
    assert a[0] > 0 and a[1] > 0
    assert a[2] == pytest.approx(0.0)  # not centered: no descend


def test_grasp_needs_both_centered_and_low():
    m = _machine()
    m.step(_Det(), _Det(), _proprio(z=GRASP_Z + 0.1))
    assert m.phase == "servo_src"  # centered but too high
