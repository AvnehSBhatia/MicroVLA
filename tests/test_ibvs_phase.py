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


class TestBoxTracker:
    def _tracker(self):
        from eval.ibvs_phase import _BoxTracker
        return _BoxTracker(gate=0.15, persist=3, hold_ticks=5)

    def test_small_movement_tracks(self):
        t = self._tracker()
        assert t.update((0.5, 0.5)) == (0.5, 0.5)
        assert t.update((0.51, 0.5)) == (0.51, 0.5)

    def test_teleport_rejected_and_held(self):
        t = self._tracker()
        t.update((0.5, 0.5))
        # A single teleporting fix is rejected; servo keeps the held box.
        assert t.update((0.9, 0.1)) == (0.5, 0.5)

    def test_persistent_new_location_rebinds(self):
        t = self._tracker()
        t.update((0.5, 0.5))
        t.update((0.9, 0.1))
        t.update((0.9, 0.11))
        assert t.update((0.89, 0.1)) == (0.89, 0.1)  # 3rd consistent fix wins

    def test_flicker_between_two_objects_stays_bound(self):
        t = self._tracker()
        t.update((0.5, 0.5))
        for _ in range(2):  # alternating fixes never persist long enough
            assert t.update((0.9, 0.1)) == (0.5, 0.5)
            assert t.update((0.5, 0.5)) == (0.5, 0.5)

    def test_held_box_goes_stale(self):
        t = self._tracker()
        t.update((0.5, 0.5))
        out = (0.5, 0.5)
        for i in range(7):
            out = t.update((0.9, 0.1 + i * 0.2))  # incoherent teleports
            if out is None:
                break
        assert out is None  # stale hold dropped rather than served forever

    def test_machine_with_gate_survives_flicker(self):
        m = PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                       target_uv=(0.5, 0.55), conf_floor=0.005,
                       track_gate=0.15)
        # Centered fixes interleaved with teleports: the servo keeps acting
        # on the tracked (centered) box, so descend keeps engaging.
        for i in range(6):
            center = (0.9, 0.1) if i % 2 else (0.5, 0.55)
            a = m.step(_Det(center=center), _Det(), _proprio(z=0.2))
            assert a[2] < 0  # still descending on the held/centered box


class TestClipRerank:
    def test_picks_role_matched_proposal_rejects_other_role(self):
        import torch
        from eval.ibvs_phase import clip_rerank_box

        class _P:
            def __init__(self, emb, conf, center):
                self.emb = torch.tensor(emb, dtype=torch.float32)
                self.confidence = conf
                self.center = center

        src_dir = [1.0, 0.0, 0.0]
        tgt_dir = [0.0, 1.0, 0.0]
        wrong = _P(tgt_dir, 0.9, (0.2, 0.2))   # looks like target (basket)
        right = _P(src_dir, 0.2, (0.7, 0.7))  # looks like source, lower conf
        pick = clip_rerank_box(
            [wrong, right], role_emb=src_dir, conf_floor=0.05,
            reject_emb=tgt_dir)
        assert pick is right

    def test_conf_floor_filters(self):
        import torch
        from eval.ibvs_phase import clip_rerank_box

        class _P:
            def __init__(self, emb, conf):
                self.emb = torch.tensor(emb, dtype=torch.float32)
                self.confidence = conf

        assert clip_rerank_box(
            [_P([1.0, 0.0], 0.01)], role_emb=[1.0, 0.0], conf_floor=0.05
        ) is None


class TestDescendHysteresis:
    def _machine(self, hyst=0.35):
        return PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                          target_uv=(0.5, 0.55), conf_floor=0.005,
                          descend_hyst=hyst)

    def test_ratchet_keeps_descending_through_parallax_growth(self):
        m = self._machine()
        # Engage: centered fix.
        a = m.step(_Det(center=(0.5, 0.55)), _Det(), _proprio(z=0.3))
        assert a[2] < 0
        # Parallax pushes err to 0.25 (> old 0.2 gate, < 0.35 release):
        # the old gate would stop descending; the ratchet must not.
        a = m.step(_Det(center=(0.75, 0.55)), _Det(), _proprio(z=0.2))
        assert a[2] < 0
        assert a[0] != 0.0  # still steering laterally on the way down

    def test_ratchet_releases_beyond_bound(self):
        m = self._machine()
        m.step(_Det(center=(0.5, 0.55)), _Det(), _proprio(z=0.3))
        a = m.step(_Det(center=(0.95, 0.55)), _Det(), _proprio(z=0.2))
        assert a[2] == 0.0  # err 0.45 > 0.35: released

    def test_grasp_gate_widens_with_hysteresis(self):
        m = self._machine()
        m.step(_Det(center=(0.5, 0.55)), _Det(), _proprio(z=0.3))
        # err 0.2 (> old 0.10 gate, < hyst) at grasp height -> grasp fires.
        m.step(_Det(center=(0.7, 0.55)), _Det(), _proprio(z=GRASP_Z - 0.01))
        assert m.phase == "grasp"

    def test_zero_hyst_preserves_old_gate(self):
        m = PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                       target_uv=(0.5, 0.55), conf_floor=0.005)
        a = m.step(_Det(center=(0.75, 0.55)), _Det(), _proprio(z=0.3))
        assert a[2] == 0.0  # err 0.25 > 0.2: old behavior, no descend
