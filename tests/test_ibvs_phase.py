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
    assert m.phase == "rise"
    assert a[6] == -1.0 and a[2] > 0  # reopen and rise to retry
    a = m.step(_Det(), _Det(), _proprio(z=0.05, jaws=1.0))
    assert a[2] > 0 and a[6] == -1.0  # rise tick
    assert m.phase == "servo_src"     # then back to the servo


class TestCalibratedAlign:
    """Hand-eye handoff (paper.md 5r): vision gates, proprio finishes."""

    def _machine(self, **kw):
        defaults = dict(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                        target_uv=(0.5, 0.55), conf_floor=0.005,
                        grasp_offset=(0.08, -0.04), close_z=0.015,
                        press=0.15, retry_rise=6)
        defaults.update(kw)
        return PhasedIBVS(**defaults)

    def test_gate_crossing_enters_align_not_grasp(self):
        m = self._machine()
        m.step(_Det(), _Det(), _proprio(z=GRASP_Z - 0.01))
        assert m.phase == "align"
        assert m._align_tgt is not None

    def test_align_servos_toward_calibrated_point_in_world(self):
        m = self._machine()
        p = _proprio(z=GRASP_Z - 0.01)
        m.step(_Det(), _Det(), p)
        # eef at origin, target at (+0.08, -0.04): drive +x, -y, hold z.
        a = m.step(_Det(), _Det(), p)
        assert a[0] > 0 and a[1] < 0
        assert a[2] == pytest.approx(0.0)
        assert a[6] == -1.0  # jaws stay open until seated

    def test_aligned_descends_to_close_z_then_grasps(self):
        m = self._machine()
        m.step(_Det(), _Det(), _proprio(z=0.05))
        # Pretend the eef arrived at the calibrated point but is still high.
        p = _proprio(z=0.05)
        p[0], p[1] = m._align_tgt
        a = m.step(_Det(), _Det(), p)
        assert m.phase == "align" and a[2] < 0  # aligned: descending
        p2 = _proprio(z=0.014)
        p2[0], p2[1] = m._align_tgt
        m.step(_Det(), _Det(), p2)
        assert m.phase == "grasp"  # below close_z

    def test_contact_stall_triggers_close_above_close_z(self):
        m = self._machine(close_z=0.0)  # unreachable: force the stall path
        m.step(_Det(), _Det(), _proprio(z=0.05))
        p = _proprio(z=0.03)
        p[0], p[1] = m._align_tgt
        for _ in range(4):  # commanded descend, z frozen: contact
            m.step(_Det(), _Det(), p)
        assert m.phase == "grasp"

    def test_press_seats_low_during_close(self):
        m = self._machine()
        m.phase = "grasp"
        a = m.step(_Det(), _Det(), _proprio(z=0.02, jaws=0.0))
        assert a[6] == 1.0 and a[2] < 0  # closing while pressing down

    def test_air_close_rises_for_retry_rise_ticks(self):
        m = self._machine()
        m.phase = "grasp"
        for _ in range(12):
            m.step(_Det(), _Det(), _proprio(z=0.02, jaws=0.0))
        assert m.phase == "rise"
        for _ in range(6):
            a = m.step(_Det(), _Det(), _proprio(z=0.05, jaws=1.0))
        assert m.phase == "servo_src"
        assert m._align_tgt is None  # stale target dropped

    def test_zero_offset_preserves_direct_grasp(self):
        m = self._machine(grasp_offset=(0.0, 0.0))
        m.step(_Det(), _Det(), _proprio(z=GRASP_Z - 0.01))
        assert m.phase == "grasp"

    def test_place_at_transports_then_lowers_then_releases(self):
        m = self._machine(place_at=(0.0, 0.30), drop_z=0.18)
        m.phase = "lift"
        p = _proprio(z=LIFT_Z + 0.01, jaws=0.6)
        m.step(_Det(), _Det(), p)
        assert m.phase == "transport"
        # Far from the basket in +y: traverse drives +y, grip stays closed.
        p2 = _proprio(z=LIFT_Z + 0.01, jaws=0.6)
        p2[0], p2[1] = 0.0, 0.0
        a = m.step(_Det(confidence=0.0), _Det(confidence=0.0), p2)
        assert a[1] > 0 and a[6] == 1.0  # vision-free traverse
        # Arrived: release lowers while still holding...
        p3 = _proprio(z=LIFT_Z + 0.01, jaws=0.6)
        p3[0], p3[1] = 0.0, 0.30
        m.step(_Det(confidence=0.0), _Det(confidence=0.0), p3)
        assert m.phase == "release"
        a = m.step(_Det(confidence=0.0), _Det(confidence=0.0), p3)
        assert a[6] == 1.0 and a[2] < 0  # lowering, not yet open
        # ...and opens once at drop height.
        p4 = _proprio(z=0.17, jaws=0.6)
        p4[0], p4[1] = 0.0, 0.30
        a = m.step(_Det(confidence=0.0), _Det(confidence=0.0), p4)
        assert a[6] == -1.0

    def test_transport_drop_restarts_pick(self):
        m = self._machine(place_at=(0.0, 0.30))
        m.phase = "transport"
        m.step(_Det(), _Det(), _proprio(z=LIFT_Z, jaws=0.0))
        assert m.phase == "servo_src"


class TestGateVerify:
    class _P:
        def __init__(self, emb, conf, center):
            import torch
            self.emb = torch.tensor(emb, dtype=torch.float32)
            self.confidence = conf
            self.center = center

    def _machine(self):
        return PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                          target_uv=(0.5, 0.55), conf_floor=0.005,
                          grasp_offset=(0.08, -0.05), gate_verify=True)

    def test_role_confusion_vetoes_gate(self):
        import torch
        m = self._machine()
        det = _Det()
        det.emb = torch.tensor([0.0, 1.0, 0.0])  # looks like the TARGET
        a = m.step(det, _Det(), _proprio(z=GRASP_Z - 0.01),
                   source_emb=[1.0, 0.0, 0.0], target_emb=[0.0, 1.0, 0.0])
        assert m.phase == "servo_src"  # vetoed
        assert a[2] > 0  # rises to reacquire

    def test_better_proposal_elsewhere_vetoes_gate(self):
        import torch
        m = self._machine()
        det = _Det(center=(0.5, 0.55))
        det.emb = torch.tensor([0.5, 0.0, 0.5])   # mediocre source match
        best = self._P([1.0, 0.0, 0.0], 0.5, (0.9, 0.2))  # true source, far
        m.step(det, _Det(), _proprio(z=GRASP_Z - 0.01), proposals=[best],
               source_emb=[1.0, 0.0, 0.0], target_emb=[0.0, 1.0, 0.0])
        assert m.phase == "servo_src"  # vetoed: cream cheese is elsewhere

    def test_correct_bind_passes_gate(self):
        import torch
        m = self._machine()
        det = _Det()
        det.emb = torch.tensor([1.0, 0.0, 0.0])
        m.step(det, _Det(), _proprio(z=GRASP_Z - 0.01),
               source_emb=[1.0, 0.0, 0.0], target_emb=[0.0, 1.0, 0.0])
        assert m.phase == "align"

    def test_repeated_vetoes_escalate_to_rerank_and_reset_clears(self):
        import torch
        m = self._machine()
        det = _Det()
        det.emb = torch.tensor([0.0, 1.0, 0.0])  # persistent role confusion
        for _ in range(3):
            m.step(det, _Det(), _proprio(z=GRASP_Z - 0.01),
                   source_emb=[1.0, 0.0, 0.0], target_emb=[0.0, 1.0, 0.0])
        assert m.clip_rerank is True   # escalated: servo now re-binds
        m.reset()
        assert m.clip_rerank is False  # per-episode escalation only

    def test_retry_probes_around_base_target_without_vision(self):
        # (lives here since the class shuffle; needs align-config machine)
        m = PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                       target_uv=(0.5, 0.55), conf_floor=0.005,
                       grasp_offset=(0.08, -0.04), close_z=0.015,
                       press=0.15, retry_rise=2)
        m.step(_Det(), _Det(), _proprio(z=GRASP_Z - 0.01))  # gate: base fixed
        base = m._base_tgt
        assert m._align_tgt == (base[0], base[1])  # attempt 0: probe dx=0
        m.phase = "grasp"
        for _ in range(12):  # close on air
            m.step(_Det(), _Det(), _proprio(z=0.02, jaws=0.0))
        assert m.phase == "rise"
        # Rise with a LOST detection: retries must not need the visual gate.
        for _ in range(2):
            m.step(_Det(confidence=0.0), _Det(), _proprio(z=0.05, jaws=1.0))
        assert m.phase == "align"
        assert m._align_tgt == (base[0] + 0.02, base[1])  # attempt 1 probes +x
        assert m._base_tgt == base  # base itself never drifts


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


def test_swap_uv_exchanges_axes():
    m = PhasedIBVS(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                   target_uv=(0.5, 0.55), conf_floor=0.005, swap_uv=True)
    # Pure u error: with swap, it must drive action[1], not action[0].
    a = m.step(_Det(center=(0.8, 0.55)), _Det(), _proprio(z=0.3))
    assert a[0] == pytest.approx(0.0) and a[1] > 0


class _DetXY(_Det):
    """Detection with pixel xyxy so box_fill_frac can fire."""

    def __init__(self, center=(0.5, 0.5), confidence=1.0, xyxy=None):
        super().__init__(center=center, confidence=confidence)
        # Default: 256x256 frame, box height = half_fill * 256 at center.
        if xyxy is None:
            cy, H, fill = float(center[1]), 256.0, 0.55
            h = fill * H
            y1 = cy * H - h / 2
            y2 = cy * H + h / 2
            xyxy = (80.0, y1, 176.0, y2)
        self.xyxy = xyxy


class TestCenterFirst:
    def _machine(self, **kw):
        defaults = dict(gain=0.5, sign=(1.0, 1.0, 0.0), descend=-0.3,
                        target_uv=(0.5, 0.5), conf_floor=0.005,
                        center_first=True, center_tol=0.06, center_persist=3,
                        half_fill=0.50, drift_tol=0.12)
        defaults.update(kw)
        return PhasedIBVS(**defaults)

    def test_off_center_never_descends(self):
        m = self._machine()
        a = m.step(_Det(center=(0.75, 0.5)), _Det(), _proprio(z=0.3))
        assert m.phase == "center_src"
        assert a[2] == pytest.approx(0.0)
        assert a[0] != 0.0  # lateral correction only

    def test_centered_persist_enters_descend(self):
        m = self._machine()
        for _ in range(3):
            a = m.step(_Det(center=(0.5, 0.5)), _Det(), _proprio(z=0.3))
            assert a[2] == pytest.approx(0.0)
        assert m.phase == "descend_src"

    def test_descend_waits_for_half_fill(self):
        m = self._machine()
        m.phase = "descend_src"
        # Small box (fill ~0.20): keep descending, do not grasp.
        small = _DetXY(center=(0.5, 0.5), xyxy=(100, 100, 140, 150))  # h=50/256≈0.20
        a = m.step(small, _Det(), _proprio(z=0.25))
        assert m.phase == "descend_src"
        assert a[2] < 0

    def test_half_fill_triggers_grasp(self):
        m = self._machine()
        m.phase = "descend_src"
        big = _DetXY(center=(0.5, 0.5))  # fill 0.55 >= 0.50
        m.step(big, _Det(), _proprio(z=0.20))
        assert m.phase == "grasp"

    def test_drift_returns_to_center(self):
        m = self._machine()
        m.phase = "descend_src"
        a = m.step(_Det(center=(0.8, 0.5)), _Det(), _proprio(z=0.25))
        assert m.phase == "center_src"
        assert a[2] > 0  # rise while re-centering

    def test_box_fill_frac_infers_height(self):
        from eval.ibvs_phase import box_fill_frac
        d = _DetXY(center=(0.5, 0.5), xyxy=(80, 64, 176, 192))  # h=128, H≈256
        assert box_fill_frac(d) == pytest.approx(0.5, abs=0.02)
