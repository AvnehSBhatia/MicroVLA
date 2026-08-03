"""Structured-control unit tests: goal heads, latch machine, label derivation.

CPU-only, mock-only. The centerpiece is a toy kinematic sim
(`_run_sim`) driving GoalServoMachine end-to-end: it must convert a
DELIBERATELY WRONG goal estimate into a completed pick-and-place via the
probe search — the mechanism the WHY analysis credits for the teacher's
robustness to its own calibration tail.
"""
import numpy as np
import pytest
import torch

from microvla.control import (GoalServoMachine, GraspPointHead, PlaceHead,
                              build_grasp_features, load_goal_heads,
                              save_goal_heads)
from train.train_goal import derive_labels


def _proprio(eef, jaw=0.02):
    return np.array([eef[0], eef[1], eef[2], 0.0, 0.0, 0.0, 1.0,
                     jaw, -jaw, 1.0], dtype=np.float64)


# --------------------------------------------------------------- machine unit
class TestMachine:
    def test_no_estimate_rises_open(self):
        m = GoalServoMachine()
        a = m.step(_proprio((0.0, 0.0, 0.3)), 7)
        assert m.phase == "approach"
        assert a[6] == -1.0 and a[2] > 0.0

    def test_none_proprio_is_safe(self):
        m = GoalServoMachine()
        a = m.step(None, 7)
        assert a.shape == (7,) and a[6] == -1.0
        assert np.all(a[:6] == 0.0)

    def test_latch_needs_stability_and_centering(self):
        m = GoalServoMachine(latch_k=3, latch_spread=0.03)
        p = _proprio((0.0, 0.0, 0.25))
        # Scattered estimates: spread 10 cm >> latch_spread -> no latch.
        for x in (0.0, 0.10, -0.10):
            m.observe((x, 0.0), 0.01, sigma=0.01)
            m.step(p, 7)
        assert m.phase == "approach" and m.base_tgt is None
        m.reset()
        for _ in range(3):
            m.observe((0.0, 0.0), 0.01, sigma=0.01)
        m.step(p, 7)                    # centered over a stable estimate
        assert m.phase == "descend"
        assert m.base_tgt == pytest.approx((0.0, 0.0))

    def test_high_sigma_estimates_rejected(self):
        m = GoalServoMachine(latch_sigma=0.05)
        for _ in range(5):
            m.observe((0.0, 0.0), 0.01, sigma=0.2)
        assert not m._est
        assert m._weak                 # ...but retained for the deadlock-breaker

    def test_force_latch_from_weak_pool(self):
        # An over-conservative sigma head must cost LATENCY, not the episode:
        # with every estimate sigma-rejected, the timeout latch still fires
        # from the weak pool, and the servo still tracks it meanwhile.
        m = GoalServoMachine(latch_sigma=0.05, force_latch_ticks=10)
        p = _proprio((0.0, 0.0, 0.25))
        for _ in range(3):
            m.observe((0.02, 0.0), 0.01, sigma=0.2)
        a = m.step(p, 7)
        assert a[0] > 0.0              # servoing toward the weak median
        for _ in range(10):
            m.step(p, 7)
        assert m.phase == "descend"
        assert m.base_tgt == pytest.approx((0.02, 0.0))

    def test_latch_is_one_way_after_freeze(self):
        m = GoalServoMachine(latch_k=1)
        m.observe((0.0, 0.0), 0.01, sigma=0.01)
        m.step(_proprio((0.0, 0.0, 0.25)), 7)
        assert m.phase == "descend"
        m.step(_proprio((0.0, 0.0, 0.05)), 7)      # below z_freeze: frozen
        m.observe((0.5, 0.5), 0.01, sigma=0.001)   # wild new estimate: ignored
        assert m.base_tgt == pytest.approx((0.0, 0.0))

    def test_force_latch_breaks_deadlock(self):
        m = GoalServoMachine(latch_k=3, force_latch_ticks=10)
        m.observe((0.2, 0.2), 0.01, sigma=0.01)    # one estimate, far away
        p = _proprio((0.0, 0.0, 0.25))
        for _ in range(11):
            m.step(p, 7)
        assert m.phase == "descend"                # timed-out latch fired

    def test_p_law_scales_and_clips(self):
        m = GoalServoMachine(latch_k=1)
        m.observe((0.0, 0.0), 0.01, sigma=0.01)
        m.step(_proprio((0.0, 0.0, 0.25)), 7)      # latch, enter descend
        far = m.step(_proprio((-0.2, 0.0, 0.25)), 7)
        assert far[0] == pytest.approx(0.6)        # clipped at 0.6
        near = m.step(_proprio((-0.01, 0.0, 0.25)), 7)
        assert near[0] == pytest.approx(0.12, abs=1e-6)   # 12 * 0.01
        m.step(_proprio((0.0, 0.0, 0.25)), 7)
        at = m.step(_proprio((0.0, 0.0, 0.25), jaw=0.02), 7)
        assert abs(at[0]) < 1e-9 and abs(at[1]) < 1e-9

    def _latched(self, **kw):
        m = GoalServoMachine(latch_k=1, **kw)
        m.observe((0.0, 0.0), 0.015, sigma=0.01)
        m.step(_proprio((0.0, 0.0, 0.25)), 7)
        assert m.phase == "descend"
        return m

    def test_descend_grasp_retry_probes(self):
        m = self._latched()
        a = m.step(_proprio((0.0, 0.0, 0.20)), 7)
        assert a[2] < 0.0                          # aligned: descending
        m.step(_proprio((0.0, 0.0, 0.01)), 7)      # z <= close_z -> grasp
        assert m.phase == "grasp"
        for _ in range(m.close_ticks):
            a = m.step(_proprio((0.0, 0.0, 0.01), jaw=0.02), 7)  # closed on air
        assert m.phase == "rise" and m.attempt == 1
        for _ in range(m.retry_rise):
            a = m.step(_proprio((0.0, 0.0, 0.10)), 7)
            assert a[6] == -1.0 and (m.phase == "rise" or a[2] >= 0.0)
        assert m.phase == "descend"
        assert m._align_tgt[0] == pytest.approx(0.02)   # probe[1] applied

    def test_grasp_hold_to_lift_transport_release_done(self):
        m = self._latched()
        m.set_place((0.0, 0.30))
        m.step(_proprio((0.0, 0.0, 0.005)), 7)
        assert m.phase == "grasp"
        for _ in range(m.close_ticks):
            m.step(_proprio((0.0, 0.0, 0.005), jaw=0.5), 7)      # jaws stalled
        assert m.phase == "lift"
        a = m.step(_proprio((0.0, 0.0, 0.31), jaw=0.5), 7)
        assert m.phase == "transport" and a[6] == 1.0
        a = m.step(_proprio((0.0, 0.30, 0.31), jaw=0.5), 7)      # at the basket
        assert m.phase == "release"
        a = m.step(_proprio((0.0, 0.30, 0.31), jaw=0.5), 7)
        assert a[6] == 1.0 and a[2] < 0.0          # lower before opening
        for _ in range(9):
            a = m.step(_proprio((0.0, 0.30, 0.10), jaw=0.5), 7)
            assert a[6] == -1.0                    # at drop height: open
        assert m.phase == "done"

    def test_first_descent_refinement_until_freeze(self):
        m = self._latched()
        # Confident estimate mid-descend (attempt 0, above z_freeze): the
        # stored goal follows vision down, EMA-blended (0.5) to damp jitter.
        m.observe((0.03, 0.01), 0.02, sigma=0.01)
        assert m.base_tgt == pytest.approx((0.015, 0.005))
        assert m._align_tgt == pytest.approx((0.015, 0.005))
        m.step(_proprio((0.015, 0.005, 0.08)), 7)   # crosses z_freeze: frozen
        m.observe((0.10, 0.10), 0.02, sigma=0.01)
        assert m.base_tgt == pytest.approx((0.015, 0.005))

    def test_refinement_wide_band_descends(self):
        # unaided_goal2 deadlock regression: while the goal refines, a 3 cm
        # lateral error must NOT stall the descent (wide band, steer down);
        # the grasp transition itself stays strict.
        m = self._latched()
        a = m.step(_proprio((0.03, 0.0, 0.20)), 7)   # 3 cm off, high up
        assert a[2] < 0.0                            # descending anyway
        assert m.phase == "descend"
        a = m.step(_proprio((0.03, 0.0, 0.02)), 7)   # 3 cm off, near close_z
        assert a[2] == 0.0 and m.phase == "descend"  # steer, don't dig/close
        m.step(_proprio((0.0, 0.0, 0.02)), 7)
        a = m.step(_proprio((0.0, 0.0, 0.01)), 7)    # centered at close height
        assert m.phase == "grasp"

    def test_align_hysteresis_keeps_contact_accumulating(self):
        # unaided_goal3 trial-1 signature: lateral error chattering at the
        # 0.015 boundary must NOT flicker the descend or starve the contact
        # window. Frozen machine: once aligned, a 2 cm excursion still
        # descends; only >2.5 cm drops alignment.
        m = self._latched()
        m.step(_proprio((0.0, 0.0, 0.05)), 7)      # below z_freeze: frozen
        assert m._aligned
        a = m.step(_proprio((0.02, 0.0, 0.045)), 7)   # inside hysteresis band
        assert a[2] < 0.0                             # still descending
        a = m.step(_proprio((0.03, 0.0, 0.045)), 7)   # beyond exit threshold
        assert a[2] == 0.0 and not m._aligned

    def test_contact_fires_through_boundary_chatter(self):
        # Sitting ON the object (z stalled ~0.04) with lateral flicking
        # between 0.01 and 0.02: contact must fire within a few ticks.
        m = self._latched()
        m.step(_proprio((0.0, 0.0, 0.05)), 7)
        for i in range(6):
            x = 0.02 if i % 2 else 0.01
            m.step(_proprio((x, 0.0, 0.041 - 0.0001 * i)), 7)
            if m.phase == "grasp":
                break
        assert m.phase == "grasp"

    def test_probe_exhaustion_restarts_fresh(self):
        m = GoalServoMachine(latch_k=1, probe_restart=2)
        m.observe((0.0, 0.0), 0.015, sigma=0.01)
        m.step(_proprio((0.0, 0.0, 0.25)), 7)
        for _ in range(2):                           # two full air-close cycles
            assert m.phase == "descend"
            tx, ty = m._align_tgt                    # stand on the probed spot
            m.step(_proprio((tx, ty, 0.01)), 7)
            assert m.phase == "grasp"
            for _ in range(m.close_ticks):
                m.step(_proprio((tx, ty, 0.01), jaw=0.02), 7)
            if m.phase == "rise":
                for _ in range(m.retry_rise):
                    m.step(_proprio((tx, ty, 0.10)), 7)
        assert m.phase == "approach" and m.base_tgt is None   # full restart

    def test_retry_descend_never_refines(self):
        m = self._latched()
        m.step(_proprio((0.0, 0.0, 0.01)), 7)       # -> grasp
        for _ in range(m.close_ticks):
            m.step(_proprio((0.0, 0.0, 0.01), jaw=0.02), 7)   # air close
        for _ in range(m.retry_rise):
            m.step(_proprio((0.0, 0.0, 0.10)), 7)
        assert m.phase == "descend" and m.attempt == 1
        probed = m._align_tgt
        m.observe((0.2, 0.2), 0.02, sigma=0.001)    # retries are proprio-only
        assert m._align_tgt == pytest.approx(probed)

    def test_transport_hang_compensation_shifts_aim(self):
        # The eef aims at place − hang so the OBJECT (hanging at eef + hang)
        # arrives over the basket center.
        m = GoalServoMachine(latch_k=1, hang_comp=(-0.03, 0.01))
        m.observe((0.0, 0.0), 0.015, sigma=0.01)
        m.step(_proprio((0.0, 0.0, 0.25)), 7)
        m.step(_proprio((0.0, 0.0, 0.005)), 7)
        for _ in range(m.close_ticks):
            m.step(_proprio((0.0, 0.0, 0.005), jaw=0.5), 7)
        m.step(_proprio((0.0, 0.0, 0.31), jaw=0.5), 7)
        assert m.phase == "transport"
        m.set_place((0.0, 0.30))
        a = m.step(_proprio((0.03, 0.29, 0.31), jaw=0.5), 7)   # at place − hang
        assert m.phase == "release"                            # object centered
        m2 = GoalServoMachine(latch_k=1, hang_comp=(-0.03, 0.01))
        m2.observe((0.0, 0.0), 0.015, sigma=0.01)
        m2.step(_proprio((0.0, 0.0, 0.25)), 7)
        m2.step(_proprio((0.0, 0.0, 0.005)), 7)
        for _ in range(m2.close_ticks):
            m2.step(_proprio((0.0, 0.0, 0.005), jaw=0.5), 7)
        m2.step(_proprio((0.0, 0.0, 0.31), jaw=0.5), 7)
        m2.set_place((0.0, 0.30))
        a = m2.step(_proprio((0.0, 0.30, 0.31), jaw=0.5), 7)   # at raw place
        assert m2.phase == "transport" and abs(a[0]) > 0.0     # keeps steering

    def test_transport_drop_restarts_unlatched(self):
        m = self._latched()
        m.set_place((0.0, 0.30))
        m.step(_proprio((0.0, 0.0, 0.005)), 7)
        for _ in range(m.close_ticks):
            m.step(_proprio((0.0, 0.0, 0.005), jaw=0.5), 7)
        m.step(_proprio((0.0, 0.0, 0.31), jaw=0.5), 7)
        assert m.phase == "transport"
        m.step(_proprio((0.0, 0.1, 0.31), jaw=0.02), 7)          # dropped it
        assert m.phase == "approach" and m.base_tgt is None


# ------------------------------------------------------ toy end-to-end pick
def _run_sim(machine, obj_xy, est_xy, max_steps=800):
    """Point-mass arm + sticky object. Returns (done, drop_xy, attempts)."""
    eef = np.array([0.05, -0.10, 0.35])
    held, drop_xy = False, None
    machine.reset()
    machine.set_place((0.0, 0.30))
    for step in range(max_steps):
        if step % 5 == 0:
            machine.observe(est_xy, 0.015, sigma=0.01)
        jaw = (0.5 if held else 0.02)
        a = machine.step(_proprio(eef, jaw=jaw), 7)
        eef[:3] += 0.02 * np.clip(a[:3], -1.0, 1.0)
        eef[2] = max(eef[2], 0.005)
        if a[6] > 0:                    # closing / holding
            if (not held and eef[2] < 0.04
                    and abs(eef[0] - obj_xy[0]) < 0.02
                    and abs(eef[1] - obj_xy[1]) < 0.02):
                held = True
        elif held:                      # opened while holding: drop it
            held = False
            drop_xy = (eef[0], eef[1])
        if machine.phase == "done":
            return True, drop_xy, machine.attempt
    return False, drop_xy, machine.attempt


class TestEndToEnd:
    def test_perfect_estimate_completes(self):
        done, drop, attempts = _run_sim(GoalServoMachine(), (0.0, 0.0), (0.0, 0.0))
        assert done and attempts == 0
        assert drop is not None
        assert abs(drop[0] - 0.0) < 0.04 and abs(drop[1] - 0.30) < 0.04

    def test_probe_search_recovers_bad_estimate(self):
        # Estimate is 3 cm off along x — outside the 2 cm capture radius, so
        # attempt 0 MUST close on air; the probe schedule must find the object.
        done, drop, attempts = _run_sim(GoalServoMachine(), (0.03, 0.0), (0.0, 0.0))
        assert done and attempts >= 1
        assert drop is not None
        assert abs(drop[0] - 0.0) < 0.04 and abs(drop[1] - 0.30) < 0.04

    def test_probe_search_recovers_y_offset(self):
        # The unaided_goal1 killer: a lateral error along Y. The x-only
        # teacher table could NEVER reach it; the 2D radius-ordered table
        # must (index 3 = (0, +0.02)).
        done, drop, attempts = _run_sim(GoalServoMachine(), (0.0, 0.03), (0.0, 0.0))
        assert done and attempts >= 3
        assert drop is not None
        assert abs(drop[0] - 0.0) < 0.04 and abs(drop[1] - 0.30) < 0.04


# ----------------------------------------------------------------- goal heads
class TestGoalHeads:
    # One fixed embedding for every sample: the unit test isolates the
    # GEOMETRIC mapping (uv, eef -> world delta). Statistical rejection of
    # uninformative embedding dims is the offline val report's job, not a
    # 2000-sample unit test's.
    _EMB = np.random.default_rng(7).normal(size=512).astype(np.float32)

    def _features(self, rng, n, obj=(0.05, -0.02), lever=(0.08, -0.05)):
        A = np.array([[0.20, 0.0], [0.0, -0.20]])
        feats, labels = [], []
        for _ in range(n):
            eef = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                            rng.uniform(0.15, 0.40)])
            uv = rng.uniform(0.2, 0.8, size=2)
            # Planted ground truth: label = eef_xy + A(uv-0.5) + lever.
            tgt = eef[:2] + A @ (uv - 0.5) + np.asarray(lever)
            f = build_grasp_features(uv=uv, conf=0.5,
                                     proprio=_proprio(eef),
                                     box_emb=self._EMB, frame_emb=self._EMB)
            feats.append(f)
            labels.append(tgt)
        batch = {k: torch.cat([f[k] for f in feats]) for k in feats[0]}
        batch["label_xy"] = torch.tensor(np.array(labels), dtype=torch.float32)
        batch["label_z"] = torch.full((n, 1), 0.01)
        return batch

    def test_forward_shapes(self):
        rng = np.random.default_rng(0)
        b = self._features(rng, 4)
        out = GraspPointHead()(b)
        assert out["xy"].shape == (4, 2) and out["z"].shape == (4, 1)
        assert out["log_var"].shape == (4, 3)

    def test_learns_planted_mapping_and_lever(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        tr, va = self._features(rng, 2000), self._features(rng, 300)
        head = GraspPointHead()
        opt = torch.optim.Adam(head.parameters(), lr=3e-3)
        for _ in range(1200):
            loss = GraspPointHead.loss(head(tr), tr["label_xy"], tr["label_z"])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            err = (head(va)["xy"] - va["label_xy"]).norm(dim=-1)
        assert float(err.median()) < 0.015          # < 1.5 cm on the planted map
        # Lever-arm READOUT: at uv == aim center the planted map reduces to
        # eef + lever, so the mean predicted delta there must recover it.
        probe = self._features(rng, 200)
        probe["geom"][:, 0:2] = 0.5                  # uv at center
        with torch.no_grad():
            delta = head(probe)["xy"] - probe["eef_xy"]
        lever = delta.mean(dim=0)
        assert float(abs(lever[0] - 0.08)) < 0.015
        assert float(abs(lever[1] + 0.05)) < 0.015

    def test_place_head_learns_constant(self):
        torch.manual_seed(0)
        emb = torch.randn(64, 512) * 0.05 + torch.randn(1, 512)
        label = torch.tensor([[-0.005, 0.257]]).repeat(64, 1)
        head = PlaceHead()
        opt = torch.optim.Adam(head.parameters(), lr=3e-3)
        for _ in range(1500):
            loss = PlaceHead.loss(head(emb), label)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            err = (head(emb)["xy"] - label).norm(dim=-1)
        assert float(err.median()) < 0.01

    def test_save_load_roundtrip(self, tmp_path):
        g, p = GraspPointHead(), PlaceHead()
        path = tmp_path / "goal.pt"
        save_goal_heads(path, g, p, meta={"k": 1})
        g2, p2, meta = load_goal_heads(path)
        assert meta == {"k": 1}
        rng = np.random.default_rng(1)
        f = build_grasp_features(uv=(0.4, 0.6), conf=0.3,
                                 proprio=_proprio((0.1, 0.0, 0.3)),
                                 box_emb=rng.normal(size=512),
                                 frame_emb=rng.normal(size=512))
        with torch.no_grad():
            assert torch.allclose(g(f)["xy"], g2(f)["xy"])


# ---------------------------------------------------------- label derivation
def _episode(T=40, probe_close=(10, 12), final_close=20, release=33,
             grasp_z=0.01):
    grip = np.zeros(T)
    grip[probe_close[0]:probe_close[1]] = 1.0       # 2-sample probe: too short
    grip[final_close:release] = 1.0
    pwm = np.zeros((T, 5, 7), dtype=np.float32)
    pwm[:, 0, -1] = grip
    proprio = np.zeros((T, 10), dtype=np.float64)
    proprio[:, 2] = np.linspace(0.35, grasp_z, T)
    proprio[final_close] = [0.11, -0.03, grasp_z, 0, 0, 0, 1, 0.4, -0.4, 1]
    proprio[release] = [-0.005, 0.257, 0.15, 0, 0, 0, 1, 0.4, -0.4, 1]
    proprio[:, 9] = 1.0
    w = np.ones((T, 2), dtype=np.float32) * 0.5
    return {"pwm_targets": pwm, "proprio": proprio, "box_weights": w}


class TestDeriveLabels:
    def test_final_close_and_release_found(self):
        lab = derive_labels(_episode())
        assert lab is not None and lab["final_close"] == 20
        assert lab["grasp_xyz"][:2] == pytest.approx((0.11, -0.03))
        assert lab["place_xy"] == pytest.approx((-0.005, 0.257))
        # Probe-close samples are excluded from supervision; pre-grasp open
        # ticks are included.
        assert lab["tick_mask"][5] and not lab["tick_mask"][10]
        assert not lab["tick_mask"][25]              # post-grasp: never

    def test_no_close_returns_none(self):
        ep = _episode()
        ep["pwm_targets"][:, 0, -1] = 0.0
        assert derive_labels(ep) is None

    def test_airborne_grasp_rejected(self):
        assert derive_labels(_episode(grasp_z=0.30)) is None
