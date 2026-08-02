"""Phased IBVS — the falsifier finishing its job: detector-driven pick-AND-place.

``microvla/utils/ibvs.py`` measures whether frozen detection supports a servo
approach; 7f3da5a extended it to grasping. Video ground truth
(results/IBVS_SWEEP_FORENSICS.md, postscript 2) then showed the learned policy
skips the grasp phase entirely, so a residual added to its actions can never
complete the task regardless of gain. This module is the arbitration-free
version of the same question: a zero-training state machine that OWNS the
action, sequencing servo -> grasp -> lift -> servo-to-target -> release from
detector output and proprio alone.

Any nonzero ``mean_success`` here is a measurement about the FEATURES (they
carry enough geometry for the full task under trivial control), not a policy.
It is eval-only, flag-gated (``--ibvs-phase``), and never touches training or
the deployment loop.

Conventions (match the swept config, eval/libero_eval flags):
  * OSC delta actions; +z is up, ``descend`` is negative.
  * Gripper: +1 close, -1 open (``action[6]``).
  * Proprio layout ``microvla/utils/proprio.py``: ``[eef(3) | quat(4) |
    gripper_jaws(2)*25 | valid]`` — jaw width ~0..1 scaled; a closed-empty
    grip reads ~0, jaws stalled on an object read well above it.
"""
from __future__ import annotations

import numpy as np

#: Jaw reading (scaled, per proprio.py) above which a finished close is
#: holding something rather than closed on air. Full open ~1.0, closed empty
#: ~0.0; a 4-6 cm object stalls the jaws around 0.5-0.8.
HELD_JAW_MIN: float = 0.2

#: EEF height (m) below which a centered source is considered reachable by
#: closing. Sweep telemetry bottomed out at z ~ +0.01-0.05 on the table.
GRASP_Z: float = 0.06

#: Height (m) to clear before traversing to the target. Above the basket rim.
LIFT_Z: float = 0.30


def clip_rerank_box(proposals, role_emb, conf_floor: float = 0.02,
                    reject_emb=None):
    """Pick the proposal whose ROIAlign emb best matches a role text emb.

    Postscript 3 / tracking null (results/IBVS_SWEEP_FORENSICS.md): temporal
    association locks the SAME wrong box (basket for "salad dressing"). The
    remaining zero-training lever is semantic rebinding — cosine of each
    proposal's visual emb against the role's CLIP text emb, optionally
    rejecting proposals that match the OTHER role better (source vs target).

    Returns the winning proposal object, or ``None`` when nothing clears the
    floor. Pure numpy/torch-free on the hot path: callers pass tensors.
    """
    if not proposals or role_emb is None:
        return None
    import torch
    role = torch.as_tensor(role_emb, dtype=torch.float32).detach().cpu().reshape(-1)
    role = role / (role.norm() + 1e-8)
    rej = None
    if reject_emb is not None:
        rej = torch.as_tensor(reject_emb, dtype=torch.float32).detach().cpu().reshape(-1)
        rej = rej / (rej.norm() + 1e-8)
    best, best_s = None, float("-inf")
    for p in proposals:
        if float(getattr(p, "confidence", 0.0)) < conf_floor:
            continue
        e = torch.as_tensor(p.emb, dtype=torch.float32).detach().cpu().reshape(-1)
        if e.numel() != role.numel():
            continue
        e = e / (e.norm() + 1e-8)
        s = float((e * role).sum())
        if rej is not None and float((e * rej).sum()) > s:
            continue
        if s > best_s:
            best_s, best = s, p
    return best


class _BoxTracker:
    """Temporal association over per-frame argmax detections.

    Postscript 3 measured the argmax box re-binding to a DIFFERENT physical
    object on 20-26% of consecutive fixes (jump > 0.15 normalized) while
    median frame-to-frame movement was 0.008. A servo cannot converge on a
    target that teleports; this tracker holds the bound box and only accepts
    a re-bind after ``persist`` consecutive fixes agree on the new location.

    Zero training, eval-side only — perception's own role binding is
    untouched; this just decides which fix the SERVO believes.
    """

    def __init__(self, gate: float, persist: int = 5, hold_ticks: int = 15) -> None:
        self.gate = float(gate)
        self.persist = int(persist)
        self.hold_ticks = int(hold_ticks)
        self.reset()

    def reset(self) -> None:
        self._track = None
        self._cand = None
        self._cand_n = 0
        self._held = 0

    def update(self, center) -> tuple[float, float] | None:
        """Feeds one fix; returns the center the servo should believe, or
        ``None`` when there is no trustworthy fix this tick."""
        c = (float(center[0]), float(center[1]))
        if self._track is None:
            self._track = c
            self._held = 0
            return c
        d = max(abs(c[0] - self._track[0]), abs(c[1] - self._track[1]))
        if d <= self.gate:
            self._track = c
            self._cand = None
            self._cand_n = 0
            self._held = 0
            return c
        # Teleport: count agreement on the new location before re-binding.
        if (self._cand is not None
                and max(abs(c[0] - self._cand[0]), abs(c[1] - self._cand[1])) <= self.gate):
            self._cand_n += 1
        else:
            self._cand = c
            self._cand_n = 1
        if self._cand_n >= self.persist:
            self._track = c
            self._cand = None
            self._cand_n = 0
            self._held = 0
            return c
        # Reject the jump; servo the held box, but not forever — the wrist
        # camera moves with the arm, so a held center goes stale.
        self._held += 1
        if self._held > self.hold_ticks:
            self.reset()
            return None
        return self._track

    def miss(self) -> None:
        """No fix this tick; ages the held box."""
        if self._track is not None:
            self._held += 1
            if self._held > self.hold_ticks:
                self.reset()


def box_fill_frac(det) -> float | None:
    """Normalized box height in the frame, or ``None`` if unusable.

    Infers frame height from ``center`` (normalized) + ``xyxy`` (pixels):
    ``cy = (y1+y2)/(2H)`` ⇒ ``H = (y1+y2)/(2 cy)``. Used by center-first
    descend-to-half-fill: grasp when the object occupies ~half the wrist view.
    """
    xyxy = getattr(det, "xyxy", None)
    center = getattr(det, "center", None)
    if xyxy is None or center is None:
        return None
    def _as_np(x):
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy().reshape(-1)
        return np.asarray(x, dtype=np.float64).reshape(-1)

    try:
        xy = _as_np(xyxy)
        c = _as_np(center)
    except Exception:
        return None
    if xy.shape[0] < 4 or c.shape[0] < 2:
        return None
    y1, y2 = float(xy[1]), float(xy[3])
    h = y2 - y1
    if h <= 1.0:
        return None
    cy = float(c[1])
    if cy < 0.05:
        return None
    frame_h = (y1 + y2) / (2.0 * cy)
    if frame_h < h or frame_h < 8.0:
        return None
    return float(np.clip(h / frame_h, 0.0, 1.5))


class PhasedIBVS:
    """Owns the action: servo/grasp/lift/place from detections + proprio.

    One instance per episode (``reset()`` on policy reset). ``step`` returns a
    full ``[action_dim]`` raw action; the caller replaces the policy's action
    with it wholesale — no blending, so the learned prior cannot drag the
    arm off-task (the failure mode that nulled the residual sweeps).
    """

    def __init__(self, gain: float, sign: tuple[float, float, float],
                 descend: float, target_uv: tuple[float, float],
                 conf_floor: float, track_gate: float = 0.0,
                 clip_rerank: bool = False,
                 descend_hyst: float = 0.0,
                 swap_uv: bool = False,
                 center_first: bool = False,
                 center_tol: float = 0.06,
                 center_persist: int = 6,
                 half_fill: float = 0.50,
                 drift_tol: float = 0.12,
                 grasp_offset: tuple[float, float] = (0.0, 0.0),
                 close_z: float = GRASP_Z,
                 press: float = 0.0,
                 retry_rise: int = 1,
                 yaw_probe: bool = False,
                 yaw_sign: float = 1.0,
                 place_at: tuple[float, float] | None = None,
                 drop_z: float = 0.18,
                 gate_z: float = GRASP_Z,
                 approach_z: float = 0.0,
                 gate_verify: bool = False,
                 body_v: float = 1.0) -> None:
        self.gain = float(gain)
        self.sign = tuple(float(v) for v in sign)
        self.descend = float(descend)
        self.target_uv = (float(target_uv[0]), float(target_uv[1]))
        self.conf_floor = float(conf_floor)
        self.track_gate = float(track_gate)
        self.clip_rerank = bool(clip_rerank)
        self._clip_rerank_init = bool(clip_rerank)
        self.descend_hyst = float(descend_hyst)
        self.swap_uv = bool(swap_uv)
        self.center_first = bool(center_first)
        self.center_tol = float(center_tol)
        self.center_persist = int(center_persist)
        self.half_fill = float(half_fill)
        self.drift_tol = float(drift_tol)
        # Hand-eye correction (paper.md 5r): the converged eef sits a CONSTANT
        # world offset from the object across every aim-uv sweep arm (the servo
        # gate fires on z-crossing, not image convergence), so the fix is a
        # proprio-only world nudge of `grasp_offset` metres at gate crossing,
        # then descend to `close_z` (not GRASP_Z) before closing.
        self.grasp_offset = (float(grasp_offset[0]), float(grasp_offset[1]))
        self.close_z = float(close_z)
        self.press = float(press)
        self.retry_rise = max(1, int(retry_rise))
        # Residual-miss probe (paper.md 5r): the calibrated constant is right
        # on average but has ~±3 cm trial variance along x (the approach
        # axis). Each air close is a bit of feedback; successive attempts
        # probe the offset's high-variance axis deterministically. With
        # yaw_probe the schedule alternates a ~90° wrist rotation too — the
        # cream cheese box is 8.1 cm along its long axis vs the ~8 cm jaw
        # span, so a probe that never rotates can be millimetre-perfect in
        # position and still close on the ungraspable axis.
        self.yaw_probe = bool(yaw_probe)
        self.yaw_sign = float(yaw_sign)
        # Calibrated place point (paper.md 5s): the basket sits at a FIXED
        # world xy across all 50 demos (std < 2.5 cm), while the wrist camera
        # rarely sees it at altitude (tgt duty ~0 in eval) — so the place
        # leg, like the last-centimetre pick, is proprio's job, not vision's.
        self.place_at = (None if place_at is None
                         else (float(place_at[0]), float(place_at[1])))
        self.drop_z = float(drop_z)
        # Object-geometry constants (from demo statistics, per task): gate_z
        # is the height crossing that ends the visual approach — the default
        # GRASP_Z=0.06 can never fire on a 12 cm bottle; approach_z keeps the
        # align servo ABOVE a standing object so the lateral correction flies
        # over it instead of bulldozing it sideways.
        self.gate_z = float(gate_z)
        self.approach_z = float(approach_z)
        # Gate-time bind verification: the gate freezes the align target, so
        # a wrong-object bind at that ONE tick decides the episode (3/10 v4
        # failures). Veto the gate unless the bound box's ROIAlign embedding
        # matches the SOURCE phrase strictly better than the TARGET phrase.
        # Unlike --ibvs-clip-rerank (continuous rebinding, measured a wash),
        # this runs once, exactly where a wrong answer is unrecoverable.
        self.gate_verify = bool(gate_verify)
        # Self-body mask: the wrist camera always sees the robot's OWN finger
        # tabs in the bottom CORNERS (v ≈ 0.75–0.95 at u ≈ 0.15 / 0.85). On
        # the salad-dressing task the detector bound a FINGER as the bottle
        # (duty 1.000, u flipping 0.13↔0.87 at v 0.80) and the servo chased
        # its own hand for 600 ticks. A fix is discarded only when it is BOTH
        # below this v line AND inside a corner u-band — a full-width mask
        # measured 0.000 on cream (handeye_v6cream): the real box lives at
        # bottom-center during descent (aim V 0.60) and was masked with the
        # fingers. 1.0 = off; 0.72 recommended.
        self.body_v = float(body_v)
        if self.yaw_probe:
            self.probe = ((0.0, 0.0), (0.0, 1.5708), (+0.02, 0.0),
                          (+0.02, 1.5708), (-0.02, 0.0), (-0.02, 1.5708),
                          (+0.04, 1.5708))
        else:
            self.probe = tuple((dx, 0.0) for dx in
                               (0.0, +0.02, -0.02, +0.04, -0.04, +0.06))
        self.reset()

    @staticmethod
    def _yaw(proprio) -> float:
        """World yaw of the eef from the canonical proprio quat (x,y,z,w)."""
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        x, y, z, w = p[3], p[4], p[5], p[6]
        return float(np.arctan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))

    def reset(self) -> None:
        # center_first starts in center_src (XY-only); legacy starts in servo_src.
        self.phase = "center_src" if self.center_first else "servo_src"
        self._close_ticks = 0
        self._release_ticks = 0
        self._descending = False
        self._center_ok = 0
        self._align_tgt = None
        self._base_tgt = None
        self._base_yaw = 0.0
        self._tgt_yaw = None
        self._rise_ticks = 0
        self._z_hist: list[float] = []
        self._attempt = 0
        self._veto_n = 0
        self._rerank_escalated = False
        self.clip_rerank = getattr(self, "_clip_rerank_init", self.clip_rerank)
        self._trackers = ({"src": _BoxTracker(self.track_gate),
                           "tgt": _BoxTracker(self.track_gate)}
                          if self.track_gate > 0.0 else None)

    def _believe(self, role: str, det) -> tuple[float, float] | None:
        """The center the servo should act on for ``role``, after tracking."""
        in_finger_corner = (
            float(det.center[1]) > self.body_v
            and (float(det.center[0]) < 0.28 or float(det.center[0]) > 0.72))
        if float(det.confidence) < self.conf_floor or in_finger_corner:
            if self._trackers is not None:
                self._trackers[role].miss()
            return None
        if self._trackers is None:
            return (float(det.center[0]), float(det.center[1]))
        return self._trackers[role].update(det.center)

    # ------------------------------------------------------------------ utils
    def _servo_xy(self, center, out: np.ndarray) -> float:
        """Writes the xy servo into ``out``; returns the image error (inf-norm)."""
        eu = float(center[0]) - self.target_uv[0]
        ev = float(center[1]) - self.target_uv[1]
        # Measured (postscript 5): with true-object boxes, NO sign combination
        # reduces image error — min err 0.204 over 618 fixes, mean flat across
        # the episode. Sign flips span reflections only; a wrist camera rolled
        # 90° relative to base xy needs the axes SWAPPED (image-u error moves
        # the arm along world-y and vice versa). swap+signs spans all 8
        # dihedral mappings.
        a, b = (ev, eu) if self.swap_uv else (eu, ev)
        out[0] = self.gain * self.sign[0] * a
        out[1] = self.gain * self.sign[1] * b
        return max(abs(eu), abs(ev))

    @staticmethod
    def _jaws(proprio) -> float:
        # ABS before the mean: robosuite's panda finger joints are MIRRORED
        # (qpos = +q, -q), so the raw mean is identically ZERO in every state
        # — open, closed, or stalled on an object. The unsigned mean is the
        # actual jaw half-width signal this check was written against.
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        return float(np.abs(p[7:9]).mean()) if p.shape[0] >= 9 else 0.0

    @staticmethod
    def _eef_z(proprio) -> float:
        return float(np.asarray(proprio, dtype=np.float64).reshape(-1)[2])

    def _bind_ok(self, det, src_emb, tgt_emb, proposals=()) -> bool:
        """True when the bound box is the best source candidate in view.

        Two vetoes: (1) the box matches the TARGET phrase at least as well as
        the SOURCE phrase (role confusion); (2) some OTHER visible proposal
        matches the source phrase better AND sits somewhere else (distractor
        bind — the soup-can-instead-of-cream-cheese failure on film).
        """
        if not self.gate_verify or src_emb is None:
            return True
        e = getattr(det, "emb", None)
        if e is None:
            return True
        import torch

        def _unit(x):
            v = torch.as_tensor(x, dtype=torch.float32).detach().cpu().reshape(-1)
            return v / (v.norm() + 1e-8)

        v, s = _unit(e), _unit(src_emb)
        score_s = float((v * s).sum())
        if tgt_emb is not None:
            if float((_unit(tgt_emb) * v).sum()) >= score_s:
                return False
        if proposals:
            best = clip_rerank_box(proposals, src_emb,
                                   conf_floor=self.conf_floor,
                                   reject_emb=tgt_emb)
            if best is not None and getattr(best, "emb", None) is not None:
                bs = float((_unit(best.emb) * s).sum())
                bc = np.asarray(
                    best.center.detach().cpu() if hasattr(best.center, "detach")
                    else best.center, dtype=np.float64).reshape(-1)[:2]
                dc = np.asarray(
                    det.center.detach().cpu() if hasattr(det.center, "detach")
                    else det.center, dtype=np.float64).reshape(-1)[:2]
                if bs > score_s and float(np.abs(bc - dc).max()) > 0.10:
                    return False
        return True

    def _escalate_veto(self) -> None:
        """Repeated gate vetoes escalate to CLIP re-binding.

        A veto proves the perception-side bind is wrong but refuses forever
        if the binding never changes (v5: veto-blocked episodes hover at
        grip 0.000). After 3 vetoes the machine turns on the existing
        clip_rerank path for the rest of the episode, so the SERVO starts
        chasing the proposal that actually matches the source phrase.
        """
        self._veto_n += 1
        if self._veto_n >= 3 and not self._rerank_escalated:
            self._rerank_escalated = True
            self.clip_rerank = True

    def _enter_grasp_or_align(self, proprio) -> None:
        """Gate crossing: with a calibrated offset, align in world first."""
        if (self.grasp_offset != (0.0, 0.0)) and proprio is not None:
            dx, dyaw = self.probe[min(self._attempt, len(self.probe) - 1)]
            if self._base_tgt is None:
                # First gate crossing this episode: derive from the eef. The
                # calibrated lever arm is only valid from the ALTITUDE
                # approach — later retries fire the gate from near the table,
                # so they must probe around this stored target instead of
                # re-deriving (which would double-apply the offset).
                p = np.asarray(proprio, dtype=np.float64).reshape(-1)
                self._base_tgt = (float(p[0]) + self.grasp_offset[0],
                                  float(p[1]) + self.grasp_offset[1])
                self._base_yaw = self._yaw(proprio)
            self._align_tgt = (self._base_tgt[0] + dx, self._base_tgt[1])
            self._tgt_yaw = self._base_yaw + dyaw
            self._z_hist = []
            self.phase = "align"
        else:
            self.phase = "grasp"
        self._close_ticks = 0

    def _retry(self) -> None:
        """Air close: rise properly, then re-acquire (paper.md 5r: the old
        1-tick retry re-closed at the same wrong spot ~12x per episode)."""
        self._align_tgt = None
        self._rise_ticks = 0
        self._attempt += 1
        self.phase = "rise"

    def _contact(self, z: float) -> bool:
        """True when a commanded descend stopped moving the eef (table/object)."""
        self._z_hist.append(z)
        if len(self._z_hist) > 4:
            self._z_hist.pop(0)
        return (len(self._z_hist) == 4 and z < self.gate_z
                and (self._z_hist[0] - z) < 0.002)

    # ------------------------------------------------------------------- step
    def step(self, source, target, proprio, action_dim: int = 7,
             proposals=(), source_emb=None, target_emb=None) -> np.ndarray:
        """One control tick.

        Args:
            source / target: role detections with ``.center`` (normalized
                ``[0,1]^2``) and ``.confidence`` (0.0 when absent this tick).
            proprio: canonical ``[10]`` proprio vector (required — the real
                env always has it; without proprio there is no grasp check
                and the machine just servos).
            action_dim: output width; translation + zeros + gripper.
            proposals: optional class-agnostic boxes for CLIP re-rank.
            source_emb / target_emb: CLIP text embeddings for re-rank
                (``TaskEncoding.source_emb`` / ``.target_emb``).
        """
        self._last_proposals = proposals
        self._last_src_emb = source_emb
        self._last_tgt_emb = target_emb
        if self.clip_rerank and proposals:
            reb_src = clip_rerank_box(
                proposals, source_emb, conf_floor=self.conf_floor,
                reject_emb=target_emb)
            if reb_src is not None:
                source = reb_src
            reb_tgt = clip_rerank_box(
                proposals, target_emb, conf_floor=self.conf_floor,
                reject_emb=source_emb)
            if reb_tgt is not None:
                target = reb_tgt
        out = np.zeros(int(action_dim), dtype=np.float32)
        grip = out.shape[0] - 1
        z = self._eef_z(proprio) if proprio is not None else 1.0

        if self.phase == "rise":
            # Reopen and genuinely clear the failed close before retrying.
            out[grip] = -1.0
            out[2] = 0.35
            self._rise_ticks += 1
            if self._rise_ticks >= self.retry_rise:
                if self._base_tgt is not None:
                    # Vision already did its one job (the first gate). This
                    # close to the table the detector is unreliable, so later
                    # attempts go straight back to the probe-shifted world
                    # target on proprio alone.
                    dx, dyaw = self.probe[min(self._attempt, len(self.probe) - 1)]
                    self._align_tgt = (self._base_tgt[0] + dx, self._base_tgt[1])
                    self._tgt_yaw = self._base_yaw + dyaw
                    self._z_hist = []
                    self.phase = "align"
                else:
                    self.phase = "center_src" if self.center_first else "servo_src"
                    self._center_ok = 0
            return out

        if self.phase == "align":
            # Proprio-only world servo to the calibrated grasp point: the
            # camera-gripper lever arm is constant, so once the visual gate
            # fires the detector has done its job — vision hands off to
            # proprioception. No ground truth: the offset is a constant
            # calibrated offline from logged runs (paper.md 5r).
            out[grip] = -1.0
            if proprio is None or self._align_tgt is None:
                self.phase = "grasp"
                self._close_ticks = 0
                return out
            p = np.asarray(proprio, dtype=np.float64).reshape(-1)
            ex = self._align_tgt[0] - float(p[0])
            ey = self._align_tgt[1] - float(p[1])
            lateral = max(abs(ex), abs(ey))
            out[0] = float(np.clip(12.0 * ex, -0.6, 0.6))
            out[1] = float(np.clip(12.0 * ey, -0.6, 0.6))
            yaw_ok = True
            if self.yaw_probe and self._tgt_yaw is not None:
                yerr = self._tgt_yaw - self._yaw(proprio)
                yerr = float(np.arctan2(np.sin(yerr), np.cos(yerr)))
                out[5] = float(np.clip(self.yaw_sign * 2.0 * yerr, -0.5, 0.5))
                yaw_ok = abs(yerr) < 0.15
            if lateral > 0.015 and z < self.approach_z:
                # Standing object: fly the lateral correction ABOVE it.
                out[2] = 0.3
                self._z_hist = []
                return out
            if lateral < 0.015 and yaw_ok:
                # Over the object: descend to close height (or contact stall).
                out[2] = self.descend if self.descend != 0.0 else -0.4
                if z <= self.close_z or self._contact(z):
                    self.phase = "grasp"
                    self._close_ticks = 0
            else:
                self._z_hist = []
            return out

        if self.phase == "center_src":
            # XY-only: kill the ~5 cm lateral near-miss before any descent.
            # Watch clips showed cream cheese right of frame while the gripper
            # closed beside it — band-descend (hyst) was engaging too early.
            out[grip] = -1.0
            center = self._believe("src", source)
            if center is not None:
                err = self._servo_xy(center, out)
                if err < self.center_tol:
                    self._center_ok += 1
                else:
                    self._center_ok = 0
                if self._center_ok >= self.center_persist:
                    self.phase = "descend_src"
            else:
                out[2] = 0.15
                self._center_ok = 0
            return out

        if self.phase == "descend_src":
            # Descend while correcting XY. Grasp when the box fills ~half the
            # wrist frame ("only half the object visible" as we close in), or
            # fall back to GRASP_Z if xyxy is missing.
            out[grip] = -1.0
            center = self._believe("src", source)
            if center is None:
                out[2] = 0.15
                self.phase = "center_src"
                self._center_ok = 0
                return out
            err = self._servo_xy(center, out)
            if err > self.drift_tol:
                # Drifted off while descending: re-center at altitude.
                self.phase = "center_src"
                self._center_ok = 0
                out[2] = 0.12
                return out
            out[2] = self.descend
            fill = box_fill_frac(source)
            ready = False
            if fill is not None and fill >= self.half_fill and err < self.center_tol:
                ready = True
            elif fill is None and z < self.gate_z and err < self.center_tol:
                ready = True
            if ready:
                if self._bind_ok(source, source_emb, target_emb, proposals):
                    self._enter_grasp_or_align(proprio)
                else:
                    out[2] = 0.15  # vetoed bind: rise, reacquire
                    self._escalate_veto()
            return out

        if self.phase == "servo_src":
            out[grip] = -1.0
            center = self._believe("src", source)
            if center is not None:
                err = self._servo_xy(center, out)
                if self.descend_hyst > 0.0:
                    # Band descend (postscript 5): the dihedral null showed
                    # wrist-image error is height-dominated — it hovers at
                    # 0.20–0.28 and CANNOT shrink without descending, while
                    # the old err<0.2 engage gate waits on it: a deadlock.
                    # So descend whenever the error is inside the (wide)
                    # band, steering laterally the whole way down; the grasp
                    # gate widens to the band too.
                    self._descending = err < self.descend_hyst
                    if self._descending:
                        out[2] = self.descend
                    grasp_err = self.descend_hyst
                else:
                    if err < 0.2:  # centered enough: descend, scaled by centering
                        out[2] = self.descend * (1.0 - err / 0.2)
                    grasp_err = 0.10
                if err < grasp_err and z < self.gate_z:
                    if self._bind_ok(source, source_emb, target_emb, proposals):
                        self._enter_grasp_or_align(proprio)
                    else:
                        out[2] = 0.15  # vetoed bind: rise, reacquire
                        self._escalate_veto()
            else:
                # Lost the source: rise to widen the wrist view and reacquire.
                out[2] = 0.15
            return out

        if self.phase == "grasp":
            out[grip] = 1.0  # close; press keeps the fingers seating low
            if self.press != 0.0 and self._close_ticks < 6:
                out[2] = -abs(self.press)
            self._close_ticks += 1
            if self._close_ticks >= 12:
                if proprio is not None and self._jaws(proprio) >= HELD_JAW_MIN:
                    self.phase = "lift"
                else:  # closed on air: reopen, rise, retry the servo
                    self._retry()
                    out[grip] = -1.0
                    out[2] = 0.2
            return out

        if self.phase == "lift":
            out[grip] = 1.0
            out[2] = 0.4
            if z >= LIFT_Z:
                self.phase = ("transport" if self.place_at is not None
                              else "servo_tgt")
            return out

        if self.phase == "transport":
            # Proprio-only traverse to the calibrated basket point.
            out[grip] = 1.0
            if proprio is not None and self._jaws(proprio) < HELD_JAW_MIN:
                self.phase = ("center_src" if self.center_first else "servo_src")
                return out
            p = np.asarray(proprio, dtype=np.float64).reshape(-1)
            ex = self.place_at[0] - float(p[0])
            ey = self.place_at[1] - float(p[1])
            out[0] = float(np.clip(12.0 * ex, -0.6, 0.6))
            out[1] = float(np.clip(12.0 * ey, -0.6, 0.6))
            if z < LIFT_Z:
                out[2] = 0.3  # keep altitude over the traverse
            if max(abs(ex), abs(ey)) < 0.02:
                self.phase = "release"
                self._release_ticks = 0
            return out

        if self.phase == "servo_tgt":
            out[grip] = 1.0
            if proprio is not None and self._jaws(proprio) < HELD_JAW_MIN:
                # Dropped it mid-traverse: start over.
                self.phase = ("center_src" if self.center_first else "servo_src")
                return self.step(source, target, proprio, action_dim,
                                 proposals=getattr(self, "_last_proposals", ()),
                                 source_emb=getattr(self, "_last_src_emb", None),
                                 target_emb=getattr(self, "_last_tgt_emb", None))
            center = self._believe("tgt", target)
            if center is not None:
                err = self._servo_xy(center, out)
                if z < LIFT_Z:  # keep altitude over the traverse
                    out[2] = 0.2
                if err < 0.12:
                    self.phase = "release"
                    self._release_ticks = 0
            else:
                out[2] = 0.15  # rise to find the target
            return out

        # release -> done. With a calibrated place point, lower the held
        # object to drop_z over the basket before opening; otherwise (legacy
        # visual place) open immediately and hold clear of the drop.
        if (self.place_at is not None and proprio is not None
                and self._release_ticks == 0 and z > self.drop_z
                and self._jaws(proprio) >= HELD_JAW_MIN):
            out[grip] = 1.0
            out[2] = -0.35
            return out
        out[grip] = -1.0
        self._release_ticks += 1
        if self._release_ticks < 8:
            out[2] = 0.1
        else:
            self.phase = "done"
        return out
