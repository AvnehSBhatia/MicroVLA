"""GoalServoMachine: the teacher's control structure with LEARNED goals.

This is the deployment-side mirror of ``eval/ibvs_phase.py::PhasedIBVS`` with
its one hand-given quantity — the visual gate + calibrated lever arm that
produced ``_base_tgt`` — replaced by the world-goal estimates of
``GraspPointHead``/``PlaceHead``. Everything the WHY analysis credits the
teacher's success to is kept as structure, not re-learned:

  * goals are LATCHED (vision admits evidence; it never steers directly),
  * actions are a P-law on ``goal - eef`` in ONE frame (magnitude scales with
    error by construction — mean-collapse/undershoot/parking are impossible),
  * phases are one-way with debounced entry (commitment is state),
  * the hold check is the teacher's ``abs()`` jaw signal, read as a bit,
  * failure triggers the deterministic probe search, never re-inference,
  * the place leg is proprio-only servo to a latched constant.

Numeric defaults are the CALIBRATED teacher values (handoff.md §3 run:
world P-gain 12 / clip 0.6, descend -0.4, close_z 0.01, press 0.2,
retry_rise 8, drop_z 0.18, LIFT_Z 0.30, HELD_JAW_MIN 0.2, ±6 cm probe) —
they are properties of the arm and task physics, calibrated once, exactly
like the teacher's. No parameters, no torch: pure numpy runtime state,
constructed per episode by the policy.

Conventions match the teacher: OSC delta actions, +z up, gripper +1 close /
-1 open at ``action[-1]``, canonical [10] proprio
``[eef(3) | quat(4) | jaws(2) | valid]``.
"""
from __future__ import annotations

import numpy as np

#: Stable phase indexing for recorded machine-state dumps (self-play
#: supervision: the scaffold's internal state becomes labels).
PHASES: tuple[str, ...] = ("approach", "descend", "grasp", "rise", "lift",
                           "transport", "release", "done")

#: Default probe schedule (dx, dy, dyaw): radius-ordered 2D search around the
#: latched goal. The teacher's table was x-only because ITS error source (the
#: calibrated lever arm) varied along x; the learned head's error is
#: ISOTROPIC — unaided_goal1 trials burned 9–12 attempts re-probing the same
#: wrong y (latch errors 2–5 cm, any direction) against an x-only table.
PROBE_XY: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (+0.02, 0.0, 0.0), (-0.02, 0.0, 0.0), (0.0, +0.02, 0.0), (0.0, -0.02, 0.0),
    (+0.02, +0.02, 0.0), (-0.02, -0.02, 0.0),
    (+0.02, -0.02, 0.0), (-0.02, +0.02, 0.0),
    (+0.04, 0.0, 0.0), (-0.04, 0.0, 0.0), (0.0, +0.04, 0.0), (0.0, -0.04, 0.0),
    (+0.06, 0.0, 0.0), (-0.06, 0.0, 0.0))

#: Probe schedule with alternating ~90° yaw (cream-cheese-shaped objects:
#: millimetre-perfect position can still close on the ungraspable axis).
PROBE_YAW: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0), (0.0, 0.0, 1.5708),
    (+0.02, 0.0, 0.0), (+0.02, 0.0, 1.5708),
    (0.0, +0.02, 0.0), (0.0, -0.02, 1.5708),
    (-0.02, 0.0, 0.0), (+0.04, 0.0, 1.5708))


def _yaw(proprio) -> float:
    p = np.asarray(proprio, dtype=np.float64).reshape(-1)
    x, y, z, w = p[3], p[4], p[5], p[6]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _jaws(proprio) -> float:
    # ABS before the mean — the mirrored finger joints' raw mean is
    # identically zero in every state (defect 29; see PhasedIBVS._jaws).
    p = np.asarray(proprio, dtype=np.float64).reshape(-1)
    return float(np.abs(p[7:9]).mean()) if p.shape[0] >= 9 else 0.0


class GoalServoMachine:
    """Owns the action from latched world goals + proprio. One per episode.

    Runtime contract:
        * ``observe(xy, z, sigma)`` — feed one goal estimate (REAL perception
          ticks only, pre-latch; ignored once latched: vision has done its
          one job).
        * ``set_place(xy)`` — latch the place point (once, at reset time).
        * ``step(proprio, action_dim)`` — every tick; returns the raw action.

    Phase graph (one-way; the only cycle is the probe retry):
        approach -> descend -> grasp -> lift -> transport -> release -> done
                       ^          |
                       +-- rise <-+   (hold-check failure: probe search)
    A mid-transport drop is the single structured restart: unlatch, back to
    approach (mirrors the teacher's servo_src fallback).
    """

    def __init__(self,
                 p_gain: float = 12.0,
                 clip: float = 0.6,
                 descend: float = -0.4,
                 hover_z: float = 0.25,
                 z_gain: float = 6.0,
                 close_z_min: float = 0.005,
                 close_z_max: float = 0.08,
                 close_z_default: float = 0.01,
                 press: float = 0.2,
                 close_ticks: int = 12,
                 retry_rise: int = 8,
                 lift_z: float = 0.30,
                 drop_z: float = 0.12,
                 held_jaw_min: float = 0.2,
                 latch_tol: float = 0.015,
                 latch_sigma: float = 0.05,
                 latch_k: int = 3,
                 latch_spread: float = 0.03,
                 force_latch_ticks: int = 240,
                 place_tol: float = 0.02,
                 approach_z: float = 0.0,
                 ws_bound: float = 0.6,
                 z_freeze: float = 0.10,
                 descend_band: float = 0.05,
                 freeze_ticks: int = 150,
                 probe_restart: int = 8,
                 hang_comp: tuple[float, float] = (0.0, 0.0),
                 yaw_probe: bool = False,
                 yaw_sign: float = 1.0,
                 sigma_trust: float = 0.0,
                 anchor_band: float = 0.0,
                 gates: dict | None = None) -> None:
        self.p_gain = float(p_gain)
        self.clip = float(clip)
        self.descend = float(descend)
        self.hover_z = float(hover_z)
        self.z_gain = float(z_gain)
        self.close_z_min = float(close_z_min)
        self.close_z_max = float(close_z_max)
        self.close_z_default = float(close_z_default)
        self.press = float(press)
        self.close_ticks = int(close_ticks)
        self.retry_rise = max(1, int(retry_rise))
        self.lift_z = float(lift_z)
        self.drop_z = float(drop_z)
        self.held_jaw_min = float(held_jaw_min)
        self.latch_tol = float(latch_tol)
        self.latch_sigma = float(latch_sigma)
        self.latch_k = max(1, int(latch_k))
        self.latch_spread = float(latch_spread)
        self.force_latch_ticks = int(force_latch_ticks)
        self.place_tol = float(place_tol)
        self.approach_z = float(approach_z)
        self.ws_bound = float(ws_bound)
        self.z_freeze = float(z_freeze)
        self.descend_band = float(descend_band)
        self.freeze_ticks = int(freeze_ticks)
        self.probe_restart = max(1, int(probe_restart))
        # Place-side hand-eye constant: the held object hangs a MEASURED
        # near-constant offset from the eef (unaided_goal3+4 telemetry:
        # (−2.8, +1.4) cm, residual 3.3 mm over 390 transport ticks — the
        # grasp pipeline's systematic bias). The traverse aims the EEF at
        # place − hang so the OBJECT arrives over the basket center. A
        # calibrated constant in the same category as the P-gains — logged
        # as such in the paper; the learned-head accuracy track is the
        # principled fix that drives it to zero.
        self.hang_comp = (float(hang_comp[0]), float(hang_comp[1]))
        # De-skeletonization stage 1: optional LEARNED gates replace their
        # thresholds per-key ("close", "hold"); absent keys keep the
        # threshold path, so each swap is independently ablatable.
        self.gates = gates or {}
        self.yaw_probe = bool(yaw_probe)
        self.yaw_sign = float(yaw_sign)
        # Sigma-conditioned latch (the v8 lesson, 2026-08-05): when the
        # head's calibrated sigma says its FAR-VIEW estimates are already
        # trustworthy, latch before the arm moves toward them and freeze the
        # goal — the approach itself drives uv off the training manifold and
        # the estimate stream walks (butter: accurate first estimates, then
        # +10 cm of chase; 0/10 -> 10/10 under early latch). When sigma is
        # loose (soup at hover: coarse estimates that NEED approach
        # refinement — the goal1 lesson), keep the standard
        # approach-then-refine path. 0.0 disables the early path.
        self.sigma_trust = float(sigma_trust)
        # Anchor trust-region (v9, 2026-08-05). Live instrumentation showed
        # BOTH objects' first stable estimates are ~1-2 cm accurate, and the
        # failure mode is what happens NEXT: soup's later estimates refine
        # within ~2 cm of the early median, butter's walk +10 cm as the
        # approach drives uv off the training manifold and the arm chases
        # (the head's own sigma cannot arbitrate — measured 0.007-0.017 in
        # both regimes). The invariant: anchor on the first stable median and
        # admit later estimates only inside ``anchor_band`` of it — true
        # refinement passes, the chase is rejected. 0.0 disables.
        self.anchor_band = float(anchor_band)
        # v10: rejection-triggered freeze. Band rejections ARE the chase
        # detector — a stream that keeps landing outside the trust region is
        # walking with the arm, not refining. After ``freeze_rejects``
        # consecutive rejections, stop refinement entirely (pure proprio to
        # the anchor, butter-mode); a stream that refines inside the band
        # never trips it and keeps soup's approach-refinement. 0 disables.
        self.freeze_rejects = 3 if self.anchor_band > 0.0 else 0
        self.probe = PROBE_YAW if self.yaw_probe else PROBE_XY
        self.reset()

    # --------------------------------------------------------------- state
    def reset(self) -> None:
        self.phase = "approach"
        self._est: list[tuple[float, float, float]] = []  # sigma-passed window
        self._est_sig: list[float] = []                    # their sigmas
        self._anchor: tuple[float, float] | None = None    # first stable median
        self._reject_n = 0                                 # consecutive band rejections
        self._weak: list[tuple[float, float, float]] = []  # all in-bounds ests
        self._base_tgt: tuple[float, float] | None = None
        self._base_yaw = 0.0
        self._tgt_yaw: float | None = None
        self._close_z = self.close_z_default
        self._align_tgt: tuple[float, float] | None = None
        self._place_tgt: tuple[float, float] | None = None
        self._attempt = 0
        self._close_n = 0
        self._release_n = 0
        self._rise_n = 0
        self._approach_n = 0
        self._descend_n = 0
        self._z_hist: list[float] = []
        self._goal_frozen = False
        self._aligned = False

    # telemetry accessors (scorecard-compatible names)
    @property
    def base_tgt(self):
        return self._base_tgt

    @property
    def attempt(self) -> int:
        return self._attempt

    def state_row(self) -> np.ndarray:
        """One float32 row of the machine's internal state, for recording.

        Layout: [phase_idx, attempt, latched, base_x, base_y, close_z,
        aligned, frozen] — base_x/y are NaN pre-latch. This is the state the
        original BC rounds could never see (the invisible ``_base_tgt``); a
        student trained WITH it as auxiliary supervision is the
        de-skeletonization bet.
        """
        bt = self._base_tgt
        return np.array([
            float(PHASES.index(self.phase)) if self.phase in PHASES else -1.0,
            float(self._attempt),
            0.0 if bt is None else 1.0,
            np.nan if bt is None else float(bt[0]),
            np.nan if bt is None else float(bt[1]),
            float(self._close_z),
            1.0 if self._aligned else 0.0,
            1.0 if self._goal_frozen else 0.0,
        ], dtype=np.float32)

    def observe(self, xy, z: float, sigma: float) -> None:
        """One goal estimate from the head. Pre-latch, approach phase only.

        Sigma gates NORMAL admission (the strong pool that can latch on
        stability). Every in-bounds estimate also lands in a weak pool so the
        force-latch deadlock-breaker always has something to break the
        deadlock WITH — an over-conservative sigma head must cost latency,
        never the whole episode.
        """
        x, y = float(xy[0]), float(xy[1])
        if abs(x) > self.ws_bound or abs(y) > self.ws_bound:
            return                     # outside any reachable workspace: junk
        if self.phase == "approach" and self._base_tgt is None:
            self._weak.append((x, y, float(z)))
            if len(self._weak) > self.latch_k:
                self._weak.pop(0)
            if float(sigma) > self.latch_sigma:
                return                 # the head itself says "don't trust this"
            if (self.anchor_band > 0.0 and self._anchor is not None
                    and max(abs(x - self._anchor[0]),
                            abs(y - self._anchor[1])) > self.anchor_band):
                self._reject_n += 1
                if (self.freeze_rejects and not self._goal_frozen
                        and self._reject_n >= self.freeze_rejects):
                    self._goal_frozen = True   # chase detected: stop refining
                return                 # outside the far-view trust region: chase
            self._reject_n = 0
            self._est.append((x, y, float(z)))
            self._est_sig.append(float(sigma))
            if len(self._est) > self.latch_k:
                self._est.pop(0)
                self._est_sig.pop(0)
            if (self.anchor_band > 0.0 and self._anchor is None
                    and len(self._est) >= self.latch_k):
                ax, ay, _ = self._median_est()
                self._anchor = (ax, ay)
            return
        # First-descent refinement (unaided_goal1 lesson): the latch fires at
        # hover altitude where the head's error runs 2–5 cm deployed, but its
        # low-altitude estimates are ~1 cm — so on attempt 0, while the goal
        # is not yet frozen (z_freeze crossing), a confident estimate keeps
        # correcting the stored goal on the way down. EMA-blended
        # (unaided_goal2 lesson: raw overwrites jitter the target and the
        # strict align gate then deadlocks the descent). Retries never
        # refine: the probe searches around the frozen target, exactly the
        # teacher's "later attempts go straight back on proprio alone".
        if (self.phase == "descend" and self._attempt == 0
                and not self._goal_frozen and float(sigma) <= self.latch_sigma):
            if (self.anchor_band > 0.0 and self._anchor is not None
                    and max(abs(float(xy[0]) - self._anchor[0]),
                            abs(float(xy[1]) - self._anchor[1]))
                    > self.anchor_band):
                self._reject_n += 1
                if (self.freeze_rejects
                        and self._reject_n >= self.freeze_rejects):
                    self._goal_frozen = True   # chase during descent: freeze
                return                 # refinement outside the trust region
            self._reject_n = 0
            bx, by = self._base_tgt
            nx, ny = 0.5 * bx + 0.5 * x, 0.5 * by + 0.5 * y
            self._base_tgt = (nx, ny)
            self._align_tgt = (nx, ny)
            zc = float(np.clip(z, self.close_z_min, self.close_z_max))
            self._close_z = 0.5 * self._close_z + 0.5 * zc

    def set_place(self, xy) -> None:
        if xy is not None:
            self._place_tgt = (float(xy[0]), float(xy[1]))

    # --------------------------------------------------------------- utils
    def _servo_to(self, tgt_xy, proprio, out: np.ndarray) -> float:
        """P-law toward ``tgt_xy``; returns the lateral inf-norm error."""
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        ex = tgt_xy[0] - float(p[0])
        ey = tgt_xy[1] - float(p[1])
        out[0] = float(np.clip(self.p_gain * ex, -self.clip, self.clip))
        out[1] = float(np.clip(self.p_gain * ey, -self.clip, self.clip))
        return max(abs(ex), abs(ey))

    def _median_est(self, pool=None) -> tuple[float, float, float]:
        arr = np.asarray(pool if pool is not None else self._est,
                         dtype=np.float64)
        m = np.median(arr, axis=0)
        return float(m[0]), float(m[1]), float(m[2])

    def _est_spread(self) -> float:
        arr = np.asarray(self._est, dtype=np.float64)[:, :2]
        return float((arr.max(axis=0) - arr.min(axis=0)).max())

    def _latch(self, proprio, pool=None) -> None:
        x, y, z = self._median_est(pool)
        self._base_tgt = (x, y)
        self._base_yaw = _yaw(proprio) if proprio is not None else 0.0
        self._close_z = float(np.clip(z, self.close_z_min, self.close_z_max))
        self._enter_descend()

    def _enter_descend(self) -> None:
        dx, dy, dyaw = self.probe[min(self._attempt, len(self.probe) - 1)]
        self._align_tgt = (self._base_tgt[0] + dx, self._base_tgt[1] + dy)
        self._tgt_yaw = self._base_yaw + dyaw if self.yaw_probe else None
        self._z_hist = []
        self._close_n = 0
        self._descend_n = 0
        self._aligned = False
        self.phase = "descend"

    def _contact(self, z: float) -> bool:
        """A commanded descend stopped moving the eef (table/object stall)."""
        self._z_hist.append(z)
        if len(self._z_hist) > 4:
            self._z_hist.pop(0)
        return (len(self._z_hist) == 4 and z < self.close_z_max
                and (self._z_hist[0] - z) < 0.002)

    # ---------------------------------------------------------------- step
    def step(self, proprio, action_dim: int = 7) -> np.ndarray:
        out = np.zeros(int(action_dim), dtype=np.float32)
        grip = out.shape[0] - 1
        if proprio is None:
            out[grip] = -1.0
            return out
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        z = float(p[2])

        if self.phase == "approach":
            out[grip] = -1.0
            self._approach_n += 1
            pool = self._est if self._est else self._weak
            if not pool:
                out[2] = 0.15          # no estimate yet: rise to widen the view
                return out
            ex, ey, _ = self._median_est(pool)
            lateral = self._servo_to((ex, ey), proprio, out)
            out[2] = float(np.clip(self.z_gain * (self.hover_z - z),
                                   -0.3, 0.3))
            ready = (len(self._est) >= self.latch_k
                     and self._est_spread() <= self.latch_spread
                     and lateral < self.latch_tol)
            # Confident-early path: a stable window the head itself marks as
            # tight latches from the FAR view and freezes — no approach-chase,
            # no descend refinement (see sigma_trust note in __init__).
            confident = (self.sigma_trust > 0.0
                         and len(self._est) >= self.latch_k
                         and self._est_spread() <= self.latch_spread
                         and float(np.median(self._est_sig)) < self.sigma_trust)
            # Deadlock-breaker (the descend-hyst lesson): a noisy-but-present
            # estimate beats hovering forever — latch on timeout regardless,
            # from the weak pool if sigma gating admitted nothing.
            if confident:
                self._latch(proprio, self._est)
                self._goal_frozen = True
            elif ready or self._approach_n >= self.force_latch_ticks:
                self._latch(proprio, pool)
            return out

        if self.phase == "descend":
            # The teacher's align phase, goal-driven: world servo to the
            # probe-shifted latched target (vision-refined until freeze on
            # the first attempt, proprio-only after). While the goal is still
            # refining, descend inside a WIDE band and steer on the way down
            # — the descend-hyst lesson, relearned in unaided_goal2 where a
            # strict gate against a moving refined target hovered forever.
            # The grasp TRANSITION always requires the strict tolerance.
            out[grip] = -1.0
            self._descend_n += 1
            if z < self.z_freeze or self._descend_n > self.freeze_ticks:
                self._goal_frozen = True
            lateral = self._servo_to(self._align_tgt, proprio, out)
            yaw_ok = True
            if self._tgt_yaw is not None:
                yerr = self._tgt_yaw - _yaw(proprio)
                yerr = float(np.arctan2(np.sin(yerr), np.cos(yerr)))
                out[5] = float(np.clip(self.yaw_sign * 2.0 * yerr, -0.5, 0.5))
                yaw_ok = abs(yerr) < 0.15
            if lateral > 0.015 and z < self.approach_z:
                out[2] = 0.3           # standing object: fly the correction over it
                self._z_hist = []
                self._aligned = False
                return out
            # Alignment HYSTERESIS (unaided_goal3 trial-1 signature: lateral
            # error chattering at exactly the gate boundary reset the contact
            # window every other tick while the gripper sat ON the object):
            # enter tight at 0.015, exit only above 0.025 — while aligned,
            # the contact evidence keeps accumulating.
            if lateral < 0.015:
                self._aligned = True
            elif lateral > 0.025:
                self._aligned = False
            wide_ok = (not self._goal_frozen and self._attempt == 0
                       and lateral < self.descend_band)
            if self._aligned and yaw_ok:
                out[2] = self.descend
                if "close" in self.gates:
                    fire = self.gates["close"].fire(
                        z, self._close_z, lateral, self._descend_n)
                    self._contact(z)   # keep the window warm for telemetry
                else:
                    fire = z <= self._close_z or self._contact(z)
                if fire:
                    self.phase = "grasp"
                    self._close_n = 0
            elif wide_ok and z > self._close_z + 0.02:
                out[2] = self.descend          # wide band: descend + steer
                self._z_hist = []
            else:
                self._z_hist = []              # off-target: steer only
            return out

        if self.phase == "grasp":
            out[grip] = 1.0            # close; press keeps the fingers seating low
            if self.press != 0.0 and self._close_n < 6:
                out[2] = -abs(self.press)
            self._close_n += 1
            if self._close_n >= self.close_ticks:
                if "hold" in self.gates:
                    held = self.gates["hold"].held(
                        float(p[7]), float(p[8]), self._close_n)
                else:
                    held = _jaws(proprio) >= self.held_jaw_min
                if held:
                    self.phase = "lift"
                elif self._attempt + 1 >= self.probe_restart:
                    # Probe exhausted (unaided_goal2: attempts 5–10 around a
                    # 3–4.5 cm-off latch never converge): full restart. A
                    # fresh approach latches from a fresh vantage with
                    # refinement — better odds than re-probing a bad latch.
                    self._unlatch()
                    out[grip] = -1.0
                    out[2] = 0.3
                else:                  # closed on air: probe search, not hope
                    self._attempt += 1
                    self._rise_n = 0
                    self.phase = "rise"
                    out[grip] = -1.0
                    out[2] = 0.2
            return out

        if self.phase == "rise":
            out[grip] = -1.0
            out[2] = 0.35
            self._rise_n += 1
            if self._rise_n >= self.retry_rise:
                # Vision did its one job at the latch; retries are proprio-only.
                self._enter_descend()
            return out

        if self.phase == "lift":
            out[grip] = 1.0
            out[2] = 0.4
            if z >= self.lift_z:
                self.phase = "transport"
            return out

        if self.phase == "transport":
            out[grip] = 1.0
            if _jaws(proprio) < self.held_jaw_min:
                # Dropped it mid-traverse: the one structured restart.
                self._unlatch()
                return self.step(proprio, action_dim)
            if self._place_tgt is None:
                self.phase = "release"     # nothing better to do than open here
                self._release_n = 0
                return out
            aim = (self._place_tgt[0] - self.hang_comp[0],
                   self._place_tgt[1] - self.hang_comp[1])
            lateral = self._servo_to(aim, proprio, out)
            if z < self.lift_z:
                out[2] = 0.3           # keep altitude over the traverse
            if lateral < self.place_tol:
                self.phase = "release"
                self._release_n = 0
            return out

        if self.phase == "release":
            # Lower the held object over the basket before opening.
            if (self._release_n == 0 and z > self.drop_z
                    and _jaws(proprio) >= self.held_jaw_min):
                out[grip] = 1.0
                out[2] = -0.35
                return out
            out[grip] = -1.0
            self._release_n += 1
            if self._release_n < 8:
                out[2] = 0.1
            else:
                self.phase = "done"
            return out

        out[grip] = -1.0               # done: open, hold still
        return out

    def _unlatch(self) -> None:
        self.phase = "approach"
        self._est = []
        self._est_sig = []
        self._anchor = None
        self._reject_n = 0
        self._weak = []
        self._base_tgt = None
        self._align_tgt = None
        self._attempt = 0
        self._approach_n = 0
        self._goal_frozen = False
