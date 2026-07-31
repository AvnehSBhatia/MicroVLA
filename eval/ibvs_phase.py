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
                 swap_uv: bool = False) -> None:
        self.gain = float(gain)
        self.sign = tuple(float(v) for v in sign)
        self.descend = float(descend)
        self.target_uv = (float(target_uv[0]), float(target_uv[1]))
        self.conf_floor = float(conf_floor)
        self.track_gate = float(track_gate)
        self.clip_rerank = bool(clip_rerank)
        self.descend_hyst = float(descend_hyst)
        self.swap_uv = bool(swap_uv)
        self.reset()

    def reset(self) -> None:
        self.phase = "servo_src"
        self._close_ticks = 0
        self._release_ticks = 0
        self._descending = False
        self._trackers = ({"src": _BoxTracker(self.track_gate),
                           "tgt": _BoxTracker(self.track_gate)}
                          if self.track_gate > 0.0 else None)

    def _believe(self, role: str, det) -> tuple[float, float] | None:
        """The center the servo should act on for ``role``, after tracking."""
        if float(det.confidence) < self.conf_floor:
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
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        return float(p[7:9].mean()) if p.shape[0] >= 9 else 0.0

    @staticmethod
    def _eef_z(proprio) -> float:
        return float(np.asarray(proprio, dtype=np.float64).reshape(-1)[2])

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
                if err < grasp_err and z < GRASP_Z:
                    self.phase = "grasp"
                    self._close_ticks = 0
            else:
                # Lost the source: rise to widen the wrist view and reacquire.
                out[2] = 0.15
            return out

        if self.phase == "grasp":
            out[grip] = 1.0  # hold position, close
            self._close_ticks += 1
            if self._close_ticks >= 12:
                if proprio is not None and self._jaws(proprio) >= HELD_JAW_MIN:
                    self.phase = "lift"
                else:  # closed on air: reopen, rise, retry the servo
                    self.phase = "servo_src"
                    out[grip] = -1.0
                    out[2] = 0.2
            return out

        if self.phase == "lift":
            out[grip] = 1.0
            out[2] = 0.4
            if z >= LIFT_Z:
                self.phase = "servo_tgt"
            return out

        if self.phase == "servo_tgt":
            out[grip] = 1.0
            if proprio is not None and self._jaws(proprio) < HELD_JAW_MIN:
                # Dropped it mid-traverse: start over.
                self.phase = "servo_src"
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

        # release -> done: open, hold clear of the drop
        out[grip] = -1.0
        self._release_ticks += 1
        if self._release_ticks < 8:
            out[2] = 0.1
        else:
            self.phase = "done"
        return out
