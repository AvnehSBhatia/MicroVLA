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


class PhasedIBVS:
    """Owns the action: servo/grasp/lift/place from detections + proprio.

    One instance per episode (``reset()`` on policy reset). ``step`` returns a
    full ``[action_dim]`` raw action; the caller replaces the policy's action
    with it wholesale — no blending, so the learned prior cannot drag the
    arm off-task (the failure mode that nulled the residual sweeps).
    """

    def __init__(self, gain: float, sign: tuple[float, float, float],
                 descend: float, target_uv: tuple[float, float],
                 conf_floor: float) -> None:
        self.gain = float(gain)
        self.sign = tuple(float(v) for v in sign)
        self.descend = float(descend)
        self.target_uv = (float(target_uv[0]), float(target_uv[1]))
        self.conf_floor = float(conf_floor)
        self.reset()

    def reset(self) -> None:
        self.phase = "servo_src"
        self._close_ticks = 0
        self._release_ticks = 0

    # ------------------------------------------------------------------ utils
    def _servo_xy(self, center, out: np.ndarray) -> float:
        """Writes the xy servo into ``out``; returns the image error (inf-norm)."""
        eu = float(center[0]) - self.target_uv[0]
        ev = float(center[1]) - self.target_uv[1]
        out[0] = self.gain * self.sign[0] * eu
        out[1] = self.gain * self.sign[1] * ev
        return max(abs(eu), abs(ev))

    @staticmethod
    def _jaws(proprio) -> float:
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        return float(p[7:9].mean()) if p.shape[0] >= 9 else 0.0

    @staticmethod
    def _eef_z(proprio) -> float:
        return float(np.asarray(proprio, dtype=np.float64).reshape(-1)[2])

    # ------------------------------------------------------------------- step
    def step(self, source, target, proprio, action_dim: int = 7) -> np.ndarray:
        """One control tick.

        Args:
            source / target: role detections with ``.center`` (normalized
                ``[0,1]^2``) and ``.confidence`` (0.0 when absent this tick).
            proprio: canonical ``[10]`` proprio vector (required — the real
                env always has it; without proprio there is no grasp check
                and the machine just servos).
            action_dim: output width; translation + zeros + gripper.
        """
        out = np.zeros(int(action_dim), dtype=np.float32)
        grip = out.shape[0] - 1
        z = self._eef_z(proprio) if proprio is not None else 1.0

        if self.phase == "servo_src":
            out[grip] = -1.0
            if float(source.confidence) >= self.conf_floor:
                err = self._servo_xy(source.center, out)
                if err < 0.2:  # centered enough: descend, scaled by centering
                    out[2] = self.descend * (1.0 - err / 0.2)
                if err < 0.10 and z < GRASP_Z:
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
                return self.step(source, target, proprio, action_dim)
            if float(target.confidence) >= self.conf_floor:
                err = self._servo_xy(target.center, out)
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
