"""Image-based visual servo residual — a zero-training diagnostic.

If frozen detection is "the ceiling", a proportional move that drives the
detected source center toward a grasp point in the image should not help
closed-loop success: the features would not contain a usable geometric error.
If it DOES help, the features carry the signal and the failure is in the
learned objective (exposure bias / MSE-BC), not in the encoder.

Wrist / eye-in-hand convention used here (LIBERO ``eye_in_hand_rgb``):

* ``center`` is normalized image coords in ``[0, 1]^2`` (x right, y down).
* Error ``e = center - target_uv``. Driving the object toward the grasp point
  requires moving the camera in the SAME image direction (object fixed in the
  world, camera on the wrist) — so the residual added to the OSC delta is
  ``(+e_x, -e_y, ...)`` mapped through a small gain into action units.

The sign on y is opposite image-down vs robot-up for the default LIBERO table
setup; ``ibvs_sign`` lets a one-command sweep flip axes without a retrain.
Orientation and gripper are never touched.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def ibvs_residual(
    source_center: Sequence[float],
    source_confidence: float,
    *,
    gain: float,
    target_uv: Tuple[float, float] = (0.5, 0.55),
    conf_floor: float = 0.1,
    sign: Tuple[float, float, float] = (1.0, -1.0, 0.0),
    descend: float = 0.0,
    descend_tol: float = 0.2,
    action_dim: int = 7,
) -> Optional[np.ndarray]:
    """Returns a ``[action_dim]`` residual or ``None`` when evidence is absent.

    Args:
        source_center: ``(u, v)`` in ``[0, 1]``.
        source_confidence: detector confidence in ``[0, 1]`` (already faded).
        gain: metres-or-normalized-units of action per unit image error. Start
            around ``0.05``–``0.2`` and sweep; ``0`` disables.
        target_uv: desired object location in the wrist frame. Slightly below
            center so a top-down grasp approaches rather than centering.
        conf_floor: ignore detections below this weight.
        sign: per-axis multipliers into ``(dx, dy, dz)``. The image-down vs
            robot-up convention is NOT self-evident, and a wrong sign makes the
            residual push AWAY from the object — a false negative from the one
            experiment meant to falsify the encoder claim. Sweep it.
        descend: raw action units of downward motion applied once the object is
            within ``descend_tol`` of the grasp point, scaled by how centred it
            is. Zero disables. A residual that only centres can never grasp, so
            without this a null result is uninformative.
        descend_tol: image-error radius inside which ``descend`` engages.
        action_dim: full action width; residual pads with zeros past translation.
    """
    if gain == 0.0 or source_confidence < conf_floor:
        return None
    u, v = float(source_center[0]), float(source_center[1])
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    eu = u - float(target_uv[0])
    ev = v - float(target_uv[1])
    out = np.zeros(int(action_dim), dtype=np.float32)
    out[0] = float(gain) * float(sign[0]) * eu
    out[1] = float(gain) * float(sign[1]) * ev
    # Optional small approach along camera optical axis when the object is
    # near the grasp point — keeps the residual from thrusting forward while
    # the object is still off-center.
    # DESCEND once centred. Without this the residual centres the object and
    # never approaches it, so the falsifier cannot complete a grasp and a null
    # result says nothing about the features. Proportional to how centred we
    # are, so it does not thrust forward while still off to one side.
    err = max(abs(eu), abs(ev))
    if descend != 0.0 and err < descend_tol:
        out[2] = float(descend) * (1.0 - err / float(descend_tol))
    elif len(sign) > 2 and sign[2] != 0.0 and err < 0.15:
        out[2] = float(gain) * float(sign[2]) * 0.5
    return out
