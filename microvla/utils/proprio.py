"""Proprioception vector: the arm's own state, one layout everywhere.

The v6 planner conditions on the robot's end-effector state. Without it the
policy must infer arm pose from a GAP'd wrist-camera embedding — practically
unobservable — so trajectory PHASE (approach vs descend vs lift) is ambiguous
and MSE-BC collapses to the timid conditional mean (measured: ~8x understd
actions, eval/replay_probe). EEF state is the cheapest phase signal there is,
and every tick has it for free on a real robot (encoders are fast; cameras are
the slow part).

Layout (PROPRIO_DIM = 10), same in the baked npz, the trainers, and eval:

    [ eef_pos(3) | eef_ori(4: quat, zero-padded if axis-angle 3) |
      gripper(2, scaled x25 so ~0.04 m jaw widths land O(1)) | valid(1) ]

``valid`` is 1.0 when real proprio is present, 0.0 for the zero-filled
fallback (Bridge episodes, un-patched npz, mock envs) — the planner can learn
to gate on it. All helpers are numpy/torch-free-of-heavy-deps and CPU-cheap.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Total proprio vector width (see module docstring for the layout).
PROPRIO_DIM: int = 10

#: Gripper joint positions are ~0.00-0.04 m; scale to O(1) for the planner.
GRIPPER_SCALE: float = 25.0


def build_proprio(
    pos: np.ndarray, ori: Optional[np.ndarray], gripper: Optional[np.ndarray]
) -> np.ndarray:
    """Packs raw robot state into the canonical ``[PROPRIO_DIM]`` vector.

    Args:
        pos: ``[3]`` end-effector position (meters, world/base frame).
        ori: ``[4]`` quaternion or ``[3]`` axis-angle (zero-padded to 4);
            ``None`` -> zeros.
        gripper: ``[2]`` gripper joint positions (or ``[1]``/scalar, padded);
            ``None`` -> zeros.

    Returns:
        ``[PROPRIO_DIM]`` float32 vector with ``valid = 1.0``.
    """
    out = np.zeros(PROPRIO_DIM, dtype=np.float32)
    p = np.asarray(pos, dtype=np.float32).reshape(-1)
    out[:3] = p[:3]
    if ori is not None:
        o = np.asarray(ori, dtype=np.float32).reshape(-1)
        out[3 : 3 + min(4, o.shape[0])] = o[:4]
    if gripper is not None:
        g = np.asarray(gripper, dtype=np.float32).reshape(-1) * GRIPPER_SCALE
        if g.shape[0] == 1:
            g = np.repeat(g, 2)
        out[7:9] = g[:2]
    out[9] = 1.0  # valid
    return out


#: Candidate obs-dict key spellings, tried in order (LIBERO-style first, then
#: raw robosuite). LIBERO bakes use ee_pos/ee_ori/gripper_states; live
#: robosuite envs expose robot0_eef_pos/robot0_eef_quat/robot0_gripper_qpos.
_POS_KEYS = ("ee_pos", "robot0_eef_pos")
_ORI_KEYS = ("ee_ori", "robot0_eef_quat")
_GRIP_KEYS = ("gripper_states", "robot0_gripper_qpos")


def proprio_from_obs(obs) -> Optional[np.ndarray]:
    """Builds the proprio vector from an env/hdf5 obs mapping, if possible.

    Args:
        obs: Mapping of observation arrays (a live env obs dict or an hdf5
            ``obs`` group). Only position is required; orientation/gripper are
            optional extras.

    Returns:
        ``[PROPRIO_DIM]`` float32 vector, or ``None`` when no known position
        key exists (mock envs) — callers fall back to zeros (``valid = 0``).
    """
    def _first(keys):
        for k in keys:
            try:
                if k in obs:
                    return np.asarray(obs[k])
            except TypeError:  # pragma: no cover - exotic mapping types
                return None
        return None

    pos = _first(_POS_KEYS)
    if pos is None:
        return None
    return build_proprio(pos, _first(_ORI_KEYS), _first(_GRIP_KEYS))
