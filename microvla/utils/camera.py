"""Robosuite camera identity and render orientation, in ONE place.

MuJoCo/robosuite render through OpenGL, whose framebuffer origin is BOTTOM-left,
so every LIBERO camera stream — the recorded ``*_rgb`` arrays in the demo hdf5
files and the live ``*_image`` observations alike — arrives ROW-REVERSED with
respect to ordinary image convention. Putting a frame upright is a row flip and
nothing else.

Two defects in this project came from that one sentence not being written down:

* ``preprocess/libero.py`` de-rotated agentview with ``frames[:, ::-1, ::-1]``
  — a full 180° turn, which is the row flip PLUS a spurious left-right mirror.
  The images look plausible either way (a mirrored tabletop is still a
  tabletop), so nothing downstream complained, but every baked ``source_center``
  x was mirrored with respect to the world the actions move in. Measured cost
  on ``libero_object``: source detection duty **0.850 → 0.613**, proposals per
  frame 2.82 → 2.29, purely from feeding the detector a mirrored scene.
* The wrist stream was never de-flipped at all, on either side. That is
  self-consistent — bake and deploy were upside down together, so it is not a
  parity defect — but YOLO-World is not rotation invariant, and it cost target
  detection duty 0.212 → 0.419 on the same frames.

The lesson is the one the rest of ``paper.md`` keeps re-learning: a producer and
a consumer that each hold their own copy of a convention will drift, and the
drift is invisible to any test that exercises one side alone. So orientation
lives here, both sides call :func:`upright`, and
``tests/test_camera_parity.py`` pins that the bake path and the deployment path
resolve the same view to the same key and the same flip.

Pure numpy: this module is on the deployment path, where only torch + numpy are
guaranteed.
"""
from __future__ import annotations

import numpy as np

#: Canonical view names. The rest of the codebase should name a VIEW, not a
#: dataset key — the two key spellings below are an accident of LIBERO's
#: recorded-vs-live APIs and should not leak into call sites.
WRIST = "wrist"
AGENTVIEW = "agentview"
VIEWS = (WRIST, AGENTVIEW)

#: ``obs/<key>`` inside a LIBERO demonstration hdf5.
HDF5_KEY = {WRIST: "eye_in_hand_rgb", AGENTVIEW: "agentview_rgb"}
#: ``obs[<key>]`` from a live ``OffScreenRenderEnv``.
ENV_KEY = {WRIST: "robot0_eye_in_hand_image", AGENTVIEW: "agentview_image"}

_BY_KEY = {}
for _v in VIEWS:
    _BY_KEY[_v] = _v
    _BY_KEY[HDF5_KEY[_v]] = _v
    _BY_KEY[ENV_KEY[_v]] = _v


def view_of(key: str) -> str:
    """Canonical view name for a view name, an hdf5 key, or an env key.

    Args:
        key: e.g. ``"wrist"``, ``"eye_in_hand_rgb"``, or
            ``"robot0_eye_in_hand_image"``.

    Returns:
        One of :data:`VIEWS`.

    Raises:
        KeyError: If ``key`` names no known LIBERO camera. Deliberately loud —
            a silently-defaulted camera is exactly the failure this module
            exists to prevent (see ``preprocess/libero.py``'s ``--camera``,
            which is ``required=True`` for the same reason).
    """
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown LIBERO camera {key!r}; expected one of "
            f"{sorted(_BY_KEY)}"
        ) from None


def upright(frames: "np.ndarray", key: str | None = None) -> "np.ndarray":
    """Puts robosuite's bottom-up render into image convention (row flip).

    Args:
        frames: ``[..., H, W, 3]``. A single frame or a stacked episode; the
            flip applies to the height axis either way.
        key: Accepted and validated for call-site clarity, but the correction
            does NOT depend on it — both LIBERO cameras come out of the same
            OpenGL framebuffer with the same bottom-left origin. Passing the
            view you think you have is a cheap way to catch a typo'd key.

    Returns:
        A C-contiguous array, upright. Contiguity matters: the detector path
        hands this straight to OpenCV/ultralytics, which reject negative
        strides.
    """
    if key is not None:
        view_of(key)
    a = np.asarray(frames)
    if a.ndim < 3 or a.shape[-1] != 3:
        raise ValueError(f"expected [..., H, W, 3] frames, got {a.shape}")
    return np.ascontiguousarray(a[..., ::-1, :, :])
