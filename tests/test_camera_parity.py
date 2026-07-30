"""The bake and the robot must resolve a camera the same way.

Every defect in paper.md's taxonomy has one shape: a producer and a consumer
each holding a private copy of a convention. Orientation was the latest —
``preprocess/libero.py`` de-rotated agentview 180° (a row flip plus a spurious
left-right mirror) while nothing de-flipped the wrist stream at all, and
``eval/record_mp4.py`` held a third copy (``np.rot90(..., 2)``). This pins the
single copy that replaced them.

CPU-only, no sim, no cv2, no network.
"""
from __future__ import annotations

import numpy as np
import pytest

from microvla.utils.camera import (AGENTVIEW, ENV_KEY, HDF5_KEY, VIEWS, WRIST,
                                   upright, view_of)


def test_every_view_has_both_key_spellings():
    for v in VIEWS:
        assert v in HDF5_KEY and v in ENV_KEY


@pytest.mark.parametrize("view", VIEWS)
def test_view_of_accepts_all_three_spellings(view):
    assert view_of(view) == view
    assert view_of(HDF5_KEY[view]) == view
    assert view_of(ENV_KEY[view]) == view


def test_view_of_rejects_unknown_camera():
    # Loud, not defaulted: a silently-defaulted camera is the defect this
    # module exists to prevent.
    with pytest.raises(KeyError):
        view_of("front_rgb")


def test_upright_is_a_row_flip_and_nothing_else():
    f = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = upright(f)
    np.testing.assert_array_equal(out, f[::-1])
    # Explicitly NOT a 180 rotation: columns must be untouched, or every box
    # center is mirrored with respect to the action frame.
    assert not np.array_equal(out, f[::-1, ::-1])


def test_upright_is_an_involution():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, size=(5, 4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(upright(upright(f)), f)


def test_upright_handles_single_frames_and_stacks_identically():
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 256, size=(4, 6, 5, 3), dtype=np.uint8)
    np.testing.assert_array_equal(upright(stack)[2], upright(stack[2]))


def test_upright_returns_contiguous():
    # ultralytics/OpenCV reject negative strides; a lazy reversed view here
    # surfaces as an opaque failure deep in the detector.
    f = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    assert upright(f).flags["C_CONTIGUOUS"]


def test_upright_rejects_non_image_shapes():
    with pytest.raises(ValueError):
        upright(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        upright(np.zeros((4, 4, 4)))


def test_upright_validates_the_key_when_given_one():
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    for key in (AGENTVIEW, ENV_KEY[AGENTVIEW], HDF5_KEY[WRIST]):
        upright(f, key)
    with pytest.raises(KeyError):
        upright(f, "agentview_image_rgb")


def test_upright_correction_does_not_depend_on_the_camera():
    """Both LIBERO streams share one bottom-left-origin framebuffer.

    A per-camera correction is what produced the shipped asymmetry (agentview
    turned 180°, wrist untouched), so the absence of a branch is the property
    worth pinning.
    """
    rng = np.random.default_rng(2)
    f = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(upright(f, ENV_KEY[WRIST]),
                                  upright(f, ENV_KEY[AGENTVIEW]))


def test_bake_and_eval_agree_on_the_view_they_name():
    """The corpus key and the env key must resolve to ONE view.

    This is the assertion whose absence cost a bake, two stage-B trainings and
    a closed-loop eval: the corpus was built from ``agentview_rgb`` while
    ``eval/libero_eval.py`` read ``robot0_eye_in_hand_image``.
    """
    for view in VIEWS:
        assert view_of(HDF5_KEY[view]) == view_of(ENV_KEY[view])
    assert view_of("agentview_rgb") != view_of("robot0_eye_in_hand_image")


def test_eval_and_bake_import_the_same_upright():
    import eval.libero_eval as le
    import preprocess.libero as pl

    assert le.upright is upright
    assert pl.upright is upright
