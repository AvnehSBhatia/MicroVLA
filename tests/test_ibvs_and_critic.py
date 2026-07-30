"""IBVS residual + EEF progress targets — diagnostics for the frozen-ceiling claim."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def test_ibvs_residual_drives_toward_target():
    from microvla.utils.ibvs import ibvs_residual

    # Object right-and-below the grasp point -> positive dx, positive dy under
    # the default sign (1, -1) wait: ev = v - 0.55; v=0.8 -> ev=+0.25;
    # dy = gain * (-1) * (+0.25) = negative. That's intentional (image-down
    # vs robot-up). Just pin the arithmetic.
    out = ibvs_residual((0.8, 0.8), 0.9, gain=0.1, target_uv=(0.5, 0.55))
    assert out is not None
    assert out.shape == (7,)
    assert out[0] == pytest.approx(0.1 * 0.3)          # +eu
    assert out[1] == pytest.approx(0.1 * (-1.0) * 0.25)  # -ev
    assert out[3:].sum() == 0.0                        # orientation/grip untouched


def test_ibvs_residual_none_when_disabled_or_low_conf():
    from microvla.utils.ibvs import ibvs_residual

    assert ibvs_residual((0.8, 0.8), 0.9, gain=0.0) is None
    assert ibvs_residual((0.8, 0.8), 0.05, gain=0.1, conf_floor=0.1) is None


def test_progress_targets_eef_rises_toward_final_pose():
    from microvla.critic import progress_targets_eef

    B, T = 2, 5
    # Episode 0: walks from origin to (1,0,0). Episode 1: invalid proprio.
    proprio = torch.zeros(B, T, 10)
    for t in range(T):
        proprio[0, t, 0] = t / (T - 1)
    proprio[0, :, -1] = 1.0
    proprio[1, :, -1] = 0.0

    tgt = progress_targets_eef(proprio)
    assert tgt.shape == (B, T)
    # Episode 0: monotone non-decreasing toward 1.
    assert torch.all(tgt[0, 1:] >= tgt[0, :-1] - 1e-6)
    assert float(tgt[0, -1]) == pytest.approx(1.0)
    assert float(tgt[0, 0]) == pytest.approx(0.0)
    # Episode 1 falls back to pure time.
    assert torch.allclose(tgt[1], torch.arange(1, T + 1, dtype=tgt.dtype) / T)


def test_progress_targets_eef_degenerate_path_falls_back_to_time():
    from microvla.critic import progress_targets, progress_targets_eef

    B, T = 1, 4
    proprio = torch.zeros(B, T, 10)
    proprio[..., -1] = 1.0
    tgt = progress_targets_eef(proprio)
    assert torch.allclose(tgt, progress_targets(T, B, proprio.device))


class TestIbvsFalsifierIsCapableOfGrasping:
    """A falsifier that cannot grasp cannot falsify.

    The residual's default sign is (1, -1, 0): no z component, so it centres the
    object and never approaches it. A null result from that configuration says
    nothing about whether the frozen features localize well enough to act, which
    is the entire question paper.md 5k reopened. Both knobs must be reachable and
    the descend must actually engage.
    """

    def test_descend_engages_only_when_centred(self):
        import numpy as np

        from microvla.utils.ibvs import ibvs_residual

        far = ibvs_residual((0.9, 0.9), 0.9, gain=0.1, descend=-0.5, descend_tol=0.2)
        near = ibvs_residual((0.5, 0.55), 0.9, gain=0.1, descend=-0.5, descend_tol=0.2)
        assert far is not None and near is not None
        assert far[2] == 0.0, "descended while the object was still off to one side"
        assert near[2] < 0.0, "did not descend when centred on the grasp point"

    def test_descend_scales_with_centredness(self):
        from microvla.utils.ibvs import ibvs_residual

        a = ibvs_residual((0.5, 0.55), 0.9, gain=0.1, descend=-0.5, descend_tol=0.2)
        b = ibvs_residual((0.5, 0.70), 0.9, gain=0.1, descend=-0.5, descend_tol=0.2)
        assert abs(a[2]) > abs(b[2]), "descend rate should fall off with image error"

    def test_sign_is_reachable_from_the_cli(self):
        from eval.libero_eval import parse_args

        a = parse_args(["--suite", "libero_object", "--ibvs-sign", "1,1,-1",
                        "--ibvs-descend", "-0.4"])
        assert a.ibvs_sign == "1,1,-1"
        assert a.ibvs_descend == -0.4

    def test_gripper_and_orientation_are_never_touched(self):
        from microvla.utils.ibvs import ibvs_residual

        r = ibvs_residual((0.2, 0.8), 0.9, gain=0.3, descend=-0.5)
        assert all(v == 0.0 for v in r[3:]), (
            "IBVS must only add translation; orientation and the gripper belong "
            "to the policy"
        )
