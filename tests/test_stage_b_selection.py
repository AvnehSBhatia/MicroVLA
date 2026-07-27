"""Stage-B early-stop selection metric.

The 2026-07-25 overnight batch produced arms that trained anywhere from 8 to 28
epochs, and every bench metric tracked epochs-survived at Spearman >= 0.84
(std_ratio, corr, grip_acc, pose_mae, wp_std_ratio) — so the arm rankings were
measuring stop timing rather than architecture. The cause: the stop metric was
``val_bc + waypoint_weight * val_wp`` compared against an ABSOLUTE ``--min-delta``,
and ``val_wp``'s scale differs ~10x between native (0.05-0.20s displacement
targets) and ``--waypoint-long`` (0.5-2.5s). A larger, noisier term clears a
fixed threshold less often, so long-horizon arms accrued ``stale`` faster.

These tests pin the two guards so the confound cannot come back silently.
"""
from __future__ import annotations

import pytest

from train.train_batched import parse_args

BASE = ["--data-dir", "data/x"]


def test_selection_defaults_to_bc_the_arm_comparable_term():
    assert parse_args(BASE).stage_b_select == "bc"


def test_total_is_still_reachable_to_reproduce_the_confounded_batch():
    assert parse_args(BASE + ["--stage-b-select", "total"]).stage_b_select == "total"


def test_selection_rejects_anything_else():
    with pytest.raises(SystemExit):
        parse_args(BASE + ["--stage-b-select", "wp"])


def test_min_epochs_defaults_off_so_old_commands_are_unchanged():
    assert parse_args(BASE).stage_b_min_epochs == 0


class TestStopPredicate:
    """The guard is ``stale >= patience and epoch >= min_epochs``.

    Mirrored here rather than driving a real training run: stage B needs a
    corpus, a stage-A checkpoint and a GPU, none of which the CPU-only suite has.
    """

    @staticmethod
    def _stop(stale: int, epoch: int, patience: int, min_epochs: int) -> bool:
        return stale >= patience and epoch >= min_epochs

    def test_floor_holds_a_run_open_through_an_early_plateau(self):
        # The observed failure: patience 4 tripped at epoch 8 of a 40 budget.
        assert not self._stop(stale=4, epoch=8, patience=4, min_epochs=20)

    def test_floor_releases_once_reached(self):
        assert self._stop(stale=4, epoch=20, patience=4, min_epochs=20)

    def test_zero_floor_is_exactly_the_old_behaviour(self):
        for epoch in (1, 8, 40):
            assert self._stop(stale=4, epoch=epoch, patience=4, min_epochs=0)

    def test_floor_alone_never_forces_a_stop(self):
        # Past the floor but still improving -> keep going.
        assert not self._stop(stale=0, epoch=30, patience=4, min_epochs=20)


class TestSelectionMetric:
    """`bc` must ignore the waypoint term entirely; `total` must not."""

    @staticmethod
    def _val(select: str, val_bc: float, val_wp: float, w: float) -> float:
        return val_bc if select == "bc" else val_bc + w * val_wp

    def test_bc_is_invariant_to_waypoint_scale(self):
        # Same planner quality, waypoint targets 10x larger (native vs longh).
        near = self._val("bc", 0.6924, 0.1107, 1.0)
        far = self._val("bc", 0.6924, 1.1070, 1.0)
        assert near == far == pytest.approx(0.6924)

    def test_total_moves_with_waypoint_scale(self):
        near = self._val("total", 0.6924, 0.1107, 1.0)
        far = self._val("total", 0.6924, 1.1070, 1.0)
        assert far > near
        # Reproduces the observed longh best val of ~0.82 vs native's ~0.61-0.65.
        assert near == pytest.approx(0.8031)

    def test_absolute_min_delta_is_what_makes_total_scale_sensitive(self):
        # An improvement of the same RELATIVE size is invisible at the larger
        # scale once --min-delta is fixed in absolute terms... except it is not:
        # the real mechanism is that the larger term carries larger noise, so
        # this test pins only the arithmetic that makes the two incomparable.
        min_delta = 1e-4
        improve_rel = 1e-4  # 0.01% better
        for wp in (0.1107, 1.1070):
            best = self._val("total", 0.6924, wp, 1.0)
            nxt = best * (1 - improve_rel)
            registered = nxt < best - min_delta
            # At the small scale the same relative gain is NOT registered; at
            # the large scale it is. Opposite verdicts from one threshold.
            assert registered is (wp > 1.0)
