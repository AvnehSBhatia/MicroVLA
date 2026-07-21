"""Enforces the v2 trainable-parameter budget from DESIGN.md (CPU-only).

Ledger: 32M total - ~13M frozen YOLO-World-S - 10M reserved TRM leaves ~9M
for the trainable heads, split fusion <= 5.0M, drift <= 1.5M, planner <= 2.5M.
"""

from __future__ import annotations

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.mock_trm import MockTRM
from microvla.utils.param_audit import count_trainable_params

CFG = DEFAULT_CONFIG

FUSION_CAP = 5_000_000
DRIFT_CAP = 1_500_000
PLANNER_CAP = 2_500_000
MOCK_TRM_STUB_CAP = 300_000  # MockTRM is a stub, not the ~10M real TRM


def test_trainable_heads_fit_total_budget(capsys):
    fusion = count_trainable_params(SlotResonanceFusion(CFG))
    drift = count_trainable_params(AnchoredDriftEncoder(CFG))
    planner = count_trainable_params(ChronoQueryPlanner(CFG))
    total = fusion + drift + planner

    with capsys.disabled():
        print("\nTrainable-head parameter breakdown:")
        print(f"  fusion  (SlotResonanceFusion): {fusion:>10,d} ({fusion / 1e6:.3f}M)")
        print(f"  drift   (AnchoredDriftEncoder): {drift:>9,d} ({drift / 1e6:.3f}M)")
        print(f"  planner (ChronoQueryPlanner):  {planner:>10,d} ({planner / 1e6:.3f}M)")
        print(f"  TOTAL:                         {total:>10,d} ({total / 1e6:.3f}M)")
        print(f"  budget:                        {CFG.trainable_param_budget:>10,d}")

    assert total < CFG.trainable_param_budget, (
        f"fusion + drift + planner = {total:,d} params, which exceeds the "
        f"budget of {CFG.trainable_param_budget:,d}"
    )


def test_fusion_fits_its_individual_cap():
    fusion = count_trainable_params(SlotResonanceFusion(CFG))
    assert fusion <= FUSION_CAP, (
        f"SlotResonanceFusion has {fusion:,d} params, exceeding its "
        f"{FUSION_CAP:,d} cap."
    )


def test_drift_fits_its_individual_cap():
    drift = count_trainable_params(AnchoredDriftEncoder(CFG))
    assert drift <= DRIFT_CAP, (
        f"AnchoredDriftEncoder has {drift:,d} params, exceeding its "
        f"{DRIFT_CAP:,d} cap."
    )


def test_planner_fits_its_individual_cap():
    planner = count_trainable_params(ChronoQueryPlanner(CFG))
    assert planner <= PLANNER_CAP, (
        f"ChronoQueryPlanner has {planner:,d} params, exceeding its "
        f"{PLANNER_CAP:,d} cap."
    )


def test_mock_trm_is_a_small_stub():
    mock_trm = count_trainable_params(MockTRM(CFG))
    assert mock_trm < MOCK_TRM_STUB_CAP, (
        f"MockTRM has {mock_trm:,d} params; it must stay a tiny stub "
        f"(< {MOCK_TRM_STUB_CAP:,d}) — the real ~10M TRM is built externally."
    )
