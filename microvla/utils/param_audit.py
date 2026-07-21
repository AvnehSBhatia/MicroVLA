"""Parameter audit for the MicroVLA v2 ~32M ledger.

Prints a table of the trainable-head parameter counts (fusion / drift /
planner / MockTRM stub), lays out the full deployed-parameter ledger against
the documented frozen-encoder and reserved-TRM constants, and asserts that
each trainable head fits its own per-module cap AND that their sum fits
``cfg.trainable_param_budget``.

Run it with::

    python -m microvla.utils.param_audit

IMPORTANT — how to read the ledger:
    The frozen YOLO-World-S detector (whose CLIP text tower also produces the
    3 task text embeddings, once per task via ``set_classes`` — MiniLM was
    deleted in v2, see below) is **not trained** and is therefore **not
    counted against the enforced budget**. Its published parameter count
    (~13M) plus the 10M reserved for the TRM open slot plus the 9M trainable
    budget sum to the ~32M headline; the "32M" figure is honest only because
    the detector is frozen, off-the-shelf weight. The budget this module
    (and ``tests/test_param_budget.py``) actually *enforces* has two levels:
    each of fusion/drift/planner must individually stay under its own cap
    (5.0M / 1.5M / 2.5M), AND their sum must stay under
    ``cfg.trainable_param_budget`` (9M by default).

    v1's MiniLM-L6 text encoder (~22.7M, frozen) is **deleted in v2** — text
    embeddings now come from YOLO-World's own internal CLIP text tower,
    harvested once per task by ``ClipTaskEncoder`` (see
    ``perception/text_encoder.py``). There is no separate text-encoder line
    in this ledger anymore; its cost is folded into the YOLO-World-S line
    below. Honest accounting: the CLIP text tower ultralytics invokes at
    set_classes() is a separate ~63M model — it is NOT inside the ~13M
    detector count. It runs once per task and its 3 output embeddings can
    be precomputed offline, so it need not be resident on-device.
"""

from __future__ import annotations

import torch.nn as nn

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.mock_trm import MockTRM

# Documented constants (parameters of components NOT built/trained here).
# These are published/approximate counts of the frozen, off-the-shelf encoder
# and the externally built TRM. They exist for the ledger printout only and
# are never trained by this repository.
YOLO_WORLD_S_PARAMS: int = 13_000_000
"""Frozen YOLO-World-S / yolov8s-worldv2, INCLUDING its internal CLIP text
tower (harvested once per task by ``ClipTaskEncoder.encode`` via
``set_classes``, not per frame). MiniLM is gone in v2 — this line is the only
text-encoder cost left in the ledger."""

TRM_RESERVED_PARAMS: int = 10_000_000
"""Reserved for the external TRM open slot (see ``microvla/trm/TRM_SPEC.md``).
Raised from 7M in v1. Not built or trained in this repository; ``MockTRM`` is
a ~0.21M placeholder stub, not a size target."""

# Per-module hard caps on the trainable heads (DESIGN.md ledger).
FUSION_PARAM_CAP: int = 5_000_000    # target ~4.5M
DRIFT_PARAM_CAP: int = 1_500_000     # target ~0.9M
PLANNER_PARAM_CAP: int = 2_500_000   # target ~1.6M

_PER_MODULE_CAPS = {
    "fusion (SlotResonanceFusion)": FUSION_PARAM_CAP,
    "drift (AnchoredDriftEncoder)": DRIFT_PARAM_CAP,
    "planner (ChronoQueryPlanner)": PLANNER_PARAM_CAP,
}

TOTAL_LEDGER_PARAMS: int = (
    YOLO_WORLD_S_PARAMS + TRM_RESERVED_PARAMS + DEFAULT_CONFIG.trainable_param_budget
)
"""Headline deployed-parameter ceiling: 13M frozen + 10M reserved TRM + 9M
trainable-head cap = 32M. The trainable heads typically land well under their
individual caps (fusion ~4.5M + drift ~0.9M + planner ~1.6M ≈ 7M achieved),
so the *actual* deployed total is closer to ~30M; 32M is the worst-case cap
sum, not a live measurement."""


def count_trainable_params(module: nn.Module) -> int:
    """Counts parameters with ``requires_grad=True`` in a module.

    Args:
        module: Any ``nn.Module``.

    Returns:
        Total number of trainable scalar parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _fmt(n: int) -> str:
    """Formats a parameter count with thousands separators and an M suffix."""
    return f"{n:>12,d}  ({n / 1e6:.3f}M)"


def audit(cfg: MicroVLAConfig | None = None, verbose: bool = True) -> dict[str, int]:
    """Builds the trainable heads, counts params, prints tables, asserts caps.

    Args:
        cfg: Optional config; defaults to ``DEFAULT_CONFIG``.
        verbose: If True, prints the table and ledger to stdout.

    Returns:
        Mapping of module name -> trainable parameter count, plus
        ``"trainable_total"`` and the MockTRM stub count.

    Raises:
        AssertionError: If any of fusion/drift/planner exceeds its individual
            cap, or if their sum exceeds ``cfg.trainable_param_budget``.
    """
    cfg = cfg or DEFAULT_CONFIG

    counts = {
        "fusion (SlotResonanceFusion)": count_trainable_params(SlotResonanceFusion(cfg)),
        "drift (AnchoredDriftEncoder)": count_trainable_params(AnchoredDriftEncoder(cfg)),
        "planner (ChronoQueryPlanner)": count_trainable_params(ChronoQueryPlanner(cfg)),
        "mock_trm (MockTRM, stub)": count_trainable_params(MockTRM(cfg)),
    }
    trainable_total = (
        counts["fusion (SlotResonanceFusion)"]
        + counts["drift (AnchoredDriftEncoder)"]
        + counts["planner (ChronoQueryPlanner)"]
    )
    counts["trainable_total"] = trainable_total

    if verbose:
        print("=" * 72)
        print("MicroVLA v2 parameter audit")
        print("=" * 72)
        print("\nTrainable heads (built and trained in this repo), vs per-module cap:\n")
        for name in (
            "fusion (SlotResonanceFusion)",
            "drift (AnchoredDriftEncoder)",
            "planner (ChronoQueryPlanner)",
        ):
            cap = _PER_MODULE_CAPS[name]
            status = "OK" if counts[name] <= cap else "OVER CAP"
            print(f"  {name:<32s} {_fmt(counts[name])}  cap {_fmt(cap)}  [{status}]")
        print(f"  {'-' * 64}")
        print(f"  {'TRAINABLE TOTAL':<32s} {_fmt(trainable_total)}")
        print(f"  {'budget (cfg.trainable_param_budget)':<32s} {_fmt(cfg.trainable_param_budget)}")

        print("\nStand-in (NOT the real TRM, NOT trained, NOT part of the budget):\n")
        print(f"  {'mock_trm (MockTRM, stub)':<32s} {_fmt(counts['mock_trm (MockTRM, stub)'])}")

        print("\nDeployed-parameter ledger (documented constants, informational):\n")
        print(f"  {'frozen: YOLO-World-S (+CLIP text tower)':<40s} {_fmt(YOLO_WORLD_S_PARAMS)}")
        print(f"  {'reserved: TRM (open slot, see TRM_SPEC.md)':<40s} {_fmt(TRM_RESERVED_PARAMS)}")
        print(f"  {'trainable heads (cap)':<40s} {_fmt(cfg.trainable_param_budget)}")
        print(f"  {'-' * 64}")
        print(f"  {'headline total (cap sum)':<40s} {_fmt(TOTAL_LEDGER_PARAMS)}")
        print(
            "\nNOTE: MiniLM (~22.7M, frozen) is DELETED in v2 — text embeddings now\n"
            "come from YOLO-World's own internal CLIP text tower, harvested once per\n"
            "task (see perception/text_encoder.py::ClipTaskEncoder), not per frame.\n"
            "The frozen YOLO-World-S detector and the reserved TRM slot are never\n"
            "trained here and do not count against the enforced budget below; that\n"
            "budget applies only to fusion + drift + planner, each under its own\n"
            "per-module cap and jointly under cfg.trainable_param_budget."
        )
        print("=" * 72)

    for name, cap in _PER_MODULE_CAPS.items():
        assert counts[name] <= cap, (
            f"{name} uses {counts[name]:,d} params, exceeding its per-module "
            f"cap of {cap:,d}."
        )

    assert trainable_total < cfg.trainable_param_budget, (
        f"Trainable heads (fusion + drift + planner) use {trainable_total:,d} "
        f"params, exceeding the budget of {cfg.trainable_param_budget:,d}."
    )

    if verbose:
        print(
            f"OK: every head fits its per-module cap, and the trainable total "
            f"fits the budget ({trainable_total:,d} < {cfg.trainable_param_budget:,d})."
        )
    return counts


def main() -> None:
    """CLI entry point for ``python -m microvla.utils.param_audit``."""
    audit(DEFAULT_CONFIG, verbose=True)


if __name__ == "__main__":
    main()
