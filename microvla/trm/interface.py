"""Abstract interface for the TRM (Tiny Recursive Model) open slot.

The TRM is the world-model core of MicroVLA: it consumes the fused
task/perception matrix and the accumulated scene-drift code, and predicts the
YOLO-World frame embedding of the *next* sampled frame — at inference this
prediction also drives the JEPA loop's dream ticks (see
``microvla/jepa/loop.py``), fed back through fusion's dream path via the
:class:`~microvla.jepa.corrector.InnovationCorrector`. The real ~10M-param TRM
is built externally (see ``TRM_SPEC.md`` in this directory); any
implementation must subclass :class:`TRMBase` so it drops into
``MicroVLAPipeline`` / ``JEPALoop`` unchanged.
"""

from __future__ import annotations

import abc

import torch
from torch import nn


class TRMBase(nn.Module, abc.ABC):
    """Interface contract for TRM implementations.

    Contract (binding — see DESIGN.md and TRM_SPEC.md). All dimensions below
    are read from :class:`microvla.config.MicroVLAConfig` — implementations
    should take ``cfg: MicroVLAConfig`` as their first constructor argument
    (repo convention) rather than hardcoding sizes:

    Inputs:
        * ``fused``: ``[B, cfg.fused_rows=32, cfg.fused_cols=5]`` float32 —
          the output of
          :class:`~microvla.fusion.slot_fusion.SlotResonanceFusion`. Each of
          the 32 rows is a learned slot's low-rank summary of the current
          (text, frame, box, geometry) observation on a real tick, or of the
          dream-mode observation (frame token = corrected TRM prediction,
          box/geometry tokens zeroed) on a dream tick; values are
          unconstrained reals.
        * ``state_delta``: ``[B, cfg.state_dim=256]`` float32 — the output of
          :class:`~microvla.aux_state.drift_encoder.AnchoredDriftEncoder`, a
          LayerNorm'd code summarizing how far the scene has drifted from the
          episode's anchor (first REAL) frame. Exactly zero on the anchor
          frame itself (zero drift by definition). At dream ticks the drift
          encoder is fed the corrected latent, so this runs at the full
          ``cfg.tick_hz`` (30 Hz), not just at real-frame rate.

    Output:
        * ``next_emb``: ``[B, cfg.vis_dim=512]`` float32 — the predicted
          YOLO-World-S frame embedding (GAP of the SPPF/P5 map) of the frame
          expected **one tick ahead**. This tensor is fed directly to
          :class:`~microvla.planner.chrono_planner.ChronoQueryPlanner` and,
          at inference, becomes the next tick's "pending prediction" —
          corrected and re-fed through ``fused`` on the following dream tick,
          or checked against the next real measurement by the
          :class:`~microvla.jepa.corrector.InnovationCorrector`.

    Behavioral requirements:
        * Pure function of its inputs — no cross-call recurrent state. Episode
          memory lives in the drift encoder; the TRM must be stateless so the
          pipeline/loop can reset episodes without touching it.
        * Must accept any batch size ``B >= 1`` and preserve it.
        * Must run on CPU; device placement follows the module's parameters as
          usual for ``nn.Module``.
        * Trainable parameter budget for the real implementation: ~10M
          (reserved in the ~32M deployed ledger; not counted against the 9M
          fusion + drift + planner budget — see ``TRM_SPEC.md``).
        * Must be trained for multi-step open-loop rollout (see
          ``TRM_SPEC.md`` section 5) since at inference it runs up to
          ``cfg.dream_ticks_per_real`` (14) consecutive ticks without a real
          measurement.
    """

    @abc.abstractmethod
    def forward(self, fused: torch.Tensor, state_delta: torch.Tensor) -> torch.Tensor:
        """Predict the next-tick YOLO embedding.

        Args:
            fused: ``[B, 32, 5]`` fused slot matrix from SlotResonanceFusion.
            state_delta: ``[B, 256]`` drift code from AnchoredDriftEncoder.

        Returns:
            ``[B, 512]`` predicted next-tick YOLO-World frame embedding.
        """
        raise NotImplementedError
