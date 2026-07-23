"""Mock stand-in for the TRM open slot.

.. warning::
    PLACEHOLDER ONLY. THIS IS NOT THE TRM. It exists solely so the MicroVLA
    pipeline and JEPA loop run end-to-end (tests, smoke training, param
    audits) before the real ~10M-parameter Tiny Recursive Model is
    delivered. It is a single linear map with zero recursion, zero
    attention, zero conditioning depth, and no predictive power — it will
    NOT produce a usable next-frame prediction, and multi-step JEPA dream
    rollouts driven by it will drift immediately. Replace it with a real
    ``TRMBase`` subclass built to ``TRM_SPEC.md`` before deploying anything
    that depends on dream-tick quality.
"""

from __future__ import annotations

import torch
from torch import nn

from microvla.config import MicroVLAConfig
from microvla.trm.interface import TRMBase


class MockTRM(TRMBase):
    """Trivial linear placeholder for the TRM slot (~0.21M params).

    .. warning::
        PLACEHOLDER ONLY — replace with the real ~10M TRM (see
        ``TRM_SPEC.md``). This class is deliberately as dumb as possible: it
        exists to unblock everything downstream of the TRM slot, not to
        predict anything meaningful.

    Flattens the fused slot matrix ``[B, 32, 5]`` to ``[B, 160]``,
    concatenates the drift code ``[B, 256]`` to get ``[B, 416]``, applies one
    ``Linear(416, 512)``, and adds the result to ``current_emb`` (the
    residual convention from the TRMBase contract — even the mock predicts a
    change, not a reconstruction). Honors the
    :class:`~microvla.trm.interface.TRMBase` I/O contract exactly so it is
    drop-in swappable.

    Args:
        cfg: Shared MicroVLA configuration; supplies all dimensions.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.fused_rows * cfg.fused_cols + cfg.state_dim  # 32*5 + 256 = 416
        self.proj = nn.Linear(in_dim, cfg.vis_dim)
        # Placeholder box head (v4 contract parity) — low-rank so the mock stays
        # a tiny stub (the real box head lives in TRM.py::RecursiveTRM).
        self.box_proj = nn.Sequential(
            nn.Linear(in_dim, 32), nn.Linear(32, cfg.vis_dim)
        )

    def forward(
        self,
        fused: torch.Tensor,
        state_delta: torch.Tensor,
        current_emb: torch.Tensor,
        context: torch.Tensor | None = None,
        return_box: bool = False,
    ) -> torch.Tensor:
        """Map inputs to a placeholder residual next-tick embedding.

        Args:
            fused: ``[B, 32, 5]`` fused slot matrix.
            state_delta: ``[B, 256]`` drift code.
            current_emb: ``[B, 512]`` current standardized frame embedding.
            context: Optional latent context window — accepted for contract
                compatibility and IGNORED (this is a stub).
            return_box: When ``True`` also return a placeholder next-tick box
                embedding (contract parity with the real TRM v4).

        Returns:
            ``[B, 512]`` placeholder next-tick embedding
            (``current_emb + linear delta``), or ``(next_emb, next_box)`` when
            ``return_box``.
        """
        flat = fused.flatten(start_dim=1)                 # [B, 160]
        x = torch.cat([flat, state_delta], dim=-1)        # [B, 416]
        next_emb = current_emb + self.proj(x)             # [B, 512]
        if return_box:
            return next_emb, self.box_proj(x)             # [B, 512], [B, 512]
        return next_emb
