"""Foil ``TRMBase`` implementations for the perception-rate sweep (paper E4).

Claim 2 ("perception is not the clock") only means something relative to
what happens *without* a trained world model dreaming between real
perceptions. These two stateless, zero-training foils slot into
``eval.sweep`` via ``MicroVLAPolicy(..., trm=<foil>)`` exactly like the real
``TRM.py::RecursiveTRM`` — same ``TRMBase`` contract
(``(fused [B,32,5], state_delta [B,256], current_emb [B,512],
context=None [B,K,512]) -> [B,512]``), so the JEPA loop and
``InnovationCorrector`` never need to know which one is plugged in:

* :class:`PersistenceTRM` — the "hold-last / no-world-model" foil. With it,
  dream latents never move: the policy keeps acting on the last real
  perception for the entire dream window. If ``ours`` and this baseline
  degrade identically as ``perception_period`` grows, the world model adds
  nothing (paper.md's Claim-2 kill bar).
* :class:`LinearExtrapolationTRM` — the "cheap dreamer" foil. Extrapolates
  the last observed velocity in latent space (``current + (current -
  previous)``) rather than predicting nothing. The TRM must beat this to be
  more than decoration (paper.md's baselines section).

Neither has learned parameters, so neither needs a checkpoint or training:
construct with just ``cfg`` and pass straight into ``MicroVLAPolicy``'s
``trm=`` override.
"""

from __future__ import annotations

import torch

from microvla.config import MicroVLAConfig
from microvla.trm.interface import TRMBase


class PersistenceTRM(TRMBase):
    """Hold-last-observation foil: predicts no change at all.

    ``forward`` returns ``current_emb`` unchanged, so every dream tick
    downstream (fusion's dream path, the planner) keeps re-deriving a plan
    from the same frozen latent the last real perception produced. Zero
    learned parameters — ``state_dict()`` is empty and there is nothing to
    checkpoint or train.

    Args:
        cfg: Shared MicroVLA configuration (kept for the ``TRMBase``
            constructor convention; unused since the module has no shaped
            weights).
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        fused: torch.Tensor,
        state_delta: torch.Tensor,
        current_emb: torch.Tensor,
        context: torch.Tensor | None = None,
        return_box: bool = False,
    ) -> torch.Tensor:
        """Returns ``current_emb`` unchanged (``fused``/``state_delta``/
        ``context`` accepted for contract compatibility and ignored).

        Args:
            fused: ``[B, 32, 5]`` fused slot matrix. Ignored.
            state_delta: ``[B, 256]`` drift code. Ignored.
            current_emb: ``[B, 512]`` current standardized frame embedding.
            context: Optional ``[B, K, 512]`` latent context window. Ignored.
            return_box: When ``True`` also returns a "no-change" box prediction
                (``current_emb``), for v4 contract parity.

        Returns:
            ``current_emb``, verbatim — the zero-delta prediction (or
            ``(current_emb, current_emb)`` when ``return_box``).
        """
        return (current_emb, current_emb) if return_box else current_emb


class LinearExtrapolationTRM(TRMBase):
    """"Cheap dreamer" foil: constant-velocity extrapolation in latent space.

    Given a context window of recent tick latents, predicts
    ``current_emb + (current_emb - context[:, -1])`` — i.e. it continues
    whatever motion the last two observed latents implied. With no context
    (episode start, or a caller that never supplies one) it falls back to
    :class:`PersistenceTRM`'s zero-delta prediction, since there is nothing
    to extrapolate from. Zero learned parameters, stateless.

    Args:
        cfg: Shared MicroVLA configuration (kept for the ``TRMBase``
            constructor convention; unused).
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        fused: torch.Tensor,
        state_delta: torch.Tensor,
        current_emb: torch.Tensor,
        context: torch.Tensor | None = None,
        return_box: bool = False,
    ) -> torch.Tensor:
        """Linearly extrapolates from the last two latents in ``context``.

        Args:
            fused: ``[B, 32, 5]`` fused slot matrix. Ignored.
            state_delta: ``[B, 256]`` drift code. Ignored.
            current_emb: ``[B, 512]`` current standardized frame embedding.
            context: Optional ``[B, K, 512]`` latent context window, oldest
                -> newest. When ``K >= 1``, ``context[:, -1]`` is treated as
                the previous tick's latent and the "velocity"
                ``current_emb - context[:, -1]`` is added again. Ignored
                (falls back to zero-delta) when ``None`` or ``K == 0``.
            return_box: When ``True`` also returns a "no-change" box prediction
                (``current_emb``), for v4 contract parity.

        Returns:
            ``[B, 512]``: ``current_emb + (current_emb - context[:, -1])``
            with context, else ``current_emb`` (or a ``(next_emb, box)`` pair
            when ``return_box``, ``box = current_emb``).
        """
        if context is None or context.shape[1] == 0:
            return (current_emb, current_emb) if return_box else current_emb
        previous = context[:, -1]
        next_emb = current_emb + (current_emb - previous)
        return (next_emb, current_emb) if return_box else next_emb
