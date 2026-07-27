"""v8 drop-in adapters, so the v8 stack reaches every existing call site.

``EvidenceEncoder`` and ``HRMBackbone`` have their own, cleaner signatures. The
training loop, the JEPA loop and the 2 Hz pipeline between them call
``fusion(...)`` and ``drift(...)`` at a dozen sites, each with the v7 signature.
Branching every one of those on ``args.v8`` would duplicate the control flow of
the whole trainer and give the two stacks separate code paths to drift apart in.

So the v8 modules are wrapped to PRESENT the v7 signature instead. Every
existing site keeps working untouched, the two stacks share one control flow,
and the only genuinely new call — the relational head, which runs after the TRM
and therefore has no v7 counterpart — is the only thing the caller adds.

The adapters are thin by design: they do shape adaptation and nothing else. All
learned behaviour lives in the wrapped module.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from microvla.config import MicroVLAConfig
from microvla.hrm import HRMBackbone
from microvla.relational.evidence import EvidenceEncoder


class FusionAdapter(nn.Module):
    """Presents :class:`EvidenceEncoder` with ``SlotResonanceFusion``'s signature.

    Maps the baked two-role box evidence (source, target) into K object slots
    padded to ``cfg.max_objects``, then encodes to the TRM's ``[B, 32, 5]`` port.
    Pad slots carry weight 0.0 and are bit-identically inert.

    The ``box_weight`` argument keeps its exact v7 meaning — confidence x
    freshness in [0, 1], faded by ``staleness_decay**k`` on dream ticks and by
    ``modality_dropout`` at train time — so the graded-evidence contract is
    unchanged by the swap.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = EvidenceEncoder(cfg)

    def forward(
        self,
        text_tokens: torch.Tensor,
        frame_emb: torch.Tensor,
        source_box_emb: torch.Tensor,
        target_box_emb: torch.Tensor,
        source_center: torch.Tensor,
        target_center: torch.Tensor,
        box_weight: Optional[torch.Tensor] = None,
        last_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obj, ctr, w = pack_objects(
            source_box_emb, target_box_emb, source_center, target_center,
            box_weight, self.cfg,
        )
        return self.encoder(obj, ctr, w, frame_emb, text_tokens,
                            last_action=last_action)


class DriftAdapter(nn.Module):
    """Presents :class:`HRMBackbone` with ``AnchoredDriftEncoder``'s signature.

    ``is_real=True`` always: the trainer runs at the baked 2 Hz data rate, where
    every step IS a real perception tick. The fast/slow split only becomes
    observable in the 30 Hz deployment loop, which steps the fast module alone
    on dream ticks — that path is the JEPA loop's, not the trainer's.

    ``gains`` (the learned control law) is dropped here because the v7 signature
    has nowhere to put it; callers wanting it hold the ``HRMBackbone`` directly
    via :attr:`hrm`.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.hrm = HRMBackbone(cfg)

    def reset(self) -> None:
        self.hrm.reset()

    def forward(self, frame_emb: torch.Tensor) -> torch.Tensor:
        return self.hrm(frame_emb, is_real=True).state


def objects_from_batch(batch, idx, fade, cfg):
    """Per-step object tokens, preferring the BAKED class-agnostic scene.

    A v8 corpus carries ``obj_*``: every box the detector's forward produced,
    padded to ``cfg.max_objects``. A v7 corpus does not, so the two role slots
    are packed instead and effective K is 2.

    ``fade`` scales the weights exactly as ``_boxes`` does — held dream evidence
    decays by ``staleness_decay**k`` — so the graded-evidence contract is the
    same whichever corpus is loaded.

    Returns ``(obj_emb [B,K,vis_dim], obj_center [B,K,2], obj_weight [B,K])``.
    """
    # `has_objects`, not an all-zero check: a v7 corpus is zero-filled, and a
    # legitimately empty frame in a v8 corpus is ALSO all-zero. Only provenance
    # separates them, and guessing wrong silently starves the relational head.
    if "obj_embs" in batch and float(batch.get("has_objects", torch.zeros(1)).max()) > 0.5:
        return (batch["obj_embs"][:, idx],
                batch["obj_centers"][:, idx],
                batch["obj_weights"][:, idx] * fade)
    return pack_objects(
        batch["source_box_embs"][:, idx], batch["target_box_embs"][:, idx],
        batch["source_centers"][:, idx], batch["target_centers"][:, idx],
        batch["box_weights"][:, idx] * fade, cfg,
    )


def pack_objects(
    source_box_emb: torch.Tensor,
    target_box_emb: torch.Tensor,
    source_center: torch.Tensor,
    target_center: torch.Tensor,
    box_weight: Optional[torch.Tensor],
    cfg: MicroVLAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two baked roles -> K padded object slots.

    Returns ``(obj_emb [B,K,vis_dim], obj_center [B,K,2], obj_weight [B,K])``
    with ``K = cfg.max_objects``. Slots 0 and 1 are source and target; the rest
    are zeros at weight 0.0.

    This is where v8's effective K is 2 rather than ``cfg.max_objects``: the v7
    bake ran YOLO-World with ``set_classes([source, target])`` and kept one box
    per role, so the corpus holds no other proposals to put in the remaining
    slots. ``microvla/perception/text_region.py`` is what supplies K
    class-agnostic proposals; using it requires re-baking the corpus.
    """
    b, k = source_box_emb.shape[0], cfg.max_objects
    if k < 2:
        raise ValueError(f"cfg.max_objects must be >= 2 to hold the source and "
                         f"target roles; got {k}.")
    obj = source_box_emb.new_zeros(b, k, cfg.vis_dim)
    ctr = source_box_emb.new_zeros(b, k, 2)
    w = source_box_emb.new_zeros(b, k)
    obj[:, 0], obj[:, 1] = source_box_emb, target_box_emb
    ctr[:, 0], ctr[:, 1] = source_center, target_center
    if box_weight is not None:
        w[:, :2] = box_weight
    else:
        w[:, :2] = 1.0
    return obj, ctr, w
