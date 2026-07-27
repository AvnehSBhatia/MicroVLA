"""Evidence port for the TRM, in the v8 stack.

Removing ``SlotResonanceFusion`` leaves the TRM's evidence input unfed. This
module fills exactly that hole and nothing more.

**Why the TRM's ``[B, 32, 5]`` contract is preserved rather than widened.** The
TRM is the only component of this project with a positive measured result
(``wm_margin`` +19.8%, paper.md 4g/4m). Widening its evidence port changes
``RecursiveTRM.embed`` and ``pos``, which forfeits that result and forces it to
be re-earned. The v8 redesign is motivated by a *planner-side* failure — a
closed-loop policy emitting a near-constant direction (per-axis ``|cmd|`` x
0.1186, y 0.8550, z 0.2420 at 0% clipping, paper.md 4m) — so the richness
belongs where the reasoning happens, not at the world model's input.

**So "data rich" is honoured where it pays.** ``RelationalHead`` receives the
full ``[B, K, vis_dim]`` object tokens — 8 x 512 = 4096 floats. This encoder
compresses to the TRM's 160-float port and is deliberately the *only* narrow
point in the stack. It performs no cross-object reasoning: pairwise structure is
``RelationalHead``'s job, downstream of the TRM.

The graded evidence fade is applied here on the same terms as everywhere else
(CLAUDE.md): ``obj_weight`` multiplies PROJECTED object content, dream ticks
pass held boxes at ``confidence * staleness_decay**k``, misses pass 0.0, and the
last-executed-action token is never faded.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from microvla.config import MicroVLAConfig

#: Width each source is projected to before assembly. Small on purpose: this is
#: a port, not a representation, and every extra unit here buys nothing the
#: relational head cannot do better downstream.
_PROJ_DIM: int = 32


class EvidenceEncoder(nn.Module):
    """Packs perception evidence into the TRM's ``[B, fused_rows, fused_cols]`` port.

    Args:
        cfg: Canonical config. Reads ``max_objects``, ``vis_dim``, ``text_dim``,
            ``num_servos``, ``fused_rows`` and ``fused_cols`` — the output shape
            is whatever the TRM contract says it is, never a literal.

    Shape:
        - Input: ``obj_emb [B, K, vis_dim]``, ``obj_center [B, K, 2]``,
          ``obj_weight [B, K]``, ``frame_emb [B, vis_dim]``,
          ``text_tokens [B, n_text, text_dim]``,
          ``last_action [B, num_servos]`` (optional).
        - Output: ``[B, cfg.fused_rows, cfg.fused_cols]``.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.k = cfg.max_objects
        out_dim = cfg.fused_rows * cfg.fused_cols

        # Object tokens carry their normalized center alongside the embedding:
        # "which thing" and "where it is" enter through one projection, so a
        # faded object loses both together rather than leaving a live position
        # attached to dead content.
        self.obj_proj = nn.Linear(cfg.vis_dim + 2, _PROJ_DIM)
        self.frame_proj = nn.Linear(cfg.vis_dim, _PROJ_DIM)
        self.text_proj = nn.Linear(cfg.text_dim, _PROJ_DIM)
        self.action_proj = nn.Linear(cfg.num_servos, _PROJ_DIM)

        n_sources = self.k + 1 + cfg.n_text_tokens + 1
        self.assemble = nn.Linear(n_sources * _PROJ_DIM, out_dim)

    def forward(
        self,
        obj_emb: torch.Tensor,
        obj_center: torch.Tensor,
        obj_weight: torch.Tensor,
        frame_emb: torch.Tensor,
        text_tokens: torch.Tensor,
        last_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = obj_emb.shape[0]
        if obj_emb.shape[1] != self.k:
            raise ValueError(
                f"obj_emb has {obj_emb.shape[1]} objects, expected "
                f"cfg.max_objects ({self.k})."
            )

        # Fade applied to PROJECTED content, never by zeroing the raw input:
        # the same graded path serves dream staleness and train-time modality
        # dropout, which is the training-inference alignment claim.
        obj = self.obj_proj(torch.cat([obj_emb, obj_center], dim=-1))  # [B,K,P]
        obj = obj * obj_weight.unsqueeze(-1)

        frame = self.frame_proj(frame_emb).unsqueeze(1)                # [B,1,P]
        text = self.text_proj(text_tokens)                             # [B,n_text,P]

        if last_action is None:
            last_action = obj_emb.new_zeros(b, self.cfg.num_servos)
        # Never faded: the previously executed action is proprioceptive fact,
        # not perceptual evidence, so staleness does not apply to it.
        action = self.action_proj(last_action).unsqueeze(1)            # [B,1,P]

        flat = torch.cat([obj, frame, text, action], dim=1).flatten(1)
        return self.assemble(flat).view(b, self.cfg.fused_rows, self.cfg.fused_cols)
