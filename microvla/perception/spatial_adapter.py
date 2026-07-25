"""Text-Queried Spatial Adapter (TQSA, v7): trainable spatial perception.

The diagnosed ceiling of the BC policy is the observation: GAP destroys the
spatial structure of the frozen backbone's feature map, and no amount of
downstream conditioning can recover "WHERE is the thing I must reach". TQSA is
a small (~0.13M) trainable head on the FROZEN YOLO-World backbone's hooked
SPPF map that produces task-conditioned spatial features:

1. ``v_proj``: 1x1 conv, backbone channels (512) -> ``cfg.tqsa_dim`` (128).
2. Per-role text queries: the three CLIP text embeddings (command, source,
   target) are projected to the same width; ``attn_j = softmax_HW(t_j . v_hw /
   sqrt(d))`` gives one ATTENTION MAP per role — "where does 'black bowl'
   light up" — the task-conditioned spatial signal the user asked for, learned
   end-to-end with the policy.
3. Outputs consumed by the planner (fusion integration deferred — it feeds the
   frozen world model, whose input space must not shift under it):
   * ``pooled  [B, 3, d]``   — attention-pooled feature per role,
   * ``tokens  [B, g*g, d]`` — a g x g grid of spatial tokens (structure GAP
     destroyed; the planner cross-attends them),
   * ``heatmaps [B, 3, h*h]`` — the per-role attention maps, area-averaged to
     h x h and re-normalized (a compact "where" the planner reads directly).

The backbone stays frozen; only this adapter trains (stage B). At 30 Hz the
loop runs TQSA on REAL ticks and holds its outputs across dream ticks, exactly
like held box evidence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from microvla.config import MicroVLAConfig


class TextQueriedSpatialAdapter(nn.Module):
    """Trainable text-queried attention over a frozen backbone feature map.

    Args:
        cfg: Canonical config; supplies ``vis_dim`` (backbone channels),
            ``text_dim``, ``tqsa_dim``, ``tqsa_grid``, ``tqsa_heat``,
            ``n_text_tokens``.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.tqsa_dim
        self.v_proj = nn.Conv2d(cfg.vis_dim, d, kernel_size=1)
        self.t_proj = nn.Linear(cfg.text_dim, d)
        self.v_norm = nn.GroupNorm(1, d)   # layer-norm-ish over the map
        self.out_norm = nn.LayerNorm(d)

    def forward(
        self, feat_map: torch.Tensor, text_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Computes attention maps + pooled features + spatial tokens.

        Args:
            feat_map: ``[B, vis_dim, Hf, Wf]`` hooked (frozen) backbone map.
            text_tokens: ``[B, n_text_tokens, text_dim]`` ordered CLIP text
                embeddings (command, source, target).

        Returns:
            Dict with ``pooled [B, 3, d]``, ``tokens [B, g*g, d]``,
            ``heatmaps [B, 3, h*h]`` (each role's map sums to 1).
        """
        cfg = self.cfg
        v = self.v_norm(self.v_proj(feat_map))            # [B, d, Hf, Wf]
        B, d, Hf, Wf = v.shape
        flat = v.flatten(2).transpose(1, 2)               # [B, HW, d]

        t = self.t_proj(text_tokens)                      # [B, 3, d]
        logits = torch.einsum("brd,bnd->brn", t, flat) / math.sqrt(d)  # [B, 3, HW]
        attn = torch.softmax(logits, dim=-1)              # per-role map over HW

        pooled = self.out_norm(torch.einsum("brn,bnd->brd", attn, flat))  # [B, 3, d]

        g = cfg.tqsa_grid
        tokens = F.adaptive_avg_pool2d(v, (g, g))         # [B, d, g, g]
        tokens = self.out_norm(tokens.flatten(2).transpose(1, 2))  # [B, g*g, d]

        h = cfg.tqsa_heat
        heat = attn.reshape(B, text_tokens.shape[1], Hf, Wf)
        heat = F.adaptive_avg_pool2d(heat, (h, h)).flatten(2)      # [B, 3, h*h]
        heat = heat / heat.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return {"pooled": pooled, "tokens": tokens, "heatmaps": heat}


if __name__ == "__main__":
    from microvla.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    tqsa = TextQueriedSpatialAdapter(cfg)
    n = sum(p.numel() for p in tqsa.parameters() if p.requires_grad)
    out = tqsa(torch.randn(2, cfg.vis_dim, 12, 20), torch.randn(2, 3, cfg.text_dim))
    print(f"params {n:,} | pooled {tuple(out['pooled'].shape)} "
          f"tokens {tuple(out['tokens'].shape)} heat {tuple(out['heatmaps'].shape)}")
    assert out["pooled"].shape == (2, 3, cfg.tqsa_dim)
    assert out["tokens"].shape == (2, cfg.tqsa_grid**2, cfg.tqsa_dim)
    assert out["heatmaps"].shape == (2, 3, cfg.tqsa_heat**2)
    assert torch.allclose(out["heatmaps"].sum(-1), torch.ones(2, 3), atol=1e-5)
    assert n <= 500_000, f"TQSA over its 0.5M cap: {n:,}"
