"""Slot Resonance Fusion v3 — slot competition over FiLM-conditioned, role-tagged tokens.

This module implements the fusion stage of the MicroVLA stack. The core idea
is **slot competition over FiLM-conditioned, role-tagged modality tokens**:

1.  Eight modality tokens are built at ``d_model``: the three ordered CLIP
    text embeddings (command, source noun phrase, target noun phrase — one
    *shared* ``Linear(text_dim, d_model)`` projects all three), the full-frame
    visual embedding, the source-box and target-box visual embeddings (one
    *shared* ``Linear(vis_dim, d_model)`` projects all three visual inputs),
    a geometry token built from Fourier features of the source center, the
    target center, and their relative displacement (``target − source``)
    plus the two box-evidence weights, and an **action token** projecting the
    previously executed servo command (``Linear(num_servos, d_model)``) — the
    world-model path downstream cannot predict controlled dynamics without
    knowing what the controller just did.
2.  A learned **role-embedding table** ``[8, d_model]`` is added per fixed
    position (``cmd, src, tgt, frame, src-box, tgt-box, geometry, action``)
    so slots can key on "which stream is this" as well as "what does it say".
3.  The command token doubles as a **FiLM** generator: ``Linear(text_dim,
    2*d_model)`` on the command embedding produces scale/shift applied to the
    three *visual* tokens — language re-tunes what the visual evidence means
    before any attention happens.
4.  ``fused_rows`` (32) learned slot queries run ``n_fusion_blocks`` rounds of
    pre-LN multi-head cross-attention over the 8 tokens (slots as queries,
    tokens as keys/values), each round followed by a pre-LN GELU MLP, all
    residual.
5.  A shared head (``Linear(d_model, 64) -> GELU -> Linear(64, fused_cols)``)
    reads every slot into a 5-vector, producing ``[B, 32, 5]``.

Evidence weighting (v3 — replaces v2's binary dream/zeroing):
    ``box_weight [B, 2]`` scales the source-box and target-box tokens (and the
    geometry token, by their mean), and is also appended raw to the geometry
    features so the network *knows* how much to trust the boxes. Callers set
    it to ``confidence`` on real ticks, ``confidence * staleness_decay**k`` on
    dream ticks (held-last boxes fade smoothly instead of vanishing), and
    ``0`` for a genuinely missed detection — which also disambiguates the
    missed-detection fallback (center 0.5, 0.5) from a real object at frame
    center. Train-time ``modality_dropout`` samples the SAME continuum: with
    probability ``p`` per sample, the weights are multiplied by a uniform
    fade in ``[0, 1)``. Dream ticks are therefore a *trained* regime — the
    network has seen every level of box-evidence decay during training —
    without discarding the near-perfect 33 ms-old boxes v2 used to zero.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from microvla.config import MicroVLAConfig

#: Explicit token order — also the order the role-embedding table indexes.
_TOKEN_ORDER = (
    "command",
    "source_text",
    "target_text",
    "frame",
    "source_box",
    "target_box",
    "geometry",
    "action",
)
_N_TOKENS = len(_TOKEN_ORDER)


class SlotResonanceFusion(nn.Module):
    """Fuses text + frame + dual-box + geometry + last action into ``[B, 32, 5]``.

    See the module docstring for the method description. All dimensions come
    from :class:`~microvla.config.MicroVLAConfig`; nothing is hardcoded.

    Attributes:
        slots: Learned slot queries, ``nn.Parameter [fused_rows, d_model]``.
        role_emb: Learned per-position role embeddings, ``[8, d_model]``.
        modality_dropout: Train-time probability of fading the box evidence
            (one Bernoulli draw + one uniform fade per sample, shared across
            both boxes — box and geometry evidence degrade together, exactly
            as they do across a dream rollout).
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        """Builds the fusion module.

        Args:
            cfg: Canonical MicroVLA configuration; supplies ``text_dim``,
                ``vis_dim``, ``num_servos``, ``d_model``, ``n_heads``,
                ``n_fusion_blocks``, ``n_fourier``, ``fused_rows``,
                ``fused_cols``, and ``modality_dropout``.
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.modality_dropout = cfg.modality_dropout

        # --- Shared modality projections ------------------------------------
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.visual_proj = nn.Linear(cfg.vis_dim, d)
        # Previously executed servo command (row 0 of the last emitted plan).
        self.action_proj = nn.Linear(cfg.num_servos, d)

        # Fourier encoding of a 2D point: frequencies 2**k * pi, k in
        # range(n_fourier); sin + cos per coordinate -> 4 * n_fourier features
        # per point; 3 points (src, tgt, tgt - src) + the 2 raw evidence
        # weights -> 12 * n_fourier + 2 geometry features.
        freqs = torch.tensor(
            [2.0**k * math.pi for k in range(cfg.n_fourier)], dtype=torch.float32
        )
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        self.geom_proj = nn.Linear(6 * 2 * cfg.n_fourier + 2, d)

        # --- FiLM conditioning of the 3 visual tokens from the command -----
        self.film = nn.Linear(cfg.text_dim, 2 * d)

        # --- Role embeddings (explicit fixed order, see _TOKEN_ORDER) ------
        self.role_emb = nn.Parameter(torch.randn(_N_TOKENS, d) * d**-0.5)

        # --- Learned slot queries -------------------------------------------
        self.slots = nn.Parameter(torch.randn(cfg.fused_rows, d) * d**-0.5)

        # --- Cross-attention + MLP blocks (pre-LN, residual) ---------------
        self.attn_slot_norms = nn.ModuleList()
        self.attn_kv_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(cfg.n_fusion_blocks):
            self.attn_slot_norms.append(nn.LayerNorm(d))
            self.attn_kv_norms.append(nn.LayerNorm(d))
            self.attns.append(nn.MultiheadAttention(d, cfg.n_heads, batch_first=True))
            self.mlp_norms.append(nn.LayerNorm(d))
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(d, d * 2),
                    nn.GELU(),
                    nn.Linear(d * 2, d),
                )
            )

        # --- Shared read-out head --------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Linear(64, cfg.fused_cols),
        )

    def _fourier(self, point: torch.Tensor) -> torch.Tensor:
        """Encodes a 2D point with sin/cos Fourier features.

        Args:
            point: ``[B, 2]`` tensor (a center in ``[0, 1]``, or an
                unconstrained displacement such as ``target - source``).

        Returns:
            ``[B, 4 * n_fourier]`` tensor of concatenated sin/cos features.
        """
        angles = point.unsqueeze(-1) * self.fourier_freqs
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return feats.flatten(start_dim=1)

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
        """Fuses the 8 modality inputs into the slot matrix.

        Args:
            text_tokens: ``[B, 3, text_dim]`` ordered (command, source,
                target) CLIP text embeddings.
            frame_emb: ``[B, vis_dim]`` standardized full-frame embedding
                (real YOLO GAP feature on real ticks, corrected TRM latent
                on dream ticks).
            source_box_emb: ``[B, vis_dim]`` source-object box embedding
                (held from the last real tick during dreams).
            target_box_emb: ``[B, vis_dim]`` target-object box embedding.
            source_center: ``[B, 2]`` normalized (cx, cy) of the source box.
            target_center: ``[B, 2]`` normalized (cx, cy) of the target box.
            box_weight: ``[B, 2]`` evidence weight per role in ``[0, 1]``
                (confidence x freshness; ``None`` -> full weight).
            last_action: ``[B, num_servos]`` previously executed servo
                command in ``[-1, 1]`` (``None`` -> zeros, episode start).

        Returns:
            Fused matrix ``[B, fused_rows, fused_cols]`` (``[B, 32, 5]``).
        """
        batch = text_tokens.shape[0]
        if box_weight is None:
            box_weight = frame_emb.new_ones(batch, 2)
        if last_action is None:
            last_action = frame_emb.new_zeros(batch, self.cfg.num_servos)

        # Train-time evidence fade: per sample, with prob modality_dropout,
        # multiply BOTH box weights by one uniform fade in [0, 1) — the same
        # degradation continuum dream ticks produce with staleness decay.
        if self.training and self.modality_dropout > 0.0:
            drop = torch.bernoulli(
                torch.full((batch, 1), self.modality_dropout,
                           device=frame_emb.device, dtype=frame_emb.dtype)
            )
            fade = torch.rand(batch, 1, device=frame_emb.device, dtype=frame_emb.dtype)
            box_weight = box_weight * (1.0 - drop * (1.0 - fade))

        # Shared text projection over the 3 ordered tokens: [B, 3, d].
        text_toks = self.text_proj(text_tokens)
        command_tok = text_toks[:, 0]
        source_text_tok = text_toks[:, 1]
        target_text_tok = text_toks[:, 2]

        # Shared visual projection over frame / source-box / target-box.
        frame_tok = self.visual_proj(frame_emb)
        source_box_tok = self.visual_proj(source_box_emb)
        target_box_tok = self.visual_proj(target_box_emb)

        # Geometry token: Fourier(src) | Fourier(tgt) | Fourier(tgt - src)
        # plus the raw evidence weights, so "how fresh is this geometry" is
        # itself an input feature, not just a multiplier.
        geom_feat = torch.cat(
            [
                self._fourier(source_center),
                self._fourier(target_center),
                self._fourier(target_center - source_center),
                box_weight,
            ],
            dim=-1,
        )
        geom_tok = self.geom_proj(geom_feat)

        # Action token: what the controller actually just did.
        action_tok = self.action_proj(last_action)

        # FiLM: the command token re-tunes the 3 visual tokens.
        scale, shift = self.film(text_tokens[:, 0]).chunk(2, dim=-1)  # each [B, d]
        frame_tok = frame_tok * (1.0 + scale) + shift
        source_box_tok = source_box_tok * (1.0 + scale) + shift
        target_box_tok = target_box_tok * (1.0 + scale) + shift

        # Evidence weighting: box tokens scale with their own weight, the
        # geometry token with the mean (it mixes both boxes). Weight 0 ->
        # token contributes nothing, exactly like v2's zeroing, but every
        # level in between is now representable and trained.
        source_box_tok = source_box_tok * box_weight[:, 0:1]
        target_box_tok = target_box_tok * box_weight[:, 1:2]
        geom_tok = geom_tok * box_weight.mean(dim=1, keepdim=True)

        tokens = torch.stack(
            [
                command_tok,
                source_text_tok,
                target_text_tok,
                frame_tok,
                source_box_tok,
                target_box_tok,
                geom_tok,
                action_tok,
            ],
            dim=1,
        )
        tokens = tokens + self.role_emb.unsqueeze(0)

        # Slot competition: cross-attention rounds (pre-LN, residual).
        slots = self.slots.unsqueeze(0).expand(batch, -1, -1)  # [B, 32, d]
        for slot_norm, kv_norm, attn, mlp_norm, mlp in zip(
            self.attn_slot_norms, self.attn_kv_norms, self.attns, self.mlp_norms, self.mlps
        ):
            q = slot_norm(slots)
            kv = kv_norm(tokens)
            attended, _ = attn(q, kv, kv, need_weights=False)
            slots = slots + attended
            slots = slots + mlp(mlp_norm(slots))

        # Shared read-out, applied per slot: [B, 32, 5].
        return self.head(slots)


if __name__ == "__main__":
    cfg = MicroVLAConfig()
    model = SlotResonanceFusion(cfg)
    model.eval()

    for B in (1, 4):
        text_tokens = torch.randn(B, cfg.n_text_tokens, cfg.text_dim)
        frame_emb = torch.randn(B, cfg.vis_dim)
        source_box_emb = torch.randn(B, cfg.vis_dim)
        target_box_emb = torch.randn(B, cfg.vis_dim)
        source_center = torch.rand(B, 2)
        target_center = torch.rand(B, 2)

        fused = model(
            text_tokens, frame_emb, source_box_emb, target_box_emb,
            source_center, target_center,
        )
        assert fused.shape == (B, cfg.fused_rows, cfg.fused_cols)

        faded = model(
            text_tokens, frame_emb, source_box_emb, target_box_emb,
            source_center, target_center,
            box_weight=torch.full((B, 2), 0.3),
            last_action=torch.zeros(B, cfg.num_servos),
        )
        assert faded.shape == (B, cfg.fused_rows, cfg.fused_cols)
        print(f"B={B}: output shape {tuple(fused.shape)} ok (full + faded evidence)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameter count: {n_params:,} ({n_params / 1e6:.3f}M)")
    assert n_params <= 5_000_000, f"SlotResonanceFusion exceeds its 5.0M param budget: {n_params:,}"
