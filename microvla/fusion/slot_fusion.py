"""Slot Resonance Fusion v2 — slot competition over FiLM-conditioned, role-tagged tokens.

This module implements the novel fusion stage of the MicroVLA v2 stack. The
core idea is **slot competition over FiLM-conditioned, role-tagged modality
tokens**:

1.  Seven modality tokens are built at ``d_model``: the three ordered CLIP
    text embeddings (command, source noun phrase, target noun phrase — one
    *shared* ``Linear(text_dim, d_model)`` projects all three), the full-frame
    visual embedding, the source-box and target-box visual embeddings (one
    *shared* ``Linear(vis_dim, d_model)`` projects all three visual inputs),
    and a geometry token built from Fourier features of the source center,
    the target center, and their relative displacement
    (``target - source``) — so the geometry token encodes not just where each
    object is but how far apart they are, without any extra learned
    parameters beyond one linear readout.
2.  Because slot attention has no notion of token identity beyond content, a
    small learned **role-embedding table** ``[7, d_model]`` is added to each
    token by its fixed position (``cmd, src, tgt, frame, src-box, tgt-box,
    geometry``) so slots can key on "which stream is this" as well as "what
    does it say".
3.  The command token doubles as a **FiLM** (feature-wise linear modulation)
    generator: a single ``Linear(text_dim, 2*d_model)`` on the command
    embedding produces a scale/shift pair applied to the three *visual*
    tokens (frame, source-box, target-box) — language re-tunes what the
    visual evidence means before any attention happens.
4.  ``fused_rows`` (32) learned slot queries run ``n_fusion_blocks`` rounds of
    pre-LN multi-head cross-attention over the 7 tokens (slots as queries,
    tokens as keys/values), each round followed by a pre-LN GELU MLP, all
    residual. Slots *compete* via softmax attention to bind different
    mixtures of the 7-token scene memory.
5.  A shared head (``Linear(d_model, 64) -> GELU -> Linear(64, fused_cols)``)
    reads every slot into a 5-vector, producing ``[B, 32, 5]``.

Dream mode is a **trained** mode, not a special-cased inference hack: at both
training time (``modality_dropout``, per-sample Bernoulli) and dream ticks
(``dream=True``, the whole batch) the source-box, target-box, and geometry
tokens are zeroed through the exact same multiplicative masking step. The
network therefore learns, during training, to fall back onto text + frame
evidence alone in precisely the situation it will face on every one of the 14
dream ticks between real 2 Hz YOLO perceptions — the caller substitutes the
corrected TRM latent for ``frame_emb`` and zeros for the box/center inputs on
those ticks (see ``JEPALoop.tick``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from microvla.config import MicroVLAConfig

#: Explicit token order — also the order the role-embedding table indexes.
_TOKEN_ORDER = ("command", "source_text", "target_text", "frame", "source_box", "target_box", "geometry")
_N_TOKENS = len(_TOKEN_ORDER)


class SlotResonanceFusion(nn.Module):
    """Fuses 3 text tokens + frame + dual-box + geometry into a ``[B, 32, 5]`` matrix.

    See the module docstring for the method description. All dimensions come
    from :class:`~microvla.config.MicroVLAConfig`; nothing is hardcoded.

    Attributes:
        slots: Learned slot queries, ``nn.Parameter`` of shape
            ``[fused_rows, d_model]``.
        role_emb: Learned per-position role embeddings,
            ``nn.Parameter`` of shape ``[7, d_model]``.
        modality_dropout: Train-time probability of zeroing the source-box,
            target-box, and geometry tokens (one Bernoulli draw per sample,
            shared across the three tokens).
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        """Builds the fusion module.

        Args:
            cfg: Canonical MicroVLA configuration; supplies ``text_dim``,
                ``vis_dim``, ``d_model``, ``n_heads``, ``n_fusion_blocks``,
                ``n_fourier``, ``fused_rows``, ``fused_cols``, and
                ``modality_dropout``.
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.modality_dropout = cfg.modality_dropout

        # --- Shared modality projections ------------------------------------
        # One projection for all 3 ordered text tokens (applies to the last
        # dim of [B, 3, text_dim] -> [B, 3, d_model]).
        self.text_proj = nn.Linear(cfg.text_dim, d)
        # One projection shared by frame / source-box / target-box embeddings.
        self.visual_proj = nn.Linear(cfg.vis_dim, d)

        # Fourier encoding of a 2D point: frequencies 2**k * pi, k in
        # range(n_fourier); sin + cos of each frequency for each of the two
        # coordinates -> 2 (coords) * 2 (sin/cos) * n_fourier features per
        # point. Geometry token concatenates 3 such points (source center,
        # target center, target - source) -> 6 * 2 * n_fourier total.
        freqs = torch.tensor(
            [2.0**k * math.pi for k in range(cfg.n_fourier)], dtype=torch.float32
        )
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        self.geom_proj = nn.Linear(6 * 2 * cfg.n_fourier, d)

        # --- FiLM conditioning of the 3 visual tokens from the command -----
        # Produces per-sample (scale, shift); applied as x * (1 + scale) + shift.
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
        # [B, 2, 1] * [n_fourier] -> [B, 2, n_fourier]
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
        dream: bool = False,
    ) -> torch.Tensor:
        """Fuses the 7 modality inputs into the slot matrix.

        Args:
            text_tokens: ``[B, 3, text_dim]`` ordered (command, source,
                target) CLIP text embeddings.
            frame_emb: ``[B, vis_dim]`` full-frame visual embedding (real
                YOLO GAP feature on real ticks, corrected TRM latent on
                dream ticks).
            source_box_emb: ``[B, vis_dim]`` source-object box embedding.
            target_box_emb: ``[B, vis_dim]`` target-object box embedding.
            source_center: ``[B, 2]`` normalized (cx, cy) of the source box.
            target_center: ``[B, 2]`` normalized (cx, cy) of the target box.
            dream: When ``True``, zero the source-box, target-box, and
                geometry tokens for every sample in the batch — the same
                code path used by train-time ``modality_dropout``, so the
                network has already learned to operate in this regime.

        Returns:
            Fused matrix of shape ``[B, fused_rows, fused_cols]``
            (``[B, 32, 5]`` with the default config).
        """
        batch = text_tokens.shape[0]

        # Shared text projection over the 3 ordered tokens: [B, 3, d].
        text_toks = self.text_proj(text_tokens)
        command_tok = text_toks[:, 0]
        source_text_tok = text_toks[:, 1]
        target_text_tok = text_toks[:, 2]

        # Shared visual projection over frame / source-box / target-box.
        frame_tok = self.visual_proj(frame_emb)
        source_box_tok = self.visual_proj(source_box_emb)
        target_box_tok = self.visual_proj(target_box_emb)

        # Geometry token: Fourier(src) | Fourier(tgt) | Fourier(tgt - src).
        geom_feat = torch.cat(
            [
                self._fourier(source_center),
                self._fourier(target_center),
                self._fourier(target_center - source_center),
            ],
            dim=-1,
        )
        geom_tok = self.geom_proj(geom_feat)

        # FiLM: the command token re-tunes the 3 visual tokens.
        scale, shift = self.film(text_tokens[:, 0]).chunk(2, dim=-1)  # each [B, d]
        frame_tok = frame_tok * (1.0 + scale) + shift
        source_box_tok = source_box_tok * (1.0 + scale) + shift
        target_box_tok = target_box_tok * (1.0 + scale) + shift

        # Dream mode / modality dropout: ONE shared code path. dream=True
        # zeroes the whole batch; train-time modality_dropout zeroes a
        # per-sample Bernoulli subset. Either way the source-box, target-box,
        # and geometry tokens are multiplied by the same {0, 1} mask.
        if dream:
            drop_mask = frame_emb.new_zeros(batch, 1)
        elif self.training and self.modality_dropout > 0.0:
            keep = 1.0 - self.modality_dropout
            drop_mask = torch.bernoulli(
                torch.full((batch, 1), keep, device=frame_emb.device, dtype=frame_emb.dtype)
            )
        else:
            drop_mask = frame_emb.new_ones(batch, 1)

        source_box_tok = source_box_tok * drop_mask
        target_box_tok = target_box_tok * drop_mask
        geom_tok = geom_tok * drop_mask

        # Stack the 7 modality tokens in the explicit, fixed order and add
        # the learned per-position role embedding: [B, 7, d].
        tokens = torch.stack(
            [command_tok, source_text_tok, target_text_tok, frame_tok, source_box_tok, target_box_tok, geom_tok],
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
            text_tokens, frame_emb, source_box_emb, target_box_emb, source_center, target_center, dream=False
        )
        assert fused.shape == (B, cfg.fused_rows, cfg.fused_cols)

        dream_fused = model(
            text_tokens,
            frame_emb,
            torch.zeros(B, cfg.vis_dim),
            torch.zeros(B, cfg.vis_dim),
            torch.zeros(B, 2),
            torch.zeros(B, 2),
            dream=True,
        )
        assert dream_fused.shape == (B, cfg.fused_rows, cfg.fused_cols)
        print(f"B={B}: output shape {tuple(fused.shape)} ok (real + dream)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameter count: {n_params:,} ({n_params / 1e6:.3f}M)")
    assert n_params <= 5_000_000, f"SlotResonanceFusion exceeds its 5.0M param budget: {n_params:,}"
