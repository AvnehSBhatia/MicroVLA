"""Anchored Drift Encoder v2 — auxiliary scene-change state model.

This module implements *anchored drift coding*: instead of handing the TRM a
raw per-frame embedding, we summarize **how the scene has changed since the
very first REAL frame of the episode**. On the first forward after
:meth:`reset`, the incoming frame embedding is frozen as the episode *anchor*
and the returned code is **exactly zero** (the scene has not drifted from
itself). Every subsequent frame is compared against that anchor via two
complementary drift features — the additive difference
``frame_emb - anchor`` (what moved / what appeared or vanished) and the
multiplicative agreement ``frame_emb * anchor`` (what stayed correlated with
the initial scene). The concatenated features are projected down, passed
through a learned sigmoid gate computed from the projected drift (so
uninformative drift channels are squashed), and accumulated over time by a
GRU cell whose hidden state is detached between steps to keep backpropagation
local to each step. The LayerNormed hidden state is the ``state_delta`` code
consumed by the TRM.

v2 note: at JEPA dream ticks the pipeline feeds this encoder the corrected
TRM latent (not a real YOLO embedding) so it runs at the full 30 Hz tick
rate; its anchor is always set from the first REAL frame of the episode
(``AnchoredDriftEncoder`` itself has no notion of real vs. dream — that
distinction lives in the caller, ``JEPALoop``).

All runtime state (anchor, GRU hidden) lives in plain Python attributes — not
buffers or parameters — because it is per-episode scratch, not model weights.
"""

from __future__ import annotations

import torch
from torch import nn

from microvla.config import MicroVLAConfig


class AnchoredDriftEncoder(nn.Module):
    """Encodes scene drift relative to the first real frame of an episode.

    Pipeline per step (``frame_emb`` is the ``[B, vis_dim]`` embedding — real
    YOLO GAP feature on real ticks, corrected TRM latent on dream ticks):

    1. First forward after :meth:`reset` stores ``anchor = frame_emb.detach()``,
       zero-initializes the GRU hidden state, and returns an **exactly-zero**
       code (contract: zero drift on the anchor frame maps to a zero
       ``state_delta``) without stepping the GRU.
    2. Drift features: ``cat([frame_emb - anchor, frame_emb * anchor])``
       -> ``[B, 2 * vis_dim]``.
    3. ``Linear(2 * vis_dim, state_dim)`` -> GELU, elementwise-gated by a
       sigmoid gate from a ``Linear(state_dim, state_dim)`` on the projected
       drift (gating from the low-dim projection instead of the raw features
       keeps the module well under its 1.5M parameter budget).
    4. ``nn.GRUCell(state_dim, state_dim)`` accumulates the gated drift; the
       hidden state is detached after each step (local BPTT).
    5. Output = ``LayerNorm(hidden)`` -> ``[B, state_dim]``.

    If a batch arrives whose size differs from the stored runtime state, the
    encoder silently resets first (the new batch becomes a fresh anchor).

    Attributes:
        drift_proj: Linear projection of the drift features.
        drift_gate: Linear producing sigmoid gate logits from the projected
            drift.
        gru: GRU cell accumulating gated drift across steps.
        out_norm: LayerNorm applied to the hidden state before output.
    """

    def __init__(self, cfg: MicroVLAConfig):
        """Builds the encoder from the canonical config.

        Args:
            cfg: Shared MicroVLA configuration; uses ``vis_dim`` (input
                embedding width) and ``state_dim`` (output code width).
        """
        super().__init__()
        self.cfg = cfg
        feat_dim = 2 * cfg.vis_dim  # [emb - anchor, emb * anchor]

        self.drift_proj = nn.Linear(feat_dim, cfg.state_dim)
        # Gate from the projected drift (state_dim), not the raw 2*vis_dim
        # features: same squashing role at a fraction of the parameter cost.
        self.drift_gate = nn.Linear(cfg.state_dim, cfg.state_dim)
        self.act = nn.GELU()
        self.gru = nn.GRUCell(cfg.state_dim, cfg.state_dim)
        self.out_norm = nn.LayerNorm(cfg.state_dim)

        # Per-episode runtime state — plain attributes, deliberately NOT
        # registered as buffers/parameters (they must not be saved, moved, or
        # trained; they are scratch that reset() clears).
        self._anchor: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None

    def reset(self) -> None:
        """Clears the anchor and GRU hidden state (call at episode start)."""
        self._anchor = None
        self._hidden = None

    def forward(self, frame_emb: torch.Tensor) -> torch.Tensor:
        """Advances the drift state by one frame.

        Args:
            frame_emb: ``[B, vis_dim]`` float frame embedding for the current
                step (real YOLO embedding or, on dream ticks, the corrected
                TRM latent).

        Returns:
            ``[B, state_dim]`` state-delta code (LayerNormed GRU hidden).
            Exactly zero on the first step after :meth:`reset` (the anchor
            frame has zero drift by definition).
        """
        batch = frame_emb.shape[0]

        # Stale state from a different batch size cannot be advanced — start
        # a fresh episode silently, per the contract.
        if self._anchor is not None and self._anchor.shape[0] != batch:
            self.reset()

        if self._anchor is None:
            # First step of the episode: this frame IS the reference scene.
            # The contract requires a well-defined exactly-zero code for zero
            # drift, so skip the GRU update entirely: store the anchor,
            # zero-init the hidden state, and return an exactly-zero code.
            self._anchor = frame_emb.detach()
            self._hidden = frame_emb.new_zeros(batch, self.cfg.state_dim)
            return frame_emb.new_zeros(batch, self.cfg.state_dim)

        anchor = self._anchor
        features = torch.cat([frame_emb - anchor, frame_emb * anchor], dim=-1)

        proj = self.drift_proj(features)
        drift = self.act(proj)
        gate = torch.sigmoid(self.drift_gate(proj))
        gated = drift * gate

        hidden_prev = self._hidden

        hidden = self.gru(gated, hidden_prev)
        # Detach between steps: gradients stay local to the current step.
        self._hidden = hidden.detach()

        return self.out_norm(hidden)


if __name__ == "__main__":
    from microvla.config import DEFAULT_CONFIG

    encoder = AnchoredDriftEncoder(DEFAULT_CONFIG)
    encoder.reset()

    torch.manual_seed(0)
    for step in range(4):
        emb = torch.randn(2, DEFAULT_CONFIG.vis_dim)
        out = encoder(emb)
        print(f"step {step}: state_delta shape = {tuple(out.shape)}, norm = {out.norm().item():.4f}")
        if step == 0:
            assert torch.all(out == 0), "first-frame state_delta must be zero"

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"param count: {n_params:,} ({n_params / 1e6:.3f}M)")
    assert n_params <= 1_500_000, f"AnchoredDriftEncoder over budget: {n_params:,}"
