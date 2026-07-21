"""Anchored Drift Encoder v4 — windowed multi-horizon scene-change state model.

This module implements *anchored drift coding with a context window*: instead
of handing the TRM a raw per-frame embedding — or (as in earlier versions)
only a GRU-compressed running code — it summarizes **how the scene has
changed over multiple timescales**:

* against the episode **anchor** (the very first REAL frame — the task's
  starting state), and
* against a **rolling context window** of the last ``cfg.context_window``
  real-frame embeddings, at the lag offsets in ``cfg.drift_horizons``
  (default 1, 2, 4, 8 frames ≈ 0.5 s … 4 s at 2 Hz).

Per step, each horizon produces drift features
``[emb − ref, emb ⊙ ref]`` → a shared ``Linear(2·vis_dim, state_dim)`` plus a
learned per-horizon embedding → one *drift token*. A single learned query
attention-pools the horizon tokens (the "context window read"), the pooled
code is sigmoid-gated, and a ``GRUCell`` still accumulates it across steps so
context older than the window survives in compressed form. Output =
``LayerNorm(hidden)`` — the same ``[B, state_dim]`` contract as before; the
TRM interface is unchanged.

Contract semantics preserved from earlier versions:
    * First forward after :meth:`reset` stores the anchor, seeds the window,
      zero-inits the GRU hidden, and returns an **exactly-zero** code without
      stepping the GRU.
    * The GRU hidden is detached between steps (local BPTT).
    * A batch-size change silently resets (with a debug log).
    * All runtime state (anchor, window, hidden) lives in plain Python
      attributes — per-episode scratch, not weights.

v3+ note: the JEPA loop calls this on REAL ticks only and holds the code
constant across dream ticks, so the window is a history of *measured*
evidence at ``real_frame_hz``, never accumulated imagination.
"""

from __future__ import annotations

import logging
from collections import deque

import torch
from torch import nn

from microvla.config import MicroVLAConfig

logger = logging.getLogger(__name__)


class AnchoredDriftEncoder(nn.Module):
    """Encodes multi-timescale scene drift relative to an anchor and a window.

    Pipeline per step (``frame_emb`` is the ``[B, vis_dim]`` standardized
    embedding of the current REAL frame):

    1. First forward after :meth:`reset`: store ``anchor``, seed the window,
       zero-init hidden, return exactly-zero ``[B, state_dim]``.
    2. Build one drift token per reference: the anchor plus each available
       lag in ``cfg.drift_horizons`` (clamped to the filled window; short
       histories reuse the oldest entry). Token = shared
       ``Linear(2*vis_dim, state_dim)`` of ``[emb − ref, emb ⊙ ref]`` + a
       learned horizon embedding.
    3. Attention-pool the tokens with a single learned query (softmax over
       ``Linear(state_dim, 1)`` scores) — the "context window read".
    4. Sigmoid-gate the pooled code, step ``GRUCell(state_dim, state_dim)``
       (hidden detached afterward), output ``LayerNorm(hidden)``.
    5. Append ``frame_emb`` to the rolling window (maxlen
       ``cfg.context_window``).

    Attributes:
        drift_proj: Shared projection of per-horizon drift features.
        horizon_emb: Learned embeddings, ``[n_horizons + 1, state_dim]``
            (index 0 = anchor, then one per configured lag).
        pool_score: Scoring head for the attention pool.
        drift_gate: Sigmoid gate on the pooled code.
        gru: GRU cell accumulating gated drift across steps.
        out_norm: LayerNorm applied to the hidden state before output.
    """

    def __init__(self, cfg: MicroVLAConfig):
        """Builds the encoder from the canonical config.

        Args:
            cfg: Shared MicroVLA configuration; uses ``vis_dim``,
                ``state_dim``, ``context_window``, and ``drift_horizons``.
        """
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(cfg.drift_horizons)
        feat_dim = 2 * cfg.vis_dim  # [emb - ref, emb * ref]

        self.drift_proj = nn.Linear(feat_dim, cfg.state_dim)
        self.horizon_emb = nn.Parameter(
            torch.randn(len(self.horizons) + 1, cfg.state_dim) * cfg.state_dim**-0.5
        )
        self.pool_score = nn.Linear(cfg.state_dim, 1)
        self.act = nn.GELU()
        self.drift_gate = nn.Linear(cfg.state_dim, cfg.state_dim)
        self.gru = nn.GRUCell(cfg.state_dim, cfg.state_dim)
        self.out_norm = nn.LayerNorm(cfg.state_dim)

        # Per-episode runtime state — plain attributes, deliberately NOT
        # registered as buffers/parameters.
        self._anchor: torch.Tensor | None = None
        self._window: deque[torch.Tensor] | None = None
        self._hidden: torch.Tensor | None = None

    def reset(self) -> None:
        """Clears the anchor, context window, and GRU hidden state."""
        self._anchor = None
        self._window = None
        self._hidden = None

    def _drift_token(self, emb: torch.Tensor, ref: torch.Tensor, idx: int) -> torch.Tensor:
        """One horizon's drift token: shared projection + horizon embedding."""
        features = torch.cat([emb - ref, emb * ref], dim=-1)
        return self.act(self.drift_proj(features)) + self.horizon_emb[idx]

    def forward(self, frame_emb: torch.Tensor) -> torch.Tensor:
        """Advances the drift state by one (real) frame.

        Args:
            frame_emb: ``[B, vis_dim]`` standardized embedding of the current
                real frame.

        Returns:
            ``[B, state_dim]`` state-delta code (LayerNormed GRU hidden).
            Exactly zero on the first step after :meth:`reset`.
        """
        batch = frame_emb.shape[0]

        if self._anchor is not None and self._anchor.shape[0] != batch:
            logger.debug(
                "AnchoredDriftEncoder: batch size changed %d -> %d; silently "
                "resetting (new anchor). If this fires mid-episode, a caller "
                "forgot an explicit reset().",
                self._anchor.shape[0],
                batch,
            )
            self.reset()

        if self._anchor is None:
            # First step of the episode: this frame IS the reference scene.
            self._anchor = frame_emb.detach()
            self._window = deque([frame_emb.detach()], maxlen=self.cfg.context_window)
            self._hidden = frame_emb.new_zeros(batch, self.cfg.state_dim)
            return frame_emb.new_zeros(batch, self.cfg.state_dim)

        assert self._window is not None and self._hidden is not None
        history = list(self._window)  # oldest .. newest

        # Drift tokens: anchor (idx 0) + each configured lag (clamped to the
        # available history; a short history reuses its oldest entry).
        tokens = [self._drift_token(frame_emb, self._anchor, 0)]
        for i, h in enumerate(self.horizons, start=1):
            ref = history[-min(h, len(history))]
            tokens.append(self._drift_token(frame_emb, ref, i))
        stacked = torch.stack(tokens, dim=1)  # [B, n_tokens, state_dim]

        # Context-window read: softmax attention pool over the drift tokens.
        weights = torch.softmax(self.pool_score(stacked), dim=1)  # [B, n_tokens, 1]
        pooled = (stacked * weights).sum(dim=1)  # [B, state_dim]

        gated = pooled * torch.sigmoid(self.drift_gate(pooled))

        hidden = self.gru(gated, self._hidden)
        self._hidden = hidden.detach()  # local BPTT

        self._window.append(frame_emb.detach())
        return self.out_norm(hidden)


if __name__ == "__main__":
    from microvla.config import DEFAULT_CONFIG

    encoder = AnchoredDriftEncoder(DEFAULT_CONFIG)
    encoder.reset()

    torch.manual_seed(0)
    for step in range(10):
        emb = torch.randn(2, DEFAULT_CONFIG.vis_dim)
        out = encoder(emb)
        if step == 0:
            assert torch.all(out == 0), "first-frame state_delta must be zero"
        print(f"step {step}: shape={tuple(out.shape)}, norm={out.norm().item():.4f}, "
              f"window={len(encoder._window)}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"param count: {n_params:,} ({n_params / 1e6:.3f}M)")
    assert n_params <= 1_500_000, f"AnchoredDriftEncoder over budget: {n_params:,}"
