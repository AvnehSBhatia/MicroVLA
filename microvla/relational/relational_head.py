"""Relational Head (v8) — object-to-object attention over the TRM's prediction.

This module replaces :class:`~microvla.fusion.slot_fusion.SlotResonanceFusion`
and moves to the other side of the world model: fusion ran BEFORE the TRM and
compressed everything it knew into a ``[32, 5]`` (160-float) matrix trained for
frame prediction; the relational head runs AFTER the TRM, on the predicted
latent, so the tokens the planner reads describe the same state the planner is
conditioned on, at full width (``cfg.rel_dim``), with no 160-float bottleneck
in between.

Two things v7 could not do, and this exists to do:

1.  **Objects are compared against each other.** v7 pre-assigned exactly two
    roles in perception (source box, target box) and fusion saw them as two
    fixed slots — nothing anywhere in the stack could relate object *i* to
    object *j*. A relational phrase like "the black bowl BETWEEN the plate and
    the ramekin" was handled by asking the FROZEN region-text head for the
    whole phrase and falling back to the bare noun (DESIGN.md, "Spatial
    grounding (Feature 1)"); when that fallback fires, the role becomes
    whichever same-noun box won on raw confidence, and no module downstream
    can revisit the choice. Here ``cfg.max_objects`` proposals arrive intact
    as full ``vis_dim`` embeddings and every layer is full self-attention over
    the whole token set, so binding phrases to proposals is trainable instead
    of delegated to a frozen head's recall.
2.  **Spatial relations are first-class.** Object-object attention logits get
    an additive bias read off the Fourier code of the *displacement* between
    the two centers (``rel_bias``), so relative geometry does not have to be
    decoded out of two sinusoidal position codes inside an MLP. This is the
    same lesson v5 learned when it handed raw centers to the planner: for a
    wrist camera, displacement IS the visual-servo error vector.

Token layout (one flat sequence, full self-attention; the read-out queries are
part of the sequence so they refine over all ``cfg.rel_layers`` layers rather
than pooling once at the end)::

    [ rel_tokens learned read-out queries | predicted latent | 3 text tokens
      | K object tokens | last executed action ]

Only the read-out query positions are returned: ``[B, rel_tokens, rel_dim]``.

Deliberate omissions:
    *No FiLM.* Fusion needed ``Linear(text_dim, 2*d)`` (0.39M) to let language
    re-tune the visual tokens because its text only entered as an attention
    key/value for slot queries and could never modulate a visual token
    directly. Here text tokens and visual tokens sit in the same
    self-attention, which subsumes it; the 0.39M buys a second transformer
    layer instead.
    *No per-slot index embedding for objects.* Proposal order is not a stable
    identity — the detector re-orders proposals between frames — so keying on
    slot index would teach the head a feature that changes meaning every real
    tick. The head is therefore permutation-equivariant over the object set
    (locked in by ``tests/test_relational.py``); identity comes from content
    and geometry only.

Evidence weighting (carried over from fusion v3 verbatim — this is a design
claim, not an implementation detail):
    ``obj_weight [B, K]`` (confidence x freshness, in ``[0, 1]``) multiplies
    each object's PROJECTED CONTENT — visual projection plus geometry
    projection — before the shared object type embedding is added, and gates
    the object-object geometry bias by ``w_i * w_j``. Callers pass
    ``confidence`` on real ticks, ``confidence * staleness_decay**k`` on dream
    ticks, and ``0.0`` for a missed detection; train-time
    ``cfg.modality_dropout`` fades the SAME weights by a random factor. There
    is no dream flag and no binary zeroing: dream ticks are a *trained*
    regime because training samples the whole fade continuum. At weight 0 an
    object contributes only its (content-free) type embedding, so its
    embedding and center provably cannot reach the output — and every level in
    between is representable and trained.

    The bias gate is quadratic (``w_i * w_j``) rather than key-side only for a
    concrete reason: gating on the key alone still lets a faded object's own
    center steer its outgoing logits at layer 1, and layer 2 then reads that
    token as a key/value — the center leaks. Gating both ends makes a
    zero-weight object inert at every depth.

    The last-executed-action token is NEVER faded (fusion's 8th token, same
    reason: the world model and the planner cannot reason about controlled
    dynamics without knowing what the controller just did, and that fact is
    always known exactly — it is not perception).

Parameter count (defaults: ``rel_dim=384``, ``rel_layers=2``, ``rel_heads=8``,
``max_objects=8``, ``rel_tokens=12``, ``vis_dim=text_dim=512``,
``n_fourier=16``) — 2.355M total, inside the 2.4M share of the 9M trainable
budget (hard cap 5.0M, inherited from fusion):

======================================  =========
component                                  params
======================================  =========
2 x (self-attention 0.591M + MLP 0.370M   1.925M
     + 2 LayerNorm)
``visual_proj`` (shared: latent + boxes)   0.197M
``text_proj`` (shared over 3 tokens)       0.197M
``geom_proj`` / ``action_proj``            0.028M
queries, type embeddings, ``rel_bias``,    0.008M
final LayerNorm
======================================  =========

The MLP hidden width is ``1.25 * rel_dim``, not the conventional ``4 *``:
attention over a 25-token sequence is where this module's capacity has to go,
and a 4x MLP would put the head at 4.6M — the whole trainable budget minus the
planner, leaving nothing for the HRM backbone.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from microvla.config import MicroVLAConfig

#: Token *types* (not positions): objects all share one type embedding, which
#: is what makes the object set permutation-equivariant. The three text
#: positions get distinct types because their order IS meaningful and stable
#: (command, source phrase, target phrase — "move can to ball" must not read
#: the same as "move ball to can").
_TOKEN_TYPES = (
    "latent",
    "command",
    "source_text",
    "target_text",
    "object",
    "action",
)
_TYPE_INDEX = {name: i for i, name in enumerate(_TOKEN_TYPES)}

#: The ordered text roles, in the order ``ClipTaskEncoder`` emits them.
_TEXT_TYPES = ("command", "source_text", "target_text")

#: MLP hidden width as a multiple of ``rel_dim`` — see the module docstring's
#: budget arithmetic for why this is not 4.
_MLP_RATIO = 1.25


class RelationalHead(nn.Module):
    """Relates ``cfg.max_objects`` proposals to each other, to the TRM's
    predicted latent, to the task phrases, and to the last executed action.

    Consumes the TRM's output (the head runs post-world-model) and emits
    ``[B, cfg.rel_tokens, cfg.rel_dim]`` tokens for the planner to
    cross-attend. All dimensions come from
    :class:`~microvla.config.MicroVLAConfig`; nothing is hardcoded.

    Attributes:
        queries: Learned read-out queries,
            ``nn.Parameter [rel_tokens, rel_dim]``. They live inside the
            self-attention sequence, so they are refined by every layer.
        type_emb: Learned per-type embeddings ``[6, rel_dim]`` (see
            ``_TOKEN_TYPES``), added AFTER the evidence fade so a
            zero-weight object still occupies a well-formed, content-free
            slot instead of vanishing from the sequence (a vanishing token
            would be binary masking, which the fade path exists to avoid).
        modality_dropout: Train-time probability of fading the object
            evidence. One Bernoulli draw and one uniform fade per SAMPLE,
            shared across all K objects — a dream rollout stales every held
            box together, so training must too.
        latent_index, text_slice, object_slice, action_index: Positions of
            each group in the flat token sequence, for tests and debugging.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        """Builds the relational head.

        Args:
            cfg: Canonical MicroVLA configuration; supplies ``vis_dim``,
                ``text_dim``, ``n_text_tokens``, ``num_servos``,
                ``max_objects``, ``rel_dim``, ``rel_tokens``, ``rel_layers``,
                ``rel_heads``, ``n_fourier``, and ``modality_dropout``.
        """
        super().__init__()
        if cfg.n_text_tokens != len(_TEXT_TYPES):
            # Each text position owns a distinct type embedding, so a fourth
            # phrase would silently borrow the OBJECT type. Fail loudly rather
            # than mis-tag a token.
            raise ValueError(
                f"RelationalHead expects the {len(_TEXT_TYPES)} ordered text "
                f"tokens {_TEXT_TYPES}; cfg.n_text_tokens is {cfg.n_text_tokens}"
            )
        self.cfg = cfg
        d = cfg.rel_dim
        k = cfg.max_objects
        self.modality_dropout = cfg.modality_dropout
        # Plain list of ints, not a buffer: it is a constant index, and the
        # module must own no state the JEPA loop would have to reset.
        self._text_types = [_TYPE_INDEX[name] for name in _TEXT_TYPES]

        # --- Shared projections ---------------------------------------------
        # ONE visual projection for the predicted latent and all K object
        # embeddings: perception standardizes every visual vector into the
        # same canonical space (utils/embedding.standardize) and the TRM
        # predicts inside it, so they are the same space and a shared
        # projection is the honest parameterization. The type embedding, not a
        # separate matrix, is what tells the layers "this one is the frame".
        self.visual_proj = nn.Linear(cfg.vis_dim, d)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.action_proj = nn.Linear(cfg.num_servos, d)

        # Fourier features of a 2D point: frequencies 2**i * pi, sin + cos per
        # coordinate -> 4 * n_fourier features (same encoding fusion used, so
        # a checkpointed geometry statistic means the same thing across v7/v8).
        freqs = torch.tensor(
            [2.0**i * math.pi for i in range(cfg.n_fourier)], dtype=torch.float32
        )
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        n_fourier_feats = 4 * cfg.n_fourier
        # Per-object geometry: Fourier(center) plus the RAW evidence weight, so
        # "how much do I trust this center" is an input feature and not only a
        # multiplier (fusion appended box_weight to its geometry token for the
        # same reason: it disambiguates a real object from the 0.5/0.5
        # missed-detection fallback).
        self.geom_proj = nn.Linear(n_fourier_feats + 1, d)
        # Object-object attention bias from the Fourier code of (c_j - c_i),
        # one scalar per head: relative geometry enters the softmax directly.
        self.rel_bias = nn.Linear(n_fourier_feats, cfg.rel_heads)

        self.type_emb = nn.Parameter(torch.randn(len(_TOKEN_TYPES), d) * d**-0.5)
        self.queries = nn.Parameter(torch.randn(cfg.rel_tokens, d) * d**-0.5)

        # --- Token layout (flat sequence, full self-attention) --------------
        self.latent_index = cfg.rel_tokens
        self.text_slice = slice(cfg.rel_tokens + 1, cfg.rel_tokens + 1 + cfg.n_text_tokens)
        obj_start = self.text_slice.stop
        self.object_slice = slice(obj_start, obj_start + k)
        self.action_index = obj_start + k
        self.n_tokens = self.action_index + 1

        # --- Pre-LN transformer layers ---------------------------------------
        hidden = int(d * _MLP_RATIO)
        self.attn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(cfg.rel_layers):
            self.attn_norms.append(nn.LayerNorm(d))
            self.attns.append(
                nn.MultiheadAttention(d, cfg.rel_heads, batch_first=True)
            )
            self.mlp_norms.append(nn.LayerNorm(d))
            self.mlps.append(
                nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
            )
        # Pre-LN stacks need a final norm or the read-out inherits the
        # unnormalized residual scale of the last layer.
        self.out_norm = nn.LayerNorm(d)

    # ------------------------------------------------------------------ utils

    def _fourier(self, points: torch.Tensor) -> torch.Tensor:
        """Encodes 2D points with sin/cos Fourier features.

        Args:
            points: ``[..., 2]`` centers in ``[0, 1]`` or unconstrained
                displacements (``c_j - c_i``).

        Returns:
            ``[..., 4 * n_fourier]`` concatenated sin/cos features.
        """
        angles = points.unsqueeze(-1) * self.fourier_freqs
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return feats.flatten(start_dim=-2)

    def _fade_evidence(self, obj_weight: torch.Tensor) -> torch.Tensor:
        """Applies the train-time evidence fade to the caller's weights.

        One Bernoulli draw and one uniform fade per sample, shared across all
        K objects: this is the same continuum a dream rollout walks down when
        every held box decays by ``staleness_decay**k`` together, so the
        network sees at train time exactly the degradations it meets at 30 Hz.
        Eval mode passes the weights through untouched.

        Args:
            obj_weight: ``[B, K]`` caller weights in ``[0, 1]``.

        Returns:
            ``[B, K]`` faded weights (the same tensor when not training).
        """
        if not self.training or self.modality_dropout <= 0.0:
            return obj_weight
        batch = obj_weight.shape[0]
        kwargs = {"device": obj_weight.device, "dtype": obj_weight.dtype}
        drop = torch.bernoulli(
            torch.full((batch, 1), self.modality_dropout, **kwargs)
        )
        fade = torch.rand(batch, 1, **kwargs)
        return obj_weight * (1.0 - drop * (1.0 - fade))

    def _build_tokens(
        self,
        next_emb: torch.Tensor,
        obj_emb: torch.Tensor,
        obj_center: torch.Tensor,
        obj_weight: torch.Tensor,
        text_tokens: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        """Builds the flat token sequence (no attention yet).

        Split out of :meth:`forward` so the evidence-fade and never-fade
        properties can be asserted on the tokens themselves rather than
        inferred from the output.

        Args:
            next_emb: ``[B, vis_dim]`` predicted latent.
            obj_emb: ``[B, K, vis_dim]`` object embeddings.
            obj_center: ``[B, K, 2]`` normalized centers.
            obj_weight: ``[B, K]`` evidence weights, already faded.
            text_tokens: ``[B, n_text_tokens, text_dim]``.
            last_action: ``[B, num_servos]``.

        Returns:
            ``[B, n_tokens, rel_dim]`` sequence in the documented order.
        """
        batch = next_emb.shape[0]

        query_toks = self.queries.unsqueeze(0).expand(batch, -1, -1)

        latent_tok = self.visual_proj(next_emb) + self.type_emb[_TYPE_INDEX["latent"]]
        latent_tok = latent_tok.unsqueeze(1)

        text_toks = self.text_proj(text_tokens) + self.type_emb[self._text_types]

        # Object content: appearance + geometry, faded TOGETHER by the single
        # per-object evidence weight. Multiplying the content (not masking the
        # token) is what makes a missed detection a smooth limit of a stale one.
        geom_feat = torch.cat(
            [self._fourier(obj_center), obj_weight.unsqueeze(-1)], dim=-1
        )
        obj_content = self.visual_proj(obj_emb) + self.geom_proj(geom_feat)
        obj_content = obj_content * obj_weight.unsqueeze(-1)
        obj_toks = obj_content + self.type_emb[_TYPE_INDEX["object"]]

        # Never faded: the controller's own last command is known exactly.
        action_tok = self.action_proj(last_action) + self.type_emb[_TYPE_INDEX["action"]]
        action_tok = action_tok.unsqueeze(1)

        return torch.cat([query_toks, latent_tok, text_toks, obj_toks, action_tok], dim=1)

    def _geometry_bias(self, obj_center: torch.Tensor, obj_weight: torch.Tensor) -> torch.Tensor:
        """Builds the additive attention bias over object-object pairs.

        The bias is a learned per-head function of the Fourier code of the
        displacement ``c_j - c_i``, gated by ``w_i * w_j`` so a faded object
        can neither receive nor emit geometric influence (see the module
        docstring on why key-side gating alone leaks at layer 2). All
        non-object pairs get exactly 0, the neutral element of a pre-softmax
        additive bias.

        Args:
            obj_center: ``[B, K, 2]`` normalized centers.
            obj_weight: ``[B, K]`` evidence weights, already faded.

        Returns:
            ``[B * rel_heads, n_tokens, n_tokens]`` float bias, laid out the
            way ``nn.MultiheadAttention`` wants a 3D ``attn_mask``.
        """
        batch, k, _ = obj_center.shape
        disp = obj_center.unsqueeze(1) - obj_center.unsqueeze(2)  # [B, K(i), K(j), 2]
        pair = self.rel_bias(self._fourier(disp))                 # [B, K, K, heads]
        gate = obj_weight.unsqueeze(2) * obj_weight.unsqueeze(1)  # [B, K(i), K(j)]
        pair = pair * gate.unsqueeze(-1)
        pair = pair.permute(0, 3, 1, 2)                           # [B, heads, K, K]

        bias = obj_center.new_zeros(batch, self.cfg.rel_heads, self.n_tokens, self.n_tokens)
        bias[:, :, self.object_slice, self.object_slice] = pair
        return bias.reshape(batch * self.cfg.rel_heads, self.n_tokens, self.n_tokens)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        next_emb: torch.Tensor,
        obj_emb: torch.Tensor,
        obj_center: torch.Tensor,
        obj_weight: torch.Tensor,
        text_tokens: torch.Tensor,
        last_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Relates the objects to each other, the prediction, and the task.

        Args:
            next_emb: ``[B, vis_dim]`` the TRM's predicted next-frame latent,
                standardized (canonical space — this module must NOT
                re-normalize it).
            obj_emb: ``[B, K, vis_dim]`` per-proposal embeddings,
                ``K = cfg.max_objects``, standardized; held from the last real
                tick during dreams.
            obj_center: ``[B, K, 2]`` normalized ``(cx, cy)`` per proposal.
            obj_weight: ``[B, K]`` evidence weight per proposal in ``[0, 1]``
                (confidence x freshness; ``0.0`` for an empty/missed slot).
            text_tokens: ``[B, n_text_tokens, text_dim]`` ordered CLIP text
                embeddings (command, source phrase, target phrase).
            last_action: ``[B, num_servos]`` previously executed command in
                ``[-1, 1]`` (``None`` -> zeros, i.e. episode start).

        Returns:
            ``[B, cfg.rel_tokens, cfg.rel_dim]`` relational tokens.
        """
        if last_action is None:
            last_action = next_emb.new_zeros(next_emb.shape[0], self.cfg.num_servos)

        weight = self._fade_evidence(obj_weight)
        x = self._build_tokens(
            next_emb, obj_emb, obj_center, weight, text_tokens, last_action
        )
        bias = self._geometry_bias(obj_center, weight)

        for attn_norm, attn, mlp_norm, mlp in zip(
            self.attn_norms, self.attns, self.mlp_norms, self.mlps
        ):
            h = attn_norm(x)
            attended, _ = attn(h, h, h, attn_mask=bias, need_weights=False)
            x = x + attended
            x = x + mlp(mlp_norm(x))

        return self.out_norm(x[:, : self.cfg.rel_tokens])


if __name__ == "__main__":
    cfg = MicroVLAConfig()
    model = RelationalHead(cfg)
    model.eval()

    for B in (1, 4):
        next_emb = torch.randn(B, cfg.vis_dim)
        obj_emb = torch.randn(B, cfg.max_objects, cfg.vis_dim)
        obj_center = torch.rand(B, cfg.max_objects, 2)
        text_tokens = torch.randn(B, cfg.n_text_tokens, cfg.text_dim)

        full = model(
            next_emb, obj_emb, obj_center,
            torch.ones(B, cfg.max_objects), text_tokens,
        )
        assert full.shape == (B, cfg.rel_tokens, cfg.rel_dim)

        # Dream tick, 3 ticks stale, two slots empty.
        weight = torch.full((B, cfg.max_objects), 0.8 * cfg.staleness_decay**3)
        weight[:, -2:] = 0.0
        faded = model(
            next_emb, obj_emb, obj_center, weight, text_tokens,
            last_action=torch.zeros(B, cfg.num_servos),
        )
        assert faded.shape == (B, cfg.rel_tokens, cfg.rel_dim)
        print(f"B={B}: output shape {tuple(full.shape)} ok (full + faded evidence)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameter count: {n_params:,} ({n_params / 1e6:.3f}M)")
    assert n_params <= 2_400_000, f"RelationalHead exceeds its 2.4M target: {n_params:,}"
