"""TRM: RecursiveTRM — step2.py's ActionTRM retargeted at the MicroVLA TRM slot.

v3 contract (see microvla/trm/interface.py and TRM_SPEC.md "CONTRACT CHANGE"):
  - I/O: (fused [B,32,5], state_delta [B,256], current_emb [B,512]) -> [B,512].
  - RESIDUAL: the model predicts the CHANGE of the scene embedding;
    forward returns `current_emb + delta`. current_emb is injected per-slot:
    it is reshaped into 32 chunks of 16 and concatenated to the 32 fused
    rows, so `embed` sees [B, 32, 5+16] — the slot structure conditions on
    the local piece of the current latent it is responsible for (+16K params
    only, total stays under the 10M reserve).
  - Canonical space: perception standardizes every embedding (zero mean /
    unit std per vector), so `spec_loss` is cosine + raw MSE — no LayerNorm
    inside the loss, which previously made it blind to scale/offset errors
    that then poisoned the JEPA feedback loop.
  - Inference cost: forward() runs a SINGLE refinement pass (`n_sup_infer`,
    default 1). The deep-supervision n_sup passes are a TRAINING device
    (use `refine_forward` in the external training loop); running them at
    inference tripled per-tick latency for no accuracy gain at eval time.
    Measured: n_sup=3 forward ≈ 57 ms on an M-series core vs ≈ 19 ms with
    n_sup_infer=1. For the Raspberry Pi 5 30 Hz target, also consider the
    fast profile `RecursiveTRM(cfg, d=512, T=2, n_inner=4)` and int8 —
    param budget is NOT compute budget; TRM_SPEC.md now carries a
    FLOPs-per-tick budget.
  - Context window (v3.1): forward() accepts an optional `context`
    [B, K, 512] — the last K tick latents, maintained by the JEPA loop.
    Two learned softmax decay profiles (a fast and a slow "context window
    read") each compress the window into a history latent, which is chunked
    per-slot (16 dims each) and concatenated to the observation alongside
    the current-emb chunks — +33K params, still under the 10M reserve. With
    no context the history latents default to current_emb (a static-scene
    prior). The model stays stateless: the window is caller-owned state.
  - No module-level side effects: seeding / thread pinning live under
    __main__ only (importing this file must not touch global RNG state).

Kept from step2.py: the weight-tied TinyNet (token-mix + channel MLP), the
y/z two-latent scheme, refine_once (n_inner z-refinements then a y-update),
deep_refine (T outer blocks, gradients only through the last), and the
deep-supervision training loop.

This file lives OUTSIDE microvla/trm/ on purpose: that package is the open
slot (interface + mock + spec only); the real TRM is built externally against
the spec and plugged in via `JEPALoop.build_real(trm=RecursiveTRM(cfg))`.

Run `python TRM.py` for: param audit, contract/statelessness checks, a
latency report, the overfit smoke test (§4 cosine+MSE loss), and a drop-in
swap into the all-mock JEPA loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.trm.interface import TRMBase


class TinyNet(nn.Module):
    """Same weights called many times. Token-mix + channel MLP over (b, L, D)."""

    def __init__(self, dim, seq_len, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Linear(seq_len, seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.chan_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, h):
        x = self.norm1(h)
        x = x.transpose(1, 2)
        x = self.token_mix(x)
        x = x.transpose(1, 2)
        h = h + x
        h = h + self.chan_mlp(self.norm2(h))
        return h


class RecursiveTRM(TRMBase):
    """Weight-tied recursive world model for the MicroVLA TRM slot (~9.5M params).

    Depth comes from recursion, not stacked layers: one TinyNet refines the
    latent pair (y, z) T*(n_inner+1) times per pass, with the FiLM-modulated
    observation re-injected at every z-step so recursion refines rather than
    drifts. v3: predicts a residual on top of `current_emb` (see module
    docstring).
    """

    def __init__(self, cfg: MicroVLAConfig, d=1024, T=3, n_inner=6, n_sup=3,
                 n_sup_infer=1, mlp_ratio=4):
        super().__init__()
        self.cfg = cfg
        self.L = cfg.fused_rows                      # 32 slot tokens
        self.d, self.T, self.n_inner = d, T, n_inner
        self.n_sup, self.n_sup_infer = n_sup, n_sup_infer

        if cfg.vis_dim % self.L != 0:
            raise ValueError(f"vis_dim ({cfg.vis_dim}) must divide into {self.L} chunks")
        self.cur_chunk = cfg.vis_dim // self.L       # 512 / 32 = 16

        # Per-slot embed of [fused row (5) | current chunk (16) |
        # fast-history chunk (16) | slow-history chunk (16)] -> d.
        self.embed = nn.Linear(cfg.fused_cols + 3 * self.cur_chunk, d)
        # Two learned "context window read" decay profiles over the latent
        # window (softmax over available entries; index -1 = newest).
        self.ctx_decay = nn.Parameter(
            torch.stack([
                torch.linspace(-2.0, 0.0, cfg.context_window),   # fast: recent-heavy
                torch.zeros(cfg.context_window),                 # slow: uniform init
            ])
        )
        self.pos = nn.Parameter(torch.randn(self.L, d) * 1e-2)
        self.film = nn.Linear(cfg.state_dim, 2 * d)  # state_delta -> (scale, shift)

        self.net = TinyNet(d, self.L, mlp_ratio)
        self.y_init = nn.Parameter(torch.randn(d) * 1e-2)
        self.z_init = nn.Parameter(torch.randn(d) * 1e-2)

        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vis_dim)        # pooled token -> residual delta [512]

    def init_states(self, b):
        y = self.y_init.expand(b, self.L, self.d)
        z = self.z_init.expand(b, self.L, self.d)
        return y, z

    def _history_latents(self, current_emb, context):
        """Compresses the latent context window into fast/slow history latents.

        context [B, K, 512] (oldest -> newest) or None. Each profile softmaxes
        its last-K learned logits over the window; None -> both histories
        default to current_emb (static-scene prior).
        """
        if context is None or context.shape[1] == 0:
            return current_emb, current_emb
        K = min(context.shape[1], self.ctx_decay.shape[1])
        window = context[:, -K:]                                # [B, K, 512]
        outs = []
        for profile in self.ctx_decay:                          # fast, slow
            w = torch.softmax(profile[-K:], dim=0)              # [K]
            outs.append(torch.einsum("k,bkd->bd", w, window))   # [B, 512]
        return outs[0], outs[1]

    def observe(self, fused, state_delta, current_emb, context=None):
        """Embeds [fused | current | history chunks], FiLM-modulated by drift."""
        fast_h, slow_h = self._history_latents(current_emb, context)
        chunks = [
            current_emb.reshape(-1, self.L, self.cur_chunk),    # [B, 32, 16]
            fast_h.reshape(-1, self.L, self.cur_chunk),
            slow_h.reshape(-1, self.L, self.cur_chunk),
        ]
        x = self.embed(torch.cat([fused, *chunks], dim=-1)) + self.pos  # [B, 32, d]
        scale, shift = self.film(state_delta).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def refine_once(self, x, y, z):
        for _ in range(self.n_inner):
            z = self.net(x + y + z)
        y = self.net(y + z)
        return y, z

    def deep_refine(self, x, y, z):
        for step in range(1, self.T + 1):
            is_last = step == self.T
            if is_last:
                y, z = self.refine_once(x, y, z)
            else:
                with torch.no_grad():
                    y, z = self.refine_once(x, y, z)
        return y, z

    def decode(self, y, current_emb):
        """Residual readout: current embedding plus the predicted change."""
        return current_emb + self.head(self.out_norm(y.mean(dim=1)))   # [B, 512]

    def refine_forward(self, fused, state_delta, current_emb, y, z, context=None):
        """One deep-supervision pass (training API, mirrors step2's forward)."""
        x = self.observe(fused, state_delta, current_emb, context)
        y, z = self.deep_refine(x, y, z)
        return self.decode(y, current_emb), y.detach(), z.detach()

    def forward(self, fused, state_delta, current_emb, context=None):
        """TRMBase contract: (fused, state_delta, current_emb, context) -> next_emb.

        Runs `n_sup_infer` passes (default 1 — the extra deep-supervision
        passes are for training; at inference they cost 3x latency).
        """
        y, z = self.init_states(fused.shape[0])
        for _ in range(self.n_sup_infer):
            next_emb, y, z = self.refine_forward(
                fused, state_delta, current_emb, y, z, context=context
            )
        return next_emb


def spec_loss(pred, tgt):
    """TRM_SPEC §4: 1.0 * (1 - cosine) + 0.5 * RAW MSE.

    Both tensors live in the canonical standardized space (perception
    standardizes at the boundary), so raw MSE is scale-honest — no LayerNorm
    inside the loss, which would forgive scale/offset errors that break the
    JEPA feedback loop at inference.
    """
    cos = 1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()
    return cos + 0.5 * F.mse_loss(pred, tgt)


if __name__ == "__main__":
    import time

    torch.set_num_threads(4)
    torch.manual_seed(0)

    cfg = DEFAULT_CONFIG
    model = RecursiveTRM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} (budget 10,000,000)")
    assert n_params < 10_000_000

    # ---- contract + statelessness checks ----
    model.eval()
    with torch.no_grad():
        for b in (1, 4):
            out = model(torch.randn(b, cfg.fused_rows, cfg.fused_cols),
                        torch.randn(b, cfg.state_dim),
                        torch.randn(b, cfg.vis_dim))
            assert out.shape == (b, cfg.vis_dim), out.shape
        f1 = torch.randn(2, cfg.fused_rows, cfg.fused_cols)
        s1 = torch.randn(2, cfg.state_dim)
        c1 = torch.randn(2, cfg.vis_dim)
        assert torch.equal(model(f1, s1, c1), model(f1, s1, c1)), "not stateless"
        ctx = torch.randn(2, cfg.context_window, cfg.vis_dim)
        with_ctx = model(f1, s1, c1, context=ctx)
        assert with_ctx.shape == (2, cfg.vis_dim)
        assert not torch.equal(with_ctx, model(f1, s1, c1)), "context must matter"
        for K in (1, 3):  # partial windows
            assert model(f1, s1, c1, context=ctx[:, -K:]).shape == (2, cfg.vis_dim)
    print("contract OK: [B,32,5] x [B,256] x [B,512] (+ctx [B,K,512]) -> [B,512], "
          "batch-preserving, stateless, residual")

    # ---- latency report (Pi 5 will be ~3-6x slower than this machine) ----
    with torch.no_grad():
        args1 = (torch.randn(1, cfg.fused_rows, cfg.fused_cols),
                 torch.randn(1, cfg.state_dim), torch.randn(1, cfg.vis_dim))
        model(*args1)  # warmup
        t0 = time.perf_counter()
        for _ in range(20):
            model(*args1)
        ms = (time.perf_counter() - t0) / 20 * 1000
    budget_ms = 1000.0 / cfg.tick_hz
    print(f"forward latency: {ms:.1f} ms/tick here (tick budget {budget_ms:.1f} ms at "
          f"{cfg.tick_hz:.0f} Hz; try d=512, T=2, n_inner=4 + int8 for the Pi 5)")

    # ---- overfit task: 10 fixed (fused, drift, current) -> next_emb pairs ----
    # Targets are built as current + small drift so the residual head starts
    # near the answer — matching the real statistics of 1/30 s scene change.
    N_PAIRS, EPOCHS = 10, 150
    FUSED = torch.randn(N_PAIRS, cfg.fused_rows, cfg.fused_cols)
    DELTA = torch.randn(N_PAIRS, cfg.state_dim)
    CUR = torch.randn(N_PAIRS, cfg.vis_dim)
    TGT = CUR + 0.3 * torch.randn(N_PAIRS, cfg.vis_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sup_w = torch.arange(1, model.n_sup + 1, dtype=torch.float32)
    sup_w = sup_w / sup_w.sum()                      # ramp weight toward final pass

    model.train()
    for epoch in range(EPOCHS):
        y, z = model.init_states(N_PAIRS)
        opt.zero_grad()
        loss_accum = 0.0
        for k in range(model.n_sup):
            pred, y, z = model.refine_forward(FUSED, DELTA, CUR, y, z)
            loss = spec_loss(pred, TGT)
            loss_accum = loss_accum + sup_w[k] * loss
        loss_accum.backward()
        opt.step()
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d} | final-pass loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(FUSED, DELTA, CUR)
        final = spec_loss(pred, TGT).item()
        cos = F.cosine_similarity(pred, TGT, dim=-1)
    print(f"\nfinal spec loss over {N_PAIRS} memorized pairs: {final:.4f}")
    print("per-pair cosine(pred, target):", [f"{c:.3f}" for c in cos.tolist()])
    print("PASS if final loss is near zero (<~0.05) and cosines ~1.0.")

    # ---- drop-in verification: swap for MockTRM in the all-mock JEPA loop ----
    import numpy as np
    from microvla.jepa.loop import JEPALoop

    loop = JEPALoop.build_mock(cfg)
    loop.trm = model
    loop.set_task("move can to ball")
    period = int(round(cfg.tick_hz / cfg.real_frame_hz))
    rng = np.random.default_rng(0)
    for i in range(2 * period):
        frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) if i % period == 0 else None
        result = loop.tick(frame)
        assert result.plan.shape == (cfg.plan_steps, cfg.num_servos)
    print(f"\nJEPA loop OK: {2 * period} ticks (2 real + {2 * period - 2} dream), "
          f"plans {cfg.plan_steps}x{cfg.num_servos}")
