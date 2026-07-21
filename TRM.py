"""TRM: RecursiveTRM — step2.py's ActionTRM retargeted at the MicroVLA TRM slot.

What changed vs step2.py (per TRM_SPEC.md):
  - I/O contract: (fused [B,32,5], state_delta [B,256]) -> next_emb [B,512].
    The TRM predicts the next-tick YOLO-World frame embedding; it does NOT
    decode actions — ChronoQueryPlanner owns that downstream.
  - Tokenization: the 32 fused slot rows are the tokens; a shared Linear(5, d)
    embeds each and a learned 32-position embedding is added (spec §3.1).
  - Drift conditioning: FiLM — Linear(256, 2d) on state_delta produces
    per-channel (scale, shift) that modulate the injected input tokens at
    every recursion step, so the drift signal cannot wash out over K steps
    (spec §3.2, recommended default).
  - Readout: mean-pool tokens -> LayerNorm -> Linear(d, 512) (spec §3.4).
    step2's flatten(L*D) head would be ~17M params alone at this width.
  - Budget: d=1024, channel-MLP hidden 4d => ~9.5M params (10M reserve).
  - Subclasses TRMBase, cfg-first constructor, stateless across calls:
    y/z latents are initialized fresh inside every forward(). The step2
    deep-supervision pattern (N_SUP outer passes with detached carry) is
    exposed as `refine_forward` for the external training loop; the
    contract-compliant forward() runs the same n_sup passes internally.
  - No einops (the repo venv is torch+numpy+pytest only).

Kept from step2.py: the weight-tied TinyNet (token-mix + channel MLP), the
y/z two-latent scheme, refine_once (n_inner z-refinements then a y-update),
deep_refine (T outer blocks, gradients only through the last), and the
deep-supervision training loop.

This file lives OUTSIDE microvla/trm/ on purpose: that package is the open
slot (interface + mock + spec only); the real TRM is built externally against
the spec and plugged in via `JEPALoop.build_real(trm=RecursiveTRM(cfg))`.

Run `python TRM.py` for: param audit, contract/statelessness checks, the
step2-style overfit smoke test (now with the §4 cosine+MSE loss), and a
drop-in swap into the all-mock JEPA loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.trm.interface import TRMBase

torch.set_num_threads(4)
torch.manual_seed(0)


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
    latent pair (y, z) T*(n_inner+1) times per supervision pass, with the
    FiLM-modulated observation re-injected at every z-step so recursion
    refines rather than drifts.
    """

    def __init__(self, cfg: MicroVLAConfig, d=1024, T=3, n_inner=6, n_sup=3, mlp_ratio=4):
        super().__init__()
        self.cfg = cfg
        self.L = cfg.fused_rows                      # 32 slot tokens
        self.d, self.T, self.n_inner, self.n_sup = d, T, n_inner, n_sup

        self.embed = nn.Linear(cfg.fused_cols, d)    # shared per-slot embed (5 -> d)
        self.pos = nn.Parameter(torch.randn(self.L, d) * 1e-2)
        self.film = nn.Linear(cfg.state_dim, 2 * d)  # state_delta -> (scale, shift)

        self.net = TinyNet(d, self.L, mlp_ratio)
        self.y_init = nn.Parameter(torch.randn(d) * 1e-2)
        self.z_init = nn.Parameter(torch.randn(d) * 1e-2)

        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vis_dim)        # pooled token -> next_emb [512]

    def init_states(self, b):
        y = self.y_init.expand(b, self.L, self.d)
        z = self.z_init.expand(b, self.L, self.d)
        return y, z

    def observe(self, fused, state_delta):
        """Embeds the fused slots and builds the FiLM-modulated injection."""
        x = self.embed(fused) + self.pos             # [B, 32, d]
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

    def decode(self, y):
        return self.head(self.out_norm(y.mean(dim=1)))   # [B, 512]

    def refine_forward(self, fused, state_delta, y, z):
        """One deep-supervision pass (training API, mirrors step2's forward)."""
        x = self.observe(fused, state_delta)
        y, z = self.deep_refine(x, y, z)
        return self.decode(y), y.detach(), z.detach()

    def forward(self, fused, state_delta):
        """TRMBase contract: (fused [B,32,5], state_delta [B,256]) -> [B,512]."""
        y, z = self.init_states(fused.shape[0])
        for _ in range(self.n_sup):
            next_emb, y, z = self.refine_forward(fused, state_delta, y, z)
        return next_emb


def spec_loss(pred, tgt):
    """TRM_SPEC §4: 1.0 * (1 - cosine) + 0.5 * MSE on LayerNorm'd tensors."""
    cos = 1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()
    mse = F.mse_loss(F.layer_norm(pred, pred.shape[-1:]),
                     F.layer_norm(tgt, tgt.shape[-1:]))
    return cos + 0.5 * mse


if __name__ == "__main__":
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
                        torch.randn(b, cfg.state_dim))
            assert out.shape == (b, cfg.vis_dim), out.shape
        f1, s1 = torch.randn(2, cfg.fused_rows, cfg.fused_cols), torch.randn(2, cfg.state_dim)
        assert torch.equal(model(f1, s1), model(f1, s1)), "not stateless"
    print("contract OK: [B,32,5] x [B,256] -> [B,512], batch-preserving, stateless")

    # ---- step2-style overfit task: 10 fixed (fused, drift) -> next_emb pairs ----
    N_PAIRS, EPOCHS = 10, 150
    FUSED = torch.randn(N_PAIRS, cfg.fused_rows, cfg.fused_cols)
    DELTA = torch.randn(N_PAIRS, cfg.state_dim)
    TGT = torch.randn(N_PAIRS, cfg.vis_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sup_w = torch.arange(1, model.n_sup + 1, dtype=torch.float32)
    sup_w = sup_w / sup_w.sum()                      # ramp weight toward final pass

    model.train()
    for epoch in range(EPOCHS):
        y, z = model.init_states(N_PAIRS)
        opt.zero_grad()
        loss_accum = 0.0
        for k in range(model.n_sup):
            pred, y, z = model.refine_forward(FUSED, DELTA, y, z)
            loss = spec_loss(pred, TGT)
            loss_accum = loss_accum + sup_w[k] * loss
        loss_accum.backward()
        opt.step()
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d} | final-pass loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(FUSED, DELTA)
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
