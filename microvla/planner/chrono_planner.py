"""Chrono-Query Planner v2: time-queried delta integration for servo trajectories.

Novel method
------------
Most action heads decode a whole trajectory in one shot (a single MLP emitting
``plan_steps * num_servos`` numbers), which gives the network no structural
reason to produce *sequential* motion — step 3 can freely contradict step 2.
The Chrono-Query Planner instead treats planning as **querying the future at
explicit points in time** and integrating motion between them:

1. The TRM's predicted next-frame embedding ``next_emb [B, vis_dim]`` is
   reshaped into a short sequence of memory tokens (8 tokens of 64 dims) and
   linearly projected to ``d_plan`` — a compact "memory" describing where the
   scene is headed.
2. ``plan_steps`` *learned time-query tokens* — one per future timestep — are
   summed with a **fixed sinusoidal monotonic time encoding** (a registered,
   non-trainable buffer over the step index). The encoding's lowest-frequency
   channels increase monotonically across the short horizon, so each query
   carries an unambiguous, ordered notion of "when" it is asking about.
3. The queries run ``n_planner_blocks`` pre-LN multi-head cross-attention
   blocks over the memory tokens (residual attention + residual GELU MLP),
   letting each timestep extract the part of the predicted future relevant to
   its moment.
4. Crucially, the per-step head predicts **deltas** (per-step servo velocity),
   not absolute positions. The final plan is
   ``tanh(cumsum(deltas, dim=1))``: cumulative integration makes step ``t`` an
   explicit function of every step before it — trajectories are *strictly
   sequential by construction* — while small per-step deltas yield smooth
   motion and the outer ``tanh`` guarantees normalized PWM in ``[-1, 1]``.

v2 scales this to ``d_plan=256``, ``n_planner_blocks=3``, ``n_heads=8``; the
caller (``JEPALoop`` / ``MicroVLAPipeline``) additionally scales the returned
plan by the corrector's trust ``tau`` — that scaling happens outside this
module, which always returns values already bounded to ``[-1, 1]``.

I/O contract (DESIGN.md):
    forward(next_emb [B, vis_dim=512]) -> plan [B, plan_steps=5, num_servos=7]
    with every value in [-1, 1] — 5 sequential updates for 7 servos.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from microvla.config import MicroVLAConfig

#: The predicted next-frame embedding is chunked into this many memory tokens.
_N_MEM_TOKENS: int = 8


def _sinusoidal_time_encoding(n_steps: int, dim: int) -> torch.Tensor:
    """Builds a fixed sinusoidal encoding over the plan-step index.

    Standard transformer sin/cos encoding evaluated at positions
    ``0 .. n_steps - 1``. Over a short horizon the low-frequency channels are
    monotonic in the step index, giving each time query an ordered "when".

    Args:
        n_steps: Number of plan timesteps (rows of the encoding).
        dim: Encoding width (must equal the planner token width).

    Returns:
        Tensor ``[n_steps, dim]`` (float32), non-trainable by construction.
    """
    position = torch.arange(n_steps, dtype=torch.float32).unsqueeze(1)  # [T, 1]
    half = (dim + 1) // 2
    div_term = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10_000.0) / half))  # [half]
    angles = position * div_term  # [T, half]
    enc = torch.zeros(n_steps, dim, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(angles[:, : (dim - dim // 2)])
    enc[:, 1::2] = torch.cos(angles[:, : (dim // 2)])
    return enc


class _CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block: queries attend over memory tokens.

    Structure (residual around each sub-layer):
        q = q + MHA(LN(q), LN(mem), LN(mem))
        q = q + MLP(LN(q))          # Linear(d, 2d) -> GELU -> Linear(2d, d)

    Args:
        d_plan: Token width.
        n_heads: Attention heads.
    """

    def __init__(self, d_plan: int, n_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_plan)
        self.norm_mem = nn.LayerNorm(d_plan)
        self.attn = nn.MultiheadAttention(d_plan, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_plan)
        self.mlp = nn.Sequential(
            nn.Linear(d_plan, d_plan * 2),
            nn.GELU(),
            nn.Linear(d_plan * 2, d_plan),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Runs one round of cross-attention + MLP.

        Args:
            queries: ``[B, plan_steps, d_plan]`` time-query tokens.
            memory: ``[B, n_mem_tokens, d_plan]`` memory tokens.

        Returns:
            Updated queries ``[B, plan_steps, d_plan]``.
        """
        mem = self.norm_mem(memory)
        attn_out, _ = self.attn(self.norm_q(queries), mem, mem, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries


class ChronoQueryPlanner(nn.Module):
    """Decodes a predicted next-frame embedding into a smooth servo plan.

    See the module docstring for the method. Conforms exactly to the
    DESIGN.md contract: ``next_emb [B, vis_dim] -> plan [B, plan_steps,
    num_servos]`` with all values guaranteed in ``[-1, 1]``.

    Args:
        cfg: Shared MicroVLA configuration (dims are read from here, never
            hardcoded).
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.vis_dim % _N_MEM_TOKENS != 0:
            raise ValueError(
                f"vis_dim ({cfg.vis_dim}) must be divisible by the memory token count ({_N_MEM_TOKENS})."
            )
        self.n_mem_tokens = _N_MEM_TOKENS
        self.mem_token_dim = cfg.vis_dim // _N_MEM_TOKENS  # 512 / 8 = 64

        # Memory path: chunk of next_emb -> d_plan token.
        self.mem_proj = nn.Linear(self.mem_token_dim, cfg.d_plan)

        # --- Rich conditioning (v2): the planner used to see ONLY the predicted
        # next_emb, a severe bottleneck for action prediction. It now ALSO
        # attends over the CURRENT latent (where things are now), the grounded
        # FUSED matrix (boxes, geometry, task text, last action — the whole
        # observation), and the drift STATE (task progress). All are available
        # at both training and 30 Hz inference. A per-source type embedding
        # tags each memory group. Passing None for any keeps the old behavior
        # (backward compatible for tests / the minimal contract).
        self.cur_proj = nn.Linear(self.mem_token_dim, cfg.d_plan)
        self.fused_proj = nn.Linear(cfg.fused_cols, cfg.d_plan)
        self.state_proj = nn.Linear(cfg.state_dim, cfg.d_plan)
        # v4: the TRM's predicted next-tick SOURCE box embedding — where the
        # grasp target is HEADED — projected into memory tokens like cur/next.
        # Chunked the same way (vis_dim -> n_mem_tokens x mem_token_dim) so the
        # planner attends over a forward-looking estimate of the object it must
        # reach, not just the held stale box carried in `fused`.
        self.box_proj = nn.Linear(self.mem_token_dim, cfg.d_plan)
        self.type_emb = nn.Parameter(torch.randn(5, cfg.d_plan) * cfg.d_plan**-0.5)

        # Learned per-timestep query tokens plus a fixed (buffer, non-trainable)
        # sinusoidal monotonic time encoding over the step index.
        self.time_queries = nn.Parameter(torch.zeros(cfg.plan_steps, cfg.d_plan))
        nn.init.normal_(self.time_queries, std=0.02)
        self.register_buffer(
            "time_encoding",
            _sinusoidal_time_encoding(cfg.plan_steps, cfg.d_plan),
            persistent=True,
        )

        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(cfg.d_plan, cfg.n_heads) for _ in range(cfg.n_planner_blocks)
        )
        self.final_norm = nn.LayerNorm(cfg.d_plan)

        # SPLIT action head. The gripper (last servo) is sharply bimodal
        # (open/close); MSE-regressing it averages the modes and the policy
        # never commits to closing -> nothing gets grasped (diagnosed at eval).
        # So the pose dims (0..num_servos-2) keep the smooth delta+cumsum
        # regression, and the gripper gets a per-step CLASSIFICATION logit
        # (BCE-trained, thresholded to +/-1 at inference) that forces a decision.
        self.n_pose = cfg.num_servos - 1
        self.pose_head = nn.Linear(cfg.d_plan, self.n_pose)   # continuous deltas
        self.grip_head = nn.Linear(cfg.d_plan, 1)             # per-step open/close logit

    def forward(self, next_emb: torch.Tensor, current_emb: torch.Tensor | None = None,
                state_delta: torch.Tensor | None = None,
                fused: torch.Tensor | None = None,
                pred_box_emb: torch.Tensor | None = None,
                return_aux: bool = False):
        """Plans a servo trajectory from the prediction + current observation.

        Args:
            next_emb: ``[B, vis_dim]`` TRM output (predicted next-frame
                embedding).
            current_emb: ``[B, vis_dim]`` the latent driving this tick (real
                standardized frame emb, or corrected dream latent). Optional.
            state_delta: ``[B, state_dim]`` drift code (task progress). Optional.
            fused: ``[B, fused_rows, fused_cols]`` grounded observation from
                SlotResonanceFusion (boxes, geometry, task text, last action).
                Optional. Passing all three is strongly recommended — the
                planner is far more accurate with the current observation than
                with the prediction alone.
            pred_box_emb: ``[B, vis_dim]`` the TRM's predicted next-tick SOURCE
                box embedding (v4, ``trm(..., return_box=True)``). Optional; a
                forward-looking estimate of the object to grasp.

            return_aux: if True, also return the per-step gripper logits
                ``[B, plan_steps]`` (needed for the BCE training loss). Callers
                that only execute the plan (loop/pipeline) leave this False.

        Returns:
            ``plan``: ``[B, plan_steps, num_servos]`` in ``[-1, 1]`` — pose dims
            are ``tanh(cumsum(deltas))`` (smooth, sequential), the gripper dim
            is a hard ``+/-1`` decision. If ``return_aux``, returns
            ``(plan, grip_logits)``.
        """
        if next_emb.dim() != 2 or next_emb.shape[1] != self.cfg.vis_dim:
            raise ValueError(f"expected next_emb of shape [B, {self.cfg.vis_dim}], got {tuple(next_emb.shape)}")
        batch = next_emb.shape[0]

        # Prediction tokens (always present): [B, 8, d_plan].
        mem_parts = [self.mem_proj(next_emb.reshape(batch, self.n_mem_tokens, self.mem_token_dim))
                     + self.type_emb[0]]
        if current_emb is not None:
            mem_parts.append(
                self.cur_proj(current_emb.reshape(batch, self.n_mem_tokens, self.mem_token_dim))
                + self.type_emb[1])
        if fused is not None:
            mem_parts.append(self.fused_proj(fused) + self.type_emb[2])   # [B, 32, d_plan]
        if state_delta is not None:
            mem_parts.append(self.state_proj(state_delta).unsqueeze(1) + self.type_emb[3])  # [B, 1, d_plan]
        if pred_box_emb is not None:
            mem_parts.append(
                self.box_proj(pred_box_emb.reshape(batch, self.n_mem_tokens, self.mem_token_dim))
                + self.type_emb[4])   # [B, 8, d_plan]
        memory = torch.cat(mem_parts, dim=1)

        # Time queries: learned tokens + fixed monotonic time encoding.
        queries = (self.time_queries + self.time_encoding).unsqueeze(0)
        queries = queries.expand(batch, -1, -1)

        for block in self.blocks:
            queries = block(queries, memory)

        feats = self.final_norm(queries)  # [B, plan_steps, d_plan]

        # Pose dims: per-step deltas -> smooth, strictly sequential trajectory.
        pose = torch.tanh(torch.cumsum(self.pose_head(feats), dim=1))  # [B, T, n_pose]

        # Gripper: per-step open/close logit -> hard +/-1 decision at inference
        # (BCE-trained via the returned logits). Hard decision is what makes
        # the hand actually close instead of hedging toward "open".
        grip_logit = self.grip_head(feats).squeeze(-1)          # [B, T]
        grip = torch.where(grip_logit > 0, torch.ones_like(grip_logit),
                           -torch.ones_like(grip_logit))         # [B, T] in {-1,+1}

        plan = torch.cat([pose, grip.unsqueeze(-1)], dim=-1)     # [B, T, num_servos]
        if return_aux:
            return plan, grip_logit
        return plan


if __name__ == "__main__":
    cfg = MicroVLAConfig()
    planner = ChronoQueryPlanner(cfg)
    planner.eval()

    with torch.no_grad():
        plan = planner(torch.randn(3, cfg.vis_dim))

    n_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    print(f"plan shape:  {tuple(plan.shape)}")
    print(f"plan range:  [{plan.min().item():+.4f}, {plan.max().item():+.4f}]")
    print(f"params:      {n_params:,} (budget 2,500,000)")
    assert plan.shape == (3, cfg.plan_steps, cfg.num_servos)
    assert plan.min() >= -1.0 and plan.max() <= 1.0
    assert n_params <= 2_500_000, f"planner over budget: {n_params:,}"
