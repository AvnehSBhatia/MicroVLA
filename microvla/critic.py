"""Progress critic: a task-aligned training signal for a BC stack.

**Training-only. Never deployed, never loaded by ``eval/policy.py``, and outside
the ``cfg.trainable_param_budget`` that governs the 9M deployed stack** — it
exists to shape the planner during stage B and is then discarded.

Why it exists. Behaviour cloning minimizes action MSE, but the metric is task
completion, and on this benchmark the two actively disagree: paper.md 4p measured
LIBERO's passing band at ~[0.95, 1.05] of demo action magnitude (ground truth at
1.00 solves 5/5, at 0.80 solves 0/4), while MSE-optimal regression shrinks toward
the conditional mean and every arm here emits ``std_ratio`` 0.26-0.42. Minimizing
the loss moves AWAY from the band. The actuation loss (4s) was the first fix of
that shape and took ``wp_std_ratio`` 0.121 -> 1.097.

The environment is not differentiable, so task completion cannot enter the loss
directly. This is the standard substitution: learn a differentiable surrogate for
"how far along the task is this state", then let the planner descend it.

``ProgressCritic`` maps a canonical (standardized) frame latent to progress in
[0, 1], supervised by a dense target rather than a terminal binary one — every
episode in the corpus succeeds, so a success classifier would be degenerate.

**Target choice matters.** Supervising against ``(t+1)/T`` (pure time) teaches
the critic to score "later-looking" frames higher. The actor term then asks for
actions that make the world model predict later-looking latents — which
reinforces the same PHASE shortcut the planner already over-uses (paper.md 4h).
Prefer ``progress_targets_eef`` when proprio is valid: fraction of EEF travel
toward the episode's final pose, which is task geometry rather than wall-clock.

The planner consumes it through the world model. Fusion's 8th token is the
previously executed action, so

    plan -> fusion(..., last_action=plan[:, 0]) -> TRM -> next latent -> V

is differentiable end to end with respect to the emitted action, with no
environment in the loop. Maximizing ``V`` of the IMAGINED next latent asks the
planner for actions that advance the task, not merely actions that look like the
demonstrator's.
"""
from __future__ import annotations

import torch
from torch import nn

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig


class ProgressCritic(nn.Module):
    """``[B, vis_dim] -> [B]`` progress in [0, 1].

    Deliberately small and unconditioned on the action: the action enters
    through the world model (see the module docstring), so an action-conditioned
    Q is not needed and would require negative actions the corpus does not
    contain.

    Args:
        cfg: Shared configuration (reads ``vis_dim``).
        hidden: Width of the two hidden layers.
    """

    def __init__(self, cfg: MicroVLAConfig = DEFAULT_CONFIG, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # The latent is already standardized (zero mean / unit std per
            # vector) at the perception boundary, but LayerNorm here keeps the
            # critic well-conditioned on IMAGINED latents too, which have no
            # such guarantee once the TRM has been rolled forward.
            nn.LayerNorm(cfg.vis_dim),
            nn.Linear(cfg.vis_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Args: latent ``[B, vis_dim]``. Returns: ``[B]`` in (0, 1)."""
        return torch.sigmoid(self.net(latent)).squeeze(-1)


def progress_targets(T: int, B: int, device, dtype=torch.float32) -> torch.Tensor:
    """``[B, T]`` fraction-of-episode-complete targets, ``(t + 1) / T``.

    Dense by construction. Prefer :func:`progress_targets_eef` when proprio is
    available — pure time is a phase signal and fights the grounding objective.
    """
    t = torch.arange(1, T + 1, device=device, dtype=dtype) / float(T)
    return t.unsqueeze(0).expand(B, T)


def progress_targets_eef(proprio: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` progress from EEF travel toward the episode's final pose.

    ``1 - ||eef_t - eef_T|| / ||eef_0 - eef_T||``, clamped to [0, 1]. Falls
    back to pure time for episodes whose validity flag is off (Bridge) or whose
    start and end coincide (degenerate denominator).

    This is still a demonstration-derived surrogate, not true task success —
    but it is geometry rather than wall-clock, so maximizing it through the
    world model asks for motions that close distance to the goal pose instead
    of motions that merely age the latent.
    """
    if proprio.ndim != 3 or proprio.shape[-1] < 4:
        raise ValueError(f"expected proprio [B,T,>=4], got {tuple(proprio.shape)}")
    B, T, _ = proprio.shape
    device, dtype = proprio.device, proprio.dtype
    time_tgt = progress_targets(T, B, device, dtype)
    eef = proprio[..., :3]
    valid = proprio[..., -1] > 0.5                                    # [B, T]
    # Episodes with any invalid tick fall back entirely — mixing would let a
    # zeroed EEF (the usual missing-sensor fill) invent a false goal.
    ep_ok = valid.all(dim=-1)                                         # [B]
    if not bool(ep_ok.any()):
        return time_tgt
    final = eef[:, -1:, :]                                            # [B,1,3]
    dist = (eef - final).norm(dim=-1)                                 # [B, T]
    d0 = dist[:, :1].clamp_min(1e-6)
    geom = (1.0 - dist / d0).clamp(0.0, 1.0)
    # Degenerate path (start ≈ end): keep time.
    degenerate = (dist[:, 0] < 1e-4).unsqueeze(-1).expand(B, T)
    out = torch.where(degenerate, time_tgt, geom)
    return torch.where(ep_ok.unsqueeze(-1), out, time_tgt)


def frozen_value(critic: ProgressCritic, latent: torch.Tensor) -> torch.Tensor:
    """``critic(latent)`` with gradient to ``latent`` but NOT to critic weights.

    The actor term maximizes the critic's output. If that gradient also reached
    the critic, the cheapest way to satisfy it would be for the critic to output
    1.0 everywhere — the value collapses and the actor term becomes a constant.
    Standard actor-critic practice is to hold the critic fixed while updating the
    actor; ``torch.func.functional_call`` with detached parameters does that in
    one pass without a target-network copy.
    """
    params = {k: v.detach() for k, v in critic.named_parameters()}
    buffers = {k: v.detach() for k, v in critic.named_buffers()}
    return torch.func.functional_call(critic, (params, buffers), (latent,))
