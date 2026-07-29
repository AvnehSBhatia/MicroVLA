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
[0, 1], supervised by position within the demonstration — a dense target rather
than a terminal binary one, which matters because every episode in the corpus
succeeds and a success classifier would therefore be degenerate.

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

    Dense by construction. Every demonstration in the corpus succeeds, so a
    terminal success label carries one bit per episode and no gradient anywhere
    else; position within the demonstration is the same information spread over
    every timestep.
    """
    t = torch.arange(1, T + 1, device=device, dtype=dtype) / float(T)
    return t.unsqueeze(0).expand(B, T)


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
