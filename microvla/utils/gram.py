"""GRAM-style stochastic guidance for the HRM and planner heads.

Baek et al. 2026 (arXiv:2605.19376) turn deterministic recursive reasoning into
a latent-variable process: after a deterministic update ``u``, sample
``ε ~ N(μ_θ(u), σ_θ²(u))`` and set ``z = u + ε``. Parallel samples give a
width axis of inference-time scaling; a target-conditioned posterior enables
an ELBO training signal.

This module is the SHARED residual primitive. It is NOT used on the TRM world
model — MicroVLA wires it into:

* ``HRMBackbone`` slow (high-level) updates — hierarchical recursion.
* ``ChronoQueryPlanner`` features before the waypoint / orient / grip /
  ``wp_disp`` heads — multi-hypothesis final decoding.

Bottlenecked ``μ/σ`` heads keep the param cost small versus a full ``d→2d``
Gaussian on the state.
"""

from __future__ import annotations

import torch
from torch import nn


class StochasticGuidance(nn.Module):
    """Learned residual ``z = u + ε`` with ``ε ~ N(μ(u), σ²(u))``.

    Args:
        dim: Width of the deterministic proposal ``u`` (and of ``z``).
        noise_dim: Bottleneck width for the Gaussian parameters.
        target_dim: If > 0, also builds a target-conditioned posterior
            ``q(ε | u, target)`` for ELBO training.
    """

    def __init__(self, dim: int, noise_dim: int = 32, target_dim: int = 0) -> None:
        super().__init__()
        if dim < 1 or noise_dim < 1:
            raise ValueError(f"dim and noise_dim must be ≥1, got {dim}, {noise_dim}")
        self.dim = dim
        self.noise_dim = noise_dim
        self.target_dim = int(target_dim)
        hidden = max(noise_dim * 2, 64)
        self.prior = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * noise_dim),
        )
        self.up = nn.Linear(noise_dim, dim)
        # Tiny init so an untrained GRAM ≈ the deterministic backbone.
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.posterior: nn.Sequential | None = None
        if self.target_dim > 0:
            self.posterior = nn.Sequential(
                nn.Linear(dim + self.target_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * noise_dim),
            )
            nn.init.zeros_(self.posterior[-1].weight)
            nn.init.zeros_(self.posterior[-1].bias)

    @staticmethod
    def _split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = raw.chunk(2, dim=-1)
        # Soft floor: avoid σ→0 (no exploration) and σ explosions.
        log_std = log_std.clamp(-6.0, 2.0)
        return mu, log_std

    def _expand(self, low: torch.Tensor) -> torch.Tensor:
        return self.up(low)

    def sample_prior(
        self,
        u: torch.Tensor,
        n_samples: int = 1,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples ``z = u + ε`` from the prior.

        Args:
            u: Deterministic proposal ``[..., dim]``.
            n_samples: Parallel trajectories (leading sample dim when > 1).
            deterministic: If True, use ``ε = μ`` (no noise) — mean path.

        Returns:
            ``(z, eps_low)`` where ``z`` matches ``u``'s shape when
            ``n_samples == 1``, else ``[n_samples, *u.shape]``, and
            ``eps_low`` is the bottleneck noise (for diagnostics / KL).
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥1, got {n_samples}")
        mu_l, log_std = self._split(self.prior(u))
        if deterministic:
            eps_l = mu_l
        else:
            eps_l = mu_l + log_std.exp() * torch.randn_like(mu_l)
        if n_samples == 1:
            return u + self._expand(eps_l), eps_l
        # Ancestral i.i.d. samples sharing the same proposal μ/σ.
        noise = torch.randn(n_samples, *mu_l.shape, device=u.device, dtype=u.dtype)
        eps_l = mu_l.unsqueeze(0) + log_std.exp().unsqueeze(0) * noise
        z = u.unsqueeze(0) + self._expand(eps_l.reshape(-1, self.noise_dim)).reshape(
            n_samples, *u.shape
        )
        return z, eps_l

    def sample_posterior(
        self,
        u: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from ``q(ε | u, target)`` and returns ``(z, kl)``.

        ``kl`` is the closed-form KL of diagonal Gaussians against the prior,
        averaged over the batch. Requires ``target_dim > 0``.
        """
        if self.posterior is None:
            raise RuntimeError("StochasticGuidance was built without a posterior "
                               "(target_dim=0)")
        if target.shape[:-1] != u.shape[:-1]:
            raise ValueError(
                f"target leading dims {tuple(target.shape[:-1])} != "
                f"u leading dims {tuple(u.shape[:-1])}"
            )
        if target.shape[-1] != self.target_dim:
            raise ValueError(
                f"target last dim {target.shape[-1]} != target_dim {self.target_dim}"
            )
        mu_p, log_std_p = self._split(self.prior(u))
        mu_q, log_std_q = self._split(self.posterior(torch.cat([u, target], dim=-1)))
        eps_l = mu_q + log_std_q.exp() * torch.randn_like(mu_q)
        z = u + self._expand(eps_l)
        # KL(q||p) for diagonal Gaussians, mean over batch (+ any mid dims).
        var_q = (2.0 * log_std_q).exp()
        var_p = (2.0 * log_std_p).exp()
        kl = 0.5 * (
            (var_q / var_p.clamp_min(1e-8))
            + (mu_q - mu_p).pow(2) / var_p.clamp_min(1e-8)
            - 1.0
            + 2.0 * (log_std_p - log_std_q)
        ).sum(dim=-1).mean()
        return z, kl

    def forward(self, u: torch.Tensor, deterministic: bool | None = None) -> torch.Tensor:
        """Default: prior sample; deterministic in ``eval()`` unless overridden."""
        if deterministic is None:
            deterministic = not self.training
        z, _ = self.sample_prior(u, deterministic=deterministic)
        return z
