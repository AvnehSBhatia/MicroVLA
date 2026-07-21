"""Canonical embedding-space normalization.

Every visual embedding in MicroVLA (frame embedding, box embeddings, TRM
predictions fed back as latents) lives in ONE canonical space: per-vector
standardized (zero mean, unit variance over the channel dim). Perception
standardizes at the boundary, so:

* the TRM's training loss (cosine + raw MSE) is well-posed — a prediction
  cannot cheat with an arbitrary scale/offset that LayerNorm-invariant
  losses would forgive;
* dream-tick feedback stays in-distribution for fusion, which only ever
  sees standardized frame tokens;
* the corrector's innovation norm and trust statistics are comparable
  across episodes and lighting conditions.
"""

from __future__ import annotations

import torch

_EPS = 1e-6


def standardize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standardizes a tensor to zero mean / unit std over ``dim``.

    Args:
        x: Any float tensor (e.g. ``[vis_dim]`` or ``[B, vis_dim]``).
        dim: Dimension to normalize over (the channel dim).

    Returns:
        ``(x - mean) / (std + eps)`` with the same shape as ``x``.
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False)
    return (x - mean) / (std + _EPS)
