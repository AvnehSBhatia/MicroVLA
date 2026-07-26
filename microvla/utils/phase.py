"""Task-phase weights: put the loss where vision is actually required.

In a pick-and-place demo the object's position only matters BEFORE the grasp.
After it, the trajectory is transport to a target that sits in the same place
every episode — reproducible from arm pose and task progress with no grounding
at all. So if most timesteps are vision-independent, MSE-BC spends most of its
gradient where vision is useless, which is one half of why the planner weights
phase over vision by an order of magnitude (paper.md 4g/4i).

The phase signal needs no new data: the demo's own gripper command transitions
open -> closed at the grasp. ``pre_grasp_weights`` turns that into a per-timestep
weight and, critically, reports which episodes the signal is USABLE for — a
corpus like Bridge whose gripper never closes, or an episode whose only closed
reading is its last sampled frame, must not be silently upweighted end to end.
"""

from __future__ import annotations

import torch


def pre_grasp_weights(
    pwm_targets: torch.Tensor,
    weight: float = 3.0,
    persist: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-timestep loss weights emphasising the pre-grasp (approach) phase.

    The gripper is the LAST servo column and is encoded so that ``> 0`` means
    closed (``train.losses.split_planner_loss`` labels it that way, and
    ``ActionNormalizer.fit_symmetric`` is a pure positive scaling, so the baked
    sign is the raw sign). Row 0 of each chunk is the action executed at that
    timestep, so ``pwm_targets[..., 0, -1]`` is the demo's gripper command
    per timestep.

    Args:
        pwm_targets: ``[B, T, plan_steps, num_servos]`` baked action chunks.
        weight: Multiplier applied to pre-grasp timesteps. Weights are
            normalized to mean 1 PER EPISODE, so the total loss scale is
            unchanged and this cannot be confused with a learning-rate change.
        persist: Extra timesteps after the first close that still count as
            pre-grasp (the grasp moment itself is vision-critical; a couple of
            steps of settling often is too).

    Returns:
        ``(w, t_close, usable)`` — ``w [B, T]`` mean-1 per usable episode and
        all-ones for unusable ones, ``t_close [B]`` the first closed timestep
        (``T`` when it never closes), ``usable [B]`` bool.

        An episode is UNUSABLE when the gripper never closes (Bridge, or a
        failed demo) or when it first closes at the LAST sampled timestep — in
        both cases every timestep would be pre-grasp and a uniform upweight of
        the whole episode is not a phase signal at all, just a per-episode
        learning-rate change.
    """
    if pwm_targets.dim() != 4:
        raise ValueError(f"expected [B, T, plan_steps, num_servos], got {tuple(pwm_targets.shape)}")
    B, T = pwm_targets.shape[0], pwm_targets.shape[1]
    closed = pwm_targets[..., 0, -1] > 0                       # [B, T]

    # First closed timestep; T when the gripper never closes.
    idx = torch.arange(T, device=pwm_targets.device).expand(B, T)
    t_close = torch.where(closed, idx, torch.full_like(idx, T)).min(dim=1).values

    # Usable needs a close that leaves at least one post-grasp step, otherwise
    # "pre-grasp" is the whole episode.
    usable = (t_close < T - 1) & (t_close > 0)

    pre = idx < (t_close + persist).unsqueeze(1)               # [B, T]
    w = torch.where(pre, torch.full_like(closed, weight, dtype=pwm_targets.dtype),
                    torch.ones_like(closed, dtype=pwm_targets.dtype))
    # Mean-1 per episode: same total loss scale, only its distribution changes.
    w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
    w = torch.where(usable.unsqueeze(1), w, torch.ones_like(w))
    return w, t_close, usable
