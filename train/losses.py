"""Losses for the MicroVLA v2 trainable heads.

Implemented:
    * ``planner_bc_loss``   — behavior-cloning MSE against PWM targets.
    * ``smoothness_loss``   — second-difference action-smoothness penalty.
    * ``total_planner_loss``— weighted sum of the two above.
    * ``waypoint_loss``     — v7.2 metric-displacement head, row/validity masked.
    * ``modality_consistency_loss`` — optional fusion modality-dropout /
      dream-mode consistency term (same code path as JEPA dream ticks).

Documented only (NOT implemented — no TRM training code exists in this repo):
    * ``trm_loss_documentation`` — returns the v2 TRM loss specification
      string; authoritative version lives in ``microvla/trm/TRM_SPEC.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def planner_bc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Behavior-cloning loss: MSE between predicted plans and PWM targets.

    Args:
        pred: Predicted plans ``[..., plan_steps, num_servos]`` in ``[-1, 1]``.
        target: Ground-truth ``pwm_targets`` with the same shape.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred, target)


def smoothness_loss(plan: torch.Tensor) -> torch.Tensor:
    """Action-smoothness penalty: mean squared second difference along time.

    Penalizes acceleration in the planned servo trajectory, i.e.
    ``plan[t+1] - 2*plan[t] + plan[t-1]``, averaged over all elements.

    Args:
        plan: Plans ``[..., plan_steps, num_servos]``; the time axis is the
            second-to-last dimension.

    Returns:
        Scalar penalty (zero tensor if ``plan_steps < 3``).
    """
    if plan.shape[-2] < 3:
        return plan.new_zeros(())
    second_diff = plan[..., 2:, :] - 2.0 * plan[..., 1:-1, :] + plan[..., :-2, :]
    return second_diff.pow(2).mean()


def total_planner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth_weight: float = 0.1,
) -> torch.Tensor:
    """Full planner training loss: BC MSE + weighted smoothness penalty.

    Args:
        pred: Predicted plans ``[..., plan_steps, num_servos]``.
        target: Ground-truth ``pwm_targets`` with the same shape.
        smooth_weight: Weight on the second-difference smoothness term.

    Returns:
        Scalar loss ``bc + smooth_weight * smoothness``.
    """
    return planner_bc_loss(pred, target) + smooth_weight * smoothness_loss(pred)


def split_planner_loss(
    plan: torch.Tensor,
    grip_logit: torch.Tensor,
    target: torch.Tensor,
    smooth_weight: float = 0.1,
    grip_weight: float = 1.0,
    row0_weight: float = 1.0,
    step_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split-head planner loss: MSE on pose dims + BCE on the gripper.

    The gripper (last servo) is a sharply bimodal open/close action that MSE
    averages into a mushy "never quite close". Training it as a per-step binary
    classification (BCE on the logit) instead forces a decision.

    Args:
        plan: Planner output ``[..., plan_steps, num_servos]`` — pose dims
            (``:num_servos-1``) are the differentiable ``tanh(cumsum)`` values;
            the last (gripper) column is a hard +/-1 (ignored by this loss,
            which supervises the gripper through ``grip_logit``).
        grip_logit: Per-step gripper logits ``[..., plan_steps]`` from
            ``ChronoQueryPlanner(..., return_aux=True)``.
        target: Ground-truth ``pwm_targets`` ``[..., plan_steps, num_servos]``.
        smooth_weight: Weight on the pose second-difference smoothness term.
        grip_weight: Weight on the gripper BCE term.
        step_weight: Optional per-EPISODE-TIMESTEP weight, flattened to match
            ``plan``'s leading dim (``[B*T]``). Composes with ``row0_weight``,
            which weights plan ROWS — orthogonal axes. See
            ``microvla.utils.phase.pre_grasp_weights``: object position only
            matters before the grasp, so without this most of the gradient is
            spent on transport to a fixed target, which needs no grounding.
        row0_weight: Extra pose-MSE weight on plan ROW 0 — the only row that is
            ever executed at deployment (the 30 Hz loop replans every tick and
            runs row 0). Weights are normalized to mean 1 so the loss scale
            stays comparable across settings; ``1.0`` = uniform (old behavior).

    Returns:
        Scalar loss ``wMSE(pose) + grip_weight*BCE(grip) + smooth_weight*smooth(pose)``.
    """
    pose_pred = plan[..., :-1]
    pose_target = target[..., :-1]
    grip_target = (target[..., -1] > 0).float()  # open(<=0) -> 0, close(>0) -> 1

    # Row weighting (plan ROW = how far into the receding horizon) and step
    # weighting (episode TIMESTEP = where in the task) are ORTHOGONAL axes and
    # compose: row0_weight says "the executed row matters most", step_weight
    # says "the approach phase matters most". Both are mean-1 normalized so
    # neither is a disguised learning-rate change.
    rw = pose_pred.new_ones(pose_pred.shape[-2])
    if row0_weight != 1.0:
        rw[0] = row0_weight
        rw = rw / rw.mean()
    w = rw.unsqueeze(-1)                                   # [rows, 1]
    if step_weight is not None:
        sw = step_weight.reshape(-1, 1, 1)                 # [B*T, 1, 1]
        if sw.shape[0] != pose_pred.shape[0]:
            raise ValueError(
                f"step_weight has {sw.shape[0]} entries but the flattened batch "
                f"is {pose_pred.shape[0]}; pass it flattened the same way.")
        w = w * sw
    if step_weight is not None or row0_weight != 1.0:
        mse = ((pose_pred - pose_target).pow(2) * w).sum() / (
            w.expand_as(pose_pred).sum().clamp_min(1e-8))
        bce = (F.binary_cross_entropy_with_logits(grip_logit, grip_target, reduction="none")
               * w.squeeze(-1)).sum() / w.squeeze(-1).expand_as(grip_logit).sum().clamp_min(1e-8)
    else:
        mse = F.mse_loss(pose_pred, pose_target)
        bce = F.binary_cross_entropy_with_logits(grip_logit, grip_target)
    smooth = smoothness_loss(pose_pred)
    return mse + grip_weight * bce + smooth_weight * smooth


def waypoint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on the v7.2 waypoint head, masked to rows and samples with targets.

    Two independent masks, and both matter:

    * ``row_mask`` ``[plan_steps]`` — a pre-v7.2 bake carries only
      ``plan_steps`` EEF rows, so the last plan row has no target (see
      ``microvla.utils.waypoint.waypoint_targets``).
    * ``valid`` — episodes with no proprioception at all are ZERO-FILLED by
      ``train.dataset``, and a zero ``eef_pos_chunk`` looks exactly like "the
      arm never moves". Training on those would teach the head to predict no
      motion, which is the collapse this head exists to fix. Pass the proprio
      validity flag (``batch["proprio"][..., -1]``).

    Args:
        pred: ``[..., plan_steps, 3]`` head output in ``[-1, 1]``.
        target: ``[..., plan_steps, 3]`` from ``waypoint_targets``.
        row_mask: ``[plan_steps]`` (broadcast over the batch) OR a full
            ``[..., plan_steps]`` mask matching ``pred``'s leading dims — the
            long-horizon mode masks per (timestep, row), because the tail of an
            episode has no future sampled frame to aim at.
        valid: Optional ``[...]`` per-sample validity in ``{0, 1}``.

    Returns:
        Scalar mean squared error over the unmasked entries; an exact zero
        tensor (still connected to ``pred``) when nothing is supervised.
    """
    if row_mask.dim() == 1:                      # [plan_steps], broadcast over batch
        w = row_mask.reshape(*([1] * (pred.dim() - 2)), -1, 1).expand_as(pred)
    else:                                        # full per-(sample, row) mask
        w = row_mask.unsqueeze(-1).expand_as(pred)
    if valid is not None:
        w = w * valid.reshape(*valid.shape, 1, 1).expand_as(pred)
    denom = w.sum()
    if float(denom) == 0.0:
        return (pred * 0.0).sum()
    return (((pred - target) ** 2) * w).sum() / denom


def modality_consistency_loss(
    fused_full: torch.Tensor,
    fused_dropped: torch.Tensor,
) -> torch.Tensor:
    """Optional fusion modality-dropout / dream-mode consistency term.

    Encourages the fusion output computed with faded box evidence (low
    ``box_weight``, or the train-time ``modality_dropout`` fade — the SAME
    weighting path in ``SlotResonanceFusion``) to stay close to the
    full-evidence (grounded) output, so predictions degrade gracefully
    across JEPA dream ticks and when the detector misses.

    Args:
        fused_full: Fusion output ``[B, fused_rows=32, fused_cols=5]`` with
            full box evidence (``box_weight`` at confidence).
        fused_dropped: Fusion output with faded evidence (same shape, low
            ``box_weight`` or dropout-triggered). Should come from a forward
            pass in train mode.

    Returns:
        Scalar MSE between the two fused outputs; ``fused_full`` is detached
        so the gradient only flows through the dropped (dream) branch.
    """
    return F.mse_loss(fused_dropped, fused_full.detach())


def trm_loss_documentation() -> str:
    """Returns the documented (NOT implemented) v2 TRM training-loss spec.

    The TRM is an open slot built externally; no TRM training code exists in
    this repository. This function only returns the specification string so
    tooling and docs can surface it. The authoritative version lives in
    ``microvla/trm/TRM_SPEC.md``.

    Returns:
        Human-readable loss specification for the future ~10M-param TRM.
    """
    return (
        "TRM training loss (DOCUMENTED ONLY — NOT IMPLEMENTED HERE; see "
        "microvla/trm/TRM_SPEC.md for the authoritative spec):\n"
        "\n"
        "  Contract (v3): y_hat = TRM(fused_t [B,32,5], state_delta_t [B,256], "
        "current_emb_t [B,512]) -> next_emb [B,512],\n"
        "  with the RESIDUAL convention y_hat = current_emb + delta.\n"
        "  Target: y = the *actual* standardized YOLO-World frame_emb of the "
        "next REAL (2 Hz) frame ([vis_dim]=512).\n"
        "\n"
        "  L = 1.0 * (1 - cosine(y_hat, y)) + 0.5 * MSE(y_hat, y)\n"
        "  on RAW vectors — perception already standardizes every embedding\n"
        "  (microvla/utils/embedding.py), so the loss is scale-honest; do NOT\n"
        "  re-normalize inside the loss (that would forgive scale/offset errors\n"
        "  that break the JEPA feedback loop at inference).\n"
        "\n"
        "  Optional: an in-batch InfoNCE term treating (y_hat_i, y_i) as the\n"
        "  positive pair and other batch targets as negatives, to sharpen the\n"
        "  predictive representation.\n"
        "\n"
        "  Collapse note: because the target encoder is the frozen YOLO-World\n"
        "  backbone, hard representation collapse is unlikely; if the target\n"
        "  encoder is ever fine-tuned, use an EMA/stop-grad target encoder\n"
        "  (momentum ~0.99-0.999) for y to avoid collapse.\n"
        "\n"
        "  MANDATORY multi-step rollout training: at inference the TRM runs\n"
        "  ~14-step open-loop dream rollouts between real (2 Hz) measurements,\n"
        "  with predictions fed back through fusion's dream path each JEPA\n"
        "  tick. Training must unroll the same feedback loop with a scheduled\n"
        "  horizon H (start at 1, grow to 14) and a discounted loss\n"
        "  sum_h 0.95^h * L_h across the rollout; single-step-only training\n"
        "  will compound error that the InnovationCorrector cannot save."
    )
