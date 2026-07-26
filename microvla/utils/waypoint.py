"""Waypoint-absolute actuation: predicted EEF positions -> translation commands.

Why this exists
---------------
The planner's translation dims are trained by MSE against normalized demo
*actions*. Teleop action commands are noisy, so the conditional mean the MSE
converges to is systematically SMALLER than any individual demo action — the
measured collapse (``eval.bench`` ``std_ratio``: 0.12 in v4-v6, 0.37 after the
v7 fixes, healthy is ~1.0). Every fix so far attacked the *inputs* to that
regression. This attacks the *output*: don't command the regressed action at
all, command a proportional move toward a predicted end-effector POSITION.

Two properties follow, and they are the whole point:

1. **Positions are a cleaner regression target than actions.** Where the arm
   WILL be is smooth and near-deterministic; the operator's per-step command is
   not. Less target noise, less conditional-mean shrinkage.
2. **Magnitude stops being the network's job.** The command is
   ``(target_position - measured_position) / gain``. If the arm lags, the error
   — and therefore the command — stays large until it actually arrives. A
   timid predictor makes the arm arrive *late*, not *never*, which is exactly
   the failure mode that a delta-action head cannot recover from.

Rate and anchoring (the two things that make or break it)
---------------------------------------------------------
``gain`` is metres per unit action per ONE control step, but a horizon-``h``
waypoint is ``h`` steps of displacement. The command must therefore be the
per-step RATE that closes the remaining error over the remaining steps —
``error / (gain * steps_left)``. Dividing by ``gain`` alone over-commands by
exactly ``h`` and pins the output at the clip, which is bang-bang control
wearing a regression's clothes.

``steps_left`` counts down, and that is where the closed-loop property lives:
an arm that falls behind has the same error to cover in fewer steps, so the
command GROWS rather than quietly under-delivering.

The target re-anchors every tick by default. Holding it across a whole
perception period only works if the period is no longer than the horizon; at
the deployment 15:5 ratio a held target is reached in ~5 ticks and the arm then
idles for 10, throwing away two thirds of its duty cycle. ``anchor_real=True``
restores the held behaviour for rates where it makes sense.

The gain (metres of EEF displacement per unit raw action per control step) is
FITTED FROM DATA — ``preprocess/fit_waypoint_gain.py`` writes
``waypoint_stats.json`` next to the episodes; pair it with the checkpoint the
same way ``norm_stats.json`` is paired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def long_horizon_targets(
    eef_pos_chunk: torch.Tensor, plan_steps: int, waypoint_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Waypoint targets at the SAMPLED (2 Hz) spacing instead of the native one.

    This changes what MSE-BC can get away with. ``waypoint_targets`` supervises
    displacement over ``plan_steps`` NATIVE steps — 0.05 to 0.20 s at LIBERO's
    20 Hz. Over 0.2 s "keep doing what you are doing" is a near-sufficient
    statistic: the target is first-order predictable from arm pose and task
    progress, and object position is only a second-order correction. A conditional
    -mean estimator consumes variance in descending order, so it takes the phase
    term and leaves the vision residual — which is the measured 12:1 phase:vision
    ratio (paper.md 4g), not a regularization accident.

    ``eef_pos_chunk[..., t, 0, :]`` is the absolute EEF position at SAMPLED frame
    ``t``, so the leading column of the baked tensor is already a 2 Hz EEF
    trajectory. Row ``k`` here is ``traj[t+k+1] - traj[t]`` — **0.5 to 2.5 s** of
    displacement. Over 2.5 s the arm has to actually ARRIVE somewhere, so where
    the object is becomes a first-order determinant of the target. No re-bake and
    no new parameters; the same ``wp_disp_head`` is reused.

    Two unit consequences, both of which are silent train/deploy mismatches if
    missed:

    * Displacements are ~10x larger, so ``cfg.waypoint_range`` must grow (0.15 m
      saturates the ``[-1, 1]`` clamp on any real reach and destroys the signal).
    * A row is now ``(k+1) * stride`` CONTROL steps out, not ``k+1``, so
      ``WaypointActuator`` needs ``row_stride`` or its per-step rate
      under-delivers by exactly the stride.

    Args:
        eef_pos_chunk: ``[B, T, rows, 3]`` — the FULL episode tensor, since row
            ``k`` needs sampled frame ``t+k+1``.
        plan_steps: ``cfg.plan_steps``.
        waypoint_range: ``cfg.waypoint_range`` (metres per unit of output).

    Returns:
        ``(target [B, T, plan_steps, 3], mask [B, T, plan_steps])`` — clamped to
        ``[-1, 1]``. The mask is PER (t, row): the tail of each episode has no
        frame ``t+k+1`` to aim at, costing ~``plan_steps/2T`` of the rows.
    """
    if eef_pos_chunk.dim() != 4:
        raise ValueError(
            f"long_horizon_targets needs the full [B, T, rows, 3] episode tensor, "
            f"got {tuple(eef_pos_chunk.shape)}")
    traj = eef_pos_chunk[..., 0, :]                               # [B, T, 3] @ 2 Hz
    B, T = traj.shape[0], traj.shape[1]
    target = traj.new_zeros(B, T, plan_steps, 3)
    mask = traj.new_zeros(B, T, plan_steps)
    for k in range(plan_steps):
        n = T - (k + 1)
        if n <= 0:
            break
        target[:, :n, k] = (traj[:, k + 1:] - traj[:, :n]) / max(waypoint_range, 1e-8)
        mask[:, :n, k] = 1.0
    return target.clamp(-1.0, 1.0), mask


def waypoint_targets(
    eef_pos_chunk: torch.Tensor, plan_steps: int, waypoint_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the head's supervision from baked absolute EEF positions.

    ``eef_pos_chunk[..., k, :]`` is the absolute EEF position at native step
    ``i_t + k`` (``preprocess.common.chunk_actions``), so row 0 is where the
    arm IS and row ``k+1`` is where it ends up after executing plan row ``k``.
    The target for plan row ``k`` is therefore ``chunk[k+1] - chunk[0]``,
    divided by ``waypoint_range`` to match the head's ``[-1, 1]`` output.

    That would need ``plan_steps + 1`` rows and the bake carries exactly
    ``plan_steps``, so the LAST plan row has no target. It is MASKED OUT rather
    than silently supervised against a shifted position: 4 of 5 rows supervised,
    including row 0 — the only row the 30 Hz loop ever executes. Widening the
    bake to buy that fifth row is not worth a shape migration that would break
    stacking against every existing episode.

    Args:
        eef_pos_chunk: ``[..., rows, 3]`` absolute EEF positions in metres.
        plan_steps: ``cfg.plan_steps``.
        waypoint_range: ``cfg.waypoint_range`` (metres per unit of output).

    Returns:
        ``(target, row_mask)`` — target ``[..., plan_steps, 3]`` clamped to
        ``[-1, 1]``, row_mask ``[plan_steps]`` float (1 = supervised).
    """
    origin = eef_pos_chunk[..., :1, :]                      # [..., 1, 3]
    disp = (eef_pos_chunk[..., 1:, :] - origin) / max(waypoint_range, 1e-8)
    n = min(disp.shape[-2], plan_steps)
    target = disp.new_zeros(*disp.shape[:-2], plan_steps, 3)
    target[..., :n, :] = disp[..., :n, :]
    row_mask = torch.zeros(plan_steps, dtype=target.dtype, device=target.device)
    row_mask[:n] = 1.0
    return target.clamp(-1.0, 1.0), row_mask


@dataclass
class WaypointGain:
    """Per-axis EEF response to a raw translation command.

    Attributes:
        gain: ``[3]`` metres of displacement per unit raw action per control
            step, per axis. Fitted by least squares over ``(action, Δeef)``
            pairs; see :mod:`preprocess.fit_waypoint_gain`.
        r2: ``[3]`` coefficient of determination of that fit — a low value
            means the axis does not respond linearly to the command and the
            waypoint controller should be trusted less on it.
        n: Number of ``(action, Δeef)`` pairs the fit used.
    """

    gain: np.ndarray
    r2: np.ndarray
    n: int

    @classmethod
    def load(cls, path: str | Path) -> "WaypointGain":
        d = json.loads(Path(path).read_text())
        return cls(
            gain=np.asarray(d["gain"], dtype=np.float64),
            r2=np.asarray(d.get("r2", [1.0] * len(d["gain"])), dtype=np.float64),
            n=int(d.get("n", 0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"gain": self.gain.tolist(), "r2": self.r2.tolist(), "n": self.n},
            indent=2,
        ))


class WaypointActuator:
    """Turns predicted EEF displacements into raw translation commands.

    Usage per control tick::

        act.reset()                                   # once per episode
        cmd_xyz = act.command(wp_disp, eef_now, is_real=tick_is_real)

    Args:
        gain: Fitted :class:`WaypointGain` (or a bare ``[3]`` array).
        waypoint_range: Metres spanned by the head's ``[-1, 1]`` output
            (``cfg.waypoint_range``) — converts the prediction to metres.
        horizon: Which predicted step to servo toward, 1-indexed
            (``cfg.waypoint_horizon``); clamped to the number of rows.
        row_stride: Control steps between waypoint rows (``cfg.waypoint_row_stride``).
        gain_scale: Extra proportional scaling (``cfg.waypoint_gain_scale``).
            1.0 means "close the whole remaining gap in one control step",
            i.e. move at whatever speed the action clip allows; lower values
            approach the waypoint more gently.
        clip: Command magnitude clamp in raw action units, applied per axis.
        anchor_real: Refresh the absolute target only on real perception ticks.
            DEFAULT IS FALSE — re-anchor every tick. Holding across a whole
            perception period only works when the period is no longer than the
            horizon; at the deployment 15:5 ratio the arm reaches the held
            target in ~5 ticks and then idles for 10, losing two thirds of its
            duty cycle. The planner replans every tick with fresh proprio
            anyway, so re-anchoring loses nothing and the step countdown keeps
            the closed-loop correction.
    """

    def __init__(
        self,
        gain: WaypointGain | np.ndarray,
        waypoint_range: float,
        horizon: int = 5,
        gain_scale: float = 1.0,
        clip: float = 1.0,
        anchor_real: bool = False,
        row_stride: int = 1,
    ) -> None:
        g = gain.gain if isinstance(gain, WaypointGain) else np.asarray(gain, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64).reshape(-1)
        # A zero/degenerate gain would divide by ~0 and emit an infinite
        # command; treat it as "this axis does not respond" -> pass the
        # displacement through unscaled rather than exploding.
        self.gain = np.where(np.abs(g) > 1e-8, g, 1.0)
        self.waypoint_range = float(waypoint_range)
        self.horizon = max(1, int(horizon))
        self.gain_scale = float(gain_scale)
        self.clip = float(clip)
        self.anchor_real = bool(anchor_real)
        # Control steps between consecutive waypoint rows. 1 for native-spaced
        # targets; tick_hz/real_frame_hz for SAMPLED-spaced ones (10 at LIBERO's
        # 20 Hz control against 2 Hz sampling). The per-step rate divides by the
        # number of CONTROL steps remaining, so getting this wrong under-delivers
        # the command by exactly the stride.
        self.row_stride = max(1, int(row_stride))
        self._target: Optional[np.ndarray] = None
        self._steps_left: int = 1

    def reset(self) -> None:
        """Clears the held absolute target (call at every episode start)."""
        self._target = None
        self._steps_left = 1

    @property
    def target(self) -> Optional[np.ndarray]:
        """The absolute EEF position currently being servoed toward, if any."""
        return None if self._target is None else self._target.copy()

    def command(
        self,
        wp_disp: np.ndarray,
        eef_now: np.ndarray,
        is_real: bool = True,
    ) -> np.ndarray:
        """Returns the ``[3]`` raw translation command for this tick.

        Args:
            wp_disp: ``[plan_steps, 3]`` predicted displacement from the
                CURRENT end-effector position, in ``[-1, 1]`` units of
                ``waypoint_range`` (the planner's ``return_wp`` output).
            eef_now: ``[3]`` measured end-effector position (metres) — the
                first three entries of the proprio vector.
            is_real: Whether this is a real perception tick. With
                ``anchor_real`` the absolute target is refreshed only on those.

        Returns:
            ``[3]`` float32 raw translation command, clipped to ``±clip``.
        """
        disp = np.asarray(wp_disp, dtype=np.float64).reshape(-1, 3)
        eef = np.asarray(eef_now, dtype=np.float64).reshape(-1)[:3]
        # The LAST row is never supervised: `waypoint_targets` builds row k from
        # chunk[k+1] - chunk[0], and the bake carries only plan_steps rows, so
        # row plan_steps-1 has no target and is masked out of the loss. Servoing
        # toward it means aiming at an output the loss never shaped — measured
        # cost: |cmd| 0.11 where the head's 0.604 vigor implies ~0.34.
        row = min(self.horizon, disp.shape[0] - 1) - 1
        row = max(0, row)
        if self._target is None or is_real or not self.anchor_real:
            self._target = eef + disp[row] * self.waypoint_range
            self._steps_left = (row + 1) * self.row_stride
        # UNITS: `gain` is metres per unit action per ONE control step, while
        # the error spans `_steps_left` steps. Dividing by the step count turns
        # a positional error into the per-step RATE that closes it — without
        # that, aiming `horizon` steps ahead over-commands by exactly `horizon`
        # and the command sits pinned at the clip.
        steps = max(1, self._steps_left)
        cmd = self.gain_scale * (self._target - eef) / (self.gain * steps)
        # Counting down is what makes this closed-loop rather than open-loop: an
        # arm that falls behind has the same error left but fewer steps to cover
        # it, so the command GROWS instead of quietly under-delivering.
        self._steps_left = max(1, self._steps_left - 1)
        return np.clip(cmd, -self.clip, self.clip).astype(np.float32)
