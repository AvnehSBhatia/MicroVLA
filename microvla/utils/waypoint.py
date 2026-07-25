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

Anchoring
---------
The target is refreshed on REAL perception ticks and held across the dream
ticks in between (``anchor_real=True``, the default): that is what makes the
error term integrate tracking failure instead of being recomputed away every
tick. Set ``anchor_real=False`` to re-anchor every tick (pure feed-forward,
useful as an ablation).

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
        gain_scale: Extra proportional scaling (``cfg.waypoint_gain_scale``).
            1.0 means "close the whole remaining gap in one control step",
            i.e. move at whatever speed the action clip allows; lower values
            approach the waypoint more gently.
        clip: Command magnitude clamp in raw action units, applied per axis.
        anchor_real: Refresh the absolute target only on real perception ticks
            (default). ``False`` re-anchors every tick.
    """

    def __init__(
        self,
        gain: WaypointGain | np.ndarray,
        waypoint_range: float,
        horizon: int = 5,
        gain_scale: float = 1.0,
        clip: float = 1.0,
        anchor_real: bool = True,
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
        self._target: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clears the held absolute target (call at every episode start)."""
        self._target = None

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
        row = min(self.horizon, disp.shape[0]) - 1
        if self._target is None or is_real or not self.anchor_real:
            self._target = eef + disp[row] * self.waypoint_range
        error = self._target - eef
        cmd = self.gain_scale * error / self.gain
        return np.clip(cmd, -self.clip, self.clip).astype(np.float32)
