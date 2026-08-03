"""Learned world-goal heads: the ONLY trainable pieces of the structured policy.

Design rationale (paper/WHY_THE_TEACHER_WORKS.md): the assisted teacher wins
because every per-tick inference it does NOT need is replaced by a constant or
a proprio feedback loop. What it still needed to be GIVEN by hand were three
task quantities: where to grasp (a visual fix + the hand-eye lever arm), where
to place (the basket constant), and the calibrated gains. This module learns
the first two from teacher rollouts; the gains stay the teacher's calibrated
values (they are properties of the arm, not the task).

``GraspPointHead`` regresses the world grasp point RELATIVE to the current
end-effector: ``grasp_xy = eef_xy + f(uv, conf, z, yaw, embs) + lever_bias``.
The label is the eef position at the episode's final close onset — where the
hand actually was when the grasp that succeeded began — so the camera-TCP
lever arm is part of the mapping BY CONSTRUCTION, and ``lever_bias`` (a bare
2-parameter ``nn.Parameter``, init 0) gives the constant component a place to
live that can be read out and compared against the physically measured
(+0.080, -0.050) m.

Heteroscedastic regression: the head predicts a per-axis log-variance and
trains with Gaussian NLL, so its ``sigma`` output is a calibrated
metres-scale uncertainty. The runtime latch (machine.py) admits a goal only
when sigma and estimate stability clear thresholds — the learned analogue of
the teacher consulting vision once, from where vision works.

Everything here is pure torch, CPU-fast (~0.25M params total), and consumes
only quantities that exist identically in the baked npz corpus and in the
deployed loop (standardized embeddings, normalized centers, canonical
proprio) — train/deploy feature symmetry is a hard requirement.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Feature layout version — bumped whenever build_grasp_features changes, and
#: checked at checkpoint load so a stale head cannot silently read shifted
#: features.
FEATURE_VERSION = 1

_EMB_PROJ = 64  # each 512-d embedding is projected to this width


def yaw_from_quat(quat) -> float:
    """World yaw from the canonical proprio quat (x, y, z, w) — teacher's formula."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    x, y, z, w = q[0], q[1], q[2], q[3]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def build_grasp_features(uv, conf, proprio, box_emb, frame_emb) -> dict:
    """Assembles the GraspPointHead input dict from raw per-tick quantities.

    One builder used by BOTH the offline trainer and the deployed policy —
    two sides computing the same features separately is the defect shape that
    produced paper.md 4t–4v.

    Args:
        uv: source box center, normalized ``[2]`` (np/tensor/list).
        conf: detector confidence scalar.
        proprio: canonical ``[10]`` proprio vector.
        box_emb: standardized source box embedding ``[512]``.
        frame_emb: standardized frame embedding ``[512]``.

    Returns:
        Dict of float32 tensors with a leading batch dim of 1:
        ``geom [1, 8]``, ``box_emb [1, 512]``, ``frame_emb [1, 512]``,
        ``eef_xy [1, 2]``.
    """
    p = np.asarray(proprio, dtype=np.float64).reshape(-1)
    c = np.asarray([float(uv[0]), float(uv[1])], dtype=np.float64)
    yaw = yaw_from_quat(p[3:7])
    geom = np.array([c[0], c[1], float(conf), p[0], p[1], p[2],
                     np.sin(yaw), np.cos(yaw)], dtype=np.float32)

    def _t(x):
        t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        return t.unsqueeze(0)

    return {"geom": _t(geom), "box_emb": _t(box_emb),
            "frame_emb": _t(frame_emb), "eef_xy": _t(p[:2].astype(np.float32))}


class GraspPointHead(nn.Module):
    """(vision fix + proprio) -> world grasp point with calibrated uncertainty.

    Outputs, given batched features from :func:`build_grasp_features`:
        ``xy [B, 2]``    — absolute world grasp xy (eef_xy + predicted delta
                           + ``lever_bias``)
        ``z [B, 1]``     — absolute world z to close at
        ``log_var [B, 3]`` — per-axis log-variance (xy, z), clamped
    """

    GEOM_DIM = 8

    def __init__(self, emb_dim: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.box_proj = nn.Linear(emb_dim, _EMB_PROJ)
        self.frame_proj = nn.Linear(emb_dim, _EMB_PROJ)
        in_dim = self.GEOM_DIM + 2 * _EMB_PROJ
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden, 3)   # delta_xy(2) + z(1)
        # Sigma reads DETACHED trunk features: the variance can describe the
        # error but never shape the representation — the joint-NLL collapse
        # (mean stalling while sigma inflates) is impossible by construction.
        self.lv_head = nn.Linear(hidden, 3)
        # The constant component of the hand-eye mapping. Init 0: the head
        # starts believing camera == TCP and must LEARN the lever arm. Its
        # converged value is directly comparable to the offline-calibrated
        # (+0.080, -0.050) m — a falsifiable readout, not just a weight.
        self.lever_bias = nn.Parameter(torch.zeros(2))

    def forward(self, feats: dict) -> dict:
        x = torch.cat([feats["geom"],
                       self.box_proj(feats["box_emb"]),
                       self.frame_proj(feats["frame_emb"])], dim=-1)
        h = self.trunk(x)
        out = self.mean_head(h)
        log_var = self.lv_head(h.detach())
        xy = feats["eef_xy"] + out[:, :2] + self.lever_bias
        return {"xy": xy, "z": out[:, 2:3],
                "log_var": torch.clamp(log_var, -10.0, 4.0)}

    @staticmethod
    def loss(pred: dict, label_xy: torch.Tensor, label_z: torch.Tensor) -> torch.Tensor:
        """Huber on the mean + NLL on DETACHED residuals for sigma.

        A joint Gaussian NLL lets the variance head mute the mean's gradient
        on hard samples — measured here as the mean stalling at ~5 cm while
        the loss "improves" by inflating sigma (the classic heteroscedastic
        collapse). Detaching the residual inside the NLL makes sigma a pure
        CALIBRATION readout: it can describe the error, never excuse it.
        """
        err = torch.cat([pred["xy"] - label_xy, pred["z"] - label_z], dim=-1)
        lv = pred["log_var"]
        mean_term = torch.nn.functional.huber_loss(
            err, torch.zeros_like(err), delta=0.05)
        nll = 0.5 * (err.detach() ** 2) * torch.exp(-lv) + 0.5 * lv
        return 10.0 * mean_term + nll.mean()

    @staticmethod
    def sigma(pred: dict) -> torch.Tensor:
        """Per-axis predictive std in metres, ``[B, 3]``."""
        return torch.exp(0.5 * pred["log_var"])


class PlaceHead(nn.Module):
    """Task text embedding -> world place point (the basket constant).

    The basket never moves within a task (std < 2.5 cm across all 50 demos)
    and is almost never wrist-visible in transit, so the honest learned
    analogue of the teacher's hardcoded ``place_at`` is a per-task constant
    read off the command embedding. Supervised by the eef position at the
    release onset of successful episodes.
    """

    def __init__(self, text_dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU())
        self.mean_head = nn.Linear(hidden, 2)
        self.lv_head = nn.Linear(hidden, 2)     # reads detached trunk feats

    def forward(self, command_emb: torch.Tensor) -> dict:
        if command_emb.dim() == 1:
            command_emb = command_emb.unsqueeze(0)
        h = self.trunk(command_emb.float())
        return {"xy": self.mean_head(h),
                "log_var": torch.clamp(self.lv_head(h.detach()), -10.0, 4.0)}

    @staticmethod
    def loss(pred: dict, label_xy: torch.Tensor) -> torch.Tensor:
        # Same decoupling as GraspPointHead.loss (sigma reads, never mutes).
        err = pred["xy"] - label_xy
        lv = pred["log_var"]
        mean_term = torch.nn.functional.huber_loss(
            err, torch.zeros_like(err), delta=0.05)
        nll = 0.5 * (err.detach() ** 2) * torch.exp(-lv) + 0.5 * lv
        return 10.0 * mean_term + nll.mean()


def save_goal_heads(path, grasp: GraspPointHead, place: PlaceHead,
                    meta: dict | None = None) -> None:
    payload = {"grasp": grasp.state_dict(), "place": place.state_dict(),
               "feature_version": FEATURE_VERSION, "meta": meta or {}}
    torch.save(payload, path)


def load_goal_heads(path, map_location: str = "cpu"
                    ) -> tuple[GraspPointHead, PlaceHead, dict]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"goal-head checkpoint {path} has feature_version "
            f"{state.get('feature_version')}, expected {FEATURE_VERSION} — "
            f"retrain rather than reading shifted features.")
    grasp, place = GraspPointHead(), PlaceHead()
    grasp.load_state_dict(state["grasp"])
    place.load_state_dict(state["place"])
    grasp.eval()
    place.eval()
    return grasp, place, state.get("meta", {})
