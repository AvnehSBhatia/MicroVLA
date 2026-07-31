"""Callable grasp tools — move to object center, then descend / place.

The learned policy gets *near* the object and still misses by a few cm
(watch_videos/bind_wrist). MSE-BC averages that last centimetre away; an
IBVS *residual* fights the policy. These tools OWN the action instead:

    reach_center(role)  → proportional move toward detector UV
    descend()           → -z once centred
    close_gripper()
    lift()
    reach_center("target")
    open_gripper()

Zero trainable params. Eval/deploy only (``--tool-phase``). A future
planner head can *select* which tool to call; this module is the
implementation those calls dispatch to — same contract either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

HELD_JAW_MIN = 0.2
GRASP_Z = 0.06
LIFT_Z = 0.30


@dataclass
class ToolObs:
    """What every tool sees — detector centers + proprio."""
    source_uv: Optional[tuple[float, float]]
    source_conf: float
    target_uv: Optional[tuple[float, float]]
    target_conf: float
    proprio: np.ndarray
    action_dim: int = 7


def _eef_z(proprio: np.ndarray) -> float:
    return float(np.asarray(proprio, dtype=np.float64).reshape(-1)[2])


def _jaws(proprio: np.ndarray) -> float:
    p = np.asarray(proprio, dtype=np.float64).reshape(-1)
    return float(p[7:9].mean()) if p.shape[0] >= 9 else 0.0


def reach_center(
    obs: ToolObs,
    *,
    role: str = "source",
    grasp_uv: tuple[float, float] = (0.5, 0.55),
    gain: float = 1.0,
    sign: tuple[float, float] = (1.0, -1.0),
    conf_floor: float = 0.05,
    grip: float = -1.0,
) -> np.ndarray:
    """Proportional move that drives ``role`` center → ``grasp_uv``.

    This is the accuracy tool: action xy cancels image error directly.
    """
    out = np.zeros(int(obs.action_dim), dtype=np.float32)
    out[-1] = float(grip)
    if role == "source":
        uv, conf = obs.source_uv, obs.source_conf
    else:
        uv, conf = obs.target_uv, obs.target_conf
    if uv is None or conf < conf_floor:
        out[2] = 0.15  # rise to reacquire
        return out
    eu = float(uv[0]) - float(grasp_uv[0])
    ev = float(uv[1]) - float(grasp_uv[1])
    out[0] = float(gain) * float(sign[0]) * eu
    out[1] = float(gain) * float(sign[1]) * ev
    return out


def image_error(obs: ToolObs, role: str = "source",
                grasp_uv: tuple[float, float] = (0.5, 0.55),
                conf_floor: float = 0.05) -> Optional[float]:
    """``max(|eu|,|ev|)`` or None when evidence is missing."""
    uv = obs.source_uv if role == "source" else obs.target_uv
    conf = obs.source_conf if role == "source" else obs.target_conf
    if uv is None or conf < conf_floor:
        return None
    return max(abs(float(uv[0]) - grasp_uv[0]), abs(float(uv[1]) - grasp_uv[1]))


def descend(obs: ToolObs, *, rate: float = -0.4, grip: float = -1.0) -> np.ndarray:
    out = np.zeros(int(obs.action_dim), dtype=np.float32)
    out[2] = float(rate)
    out[-1] = float(grip)
    return out


def close_gripper(obs: ToolObs) -> np.ndarray:
    out = np.zeros(int(obs.action_dim), dtype=np.float32)
    out[-1] = 1.0
    return out


def open_gripper(obs: ToolObs, *, rise: float = 0.1) -> np.ndarray:
    out = np.zeros(int(obs.action_dim), dtype=np.float32)
    out[2] = float(rise)
    out[-1] = -1.0
    return out


def lift(obs: ToolObs, *, rate: float = 0.4) -> np.ndarray:
    out = np.zeros(int(obs.action_dim), dtype=np.float32)
    out[2] = float(rate)
    out[-1] = 1.0
    return out


# Registry a future tool-head can index into by name.
TOOLS: dict[str, Callable[..., np.ndarray]] = {
    "reach_center": reach_center,
    "descend": descend,
    "close_gripper": close_gripper,
    "open_gripper": open_gripper,
    "lift": lift,
}


class GraspToolController:
    """Scripted tool sequence with a tight centering gate (accuracy mode).

    Phases call the tools above. Descend/grasp only fire once image error
    is below ``center_tol`` (default 0.05 ≈ tighter than PhasedIBVS's 0.10),
    which is the lever for the lateral near-miss the wrist clips show.
    """

    def __init__(
        self,
        *,
        gain: float = 1.0,
        sign: Sequence[float] = (1.0, -1.0),
        descend_rate: float = -0.4,
        grasp_uv: tuple[float, float] = (0.5, 0.55),
        conf_floor: float = 0.05,
        center_tol: float = 0.05,
        descend_tol: float = 0.08,
    ) -> None:
        self.gain = float(gain)
        self.sign = (float(sign[0]), float(sign[1]))
        self.descend_rate = float(descend_rate)
        self.grasp_uv = (float(grasp_uv[0]), float(grasp_uv[1]))
        self.conf_floor = float(conf_floor)
        self.center_tol = float(center_tol)
        self.descend_tol = float(descend_tol)
        self.reset()

    def reset(self) -> None:
        self.phase = "reach_src"
        self._close_ticks = 0
        self._release_ticks = 0
        self.last_tool = "reach_center"

    def call(self, name: str, obs: ToolObs, **kw) -> np.ndarray:
        """Dispatch a named tool — the API a learned head would use."""
        if name not in TOOLS:
            raise KeyError(f"unknown tool {name!r}; have {sorted(TOOLS)}")
        self.last_tool = name
        if name == "reach_center":
            return reach_center(
                obs, grasp_uv=self.grasp_uv, gain=self.gain, sign=self.sign,
                conf_floor=self.conf_floor, **kw)
        if name == "descend":
            return descend(obs, rate=self.descend_rate, **kw)
        return TOOLS[name](obs, **kw)

    def step(self, source, target, proprio, action_dim: int = 7) -> np.ndarray:
        """One control tick — scripted tool sequence (accuracy-first)."""
        def _uv(box):
            if box is None or float(getattr(box, "confidence", 0.0)) <= 0.0:
                return None, 0.0
            c = getattr(box, "center", None)
            if c is None:
                return None, 0.0
            return (float(c[0]), float(c[1])), float(box.confidence)

        su, sc = _uv(source)
        tu, tc = _uv(target)
        obs = ToolObs(su, sc, tu, tc, np.asarray(proprio, dtype=np.float64),
                      action_dim=action_dim)
        z = _eef_z(obs.proprio)

        if self.phase == "reach_src":
            err = image_error(obs, "source", self.grasp_uv, self.conf_floor)
            act = self.call("reach_center", obs, role="source", grip=-1.0)
            if err is not None and err < self.descend_tol:
                # Blend in descend once roughly centred; full lock still
                # required before closing.
                act[2] = self.descend_rate * (1.0 - err / self.descend_tol)
            if err is not None and err < self.center_tol and z < GRASP_Z:
                self.phase = "grasp"
                self._close_ticks = 0
            return act

        if self.phase == "grasp":
            act = self.call("close_gripper", obs)
            self._close_ticks += 1
            if self._close_ticks >= 12:
                if _jaws(obs.proprio) >= HELD_JAW_MIN:
                    self.phase = "lift"
                else:
                    self.phase = "reach_src"  # closed on air — retry
                    act = self.call("reach_center", obs, role="source", grip=-1.0)
                    act[2] = 0.2
            return act

        if self.phase == "lift":
            act = self.call("lift", obs)
            if z >= LIFT_Z:
                self.phase = "reach_tgt"
            return act

        if self.phase == "reach_tgt":
            if _jaws(obs.proprio) < HELD_JAW_MIN:
                self.phase = "reach_src"
                return self.step(source, target, proprio, action_dim)
            err = image_error(obs, "target", self.grasp_uv, self.conf_floor)
            act = self.call("reach_center", obs, role="target", grip=1.0)
            if z < LIFT_Z:
                act[2] = 0.2
            if err is not None and err < 0.10:
                self.phase = "release"
                self._release_ticks = 0
            return act

        # release
        act = self.call("open_gripper", obs)
        self._release_ticks += 1
        if self._release_ticks >= 8:
            self.phase = "done"
        return act
