"""Learned gates: de-skeletonization stage 1 (thresholds -> tiny classifiers).

The scaffold's internal state, dumped per tick during self-play
(``machine_state`` sidecars, layout per ``GoalServoMachine.state_row``),
supervises small heads that replace the machine's hand-set thresholds one
at a time — each replacement is an ablation row. Stage 1 replaces the two
most load-bearing decisions:

* **close trigger** (descend -> grasp): historically THE unlearnable bit for
  free regression (paper §6.4d). Features are fully-observed geometry
  against the latched goal — exactly why the structured machine could gate
  it with thresholds, and why a 2-layer head can learn the same boundary.
* **hold check** (grasp -> lift vs retry): replaces ``abs(jaws) >= 0.2``.

Pure torch, CPU-fast (<3K params total). The machine consumes them via an
optional ``gates`` dict; absent heads fall back to thresholds, so the swap
is per-gate ablatable.
"""
from __future__ import annotations

import torch
from torch import nn

GATE_VERSION = 1


class CloseTriggerGate(nn.Module):
    """(z, close_z, lateral, descend_ticks/100) -> fire-close logit."""

    IN = 4

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(self.IN, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def fire(self, z: float, close_z: float, lateral: float,
             descend_n: int) -> bool:
        x = torch.tensor([[z, close_z, lateral, descend_n / 100.0]],
                         dtype=torch.float32)
        with torch.no_grad():
            return bool(torch.sigmoid(self(x))[0] > 0.5)


class HoldCheckGate(nn.Module):
    """(|jaw0|, |jaw1|, close_ticks/12) -> holding-object logit."""

    IN = 3

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(self.IN, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def held(self, jaw0: float, jaw1: float, close_n: int) -> bool:
        x = torch.tensor([[abs(jaw0), abs(jaw1), close_n / 12.0]],
                         dtype=torch.float32)
        with torch.no_grad():
            return bool(torch.sigmoid(self(x))[0] > 0.5)


def save_gates(path, close: CloseTriggerGate, hold: HoldCheckGate,
               meta: dict | None = None) -> None:
    torch.save({"close": close.state_dict(), "hold": hold.state_dict(),
                "gate_version": GATE_VERSION, "meta": meta or {}}, path)


def load_gates(path, map_location: str = "cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("gate_version") != GATE_VERSION:
        raise RuntimeError(f"gate ckpt {path} version mismatch")
    close, hold = CloseTriggerGate(), HoldCheckGate()
    close.load_state_dict(state["close"])
    hold.load_state_dict(state["hold"])
    close.eval()
    hold.eval()
    return close, hold, state.get("meta", {})
