"""JEPA latent rollout: the 30 Hz deployment loop and its innovation corrector.

:class:`~microvla.jepa.loop.JEPALoop` drives real YOLO-World perception at
``cfg.real_frame_hz`` (2 Hz) and dreams the remaining ``cfg.tick_hz`` (30 Hz)
ticks by feeding the TRM's own prediction back through fusion, corrected by
:class:`~microvla.jepa.corrector.InnovationCorrector` — a parameter-free
Kalman-lite complementary filter.
"""

from __future__ import annotations

from microvla.jepa.corrector import InnovationCorrector
from microvla.jepa.loop import JEPALoop, TickResult

__all__ = ["InnovationCorrector", "JEPALoop", "TickResult"]
