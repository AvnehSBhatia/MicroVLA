"""MicroVLA: a micro vision-language-action pipeline.

Public API re-exports. The package imports with only ``torch`` + ``numpy``
installed; heavy perception dependencies (``cv2``, ``ultralytics``,
``torchvision``) are imported lazily inside the classes that need them.

v2: MiniLM is gone — text comes from YOLO-World's own CLIP text tower
(:class:`~microvla.perception.text_encoder.ClipTaskEncoder`). The deployment
path is the 30 Hz :class:`JEPALoop` (real perception at 2 Hz, dream ticks in
between via :class:`InnovationCorrector`); :class:`MicroVLAPipeline` remains
as the simple 2 Hz real-only debug harness.
"""

from __future__ import annotations

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.jepa.corrector import InnovationCorrector
from microvla.jepa.loop import JEPALoop, TickResult
from microvla.perception.command_parser import ParsedCommand, parse_command
from microvla.perception.text_encoder import TaskEncoding
from microvla.pipeline import MicroVLAPipeline, StepResult
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.interface import TRMBase
from microvla.trm.mock_trm import MockTRM

__all__ = [
    "MicroVLAConfig",
    "DEFAULT_CONFIG",
    "MicroVLAPipeline",
    "StepResult",
    "JEPALoop",
    "TickResult",
    "InnovationCorrector",
    "SlotResonanceFusion",
    "AnchoredDriftEncoder",
    "ChronoQueryPlanner",
    "TRMBase",
    "MockTRM",
    "parse_command",
    "ParsedCommand",
    "TaskEncoding",
]
