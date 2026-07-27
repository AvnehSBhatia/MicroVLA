"""Hierarchical Reasoning Model backbone (v8) — replaces ``aux_state``.

Two coupled timescales taken from the deployment loop itself: the SLOW module
steps on REAL perception ticks (``cfg.real_frame_hz``), the FAST module on every
control tick (``cfg.tick_hz``). It absorbs the v7 anchored drift code, the
hand-fitted per-axis waypoint gains, and long-horizon context-window reasoning
into one module — see :mod:`microvla.hrm.hrm_backbone` for the argument.
"""

from microvla.hrm.hrm_backbone import (
    FITTED_GAIN_PRIOR,
    GAIN_LOG_RANGE,
    LOG_GAIN_LIMITS,
    HRMBackbone,
    HRMState,
)

__all__ = [
    "HRMBackbone",
    "HRMState",
    "FITTED_GAIN_PRIOR",
    "GAIN_LOG_RANGE",
    "LOG_GAIN_LIMITS",
]
