"""Auxiliary state models (anchored drift coding).

Named ``aux_state`` rather than ``aux`` because ``aux`` is a reserved
filename on some filesystems.
"""

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder

__all__ = ["AnchoredDriftEncoder"]
