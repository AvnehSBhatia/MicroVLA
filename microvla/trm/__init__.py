"""TRM open slot: interface + mock placeholder.

The real ~10M-param Tiny Recursive Model is built externally against
``TRM_SPEC.md`` and must subclass :class:`TRMBase`. :class:`MockTRM` is a
placeholder so the pipeline runs end-to-end in the meantime.
"""

from microvla.trm.interface import TRMBase
from microvla.trm.mock_trm import MockTRM

__all__ = ["TRMBase", "MockTRM"]
