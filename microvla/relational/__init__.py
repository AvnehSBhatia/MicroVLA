"""Relational subpackage (v8) — post-TRM object-relational reasoning.

Replaces ``microvla.fusion`` in the v8 stack: the same evidence-fade contract,
but running on the TRM's predicted latent with K full-width object proposals
that attend to each other instead of two pre-assigned role slots.
"""

from microvla.relational.relational_head import RelationalHead

__all__ = ["RelationalHead"]
