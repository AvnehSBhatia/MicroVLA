"""Tests for role-ordered spatial grounding prompts (Feature 1).

The LIBERO ``libero_spatial`` suite disambiguates *which* black bowl by a
spatial clause ("between the plate and the ramekin"). The command parser
correctly extracts the full source phrase, but detection used to ground the
bare noun ("black bowl") and keep the best box by raw confidence — blind to the
clause, so among several identical bowls it grounded an arbitrary one. The fix
feeds per-role detection prompts in preference order (full phrase, then bare
noun) so the frozen region-text head can pick the box matching the clause.

These tests exercise the prompt construction + mock wiring (CPU/mock-only, no
model); the region-text preference itself is validated in the real detector.
"""

from __future__ import annotations

from microvla.config import DEFAULT_CONFIG
from microvla.jepa.loop import JEPALoop, _role_prompts
from microvla.perception.yolo_world import MockYoloWorldPerception


class TestRolePrompts:
    """``_role_prompts`` yields [full phrase, bare noun], deduped."""

    def test_spatial_clause_keeps_full_then_noun(self):
        prompts = _role_prompts("the black bowl between the plate and the ramekin")
        assert prompts == [
            "the black bowl between the plate and the ramekin",
            "black bowl",
        ]

    def test_article_only_still_two_distinct_prompts(self):
        # "the can" -> full "the can", noun "can": distinct, both kept.
        assert _role_prompts("the can") == ["the can", "can"]

    def test_already_bare_noun_collapses_to_one(self):
        assert _role_prompts("can") == ["can"]

    def test_whitespace_is_stripped(self):
        assert _role_prompts("  the red cup  ") == ["the red cup", "red cup"]


class TestMockRolePrompts:
    """The mock perception records active roles from the primary prompts."""

    def test_two_roles(self):
        p = MockYoloWorldPerception(vis_dim=DEFAULT_CONFIG.vis_dim)
        p.set_role_prompts(["black bowl between the plate", "black bowl"], ["plate"])
        assert p.active_classes == ["black bowl between the plate", "plate"]

    def test_single_role(self):
        p = MockYoloWorldPerception(vis_dim=DEFAULT_CONFIG.vis_dim)
        p.set_role_prompts(["the red block", "red block"], None)
        assert p.active_classes == ["the red block"]


class TestLoopSetTaskGrounding:
    """The JEPA loop wires role prompts through set_task for real relational text."""

    def test_relational_instruction_configures_two_roles(self):
        loop = JEPALoop.build_mock(DEFAULT_CONFIG)
        loop.set_task(
            "pick up the black bowl between the plate and the ramekin "
            "and place it on the plate"
        )
        # source role -> "black bowl ..." full phrase primary; target -> "the plate"
        # (mock records the primary/full prompt per role).
        assert loop.perception.active_classes[0].startswith("the black bowl between")
        assert loop.perception.active_classes[-1] == "the plate"

    def test_no_destination_single_role(self):
        loop = JEPALoop.build_mock(DEFAULT_CONFIG)
        loop.set_task("pick up the black bowl")
        # source == target -> one active role.
        assert len(loop.perception.active_classes) == 1
