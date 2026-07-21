"""Unit tests for the rule-based command parser (pure Python, zero deps).

Covers every pattern family listed in DESIGN.md: the ``VERB X PREP Y``
family (order-sensitive source/target extraction), the no-destination verbs
(``pick up``, ``grab``, ``grasp``, ``lift``, ``point at/to``, ``go to``,
``look at``), the directional ``push X <direction>`` family (direction word
folded into the verb, target == source), article preservation, lowercase
normalization, and the full-text fallback.
"""

from __future__ import annotations

import pytest

from microvla.perception.command_parser import ParsedCommand, parse_command


class TestVerbPrepositionSwap:
    """``(move|put|...) X (to|onto|...) Y`` — order determines source/target."""

    def test_move_to_order(self):
        parsed = parse_command("move can to ball")
        assert parsed.verb == "move"
        assert parsed.source == "can"
        assert parsed.target == "ball"

    def test_move_to_order_swapped(self):
        parsed = parse_command("move ball to can")
        assert parsed.verb == "move"
        assert parsed.source == "ball"
        assert parsed.target == "can"

    def test_put_onto(self):
        parsed = parse_command("put the cup onto the table")
        assert parsed.verb == "put"
        assert parsed.source == "the cup"
        assert parsed.target == "the table"

    def test_place_into(self):
        parsed = parse_command("place the block into the box")
        assert parsed.verb == "place"
        assert parsed.source == "the block"
        assert parsed.target == "the box"

    def test_push_toward(self):
        parsed = parse_command("push the cart toward the door")
        assert parsed.verb == "push"
        assert parsed.source == "the cart"
        assert parsed.target == "the door"

    def test_push_towards(self):
        parsed = parse_command("push the cart towards the door")
        assert parsed.verb == "push"
        assert parsed.source == "the cart"
        assert parsed.target == "the door"

    def test_bring_near(self):
        parsed = parse_command("bring the ball near the wall")
        assert parsed.verb == "bring"
        assert parsed.source == "the ball"
        assert parsed.target == "the wall"

    def test_carry_at(self):
        parsed = parse_command("carry the tray at the counter")
        assert parsed.verb == "carry"
        assert parsed.source == "the tray"
        assert parsed.target == "the counter"

    def test_slide_behind(self):
        parsed = parse_command("slide the chair behind the desk")
        assert parsed.verb == "slide"
        assert parsed.source == "the chair"
        assert parsed.target == "the desk"

    def test_drag_in_front_of(self):
        parsed = parse_command("drag the mat in front of the door")
        assert parsed.verb == "drag"
        assert parsed.source == "the mat"
        assert parsed.target == "the door"

    def test_take_by(self):
        parsed = parse_command("take the plate by the sink")
        assert parsed.verb == "take"
        assert parsed.source == "the plate"
        assert parsed.target == "the sink"

    def test_on_preposition(self):
        parsed = parse_command("place the lid on the pot")
        assert parsed.verb == "place"
        assert parsed.source == "the lid"
        assert parsed.target == "the pot"

    def test_in_preposition(self):
        parsed = parse_command("put the key in the drawer")
        assert parsed.verb == "put"
        assert parsed.source == "the key"
        assert parsed.target == "the drawer"

    def test_next_to_preposition(self):
        parsed = parse_command("move the vase next to the lamp")
        assert parsed.verb == "move"
        assert parsed.source == "the vase"
        assert parsed.target == "the lamp"


class TestNoDestinationVerbs:
    """Verbs with a single acted-on object: source == target."""

    def test_pick_up(self):
        parsed = parse_command("pick up the red block")
        assert parsed.verb == "pick up"
        assert parsed.source == "the red block"
        assert parsed.target == "the red block"

    def test_grab(self):
        parsed = parse_command("grab the mug")
        assert parsed.verb == "grab"
        assert parsed.source == parsed.target == "the mug"

    def test_grasp(self):
        parsed = parse_command("grasp the handle")
        assert parsed.verb == "grasp"
        assert parsed.source == parsed.target == "the handle"

    def test_lift(self):
        parsed = parse_command("lift the lid")
        assert parsed.verb == "lift"
        assert parsed.source == parsed.target == "the lid"

    def test_point_at(self):
        parsed = parse_command("point at the door")
        assert parsed.source == parsed.target == "the door"
        assert "point" in parsed.verb

    def test_point_to(self):
        parsed = parse_command("point to the window")
        assert parsed.source == parsed.target == "the window"
        assert "point" in parsed.verb

    def test_go_to(self):
        parsed = parse_command("go to the kitchen")
        assert parsed.source == parsed.target == "the kitchen"
        assert "go" in parsed.verb

    def test_look_at(self):
        parsed = parse_command("look at the shelf")
        assert parsed.source == parsed.target == "the shelf"
        assert "look" in parsed.verb


class TestDirectionalPush:
    """``push X <direction>`` — direction folds into the verb, target == source."""

    @pytest.mark.parametrize(
        "direction", ["left", "right", "up", "down", "forward", "back", "backward"]
    )
    def test_direction_suffix(self, direction: str):
        parsed = parse_command(f"push the box {direction}")
        assert parsed.source == parsed.target == "the box"
        assert "push" in parsed.verb
        assert direction in parsed.verb
        # The direction word is not left dangling on the object phrase.
        assert direction not in parsed.source


class TestFallback:
    def test_unmatched_command_falls_back(self):
        parsed = parse_command("dance around randomly")
        assert parsed.verb == "do"
        assert parsed.source == parsed.target == "dance around randomly"

    def test_fallback_strips_and_lowercases(self):
        parsed = parse_command("  Twirl In Place  ")
        assert parsed.verb == "do"
        assert parsed.source == parsed.target == "twirl in place"


class TestNormalization:
    def test_lowercase_normalized(self):
        parsed = parse_command("MOVE CAN TO BALL")
        assert parsed.verb == "move"
        assert parsed.source == "can"
        assert parsed.target == "ball"

    def test_articles_preserved(self):
        parsed = parse_command("move the can to the ball")
        assert parsed.source == "the can"
        assert parsed.target == "the ball"

    def test_raw_preserves_original_string(self):
        original = "  Move The Can To The Ball  "
        parsed = parse_command(original)
        assert parsed.raw == original

    def test_returns_parsed_command_instance(self):
        parsed = parse_command("grab the mug")
        assert isinstance(parsed, ParsedCommand)

    def test_parsed_command_is_frozen(self):
        parsed = parse_command("grab the mug")
        with pytest.raises(Exception):
            parsed.verb = "something else"  # type: ignore[misc]
