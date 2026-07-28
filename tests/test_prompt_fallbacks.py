"""Prompt fallback chains — the fix for the blind-corpus root cause.

YOLO-World-S returns EXACTLY 0.000 for LIBERO's task phrases ("alphabet soup",
"black bowl"), so a bake keyed on them detected the source object on 0.0% of
frames and every model trained with no object input at all (paper.md 4n). The
chain appends CONCRETE visual categories after the exact phrase, and
`set_role_prompts` takes the first prompt that actually detected something.

A single tail is not enough: libero_object is groceries and libero_spatial is
tableware. The grocery-only tail is why libero_spatial baked 500 episodes and
then failed the sighted gate.
"""
from __future__ import annotations

import pytest

from preprocess.common import _with_fallbacks

ABSTRACT = {"product", "package", "item", "object", "thing", "stuff"}


class TestChainShape:
    def test_exact_phrase_is_always_tried_first(self):
        for p in ("alphabet soup", "black bowl", "basket"):
            assert _with_fallbacks(p)[0] == p

    def test_head_noun_precedes_the_generic_tail(self):
        # A two-word product name sometimes grounds on its noun alone.
        c = _with_fallbacks("alphabet soup")
        assert c[1] == "soup"
        assert c.index("soup") < c.index("box")

    def test_no_duplicates(self):
        for p in ("box", "bowl", "basket", "bottle"):
            c = _with_fallbacks(p)
            assert len(c) == len(set(c)), c

    def test_never_falls_back_to_an_abstract_noun(self):
        # Measured 0.000 for all of these; including one would add NMS cost and
        # buy nothing.
        for p in ("alphabet soup", "black bowl", "widget"):
            assert not (set(_with_fallbacks(p)) & ABSTRACT)


class TestCategoryRouting:
    @pytest.mark.parametrize("phrase", ["black bowl", "plate", "ramekin", "white mug"])
    def test_tableware_phrases_get_a_tableware_tail(self, phrase):
        c = _with_fallbacks(phrase)
        assert "bowl" in c and "plate" in c
        assert "can" not in c and "carton" not in c, (
            f"{phrase!r} got the grocery tail; nothing in it can fire on tableware"
        )

    @pytest.mark.parametrize("phrase", ["alphabet soup", "bbq sauce", "cream cheese",
                                        "orange juice", "wine bottle"])
    def test_grocery_phrases_get_a_grocery_tail(self, phrase):
        c = _with_fallbacks(phrase)
        assert "box" in c and "can" in c

    def test_relational_phrase_routes_on_its_head_noun(self):
        # libero_spatial's real phrasing.
        c = _with_fallbacks("black bowl between the plate and the ramekin")
        assert c[0] == "black bowl between the plate and the ramekin"
        assert "bowl" in c and "plate" in c

    def test_basket_keeps_the_receptacle_tail(self):
        c = _with_fallbacks("basket")
        assert c[:2] == ["basket", "bin"]

    def test_unknown_noun_gets_both_families(self):
        c = _with_fallbacks("widget")
        assert "bowl" in c and "box" in c, "an unrecognized noun must not be left bare"


def test_every_suite_family_is_covered():
    """One representative phrase per LIBERO suite must get a usable tail."""
    for phrase, must_contain in (
        ("alphabet soup", "can"),        # libero_object
        ("black bowl", "bowl"),          # libero_spatial
        ("plate", "plate"),              # libero_goal targets
    ):
        assert must_contain in _with_fallbacks(phrase)
