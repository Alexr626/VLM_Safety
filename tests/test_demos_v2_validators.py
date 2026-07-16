"""Unit tests for demos_v2 validators and helpers (no GPU / network)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data_scripts"))

from vti_demos_v2 import config  # noqa: E402
from vti_demos_v2.number_words import false_count, false_count_at_most, number_to_word  # noqa: E402
from vti_demos_v2.validators import (  # noqa: E402
    category_mentioned,
    deterministic_existence_insert,
    find_banned_hedges,
    has_digits,
    split_sentences,
    structural_check_truthful,
    validate_minimal_pair,
    validate_record_variants,
    replace_span_in_sentence,
)


def test_number_to_word_basic():
    assert number_to_word(0) == "zero"
    assert number_to_word(8) == "eight"
    assert number_to_word(12) == "twelve"
    assert number_to_word(20) == "twenty"


def test_false_count_boundaries():
    assert false_count(2) == 4  # max(2+2, ceil(3)) = 4
    assert false_count(8) == 12  # max(10, 12) = 12
    assert false_count(14) == 20  # capped
    assert config.FALSE_COUNT(14) == min(max(16, math.ceil(21)), 20)


def test_false_count_at_most_boundaries_and_sentence_scope():
    assert [false_count_at_most(n) for n in (3, 4, 6, 9)] == [1, 2, 4, 6]
    caption = ("The room has a right side of the room. The plates are white. "
               "There are at least two cups. The cup is to the left of the bowl.")
    out = replace_span_in_sentence(caption, 3, "left", "right")
    assert "right side of the room" in out
    assert out.endswith("right of the bowl.")


def test_at_most_plurality_and_relation_phrase_swaps():
    from vti_demos_v2.validators import validate_minimal_pair
    truthful = (
        "Scene with chairs and a table. The cloth is red. "
        "There are at most three chairs near the wall. "
        "The cup is on top of the table."
    )
    variant = (
        "Scene with chairs and a table. The cloth is red. "
        "There are at most one chair near the wall. "
        "The cup is on top of the table."
    )
    assert validate_minimal_pair(
        truthful, variant, "counting",
        true_count_word="three", false_count_word="one",
        count_category="chair", allow_count_noun_morph=True,
    ) == []
    rel_t = (
        "Scene with A and B. The cloth is red. "
        "There are at least two cups. "
        "The cup is right next to the bowl."
    )
    rel_v = rel_t.replace("right next to", "far away from", 1)
    assert validate_minimal_pair(
        rel_t, rel_v, "relation", true_relation="right next to",
    ) == []


def test_validate_combined_four_regions():
    from vti_demos_v2.validators import validate_combined
    t = (
        "S1 with cups. The plates are white. "
        "There are at least two cups. The cup is to the left of the bowl."
    )
    c = (
        "S1 with a fork, cups. The plates are black. "
        "There are at least four cups. The cup is to the right of the bowl."
    )
    assert validate_combined(t, c) == []
    bad = (
        "S1 with a fork, cups. The plates are white. "
        "There are at least two cups. The cup is to the left of the bowl."
    )
    assert validate_combined(t, bad)


def test_sentence_split_no_digits():
    cap = (
        "The image shows a kitchen with plates, bowls, and cups. "
        "The plates are white. "
        "There are at least eight sandwiches. "
        "The plate is to the left of the bowl."
    )
    sents = split_sentences(cap)
    assert len(sents) == 4
    assert sents[2].startswith("There are at least")


def test_structural_pass():
    caption = (
        "The image shows a counter with food, including sandwiches, bowls, and bottles. "
        "The plates holding the food are white. "
        "There are at least eight sandwiches on one plate. "
        "The plate of sandwiches is to the left of a bowl of pasta."
    )
    spans = {
        "existence_insertion_hint": "including sandwiches, bowls, and bottles",
        "attribute": "white",
        "counting": "eight",
        "relation": "left",
    }
    errs = structural_check_truthful(
        caption,
        count_word="eight",
        category="sandwich",
        relation_word="left",
        attribute_value="white",
        distractor="fork",
        spans=spans,
    )
    assert errs == []


def test_structural_accepts_person_people_plural():
    caption = (
        "This kitchen is busy with people, a refrigerator, and a potted plant. "
        "The pendant lamp is red. "
        "There are at least two people visible in the space. "
        "The refrigerator is to the left of the potted plant."
    )
    spans = {
        "existence_insertion_hint": "with people, a refrigerator, and a potted plant",
        "attribute": "red",
        "counting": "two",
        "relation": "left",
    }
    assert category_mentioned(caption, "person")
    errs = structural_check_truthful(
        caption,
        count_word="two",
        category="person",
        relation_word="left",
        attribute_value="red",
        distractor="sink",
        spans=spans,
    )
    assert errs == []


def test_structural_rejects_digits_and_hedges():
    caption = (
        "The image shows 2 dogs. The dogs are brown. "
        "There are at least two dogs. The dog is to the left of the person."
    )
    spans = {
        "existence_insertion_hint": "shows 2 dogs",
        "attribute": "brown",
        "counting": "two",
        "relation": "left",
    }
    errs = structural_check_truthful(
        caption,
        count_word="two",
        category="dog",
        relation_word="left",
        attribute_value="brown",
        distractor="cat",
        spans=spans,
    )
    assert any("digits" in e for e in errs)


def test_hedge_detection():
    assert "possibly" in find_banned_hedges("It possibly contains a cat.")
    assert find_banned_hedges("A red ball sits on the table.") == []


def test_minimal_pair_counting():
    t = "There are at least eight sandwiches on one plate."
    truthful = (
        "Scene with sandwiches, bowls, and cups. "
        "The plates are white. "
        f"{t} "
        "The plate is to the left of the bowl."
    )
    variant = truthful.replace("eight", "twelve", 1)
    assert validate_minimal_pair(
        truthful, variant, "counting",
        true_count_word="eight", false_count_word="twelve",
    ) == []


def test_minimal_pair_relation():
    truthful = (
        "Scene with A and B. The plates are white. "
        "There are at least two cups. "
        "The cup is to the left of the bowl."
    )
    variant = truthful.replace("left", "right", 1)
    assert validate_minimal_pair(
        truthful, variant, "relation", true_relation="left",
    ) == []


def test_minimal_pair_attribute():
    truthful = (
        "Scene with plates. The plates are white. "
        "There are at least two plates. "
        "The plate is to the left of the bowl."
    )
    variant = truthful.replace("white", "black", 1)
    assert validate_minimal_pair(
        truthful, variant, "attribute",
        true_attr="white", false_attr="black",
    ) == []


def test_minimal_pair_existence():
    truthful = (
        "The image shows food, including sandwiches, bowls, and bottles. "
        "The plates are white. "
        "There are at least eight sandwiches. "
        "The plate is to the left of the bowl."
    )
    variant = truthful.replace(
        "including sandwiches, bowls, and bottles",
        "including a fork, sandwiches, bowls, and bottles",
        1,
    )
    assert validate_minimal_pair(
        truthful, variant, "existence", distractor="fork",
    ) == []


def test_minimal_pair_existence_allows_list_rewrite_two_regions():
    """Pilot failure mode: LLM drops 'and' before last item while appending distractor."""
    truthful = (
        "An outdoor cafe patio features wooden dining tables with a slice of cake, "
        "a spoon, and a knife. "
        "The cup resting on its saucer is white. "
        "There are at least two dining tables spread across the patio. "
        "The bench is to the left of the cup."
    )
    variant = (
        "An outdoor cafe patio features wooden dining tables with a slice of cake, "
        "a spoon, a knife, and a chair. "
        "The cup resting on its saucer is white. "
        "There are at least two dining tables spread across the patio. "
        "The bench is to the left of the cup."
    )
    assert validate_minimal_pair(
        truthful, variant, "existence", distractor="chair",
    ) == []


def test_deterministic_existence_insert():
    caption = (
        "Scene with sandwiches, bowls, and cups. The plates are white. "
        "There are at least two sandwiches. "
        "The plate is to the left of the bowl."
    )
    hint = "with sandwiches, bowls, and cups"
    out = deterministic_existence_insert(caption, hint, "fork")
    assert "a fork, sandwiches" in out
    assert validate_minimal_pair(caption, out, "existence", distractor="fork") == []


def test_deterministic_existence_insert_includes_verb():
    """Regression: 'includes …' must not become 'a chair, includes …'."""
    caption = (
        "This bedroom scene includes a bed, a person, and a tv arranged near the window. "
        "The bed sheets carry a blue white floral pattern. "
        "There are at least two people seated inside the room. "
        "The person is to the left of the tv."
    )
    hint = "includes a bed, a person, and a tv arranged near the window"
    out = deterministic_existence_insert(caption, hint, "chair")
    assert out.startswith(
        "This bedroom scene includes a chair, a bed, a person, and a tv"
    )
    assert "a chair, includes" not in out
    assert validate_minimal_pair(caption, out, "existence", distractor="chair") == []


def test_fail_multi_region():
    truthful = (
        "Scene with A and B. The plates are white. "
        "There are at least two cups. "
        "The cup is to the left of the bowl."
    )
    variant = truthful.replace("white", "black").replace("left", "right")
    errs = validate_minimal_pair(
        truthful, variant, "attribute",
        true_attr="white", false_attr="black",
    )
    # Sentence-scoped attribute check: extra relation edit must fail as S4 leak.
    assert any("leaked into S4" in e for e in errs)


def test_fail_wrong_sentence():
    truthful = (
        "Scene with A and B. The plates are white. "
        "There are at least two cups. "
        "The cup is to the left of the bowl."
    )
    # edit counting word but claim attribute dimension
    variant = truthful.replace("two", "four", 1)
    errs = validate_minimal_pair(
        truthful, variant, "attribute",
        true_attr="white", false_attr="black",
    )
    assert errs  # wrong sentence and/or wrong tokens


def test_distractor_leak_in_truthful():
    caption = (
        "The image includes a fork and sandwiches. The plates are white. "
        "There are at least two sandwiches. "
        "The plate is to the left of the bowl."
    )
    spans = {
        "existence_insertion_hint": "includes a fork and sandwiches",
        "attribute": "white",
        "counting": "two",
        "relation": "left",
    }
    errs = structural_check_truthful(
        caption,
        count_word="two",
        category="sandwich",
        relation_word="left",
        attribute_value="white",
        distractor="fork",
        spans=spans,
    )
    assert any("distractor" in e for e in errs)


def test_validate_record_variants_leak():
    truthful = (
        "Scene including sandwiches, bowls, and cups. The plates are white. "
        "There are at least two sandwiches. "
        "The plate is to the left of the bowl."
    )
    h_values = {
        "existence": truthful.replace(
            "including sandwiches, bowls, and cups",
            "a fork, including sandwiches, bowls, and cups",
            1,
        ),
        "attribute": truthful.replace("white", "black", 1),
        "counting": truthful.replace("two", "four", 1),
        "relation": truthful.replace("left", "right", 1),
    }
    # leak distractor into attribute variant
    h_values["attribute"] = h_values["attribute"].replace(
        "plates", "fork plates", 1
    )
    errs = validate_record_variants(
        truthful, h_values,
        distractor="fork",
        true_attr="white", false_attr="black",
        true_count_word="two", false_count_word="four",
        true_relation="left",
    )
    assert any("leaked" in e or "edit region" in e for e in errs)


def test_has_digits():
    assert has_digits("there are 3 cats")
    assert not has_digits("there are three cats")


def test_relation_geometry_helper_import():
    # stage0 helpers are importable
    from vti_demos_v2.geometry import evaluate_all_relation_types  # noqa: F401
