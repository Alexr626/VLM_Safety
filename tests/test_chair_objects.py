"""CHAIR caption parser against the vendored synonym list (CPU, no data needed)."""

from evaluation.classifiers.chair_objects import load_synonym_map, parse_caption_objects


def test_synonym_map_covers_80_coco_classes():
    _, classes = load_synonym_map()
    assert len(classes) == 80
    assert "person" in classes and "dining table" in classes


def test_parse_singular_plural_and_synonyms():
    objs = parse_caption_objects("Two men sit at a table next to a dog and some cups.")
    assert {"person", "dining table", "dog", "cup"} <= objs


def test_multiword_phrase_wins_over_single_word():
    objs = parse_caption_objects("A man wearing a bow tie.")
    assert "tie" in objs


def test_toilet_seat_does_not_add_chair():
    objs = parse_caption_objects("The toilet seat is up next to the sink.")
    assert "toilet" in objs
    assert "chair" not in objs


def test_empty_caption():
    assert parse_caption_objects("") == set()
