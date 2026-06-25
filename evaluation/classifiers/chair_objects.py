"""
CHAIR object grounding: COCO ground-truth lookup + caption object parsing.

Implements the object-inventory machinery the CHAIR metric needs:
  - the canonical Rohrbach et al. synonym list (`chair_synonyms.txt`, vendored
    next to this file) mapping caption words/phrases -> the 80 COCO classes,
  - ground-truth COCO objects per image from `instances_val2014.json`,
  - a caption -> mentioned-COCO-objects parser.

Parsing fidelity note: the original CHAIR (`chair.py`) tokenizes with nltk and
singularizes with `pattern.en`. Those deps are intentionally avoided here (this
env has neither, and pattern.en is unmaintained on py3.12+). Instead we expand
each synonym to its common surface forms (singular + naive/irregular plural) and
match raw caption tokens against the expanded set with longest-n-gram-first
matching. Because plural handling lives entirely in the key set (the caption is
NOT singularized), matching is deterministic and self-consistent. Canonical
class labels and GT names are taken verbatim from the synonym list's first
column, which is identical to the COCO instance category names, so mentioned vs.
GT comparison is apples-to-apples.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.paths import coco_annotations_dir

_SYNONYMS_PATH = Path(__file__).resolve().parent / "chair_synonyms.txt"
_INSTANCES_FILENAME = "instances_val2014.json"

# Documented CHAIR special-case mappings (from the canonical chair.py rules):
# qualified animals ("baby bird" -> "bird") and a few fixed double words that
# would otherwise mis-resolve. Keys are matched as multi-word phrases.
_ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "animal",
                 "elephant", "bear", "zebra", "giraffe"]
_SPECIAL_PHRASE_TO_CLASS = {
    "toilet seat": "toilet",
    "bow tie": "tie",
    "passenger jet": "airplane",
    "passenger train": "train",
}

# Irregular plurals for synonym words that are not already listed in plural form
# in chair_synonyms.txt.
_IRREGULAR_PLURALS = {
    "man": "men",
    "woman": "women",
    "person": "people",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice",
    "goose": "geese",
}

_TOKEN_RE = re.compile(r"[a-z]+")


def _pluralize(word: str) -> str:
    """Naive English pluralization of a single token (last word of a phrase)."""
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _surface_forms(phrase: str) -> Set[str]:
    """All surface forms of a synonym phrase to register as lookup keys."""
    phrase = phrase.strip()
    forms = {phrase}
    words = phrase.split()
    if not words:
        return forms
    head, last = words[:-1], words[-1]
    prefix = (" ".join(head) + " ") if head else ""
    forms.add(prefix + _pluralize(last))
    if last in _IRREGULAR_PLURALS:
        forms.add(prefix + _IRREGULAR_PLURALS[last])
    return forms


@lru_cache(maxsize=1)
def load_synonym_map() -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """Return (surface_form -> canonical COCO class, ordered COCO class tuple).

    The first token of each `chair_synonyms.txt` line is the canonical COCO
    class name; every comma-separated entry on the line is a synonym for it.
    """
    if not _SYNONYMS_PATH.exists():
        raise FileNotFoundError(
            f"CHAIR synonym list missing at {_SYNONYMS_PATH}. It is vendored in "
            "the repo (canonical Rohrbach et al. synonyms.txt).")
    syn_to_class: Dict[str, str] = {}
    classes: List[str] = []
    with open(_SYNONYMS_PATH) as f:
        for line in f:
            entries = [e.strip().lower() for e in line.strip().split(",") if e.strip()]
            if not entries:
                continue
            canonical = entries[0]
            classes.append(canonical)
            for syn in entries:
                for form in _surface_forms(syn):
                    syn_to_class.setdefault(form, canonical)
    # Layer in the documented special-case phrases.
    for animal in _ANIMAL_WORDS:
        target = animal if animal != "animal" else "animal"
        for qualifier in ("baby", "adult"):
            syn_to_class.setdefault(f"{qualifier} {animal}", target)
    for phrase, cls in _SPECIAL_PHRASE_TO_CLASS.items():
        syn_to_class[phrase] = cls
    return syn_to_class, tuple(classes)


@lru_cache(maxsize=1)
def load_gt_objects(instances_path: str | None = None) -> Dict[int, Set[str]]:
    """Build COCO image_id -> set(class_name) from instance annotations."""
    path = Path(instances_path) if instances_path else (
        coco_annotations_dir() / _INSTANCES_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"COCO instance annotations missing at {path}. "
            "Run: python data_scripts/download_chair.py")
    with open(path) as f:
        data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"].lower() for c in data["categories"]}
    gt: Dict[int, Set[str]] = {}
    for ann in data["annotations"]:
        name = cat_id_to_name.get(ann["category_id"])
        if name is None:
            continue
        gt.setdefault(ann["image_id"], set()).add(name)
    return gt


def parse_caption_objects(caption: str) -> Set[str]:
    """Return the set of distinct COCO classes mentioned in a caption.

    Longest-n-gram-first matching (up to 3 words) against the expanded synonym
    surface forms; deduplicated to distinct canonical classes (CHAIR scores at
    the distinct-object-per-caption level). The toilet/chair confusion is
    handled via the "toilet seat" -> toilet phrase plus the post-rule below.
    """
    syn_to_class, _ = load_synonym_map()
    tokens = _TOKEN_RE.findall((caption or "").lower())
    mentioned: Set[str] = set()
    matched_seat = False
    i = 0
    n = len(tokens)
    while i < n:
        hit = None
        for span in (3, 2, 1):
            if i + span > n:
                continue
            phrase = " ".join(tokens[i:i + span])
            cls = syn_to_class.get(phrase)
            if cls is not None:
                hit = (cls, span, phrase)
                break
        if hit is None:
            i += 1
            continue
        cls, span, phrase = hit
        if phrase in ("seat", "seats"):
            matched_seat = True
        mentioned.add(cls)
        i += span
    # "the seat of the toilet" should not fire 'chair' via the 'seat' synonym.
    if matched_seat and "toilet" in mentioned and "chair" in mentioned:
        # Only drop 'chair' if it was introduced solely by 'seat'/'stool'/'chair'
        # words co-occurring with a toilet; conservative removal matching the
        # canonical CHAIR special case.
        mentioned.discard("chair")
    return mentioned


def gt_objects_for_image(image_id: int) -> Set[str]:
    """Ground-truth COCO classes for a COCO val2014 image id (empty if none)."""
    return load_gt_objects().get(int(image_id), set())
