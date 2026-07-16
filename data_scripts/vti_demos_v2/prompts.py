"""Machine-readable prompts for the demos_v2.1 option-set pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict

from . import config
from .number_words import number_to_word


STAGE1_SYSTEM = """You are a quality-control annotator for a contrastive captioning dataset.
Your reply is machine-parsed. Output ONLY a single JSON object matching the schema described
below — no prose, no markdown fences, no commentary.

You are given an image plus COCO-derived anchors. The annotations are the primary ground truth;
you verify them visually and supply only what annotations cannot (absence checks, attributes).

Return this schema exactly:
{
  "counting_options": [{"category": "...", "count": 2, "verdict": "pass"|"fail",
    "count_complete": true|false, "reason": "..."}],
  "relation_options": [{"a": "...", "b": "...", "type": "...", "verdict": "pass"|"fail", "reason": "..."}],
  "distractors": [{"category": "...", "verdict": "absent"|"present"|"unsure", "reason": "..."}],
  "attributes": [{"type": "color"|"material"|"state"|"action"|"texture", "object": "...",
    "true_value": "<1-2 words>", "false_value": "<1-2 words>", "confidence": "high"|"medium"}]
}

Checks:
1) For EACH counting option, the category and count come from human annotation. Confirm objects
   are individually distinguishable (not a blurred mass, not mostly occluded) and that
   "at least {count} {category}" is clearly true. Do NOT recount from scratch as the primary
   signal; fail only if the image visibly contradicts the annotation or objects cannot be told apart.
`count_complete` is true only when no additional instances are visible anywhere; false does not
fail an otherwise usable option. 2) For EACH relation option, require exactly one visible instance
of each category, comparable depth, and the stated viewer-perspective geometry. Fail ambiguity.
3) Give a verdict for EACH distractor; absent means definitely not visible anywhere.
4) Return up to """ + str(config.K_ATTR) + """ attribute candidates, preferring distinct types.
Choose one clearly visible object or homogeneous group and an unambiguous attribute
value. Color, material, state, action, and texture are allowed.
   Also give one false alternative value clearly not present on that object. Both values must be
   1-2 words. Do not use color on black-and-white images. Prefer an object distinct from the
   counting-anchor category; allow overlap only if unavoidable and set overlaps_counting_category.

Prohibitions: no hedging language in any field; no attributes requiring counting or spatial
reasoning; no subjective attributes (delicious, beautiful, nice).
"""


def stage1_user(rec: Dict[str, Any]) -> str:
    return (
        f"Image id: {rec['id']}\n"
        f"Counting options: {json.dumps(rec['counting_options'])}\n"
        f"Relation options: {json.dumps(rec['relation_options'])}\n"
        f"Distractor candidates: {json.dumps(rec['distractor_candidates'])}\n"
        f"Present categories: {rec['present_categories']}\n"
        "Perform the four checks and return the JSON object."
    )


STAGE2_SYSTEM = f"""You write rigidly structured truthful captions for a contrastive dataset.
Output ONLY a single JSON object — no prose, no markdown fences.

Write EXACTLY four sentences with these roles:
- S1 (scene + existence anchors): Name the scene and enumerate ≥2 present anchor objects.
  Use an enumeration form ("…with X, Y, and Z" or "including X, Y, and Z") so a later noun
  insertion stays grammatical. Must mention the counting category and both relation categories
  (or natural noun phrases for them).
- S2 (attribute): Exactly one attribute claim using the given object and true value. The value
  word(s) must appear EXACTLY ONCE in the whole caption.
- S3 (counting): Must contain the exact supplied at_least/at_most phrase. The number
  word appears EXACTLY ONCE in the whole caption. No other quantity words (several, many, few,
  a couple). You may use a natural plural of the counting category (e.g. category "person" →
  "people"; "dining table" → "dining tables") — do not invent a different noun.
- S4 (relation): Use the supplied fixed relation template, viewer perspective. Relation categories
  must be singular everywhere in the caption.

Global rules:
- Every claim must be visibly true; nothing beyond the given anchors plus minimal scene glue
  that is trivially true.
- Forbidden: digits; these hedge words {config.BANNED_HEDGES}; the distractor word; pronouns
  that refer across sentences; subjective language.
- Output schema:
{{
  "caption": "<four sentences>",
  "spans": {{
    "existence_insertion_hint": {{"sentence_idx": 0, "text": "<verbatim S1 enumeration substring>"}},
    "attribute": {{"sentence_idx": 1, "text": "<true attribute value ONLY>"}},
    "counting": {{"sentence_idx": 2, "text": "<number word ONLY, e.g. two>"}},
    "relation": {{"sentence_idx": 3, "text": "<true relation phrase ONLY>"}}
  }}
}}
CRITICAL for spans.relation.text: it must be ONLY the short relation phrase that will be edited
later — one of: left | right | above | below | on top of | right next to | far away from.
Do NOT put the full S4 sentence in spans.relation (wrong: "The bowl is to the left of the cup.";
right: "left"). All span texts MUST be verbatim substrings of caption.
"""


def stage2_user(rec: Dict[str, Any]) -> str:
    ca, ra, attr = rec["counting"], rec["relation"], rec["attribute"]
    n_word = number_to_word(int(ca["count"]))
    count_phrase = f"{ca['mode'].replace('_', ' ')} {n_word} {ca['category']}"
    true_phrase = ra["true_phrase"]
    templates = {
        "horizontal": f"The {ra['a']} is to the {true_phrase} of the {ra['b']}.",
        "vertical": f"The {ra['a']} is {true_phrase} the {ra['b']}.",
        "support": f"The {ra['a']} is on top of the {ra['b']}.",
        "proximity": f"The {ra['a']} is {true_phrase} the {ra['b']}.",
    }
    return (
        f"Image id: {rec['id']}\n"
        f"Counting exact phrase: {count_phrase}\n"
        f"Relation exact S4: {templates[ra['type']]}\n"
        f"Relation span text MUST be exactly: {true_phrase!r}\n"
        f"Attribute: object={attr['object']}, true_value={attr['true_value']}\n"
        f"Attribute span text MUST be exactly: {attr['true_value']!r}\n"
        f"Counting span text MUST be exactly: {n_word!r}\n"
        f"FORBIDDEN distractor word (must not appear): {rec['distractor']}\n"
        f"Present categories: {rec.get('present_categories')}\n"
        "Write the truthful caption JSON."
    )


STAGE3_EXISTENCE_SYSTEM = """You edit captions with a single, minimal insertion.
Output ONLY JSON: {"caption": "..."}.

Given a full caption and a target noun {distractor}, insert ONE noun phrase naming it
(with a natural article/plural as needed, ≤4 words total) into the first sentence's
enumeration substring provided as existence_insertion_hint. Change NOTHING else —
no rewording, no punctuation changes outside the insertion. Return the full caption.
"""


def stage3_existence_user(caption: str, distractor: str, hint: str) -> str:
    return (
        f"distractor: {distractor}\n"
        f"existence_insertion_hint: {hint}\n"
        f"caption:\n{caption}\n"
        "Return JSON {\"caption\": \"...\"} with the single insertion."
    )


STAGE3_GRAMMAR_SYSTEM = """You repair grammar in a draft existence-hallucination caption.
Output ONLY JSON: {"caption": "..."}.

You receive:
- truthful: a grammatical 4-sentence caption
- draft: the same caption after inserting one distractor object into sentence 1
  (the draft may be ungrammatical, e.g. "scene a chair, includes a bed…")
- distractor: the object that must remain present in sentence 1

Rules:
1. Fix grammar/fluency of sentence 1 with the smallest possible edit.
2. Keep the distractor noun in sentence 1 (natural article/plural ok).
3. Sentences 2, 3, and 4 of your output MUST be character-identical to truthful.
4. Do not add or remove other objects; do not change attributes, counts, or relations.
5. Do not introduce digits or hedging words (possibly, appears, seems, …).
"""


def stage3_grammar_user(truthful: str, draft: str, distractor: str) -> str:
    return (
        f"distractor: {distractor}\n"
        f"truthful:\n{truthful}\n\n"
        f"draft:\n{draft}\n\n"
        "Return JSON {\"caption\": \"...\"} with a grammatical repair of draft."
    )


STAGE4_SYSTEM = """You verify factual claims about an image with targeted binary judgments.
Output ONLY JSON:
{"answers": [{"idx": 1, "verdict": "true"|"false"|"unsure", "reason": "<short>"}]}

Rules: judge only what is visible; use VIEWER perspective for left/right; prefer true/false;
use unsure only when genuinely ambiguous. Do not add extra keys.
"""


def stage4_user(rec: Dict[str, Any]) -> str:
    statements = stage4_statement_specs(rec)
    return (
        f"Image id: {rec['id']}\n"
        "Answer each numbered statement with true/false/unsure:\n"
        + "\n".join(f"{x['idx']}. (expect {x['expect']}) {x['text']}" for x in statements)
    )


def _relation_false_phrase(rel: Dict[str, Any]) -> str:
    """Prefer anchors.relation.false_phrase; accept legacy ``false`` key."""
    phrase = (rel.get("false_phrase") or rel.get("false") or "").strip()
    if not phrase:
        raise KeyError("anchors.relation missing false_phrase/false")
    return phrase


def _relation_claim(a: str, b: str, phrase: str, rel_type: str) -> str:
    """Build a natural-language relation claim matching Stage-2 templates."""
    t = (rel_type or "").strip()
    p = phrase.strip()
    if t == "horizontal" or p in ("left", "right"):
        return f"The {a} is to the {p} of the {b}."
    if t == "support" or p in ("on top of", "underneath"):
        return f"The {a} is {p} the {b}."
    return f"The {a} is {p} the {b}."


def stage4_statement_specs(rec: Dict[str, Any]) -> list[dict]:
    """Return dynamic verifier statements with explicit dimension provenance."""
    anchors = rec["anchors"]
    sents = rec.get("sentences") or rec["value"].rstrip(".").split(". ")
    specs = [{"idx": i + 1, "dimension": "truthful", "expect": "true", "text": s.rstrip(".") + "."}
             for i, s in enumerate(sents[:4])]
    rel = anchors["relation"]
    false_rel = _relation_false_phrase(rel)
    specs.extend([
        {"idx": len(specs) + 1, "dimension": "existence", "expect": "false",
         "text": f"There is a {anchors['existence']['distractor']} visible in the image."},
        {"idx": len(specs) + 2, "dimension": "attribute", "expect": "false",
         "text": f"The {anchors['attribute']['object']} is {anchors['attribute']['false_value']}."},
        {"idx": len(specs) + 3, "dimension": "counting", "expect": "false",
         "text": f"There are {anchors['counting']['mode'].replace('_', ' ')} {anchors['counting']['false_word']} {anchors['counting']['category']} visible."},
        {"idx": len(specs) + 4, "dimension": "relation_uniqueness_a", "expect": "true",
         "text": f"Exactly one {rel['a']} is visible in the image."},
        {"idx": len(specs) + 5, "dimension": "relation_uniqueness_b", "expect": "true",
         "text": f"Exactly one {rel['b']} is visible in the image."},
        {"idx": len(specs) + 6, "dimension": "relation", "expect": "false",
         "text": _relation_claim(rel["a"], rel["b"], false_rel, rel.get("type", ""))},
    ])
    return specs


def dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
