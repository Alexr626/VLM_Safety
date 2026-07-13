"""System / user prompt templates for demos_v2 MLLM stages."""

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
  "counting_check": {"verdict": "pass"|"fail", "reason": "<one sentence>"},
  "relation_check": {"verdict": "pass"|"fail", "reason": "<one sentence>"},
  "distractor": {"verdict": "pass"|"fail", "choice": "<category or empty>", "reason": "<one sentence>"},
  "attribute": {
    "verdict": "pass"|"fail",
    "object": "<object or group noun>",
    "true_value": "<1-2 words>",
    "false_value": "<1-2 words, clearly wrong for that object>",
    "overlaps_counting_category": true|false,
    "reason": "<one sentence>"
  }
}

Checks:
1) counting_check: The stated category and count come from human annotation. Confirm the objects
   are individually distinguishable (not a blurred mass, not mostly occluded) and that
   "at least {count} {category}" is clearly true. Do NOT recount from scratch as the primary
   signal; fail only if the image visibly contradicts the annotation or objects cannot be told apart.
2) relation_check: From the VIEWER's perspective, every instance of A is clearly to the left of
   every instance of B, with visible separation. Fail on ambiguity (depth overlap, partial visibility).
3) distractor: From the provided candidate list IN ORDER, pick the FIRST category that is
   (a) plausible in this scene and (b) definitely NOT visible anywhere (including background,
   partial views, reflections). Scan carefully. If none qualify, verdict=fail and choice="".
4) attribute: Choose one clearly visible object or homogeneous group and one unambiguous attribute
   value. Color is strongly preferred; material or a coarse size term only if no clean color exists.
   Also give one false alternative value clearly not present on that object. Both values must be
   1-2 words. Do not use color on black-and-white images. Prefer an object distinct from the
   counting-anchor category; allow overlap only if unavoidable and set overlaps_counting_category.

Prohibitions: no hedging language in any field; no attributes requiring counting or spatial
reasoning; no subjective attributes (delicious, beautiful, nice).
"""


def stage1_user(rec: Dict[str, Any]) -> str:
    ca = rec["counting_anchor"]
    ra = rec["relation_anchor"]
    return (
        f"Image id: {rec['id']}\n"
        f"Counting anchor: category={ca['category']}, annotated_count={ca['count']}\n"
        f"Relation anchor: A={ra['a']}, B={ra['b']}, claimed={ra['relation']} "
        f"(viewer perspective; A should be to the {ra['relation']} of B)\n"
        f"Distractor candidates (in order): {rec['distractor_candidates']}\n"
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
- S3 (counting): Must contain the exact phrase "at least {{N_word}} {{category…}}". The number
  word appears EXACTLY ONCE in the whole caption. No other quantity words (several, many, few,
  a couple). You may use a natural plural of the counting category (e.g. category "person" →
  "people"; "dining table" → "dining tables") — do not invent a different noun.
- S4 (relation): Exactly "The {{A phrase}} is to the left of the {{B phrase}}." (or "right" if
  the anchor says so), viewer perspective. The words "left"/"right" appear EXACTLY ONCE.

Global rules:
- Every claim must be visibly true; nothing beyond the given anchors plus minimal scene glue
  that is trivially true.
- Forbidden: digits; these hedge words {config.BANNED_HEDGES}; the distractor word; pronouns
  that refer across sentences; subjective language.
- Output schema:
{{
  "caption": "<four sentences>",
  "spans": {{
    "existence_insertion_hint": "<verbatim S1 enumeration substring>",
    "attribute": "<true attribute value verbatim>",
    "counting": "<number word verbatim>",
    "relation": "left"|"right"
  }}
}}
spans values MUST be verbatim substrings of caption.
"""


def stage2_user(rec: Dict[str, Any]) -> str:
    ca = rec["counting_anchor"]
    ra = rec["relation_anchor"]
    attr = rec["attribute"]
    n_word = number_to_word(int(ca["count"]))
    return (
        f"Image id: {rec['id']}\n"
        f"Counting: category={ca['category']}, N={ca['count']}, N_word={n_word}\n"
        f"Relation: A={ra['a']}, B={ra['b']}, direction={ra['relation']}\n"
        f"Attribute: object={attr['object']}, true_value={attr['true_value']}\n"
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
    sents = rec["value"].rstrip(".").split(". ")
    # Prefer stored sentences if present
    if rec.get("sentences"):
        sents = rec["sentences"]
    anchors = rec["anchors"]
    false_rel = anchors["relation"]["false"]
    statements = []
    for i, s in enumerate(sents[:4], start=1):
        statements.append(f"{i}. (expect true) {s.rstrip('.')}.")
    statements.append(
        f"5. (expect false) There is a {anchors['existence']['distractor']} visible in the image."
    )
    statements.append(
        f"6. (expect false) The {anchors['attribute']['object']} is "
        f"{anchors['attribute']['false_value']}."
    )
    statements.append(
        f"7. (expect false) There are at least {anchors['counting']['false_word']} "
        f"{anchors['counting']['category']}."
    )
    statements.append(
        f"8. (expect false) The {anchors['relation']['a']} is to the {false_rel} of "
        f"the {anchors['relation']['b']}."
    )
    return (
        f"Image id: {rec['id']}\n"
        "Answer each numbered statement with true/false/unsure:\n"
        + "\n".join(statements)
    )


def dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
