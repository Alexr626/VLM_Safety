"""Deterministic caption / minimal-pair validators for demos_v2."""

from __future__ import annotations

import difflib
import re
from typing import List, Optional, Sequence, Tuple

from . import config
from .number_words import number_to_word

_DIGIT_RE = re.compile(r"\d")
_SENT_SPLIT_RE = re.compile(r"(?<=\.)\s+")


def split_sentences(caption: str) -> List[str]:
    """Split on '. ' / trailing period. Safe under the no-digits caption rule."""
    text = caption.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    # Normalize: keep terminal period on each sentence if present
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out


def tokenize(text: str) -> List[str]:
    """Whitespace-tokenize after light punctuation isolation for diffing."""
    # Keep commas/periods attached for existence-insertion checks; splitter is whitespace.
    return text.split()


def find_banned_hedges(caption: str) -> List[str]:
    lower = caption.lower()
    hits = []
    for h in config.BANNED_HEDGES:
        # word-ish match: allow phrase hedges via substring with word boundaries for singles
        if " " in h:
            if h in lower:
                hits.append(h)
        else:
            if re.search(rf"\b{re.escape(h)}\b", lower):
                hits.append(h)
    return hits


def has_digits(caption: str) -> bool:
    return bool(_DIGIT_RE.search(caption))


def count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return haystack.lower().count(needle.lower())


# Irregular last-word plurals for COCO-ish category tokens.
_IRREGULAR_PLURALS = {
    "person": ("people", "persons"),
    "man": ("men",),
    "woman": ("women",),
    "child": ("children",),
    "knife": ("knives",),
    "shelf": ("shelves",),
    "leaf": ("leaves",),
    "mouse": ("mice",),
    "goose": ("geese",),
    "tooth": ("teeth",),
    "foot": ("feet",),
    "sheep": ("sheep",),
}


def category_surface_forms(category: str) -> List[str]:
    """Singular + common plural surface forms for a COCO category name."""
    cat = (category or "").strip().lower()
    if not cat:
        return []
    forms = {cat}
    words = cat.split()
    last = words[-1]
    last_forms = {last, last + "s", last + "es"}
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last_forms.add(last[:-1] + "ies")
    if last.endswith("f") and not last.endswith("ff"):
        last_forms.add(last[:-1] + "ves")
    if last.endswith("fe"):
        last_forms.add(last[:-2] + "ves")
    last_forms.update(_IRREGULAR_PLURALS.get(last, ()))
    for lf in last_forms:
        forms.add(" ".join(words[:-1] + [lf]) if len(words) > 1 else lf)
    # longest first so multi-word matches prefer full phrases
    return sorted(forms, key=len, reverse=True)


def category_mentioned(caption: str, category: str) -> bool:
    """True if ``category`` or a natural plural appears as a whole phrase."""
    cap = caption or ""
    for form in category_surface_forms(category):
        if re.search(rf"\b{re.escape(form)}\b", cap, flags=re.IGNORECASE):
            return True
    return False


def distractor_noun_phrase(distractor: str) -> str:
    """Build a short noun phrase (≤4 words) for existence insertion."""
    d = (distractor or "").strip()
    if not d:
        raise ValueError("empty distractor")
    if re.match(r"^(a|an|some)\b", d, flags=re.IGNORECASE):
        return d
    first = d.split()[0].lower()
    article = "an" if first[:1] in "aeiou" else "a"
    return f"{article} {d}"


# Verb / preposition leads that introduce an enumeration in S1.
_EXISTENCE_HINT_LEAD = re.compile(
    r"^(including|with|featuring|includes|features|shows|contains|has)\s+(.+)$",
    flags=re.IGNORECASE,
)


def deterministic_existence_insert(
    caption: str, hint: str, distractor: str,
) -> str:
    """Insert ``distractor`` into the S1 enumeration hint (no LLM).

    Prefers ``includes/with/including/… X…`` → ``includes/with/… a D, X…``.
    Avoids the broken prepend ``a D, includes X…``.
    """
    if not hint or hint not in caption:
        raise ValueError(f"existence hint not found verbatim in caption: {hint!r}")
    phrase = distractor_noun_phrase(distractor)
    m = _EXISTENCE_HINT_LEAD.match(hint.strip())
    if m:
        new_hint = f"{m.group(1)} {phrase}, {m.group(2)}"
    else:
        # Last resort: prepend inside the hint span (grammar polish may repair).
        new_hint = f"{phrase}, {hint}"
    return caption.replace(hint, new_hint, 1)


def structural_check_truthful(
    caption: str,
    *,
    count_word: str,
    category: str,
    relation_word: str,
    attribute_value: str,
    distractor: str,
    spans: dict,
) -> List[str]:
    """Stage-2 structural checks. Returns a list of violation strings (empty = ok)."""
    errs: List[str] = []
    sents = split_sentences(caption)
    if len(sents) != 4:
        errs.append(f"expected 4 sentences, got {len(sents)}")
    if has_digits(caption):
        errs.append("caption contains digits")
    hedges = find_banned_hedges(caption)
    if hedges:
        errs.append(f"banned hedges: {hedges}")
    if distractor and re.search(rf"\b{re.escape(distractor)}\b", caption, re.I):
        errs.append(f"distractor token '{distractor}' appears in truthful caption")

    phrase = f"at least {count_word}"
    if phrase.lower() not in caption.lower():
        errs.append(f"missing exact phrase '{phrase}'")
    if count_occurrences(caption, count_word) != 1:
        errs.append(f"count word '{count_word}' must appear exactly once")
    if count_occurrences(caption, relation_word) != 1:
        errs.append(f"relation word '{relation_word}' must appear exactly once")
    if count_occurrences(caption, attribute_value) != 1:
        errs.append(f"attribute value '{attribute_value}' must appear exactly once")

    # span presence / sentence membership
    for key, sent_idx in (
        ("existence_insertion_hint", 0),
        ("attribute", 1),
        ("counting", 2),
        ("relation", 3),
    ):
        span = (spans or {}).get(key, "")
        if not span:
            errs.append(f"missing span '{key}'")
            continue
        if span not in caption:
            if span.lower() not in caption.lower():
                errs.append(f"span '{key}'={span!r} not found in caption")
                continue
        if len(sents) == 4 and span.lower() not in sents[sent_idx].lower():
            errs.append(f"span '{key}' not in sentence S{sent_idx + 1}")

    expected_rel = relation_word.lower()
    if expected_rel not in ("left", "right"):
        errs.append(f"invalid relation word '{relation_word}'")
    elif (spans or {}).get("relation", "").lower() != expected_rel:
        errs.append("spans.relation must equal anchor direction")

    if (spans or {}).get("counting", "").lower() != count_word.lower():
        errs.append("spans.counting must equal the true count word")
    if (spans or {}).get("attribute", "").lower() != attribute_value.lower():
        errs.append("spans.attribute must equal the true attribute value")

    if len(sents) == 4:
        s3 = sents[2].lower()
        for q in ("several", "many", "few", "a couple"):
            if q in s3:
                errs.append(f"forbidden quantity word '{q}' in S3")

    # Accept natural plurals (person/people, dining table/tables, …).
    if category and not category_mentioned(caption, category):
        errs.append(
            f"counting category '{category}' (or a natural plural) not found in caption"
        )

    return errs


def _norm_tok(t: str) -> str:
    return t.lower().strip(".,;:")


def _join_norm(toks: Sequence[str]) -> str:
    return " ".join(_norm_tok(t) for t in toks).strip()


def _opcode_regions(a_toks: Sequence[str], b_toks: Sequence[str]):
    sm = difflib.SequenceMatcher(a=list(a_toks), b=list(b_toks))
    return [op for op in sm.get_opcodes() if op[0] != "equal"]


def _sentence_index_for_token_span(caption: str, tok_start: int, tok_end: int) -> int:
    """Map a token-index span in whitespace-tokenized caption to sentence index."""
    toks = tokenize(caption)
    if tok_start >= len(toks):
        return -1
    sents = split_sentences(caption)
    cum = 0
    for i, s in enumerate(sents):
        n = len(tokenize(s))
        if tok_start < cum + n:
            return i
        cum += n
    return len(sents) - 1 if sents else -1


def _validate_existence_pair(
    truthful: str, variant: str, distractor: str,
) -> List[str]:
    """Existence: S1-only change; allow multiple local edit regions (list rewrites)."""
    errs: List[str] = []
    t_sents = split_sentences(truthful)
    v_sents = split_sentences(variant)
    if len(t_sents) != 4 or len(v_sents) != 4:
        errs.append(
            f"existence requires 4 sentences on both sides "
            f"(got {len(t_sents)} / {len(v_sents)})"
        )
        return errs
    for i in (1, 2, 3):
        if t_sents[i] != v_sents[i]:
            errs.append(f"existence edit leaked into S{i + 1}")
    if t_sents[0] == v_sents[0]:
        errs.append("existence variant S1 is identical to truthful S1")
    if distractor:
        if not re.search(rf"\b{re.escape(distractor)}\b", v_sents[0], re.I):
            errs.append(f"existence S1 must contain distractor '{distractor}'")
        if re.search(rf"\b{re.escape(distractor)}\b", t_sents[0], re.I):
            errs.append(f"distractor '{distractor}' already present in truthful S1")

    regions = _opcode_regions(tokenize(truthful), tokenize(variant))
    if not regions:
        errs.append("existence variant has no token-level edit")
    if len(regions) > 3:
        errs.append(
            f"existence has {len(regions)} edit regions (max 3 for list rewrites)"
        )
    for tag, i1, i2, j1, j2 in regions:
        sent_idx = _sentence_index_for_token_span(
            truthful, i1, i2 if i2 > i1 else i1,
        )
        if sent_idx != 0:
            errs.append(
                f"existence edit region in S{sent_idx + 1}, expected S1 only"
            )
        if tag not in ("insert", "replace", "delete"):
            errs.append(f"unexpected existence opcode {tag}")
    # Soft length budget: S1 should not grow by more than 6 tokens.
    growth = len(tokenize(v_sents[0])) - len(tokenize(t_sents[0]))
    if growth > 6:
        errs.append(f"existence S1 grew by {growth} tokens (max 6)")
    if growth < 1 and not errs:
        errs.append("existence S1 did not gain tokens")
    return errs


def validate_minimal_pair(
    truthful: str,
    variant: str,
    dimension: str,
    *,
    distractor: str = "",
    true_attr: str = "",
    false_attr: str = "",
    true_count_word: str = "",
    false_count_word: str = "",
    true_relation: str = "",
) -> List[str]:
    """Return violation strings for a (truthful, variant, dimension) triple."""
    errs: List[str] = []
    dim_to_sent = {
        "existence": 0,
        "attribute": 1,
        "counting": 2,
        "relation": 3,
    }
    if dimension not in dim_to_sent:
        return [f"unknown dimension '{dimension}'"]

    if dimension == "existence":
        return _validate_existence_pair(truthful, variant, distractor)

    a = tokenize(truthful)
    b = tokenize(variant)
    regions = _opcode_regions(a, b)
    if len(regions) != 1:
        errs.append(f"expected exactly 1 edit region, got {len(regions)}: {regions}")
        return errs

    tag, i1, i2, j1, j2 = regions[0]
    sent_idx = _sentence_index_for_token_span(truthful, i1, i2 if i2 > i1 else i1)
    if sent_idx != dim_to_sent[dimension]:
        errs.append(
            f"edit in sentence S{sent_idx + 1}, expected S{dim_to_sent[dimension] + 1}"
        )

    removed = a[i1:i2]
    inserted = b[j1:j2]

    if dimension == "attribute":
        if tag != "replace":
            errs.append(f"attribute edit must be replace, got {tag}")
        if _join_norm(removed) != true_attr.lower():
            errs.append(f"attribute removed {removed} != true '{true_attr}'")
        if _join_norm(inserted) != false_attr.lower():
            errs.append(f"attribute inserted {inserted} != false '{false_attr}'")

    elif dimension == "counting":
        if tag != "replace":
            errs.append(f"counting edit must be replace, got {tag}")
        if _join_norm(removed) != true_count_word.lower():
            errs.append(f"counting removed {removed} != '{true_count_word}'")
        if _join_norm(inserted) != false_count_word.lower():
            errs.append(f"counting inserted {inserted} != '{false_count_word}'")

    elif dimension == "relation":
        if tag != "replace":
            errs.append(f"relation edit must be replace, got {tag}")
        if len(removed) != 1 or len(inserted) != 1:
            errs.append(f"relation edit must swap one token: {removed} -> {inserted}")
        else:
            pair = {_norm_tok(removed[0]), _norm_tok(inserted[0])}
            if pair != {"left", "right"}:
                errs.append(f"relation swap must be left↔right, got {pair}")
            if true_relation and _norm_tok(removed[0]) != true_relation.lower():
                errs.append(f"relation removed '{removed[0]}' != true '{true_relation}'")

    return errs


def validate_record_variants(
    truthful: str,
    h_values: dict,
    *,
    distractor: str,
    true_attr: str,
    false_attr: str,
    true_count_word: str,
    false_count_word: str,
    true_relation: str,
) -> List[str]:
    errs: List[str] = []
    for dim, variant in h_values.items():
        errs.extend(
            f"{dim}: {e}"
            for e in validate_minimal_pair(
                truthful,
                variant,
                dim,
                distractor=distractor,
                true_attr=true_attr,
                false_attr=false_attr,
                true_count_word=true_count_word,
                false_count_word=false_count_word,
                true_relation=true_relation,
            )
        )
    # distractor only in existence
    for dim, variant in h_values.items():
        if dim == "existence":
            continue
        if distractor and re.search(rf"\b{re.escape(distractor)}\b", variant, re.I):
            errs.append(f"distractor '{distractor}' leaked into {dim} variant")
    return errs
