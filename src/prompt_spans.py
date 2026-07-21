"""Tokenization-invariant clause / question span utilities for prefixed prompts.

Used by leading-clause attention knockout and reusable by later experiments on
the same augmented prompts.

Hard-fails when a clause/question boundary merges tokens or when the shared
question span in a prefixed prompt does not exactly match the neutral question
token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PromptSpanLayout:
    """Byte-agnostic token layout for a prefixed user question.

    Indices are into the *user-question text token stream* (prefix + question),
    not the full chat/multimodal sequence. Use ``map_text_span_to_sequence`` to
    lift into processor ``input_ids`` / expanded decoder positions.
    """

    prefix_token_ids: Tuple[int, ...]
    question_token_ids: Tuple[int, ...]
    full_token_ids: Tuple[int, ...]
    clause_start: int  # inclusive; 0 when prefix empty
    clause_end: int  # exclusive; == start when prefix empty
    question_start: int
    question_end: int

    @property
    def clause_token_ids(self) -> Tuple[int, ...]:
        return self.full_token_ids[self.clause_start : self.clause_end]


def _tokenizer_of(wrapper_or_tok):
    if hasattr(wrapper_or_tok, "encode") and hasattr(wrapper_or_tok, "decode"):
        return wrapper_or_tok
    tok = getattr(wrapper_or_tok, "tokenizer", None)
    if tok is None:
        proc = getattr(wrapper_or_tok, "processor", None)
        tok = getattr(proc, "tokenizer", None) if proc is not None else None
    if tok is not None:
        return tok
    # Test doubles / callables that implement HF tokenizer __call__
    if callable(wrapper_or_tok) and not hasattr(wrapper_or_tok, "forward_vl"):
        return wrapper_or_tok
    raise AttributeError(
        f"No tokenizer on {type(wrapper_or_tok).__name__}"
    )


def encode_no_special(tokenizer, text: str) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def assert_tokenization_invariant(
    tokenizer,
    *,
    prefix: str,
    question: str,
    join: str = "",
) -> PromptSpanLayout:
    """Derive clause/question spans from ``tok(prefix+join+question)``.

    Ideal case: ``tok(question)`` is an exact suffix of the full tokenization.
    Qwen (and similar BBPE tokenizers) may merge the first question token with
    the end of the prefix, so the full string is one token shorter than
    ``tok(prefix)+tok(question)``. We allow **at most one** unmatched token at
    the left edge of the question: the longest trailing match between
    ``question_ids`` and ``full_ids`` must cover ``len(question_ids)`` or
    ``len(question_ids)-1``. The unmatched/merged boundary token is assigned to
    the **clause** span (blocked keys), so the shared question interior stays
    unblocked.
    """
    tok = _tokenizer_of(tokenizer)
    prefix_full = f"{prefix}{join}" if prefix or join else ""
    full_text = f"{prefix_full}{question}"

    prefix_ids = encode_no_special(tok, prefix_full) if prefix_full else []
    question_ids = encode_no_special(tok, question)
    full_ids = encode_no_special(tok, full_text)

    if not question_ids:
        raise AssertionError(f"Empty question tokenization for {question!r}")
    if not full_ids:
        raise AssertionError(f"Empty full tokenization for {full_text!r}")

    matched = 0
    max_k = min(len(question_ids), len(full_ids))
    for k in range(1, max_k + 1):
        if full_ids[-k:] == question_ids[-k:]:
            matched = k
    min_required = len(question_ids) if not prefix_full else max(1, len(question_ids) - 1)
    if matched < min_required:
        raise AssertionError(
            "Tokenization invariant violated: question tokens do not form a "
            f"near-exact suffix of the prefixed prompt (matched={matched}, "
            f"need>={min_required}).\n"
            f"  prefix={prefix_full!r}\n"
            f"  question={question!r}\n"
            f"  len(prefix_ids)={len(prefix_ids)} len(question_ids)="
            f"{len(question_ids)} len(full_ids)={len(full_ids)}\n"
            f"  question_ids={question_ids}\n"
            f"  full_suffix={full_ids[-len(question_ids):] if question_ids else []}\n"
            f"  full_ids={full_ids}"
        )

    # Question span in full_ids is the matched trailing tokens. Any merged
    # boundary token sits in the clause span immediately before.
    question_start = len(full_ids) - matched
    question_end = len(full_ids)
    clause_start = 0
    clause_end = question_start
    return PromptSpanLayout(
        prefix_token_ids=tuple(prefix_ids),
        question_token_ids=tuple(question_ids),
        full_token_ids=tuple(full_ids),
        clause_start=clause_start,
        clause_end=clause_end,
        question_start=question_start,
        question_end=question_end,
    )


def assert_neutral_question_match(
    tokenizer,
    *,
    neutral_question: str,
    prefixed_layout: PromptSpanLayout,
) -> None:
    """Question span token IDs must equal the neutral question tokenization."""
    tok = _tokenizer_of(tokenizer)
    neutral_ids = tuple(encode_no_special(tok, neutral_question))
    got = prefixed_layout.question_token_ids
    if got != neutral_ids:
        raise AssertionError(
            "Shared question span token IDs differ from neutral question.\n"
            f"  neutral_ids={neutral_ids}\n"
            f"  prefixed_question_ids={got}"
        )


def find_subsequence(
    haystack: Sequence[int], needle: Sequence[int]
) -> Optional[int]:
    """Return start index of ``needle`` in ``haystack``, or None."""
    if not needle:
        return 0
    n = len(needle)
    limit = len(haystack) - n + 1
    for i in range(max(limit, 0)):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return None


def expand_llava_positions(
    input_ids_1d: Sequence[int],
    *,
    image_token_id: int,
    num_image_tokens: int,
) -> List[int]:
    """Map each pre-expansion token index → first expanded index.

    Returns a list ``map_pre[i]`` = starting expanded index of pre-token ``i``.
    Length is ``len(input_ids)``; image placeholder maps to a span of
    ``num_image_tokens``.
    """
    mapping: List[int] = []
    expanded = 0
    for tok in input_ids_1d:
        mapping.append(expanded)
        if int(tok) == int(image_token_id):
            expanded += num_image_tokens
        else:
            expanded += 1
    return mapping


def map_text_span_to_sequence(
    input_ids: torch.Tensor,
    *,
    text_token_ids: Sequence[int],
    wrapper,
) -> Tuple[int, int]:
    """Locate ``text_token_ids`` in multimodal ``input_ids``; return [start, end).

    For LLaVA: if ``input_ids`` still contains a single image placeholder, map
    through expansion; if the processor already expanded to many image tokens
    (common with current HF LLaVA processors), search directly in ``input_ids``.
    For Qwen2-VL, search directly (vision pads already present).
    """
    row = input_ids[0] if input_ids.dim() == 2 else input_ids
    ids = [int(x) for x in row.tolist()]
    fam = type(wrapper).__name__

    if fam == "LLaVAWrapper":
        img_id = wrapper.image_token_id
        n_img = wrapper.num_image_tokens
        n_img_tok = sum(1 for t in ids if t == img_id)
        pre_start = find_subsequence(ids, text_token_ids)
        if pre_start is None:
            raise AssertionError(
                "Could not find text token span in LLaVA input_ids "
                f"(len_ids={len(ids)}, len_span={len(text_token_ids)}, "
                f"n_image_token_ids={n_img_tok})"
            )
        pre_end = pre_start + len(text_token_ids)
        # Already-expanded processor output: indices are final decoder indices.
        if n_img_tok > 1:
            return int(pre_start), int(pre_end)
        # Single placeholder → expand
        if img_id in ids[pre_start:pre_end]:
            raise AssertionError(
                "Matched text span overlaps the image token placeholder"
            )
        mapping = expand_llava_positions(
            ids, image_token_id=img_id, num_image_tokens=n_img
        )
        if pre_end < len(ids):
            exp_end = mapping[pre_end]
        else:
            last = mapping[pre_end - 1]
            last_tok = ids[pre_end - 1]
            exp_end = last + (n_img if last_tok == img_id else 1)
        return int(mapping[pre_start]), int(exp_end)

    start = find_subsequence(ids, text_token_ids)
    if start is None:
        raise AssertionError(
            "Could not find text token span in input_ids "
            f"(wrapper={fam}, len_ids={len(ids)}, len_span={len(text_token_ids)})"
        )
    return int(start), int(start + len(text_token_ids))


def resolve_clause_and_question_spans(
    wrapper,
    input_ids: torch.Tensor,
    layout: PromptSpanLayout,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return ((clause_start, clause_end), (question_start, question_end)) in seq space.

    Locates the matched question suffix (``layout.full_token_ids[question_start:]``,
    which equals the longest trailing match to ``tok(question)``) in the
    multimodal sequence, then takes the immediately preceding ``clause_len``
    tokens as the clause span. Interior clause tokens (all but possibly the
    leftmost) must match ``layout.clause_token_ids[1:]`` when already in final
    sequence index space.
    """
    q_toks = layout.full_token_ids[layout.question_start : layout.question_end]
    c_len = layout.clause_end - layout.clause_start
    if not q_toks:
        raise AssertionError("Empty question span in layout")

    q_span = map_text_span_to_sequence(
        input_ids, text_token_ids=q_toks, wrapper=wrapper
    )
    clause_span = (q_span[0] - c_len, q_span[0])
    if clause_span[0] < 0:
        raise AssertionError(
            f"Clause length {c_len} overruns before question span {q_span}"
        )
    if c_len > 1:
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        ids = [int(x) for x in row.tolist()]
        fam = type(wrapper).__name__
        compare_direct = True
        if fam == "LLaVAWrapper":
            img_id = wrapper.image_token_id
            n_img_tok = sum(1 for t in ids if t == img_id)
            compare_direct = n_img_tok > 1
        if compare_direct:
            got = tuple(ids[clause_span[0] : clause_span[1]])
            exp = layout.clause_token_ids
            if got[1:] != exp[1:]:
                raise AssertionError(
                    "Clause interior tokens before question do not match layout.\n"
                    f"  got={got}\n  expected={exp}"
                )
    return clause_span, q_span


def layer_windows(
    num_layers: int, *, width: int = 10, stride: int = 5
) -> List[Tuple[int, int]]:
    """Overlapping inclusive layer windows; final window right-aligned to top.

    Returns list of (start, end_inclusive) pairs.
    """
    if num_layers <= 0:
        return []
    windows: List[Tuple[int, int]] = []
    start = 0
    while start + width - 1 < num_layers - 1:
        windows.append((start, start + width - 1))
        start += stride
        if start >= num_layers:
            break
    last = (max(0, num_layers - width), num_layers - 1)
    if not windows or windows[-1] != last:
        windows.append(last)
    # Deduplicate while preserving order
    out: List[Tuple[int, int]] = []
    seen = set()
    for w in windows:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out
