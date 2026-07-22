"""Unit tests for attention knockout masks and prompt-span invariants."""

from __future__ import annotations

import torch

from src.attention_knockout import build_knockout_additive_mask, query_positions_for_scope
from src.prompt_spans import (
    assert_tokenization_invariant,
    expand_llava_positions,
    find_subsequence,
    layer_windows,
)


class _FakeTok:
    """Character tokenizer with stable ids (no cross-boundary merges)."""

    def __init__(self):
        self._vocab = {}

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        return {"input_ids": [self._id(ch) for ch in text]}

    def _id(self, s: str) -> int:
        if s not in self._vocab:
            self._vocab[s] = len(self._vocab) + 1
        return self._vocab[s]


def test_build_knockout_mask_blocks_pairs():
    mask = build_knockout_additive_mask(
        batch_size=1,
        seq_len=6,
        query_positions=[5, 4],
        key_positions=[1, 2],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 6, 6)
    assert mask[0, 0, 5, 1] == torch.finfo(torch.float32).min
    assert mask[0, 0, 4, 2] == torch.finfo(torch.float32).min
    assert mask[0, 0, 5, 0] == 0
    assert mask[0, 0, 3, 1] == 0

    # Softmax over keys for query 5: blocked keys get zero mass
    scores = torch.zeros(6)
    scores = scores + mask[0, 0, 5]
    probs = torch.softmax(scores, dim=-1)
    assert float(probs[1]) == 0.0
    assert float(probs[2]) == 0.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_query_scopes():
    assert query_positions_for_scope(
        "block_last_token_reading", clause_end=3, seq_len=10
    ) == [9]
    assert query_positions_for_scope(
        "block_all_downstream_reading", clause_end=3, seq_len=7
    ) == [3, 4, 5, 6]


def test_layer_windows_match_plan():
    llava = layer_windows(32, width=10, stride=5)
    assert llava == [(0, 9), (5, 14), (10, 19), (15, 24), (20, 29), (22, 31)]
    qwen = layer_windows(28, width=10, stride=5)
    assert qwen == [(0, 9), (5, 14), (10, 19), (15, 24), (18, 27)]


def test_tokenization_invariant_ok_and_merge_fails():
    tok = _FakeTok()
    layout = assert_tokenization_invariant(
        tok,
        prefix="hello ",
        question="world?",
    )
    assert layout.clause_end == layout.question_start
    assert layout.question_token_ids
    # Suffix property: full ends with question
    assert layout.full_token_ids[-len(layout.question_token_ids) :] == layout.question_token_ids

    class BadTok:
        def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
            if text == "ab":
                return {"input_ids": [9, 9]}  # not ending with question ids
            if text == "a":
                return {"input_ids": [1]}
            if text == "b":
                return {"input_ids": [2]}
            return {"input_ids": [9]}

    try:
        assert_tokenization_invariant(BadTok(), prefix="a", question="b")
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_find_subsequence_and_llava_expand():
    assert find_subsequence([1, 2, 3, 4], [2, 3]) == 1
    assert find_subsequence([1, 2, 3], [9]) is None
    mapping = expand_llava_positions(
        [10, 99, 11, 12], image_token_id=99, num_image_tokens=4
    )
    # pre indices: 0→0, 1→1, 2→5, 3→6
    assert mapping == [0, 1, 5, 6]
