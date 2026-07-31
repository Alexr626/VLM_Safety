# S1 HTML yn-badge bug (leading clause) — metrics not poisoned

## Bug
HTML renderer concatenated `[prompt]\n{leading clause}…\n[response]\n{model}` into the display body. `review_lib._yn_badge` ran `normalize_yes_no` on that **whole string**, so clauses like “answer is probably **no**” flipped the chip even when the model said **Yes**.

## Dump metrics (clean)
`manifest.jsonl` stores `response` separately from the augment prompt. `parsed_outcome`, `score_*`, answer mass come from capture-time scoring on the model output / logits — **not** from the HTML string.

Spot-check (S1 B5, `amber_disc_07689`, `tentative_toward_no`): response starts with Yes; `parsed_outcome=yes`; matches first-token pred. Across S1: 108 `tentative_toward_no` rows whose response starts with Yes, **108/108** have `parsed_outcome=yes`.

## Fix
- `ResponseSet.add(..., badge_text=)` in `review_lib.py`
- S1 renderer passes `badge_text=parsed_outcome or response` while keeping `[prompt]/[response]` in the body
- Regenerated all galleries under `evaluation/results/2026-07-16/_samples/llava-1.5-7b-hf/amber/`

Hard-refresh the HTML in the browser (files replaced in place).
