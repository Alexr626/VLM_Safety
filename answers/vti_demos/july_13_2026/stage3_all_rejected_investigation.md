# Stage 3 all-rejected investigation (demos_v2.1 smoke)

## Why HTML looked empty

`render_review.py --stage 3` originally rendered **only** `stage3_variants.jsonl` (passes). Rejects lived in `stage3_rejected.jsonl` with `reject_reason` / `errors`, but were never shown — so a 0-pass / all-reject run produced a nearly blank gallery with no failure clues.

**Fix:** Stage 3 review now concatenates passes + rejects and surfaces `reject_reason` + `errors`.

## Root causes (smoke, 9 Stage-2 captions)

| Count | `reject_reason` | Cause |
|------:|-----------------|--------|
| 7 | `variant_validation_failed` | `validate_record_variants` treated distractor presence in the **`all`** (combined) caption as a leak. Combined *must* include the existence distractor in S1 — same as the existence variant. |
| 2 | `deterministic_diff_failed` | Token-diff attribute check on multi-word values with a shared prefix (e.g. `turned off` → `turned on`) only saw `off.` ↔ `on.` because of shared-prefix + period attachment, so the pair failed the strict one-region replace rule. |

## Code fixes

1. **Skip `all` in distractor-leak check** in `validate_record_variants` (existence + combined may contain the distractor; attribute/counting/relation must not).
2. **Sentence-scoped `_validate_attribute_pair`** — require S2-only change and that true/false attribute phrases appear/disappear correctly, instead of raw token opcodes for attributes (handles shared prefixes).
3. **Stage 3 HTML** includes rejects + `errors`.

## Recovery / current smoke status

After validator fixes, Stage 3 was rebuilt offline from `stage2_captions.jsonl` + `calls/stage3.jsonl` grammar-polish logs (API blocked in sandbox):

- **9 pass / 0 reject** → `data/vti/v2/stage3_variants.jsonl`
- Review: `data/vti/v2/_review/stage3_review.html`

Continue smoke with Stage 4+:

```bash
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 4
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 10
python data_scripts/vti_demos_v2/render_review.py --stage final
```

Download HTML from `data/vti/v2/_review/`.
