# Stage 4 `KeyError: false_phrase`

## Symptom

```text
File ".../prompts.py", line 208, in stage4_statement_specs
    "text": f"The {rel['a']} is {rel['false_phrase']} the {rel['b']}."},
KeyError: 'false_phrase'
```

Stage 4 never wrote verdicts → Stage 5 assembled 0 finals.

## Cause

Offline Stage-3 recovery (after the validator fix) wrote relation anchors as:

```json
{"a": "...", "b": "...", "type": "...", "true": "left", "false": "right"}
```

Canonical `stage3_make_variants.py` also sets `false_phrase` (same value as `false`). Stage 4 only read `false_phrase`.

## Fix

1. `prompts.py`: resolve false relation via `false_phrase` or legacy `false`; build type-aware claim text (e.g. horizontal → "to the right of").
2. Backfilled `data/vti/v2/stage3_variants.jsonl` so each `anchors.relation` has `false_phrase`.

Re-run:

```bash
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 4
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 10
python data_scripts/vti_demos_v2/render_review.py --stage final
```
