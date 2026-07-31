# Stage-2 mass reject: `spans.relation must equal true relation phrase`

## What happened

Spans were **not empty**. Opus returned a **full S4 sentence** as `spans.relation.text`, e.g.:

- returned: `"The bowl is to the left of the cup."`
- required: `"left"` (`relation.true_phrase`)

The validator compares those literally → reject. Reject JSONL does not store the attempted caption/spans, so the review HTML looks empty for failures.

All 9 smoke rejects showed the same pattern (`left` / `above` / `on top of` / `right next to` / `far away from` buried inside a full sentence).

## Fix applied

1. **Prompt:** explicit “relation span = short phrase ONLY” + user message lists exact required span strings.
2. **Deterministic coerce** in `stage2_write_truthful.py`: if the required phrase is a substring of the model’s relation/counting/attribute span, rewrite the span text to that phrase before validation.

## Re-run stage 2 only (keep stage1 / 1b)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

rm -f data/vti/v2/stage2_*.jsonl data/vti/v2/stage2_summary.json \
      data/vti/v2/calls/stage2.jsonl

python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 2
```

Then continue 3→4→5 as before. Offline revalidation of the previous call logs: several captions may still fail *other* structural checks after coerce (e.g. category plural wording); those need a fresh model attempt under the clearer prompt.
