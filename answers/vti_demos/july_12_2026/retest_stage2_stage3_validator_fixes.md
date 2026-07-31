# Commands to retest stage-2 plural + stage-3 existence fixes

Keep the current **300** `stage0_candidates.jsonl`. Wipe stage 1–5 so resume-by-id does not skip the old pilot rejects.

Note: stage-0 has **116** `person` counting anchors. In file order the first is at index 2 (`000000072961`), so `--limit 5` already includes two `person` rows and will exercise both plural matching and deterministic existence.

## 0) Free unit tests (no API)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

python -m pytest tests/test_demos_v2_validators.py -q
# expect: 19 passed
```

## 1) Clear downstream artifacts (keep stage0 + cooccurrence)

```bash
rm -f data/vti/v2/stage1_*.jsonl \
      data/vti/v2/stage1_summary.json \
      data/vti/v2/stage2_*.jsonl \
      data/vti/v2/stage2_summary.json \
      data/vti/v2/stage3_*.jsonl \
      data/vti/v2/stage3_summary.json \
      data/vti/v2/stage4_*.jsonl \
      data/vti/v2/stage4_summary.json \
      data/vti/v2/stage5_summary.json \
      data/vti/v2/calls/stage{1,2,3,4}.jsonl \
      data/vti/demos_v2.jsonl \
      data/vti/demos_v2_*.jsonl

wc -l data/vti/v2/stage0_candidates.jsonl   # expect 300
```

## 2) Small paid smoke (recommended first) — `--limit 5`

Uses config defaults: stage1 sonnet-4-6, stage2 opus-4-8, stage3 haiku-4-5 (existence usually deterministic → little/no stage3 API), stage4 sonnet-4-6.

```bash
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 1 --open

python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 2 --open
# watch stage2_rejected.jsonl: should NOT reject for
#   "counting category '…' not found" when a natural plural is present

python data_scripts/vti_demos_v2/stage3_make_variants.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 3 --open
# watch stage3_variants.jsonl: existence_source should be "deterministic"
# and existence rejects should NOT cite "exactly 1 edit region" for S1 list rewrites

python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 4 --open

python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 5
python data_scripts/vti_demos_v2/render_review.py --stage final --open
```

Quick checks after stage 2 / 3:

```bash
# Stage 2 reject reasons (if any)
python -c "
import json
from pathlib import Path
p=Path('data/vti/v2/stage2_rejected.jsonl')
if not p.exists(): print('no stage2 rejects'); raise SystemExit
for line in p.read_text().splitlines():
    if not line.strip(): continue
    r=json.loads(line)
    print(r.get('id'), r.get('errors') or r.get('reject_reasons') or r)
"

# Stage 3 existence source + reject reasons
python -c "
import json
from pathlib import Path
for name in ['stage3_variants.jsonl','stage3_rejected.jsonl']:
    p=Path('data/vti/v2')/name
    print('===', name, '===')
    if not p.exists():
        print('(missing)'); continue
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        print(r.get('id'), 'existence_source=', r.get('existence_source'),
              'errors=', r.get('errors') or r.get('reject_reasons'))
"
```

## 3) Wider paid pass (optional, after smoke looks good)

Same wipe as §1 if you want a clean funnel, then raise `--limit` (e.g. 20 or omit limit to walk the full 300 through each stage — expensive).

```bash
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --limit 20
python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 20
python data_scripts/vti_demos_v2/stage3_make_variants.py --limit 20
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 20
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 20
```

Omit `--limit` only when you intentionally want the full stage-0 pool (token cost scales with stage-1/2/4 vision calls).
