# Paid end-to-end run for the 5 stage-0 candidates (real MLLMs)

Defaults from `config.py`: stage1 sonnet-4-6, stage2 opus-4-8, stage3 haiku-4-5, stage4 sonnet-4-6. Keys load from repo-root `.env`.

**Important:** resume-by-id skips IDs already in stage outputs. The mock dry-run wrote `stage1_model: mock` for these 5 ids. Also `stage2_captions.jsonl` / `stage3_variants.jsonl` appear beautified (multi-line), so clear stages 1–5 before re-running.

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

# Keep stage0 (5 candidates) + cooccurrence.json; wipe mock/downstream artifacts
rm -f data/vti/v2/stage1_*.jsonl \
      data/vti/v2/stage2_*.jsonl \
      data/vti/v2/stage3_*.jsonl \
      data/vti/v2/stage4_*.jsonl \
      data/vti/v2/stage5_summary.json \
      data/vti/v2/calls/stage{1,2,3,4}.jsonl \
      data/vti/demos_v2.jsonl \
      data/vti/demos_v2_*.jsonl

# Confirm stage0 still has the 5 ids
wc -l data/vti/v2/stage0_candidates.jsonl

# Stages 1–4: omit --provider mock → config defaults + real API calls
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 1 --open   # gate 1

python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 2 --open   # gate 2 (most important)

python data_scripts/vti_demos_v2/stage3_make_variants.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 3 --open   # gate 3

python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 4 --open   # gate 4

python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 5
python data_scripts/vti_demos_v2/export_vti_flat.py --dimension counting
python data_scripts/vti_demos_v2/render_review.py --stage final --open
```

Override a stage model if needed, e.g.:

```bash
python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 5 \
  --provider anthropic:claude-opus-4-8
```

Sanity check after stage 1 that calls were real:

```bash
python -c "import json; from pathlib import Path
r=json.loads(Path('data/vti/v2/calls/stage1.jsonl').read_text().splitlines()[0])
print(r['model'], r['input_tokens'], r['output_tokens'])"
# expect something like claude-sonnet-4-6 and non-zero tokens — not mock / 0
```
