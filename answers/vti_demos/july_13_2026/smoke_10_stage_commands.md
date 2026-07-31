# Smoke test: 10 examples through demos_v2.1 stages

Your stage0 pool is **1000** candidates in the v2.1 option-set schema. Live `stage1_verified.jsonl` still has **240 v2.0 rows**, so **wipe stage1–5 first** or resume-by-id will skip/mix schemas.

v2.0 finals stay safe in `data/vti/v2_poc_2026-07-12/`.

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

# Keep stage0 (1000) + cooccurrence; wipe downstream
rm -f data/vti/v2/stage1_*.jsonl \
      data/vti/v2/stage1_summary.json \
      data/vti/v2/stage1b_*.jsonl \
      data/vti/v2/stage1b_summary.json \
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

# Optional: peek at first 10 stage0 candidates
python data_scripts/vti_demos_v2/render_review.py --stage 0 --limit 10 --open

# Stage 1 (vision, sonnet) — first 10 of stage0
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 1 --open

# Stage 1b (CPU allocator) — allocates all current stage1 passes
python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/render_review.py --stage 1b --open
# also: cat data/vti/v2/stage1b_summary.json

# Stage 2 (vision, opus)
python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 2 --open

# Stage 3 (text haiku polish + deterministic edits)
python data_scripts/vti_demos_v2/stage3_make_variants.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 3 --open

# Stage 4 (vision, sonnet)
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 4 --open

# Stage 5 assemble whatever passed stage 4
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 10
python data_scripts/vti_demos_v2/render_review.py --stage final --open
```

HTML lands under `data/vti/v2/_review/` (`stage0_review.html`, `stage1_review.html`, `stage1b_review.html`, …).

**Notes**
- `--limit 10` on stage1 takes the **first 10** stage0 rows (by file order / score). Later stages limit their own inputs the same way.
- If stage1 rejects some of the 10, stages 2–4 will see fewer than 10; that is expected for a smoke.
- For mock instead of paid: add `--provider mock` on stages 1, 2, 3, 4.
