# Smoke-test blocker: stage1b `KeyError: a_side`

**Cause:** Stage-1 model replies returned relation options as `{a,b,type,verdict,reason}` and dropped stage-0 geometry (`a_side`, `gap_frac`, …). Allocator required `a_side`.

**Fix:**
1. Stage 1 now **merges** model verdicts onto original stage-0 options (relations / counting / distractors).
2. Stage 1b also backfills `a_side` from `relation_options` when missing.
3. Existing 9 stage-1 passes were repaired on disk (no re-API); `stage1b_allocate.py` wrote **9** allocations.

## Resume from stage 1b (do not re-run stage 1)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

python data_scripts/vti_demos_v2/render_review.py --stage 1b --open

python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 2 --open

python data_scripts/vti_demos_v2/stage3_make_variants.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 3 --open

python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 4 --open

python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 10
python data_scripts/vti_demos_v2/render_review.py --stage final --open
```
