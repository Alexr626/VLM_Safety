# Resume smoke test from stage 1b (no `--open`)

Allocation already written (9 rows). HTML reviews land under `data/vti/v2/_review/` — download those files to your laptop to view.

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

python data_scripts/vti_demos_v2/render_review.py --stage 1b

python data_scripts/vti_demos_v2/stage2_write_truthful.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 2

python data_scripts/vti_demos_v2/stage3_make_variants.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 3

python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --limit 10
python data_scripts/vti_demos_v2/render_review.py --stage 4

python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 10
python data_scripts/vti_demos_v2/render_review.py --stage final
```

**Download paths (WinSCP / scp):**
- `data/vti/v2/_review/stage1b_review.html`
- `data/vti/v2/_review/stage2_review.html`
- `data/vti/v2/_review/stage3_review.html`
- `data/vti/v2/_review/stage4_review.html`
- `data/vti/v2/_review/final_review.html`
