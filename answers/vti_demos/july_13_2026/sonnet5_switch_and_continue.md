# Switch Stages 1/4 to Sonnet 5 + continue full-1000

## Done

- Killed `run_full.sh` / `stage1_verify_anchors.py`.
- Config: `STAGE1_PROVIDER` / `STAGE4_PROVIDER` → `anthropic:claude-sonnet-5`.
- `MLLMClient`: `thinking: {type: "disabled"}` via `extra_body` for Sonnet 5 (adaptive thinking is on by default otherwise).
- Fixed Stage 3 `NameError: relation is not defined` (`relation = rec["relation"]`).
- Smoke: Stage 1 on 2 pending IDs (Sonnet 5); Stage 2–4 on 1 sample (Stage 4 = Sonnet 5). Stage 4 passed.

**Do not** re-run `run_full.sh` — it wipes Stages 1–5.

## Continue with nohup (resume-safe)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
conda activate vlm_hallucination_mitigation
mkdir -p logs

nohup bash -lc '
set -euo pipefail
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

# 1) Finish Stage 1 on remaining Stage-0 (Sonnet 5)
python data_scripts/vti_demos_v2/stage1_verify_anchors.py

# 2) Re-allocate from full Stage-1 pass set
python data_scripts/vti_demos_v2/stage1b_allocate.py

# 3–5) Resume Stage 2–4 (skip ids already done in smoke)
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py

# 6) Assemble up to all Stage-4 passes
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 1000
' > logs/demos_v2_1_continue_sonnet5_$(date +%Y-%m-%d).log 2>&1 &

echo $!
tail -f logs/demos_v2_1_continue_sonnet5_*.log
```

Providers after switch: Stage1/4 Sonnet 5, Stage2 Opus 4.8, Stage3 Haiku 4.5.
