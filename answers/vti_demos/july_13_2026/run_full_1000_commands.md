# Full 1000-candidate demos_v2.1 end-to-end commands

Stage-0 already has **1000** candidates at `data/vti/v2/stage0_candidates.jsonl`.
`run_full.sh` keeps Stage 0, wipes Stages 1–5 + call logs, then runs Stage 1→5.
Use `N_CANDIDATES=1000` so Stage 5 assembles up to all Stage-4 passes (not the default 300).

## Recommended (fresh downstream, keep Stage 0)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
conda activate vlm_hallucination_mitigation

mkdir -p logs
N_CANDIDATES=1000 nohup bash data_scripts/vti_demos_v2/run_full.sh \
  > logs/demos_v2_1_full1000_$(date +%Y-%m-%d).log 2>&1 &
echo $!   # note PID
tail -f logs/demos_v2_1_full1000_*.log
```

Or foreground (same wipe + pipeline):

```bash
N_CANDIDATES=1000 bash data_scripts/vti_demos_v2/run_full.sh
```

## After it finishes — reviews / exports

```bash
python data_scripts/vti_demos_v2/render_review.py --stage 1
python data_scripts/vti_demos_v2/render_review.py --stage 1b
python data_scripts/vti_demos_v2/render_review.py --stage 2 --limit 50
python data_scripts/vti_demos_v2/render_review.py --stage 3 --limit 50
python data_scripts/vti_demos_v2/render_review.py --stage 4 --limit 50
python data_scripts/vti_demos_v2/render_review.py --stage final

# optional flat exports
python data_scripts/vti_demos_v2/export_vti_flat.py --dimension all
python data_scripts/vti_demos_v2/analyze_rejections.py
```

HTML lands under `data/vti/v2/_review/`. Final pack: `data/vti/demos_v2.jsonl` (+ `stage5_summary.json`).

## Notes

- Smoke artifacts (9 Stage-1 passes, etc.) are **deleted** by `run_full.sh`; that is intentional for a clean full run.
- Stages resume by id if you re-run a single stage script without wiping; do **not** mix smoke leftovers with a full Stage-1 pool without wiping Stage 1b onward.
- Expect long wall time + Anthropic cost (Stage 1+4 ≈ 1000 vision calls each on Sonnet; Stage 2 Opus on Stage-1 passes; Stage 3 Haiku grammar polish).
- If Stage 5 reports shortfall, top up Stage 0 then re-run only the new ids (or wipe and restart downstream).
